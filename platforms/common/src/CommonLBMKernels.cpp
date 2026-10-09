/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "CommonLBMKernels.h"
#include "CommonLBMKernelSources.h"
#include "internal/D3Q19.h"
#include "internal/LBMBoundaries.h"
#include "internal/LBMDecomposition.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/IntegrationUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/OSRngSeed.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

using namespace LBMPlugin;
using namespace OpenMM;
using namespace std;

/**
 * The sort keys of the coupled particles, node*numCoupled + i: 64-bit integers, sorted by their value.
 */
class CouplingSortTrait : public ComputeSortImpl::SortTrait {
    int getDataSize() const {return 8;}
    int getKeySize() const {return 8;}
    const char* getDataType() const {return "mm_long";}
    const char* getKeyType() const {return "mm_long";}
    const char* getMinKey() const {return "0";}
    const char* getMaxKey() const {return "0x7FFFFFFFFFFFFFFF";}
    const char* getMaxValue() const {return "0x7FFFFFFFFFFFFFFF";}
    const char* getSortKey() const {return "value";}
};

/**
 * With the Centered drag scheme, the work of the force at the end of every force evaluation, when OpenMM has
 * computed all the other forces (LBMForce is the last force of the System, so this post-computation comes after
 * those of the other forces).  It respects the force groups as execute() does.  When the neighbor list has
 * overflowed, OpenMM has already marked the evaluation as invalid and will repeat it: the other forces are then
 * incomplete, and the post-computation does nothing, so that the lattice step is done in the repeated evaluation
 * with the complete forces.
 */
class CommonCalcLBMForceKernel::CenteredDragPostComputation : public ComputeContext::ForcePostComputation {
public:
    CenteredDragPostComputation(CommonCalcLBMForceKernel& owner) : owner(owner) {
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<owner.forceGroup)) == 0 || !owner.cc.getForcesValid())
            return 0.0;
        return owner.evaluate(owner.cc.getStepCount(), includeForces);
    }
private:
    CommonCalcLBMForceKernel& owner;
};

/**
 * Download an array of floats or doubles into a vector of doubles.  ComputeArray::download() converts
 * types only from OpenMM 8.4, and the plugin supports OpenMM 8.3.
 */
static void downloadAsDouble(const ComputeArray& array, vector<double>& data) {
    data.resize(array.getSize());
    if (array.getElementSize() == sizeof(double))
        array.download(data.data());
    else {
        vector<float> values(array.getSize());
        array.download(values.data());
        for (size_t i = 0; i < values.size(); i++)
            data[i] = values[i];
    }
}

int CommonCalcLBMForceKernel::globalNode(int n) const {
    int i = n%localCount[0], j = (n/localCount[0])%localCount[1], k = n/(localCount[0]*localCount[1]);
    return (localStart[0]+i) + lattice.nx*((localStart[1]+j) + lattice.ny*(localStart[2]+k));
}

int CommonCalcLBMForceKernel::localNode(int node) const {
    int i = node%lattice.nx - localStart[0], j = (node/lattice.nx)%lattice.ny - localStart[1];
    int k = node/(lattice.nx*lattice.ny) - localStart[2];
    return i + localCount[0]*(j + localCount[1]*k);
}

int CommonCalcLBMForceKernel::storageIndex(int n) const {
    int i = n%localCount[0], j = (n/localCount[0])%localCount[1], k = n/(localCount[0]*localCount[1]);
    return (i+pad[0]) + (localCount[0]+2*pad[0])*((j+pad[1]) + (localCount[1]+2*pad[1])*(k+pad[2]));
}

void CommonCalcLBMForceKernel::initialize(const System& system, const LBMForce& force, const LBMLatticeParameters& lattice) {
    ContextSelector selector(cc);
    if (cc.getNumContexts() > 1)
        throw OpenMMException("LBMForce does not support running on multiple devices");
    if (lattice.interpolationStencil != LBMForce::NearestNode)
        throw OpenMMException("LBMForce: on the " + getPlatform().getName() + " platform the interpolation stencils "
                "other than NearestNode are not available yet (in development for version 0.5.0; Reference platform only)");
    decomposition = LBMDecomposition(lattice.nx, lattice.ny, lattice.nz, lattice.procs);
    bool decomposed = decomposition.isDecomposed();
    if (decomposed) {
        // The same collective check of the platform and precision as on the Reference platform, so that ranks on
        // different platforms all stop with the same error.  The parts not available yet with the decomposition on
        // these platforms depend only on the System and the force, which are the same on every rank, so all the
        // ranks stop together.
        string precision = (cc.getUseDoublePrecision() ? "double" : cc.getUseMixedPrecision() ? "mixed" : "single");
        decomposition.requireSameOnAllRanks(getPlatform().getName() + " platform in " + precision + " precision", "platform and precision");

        // The copies of the particles stay identical only if every rank computes the same forces of OpenMM on them;
        // the CUDA and HIP platforms do so with the property DeterministicForces.  The property could differ between
        // the ranks, so the check is collective.
        decomposition.throwIfAnyError(!lattice.particles.empty() && !deterministicForces ? "LBMForce: with the domain "
                "decomposition and coupled particles the platform must compute the forces deterministically, so that the "
                "copies of the particles stay identical on every rank: set the platform property DeterministicForces to "
                "true" : "");
    }
    this->lattice = lattice;
    forceGroup = force.getForceGroup();
    bool centered = (lattice.dragScheme == LBMForce::Centered);

    // The block of the rank, the whole lattice without the decomposition.  Along the divided axes the arrays hold a
    // layer of halo nodes on each side (kernels/lbmFluid.cc); stored is their extent.

    int numNodes = lattice.getNumNodes();
    int size[3] = {lattice.nx, lattice.ny, lattice.nz}, stored[3];
    decomposition.getLocalDomain(localStart, localCount);
    for (int a = 0; a < 3; a++) {
        pad[a] = (localCount[a] < size[a] ? 1 : 0);
        stored[a] = localCount[a]+2*pad[a];
    }
    numLocal = localCount[0]*localCount[1]*localCount[2];
    numStored = stored[0]*stored[1]*stored[2];

    // The global node next to the node with coordinates index along c (wrapped around), and whether the link crosses
    // an open face; the storage index of the node next to the node of the block with local coordinates index.
    auto globalNeighbor = [&] (const int index[3], const int c[3], bool& crossesFace) {
        int target[3];
        crossesFace = false;
        for (int a = 0; a < 3; a++) {
            target[a] = index[a] + c[a];
            crossesFace = crossesFace || (lattice.isOpenAxis(a) && (target[a] < 0 || target[a] >= size[a]));
            target[a] = (target[a] + size[a])%size[a];
        }
        return target[0] + lattice.nx*(target[1] + lattice.ny*target[2]);
    };
    auto neighborStorage = [&] (const int local[3], const int c[3]) {
        int index[3];
        for (int a = 0; a < 3; a++)
            index[a] = (pad[a] ? local[a]+c[a]+1 : (local[a]+c[a]+localCount[a])%localCount[a]);
        return index[0] + stored[0]*(index[1] + stored[1]*index[2]);
    };
    auto localIndex = [&] (int node) {
        int i = node%lattice.nx - localStart[0], j = (node/lattice.nx)%lattice.ny - localStart[1];
        int k = node/(lattice.nx*lattice.ny) - localStart[2];
        return i + localCount[0]*(j + localCount[1]*k);
    };

    // The fluid is stored in the mixed type: double unless the platform runs in single precision.

    useDouble = (cc.getUseDoublePrecision() || cc.getUseMixedPrecision());
    int elementSize = (useDouble ? sizeof(double) : sizeof(float));
    blockSize = ComputeContext::ThreadBlockSize;
    numGroups = max(1, min(cc.getNumThreadBlocks(), (numLocal+blockSize-1)/blockSize));
    populations.initialize(cc, D3Q19::numVelocities*numStored, elementSize, "lbmPopulations");
    densityDeviation.initialize(cc, numStored, elementSize, "lbmDensityDeviation");
    momentum.initialize(cc, 3*numStored, elementSize, "lbmMomentum");
    piNeq.initialize(cc, 6*numStored, elementSize, "lbmPiNeq");
    partialSums.initialize(cc, 4*numGroups, elementSize, "lbmPartialSums");
    partialMax.initialize(cc, numGroups, elementSize, "lbmPartialMax");
    centerVelocity.initialize(cc, (decomposed ? 4 : 3), elementSize, "lbmCenterVelocity");

    // The fluid starts at equilibrium, with lattice density 1 and the initial velocity.  As on the Reference
    // platform, the populations are stored as deviations from the rest equilibrium, f_q - w_q.  The nodes of the
    // halo take the type of their node (those beyond an open face, which nothing reads, are fluid).

    int numSolidNodes = lattice.solidNodes.size();
    isFluidHost.assign(numNodes, 1);
    for (int node : lattice.solidNodes)
        isFluidHost[node] = 0;
    vector<int> isFluidStored = isFluidHost;
    if (decomposed) {
        isFluidStored.assign(numStored, 1);
        for (int c = 0; c < stored[2]; c++)
            for (int b = 0; b < stored[1]; b++)
                for (int a = 0; a < stored[0]; a++) {
                    int index[3] = {localStart[0]+a-pad[0], localStart[1]+b-pad[1], localStart[2]+c-pad[2]};
                    int zero[3] = {0, 0, 0};
                    bool beyondFace;
                    int node = globalNeighbor(index, zero, beyondFace);
                    if (!beyondFace)
                        isFluidStored[a + stored[0]*(b + stored[1]*c)] = isFluidHost[node];
                }
    }
    vector<double> f(D3Q19::numVelocities*numStored);
    double dfeq[D3Q19::numVelocities];
    D3Q19::equilibriumDeviation(0.0, lattice.initialVelocity[0], lattice.initialVelocity[1], lattice.initialVelocity[2], dfeq);
    for (int q = 0; q < D3Q19::numVelocities; q++)
        for (int node = 0; node < numStored; node++)
            f[q*numStored+node] = (isFluidStored[node] ? dfeq[q] : -D3Q19::w[q]);

    // Solid nodes hold no fluid: their populations start at zero, that is at the deviation -w_q.  With bounce-back
    // the part w of the populations gives the walls the same momentum in every step (the static pressure): it is
    // computed here once, in double precision, with the links from the solid nodes to the fluid nodes that do not
    // cross an open face.  The same links, seen from the fluid nodes next to the walls (wallNodes), are the bits of
    // wallLinks: along them the bounce-back returns the populations to the fluid; seen from the solid nodes, they are
    // the bits of solidLinks, for the momentum exchange.  With the decomposition a rank counts the links to its own
    // fluid nodes, and the solid node may lie in its halo; a solid node seen at two places of the halo (an axis divided
    // in two blocks of one node) has an entry for each.

    staticWallMomentum = Vec3();
    vector<int> solidSlotsHost, solidLinksHost;
    for (int node : lattice.solidNodes) {
        int index[3] = {node%lattice.nx, (node/lattice.nx)%lattice.ny, node/(lattice.nx*lattice.ny)};
        vector<pair<int, int> > entries;
        if (!decomposed)
            entries.push_back(make_pair(node, 0));
        for (int q = 1; q < D3Q19::numVelocities; q++) {
            int c[3] = {D3Q19::cx[q], D3Q19::cy[q], D3Q19::cz[q]};
            bool crossesFace;
            int target = globalNeighbor(index, c, crossesFace);
            if (crossesFace || !isFluidHost[target] || !decomposition.owns(target))
                continue;
            if (lattice.wallScheme == LBMForce::BounceBack)
                staticWallMomentum -= Vec3(c[0], c[1], c[2])*(2.0*D3Q19::w[q]);
            int slot = node;
            if (decomposed) {
                int t = localIndex(target);
                int local[3] = {t%localCount[0], (t/localCount[0])%localCount[1], t/(localCount[0]*localCount[1])};
                int back[3] = {-c[0], -c[1], -c[2]};
                slot = neighborStorage(local, back);
            }
            int e = 0;
            while (e < (int) entries.size() && entries[e].first != slot)
                e++;
            if (e == (int) entries.size())
                entries.push_back(make_pair(slot, 0));
            entries[e].second |= 1<<q;
        }
        for (auto& entry : entries) {
            solidSlotsHost.push_back(entry.first);
            solidLinksHost.push_back(entry.second);
        }
    }
    numSolidEntries = solidSlotsHost.size();

    // Boundary nodes (docs/theory.md, section 1, and internal/LBMBoundaries.h): with regularized walls the fluid
    // nodes next to the solid nodes, and the fluid nodes on the open faces.  They are fluid nodes like the others,
    // except that applyBoundaries rebuilds their unknown populations.  The part w of the populations that a wall node
    // sends into the solid nodes and receives from them gives the walls the same momentum in every step, -2 c_q w_q
    // for each solid direction q.  With the decomposition each rank keeps the boundary nodes of its block, and the
    // kernels see the local indices of the nodes.

    LBMBoundaries boundaries;
    boundaries.find(lattice, isFluidHost);
    vector<int> boundaryLocal, unknownLocal, solidLocal, kindAndFace;
    staticBoundaryMomentum = Vec3();
    for (int b = 0; b < (int) boundaries.nodes.size(); b++) {
        if (!decomposition.owns(boundaries.nodes[b]))
            continue;
        boundaryLocal.push_back(localIndex(boundaries.nodes[b]));
        unknownLocal.push_back(boundaries.unknown[b]);
        solidLocal.push_back(boundaries.solid[b]);
        kindAndFace.push_back(boundaries.kind[b] + 4*(boundaries.face[b] + 1));
        for (int q = 1; q < D3Q19::numVelocities; q++)
            if (boundaries.solid[b] & (1<<q))
                staticBoundaryMomentum -= Vec3(D3Q19::cx[q], D3Q19::cy[q], D3Q19::cz[q])*(2.0*D3Q19::w[q]);
    }
    int numBoundaryNodes = boundaryLocal.size();
    if (numBoundaryNodes > 0) {
        boundaryNodes.initialize<int>(cc, numBoundaryNodes, "lbmBoundaryNodes");
        boundaryNodes.upload(boundaryLocal);
        boundaryUnknown.initialize<int>(cc, numBoundaryNodes, "lbmBoundaryUnknown");
        boundaryUnknown.upload(unknownLocal);
        boundarySolid.initialize<int>(cc, numBoundaryNodes, "lbmBoundarySolid");
        boundarySolid.upload(solidLocal);
        boundaryKindAndFace.initialize<int>(cc, numBoundaryNodes, "lbmBoundaryKindAndFace");
        boundaryKindAndFace.upload(kindAndFace);
        boundaryExchange.initialize(cc, 3*numBoundaryNodes, elementSize, "lbmBoundaryExchange");
        boundaryExchange.upload(vector<double>(3*numBoundaryNodes, 0.0), true);
        faceParameters.initialize(cc, 24, elementSize, "lbmFaceParameters");
    }
    uploadPopulations(f);
    isFluid.initialize<int>(cc, numStored, "lbmIsFluid");
    isFluid.upload(isFluidStored);
    vector<int> wallNodesHost, wallLinksHost, wallNodesLocal, wallLinksLocal;
    LBMBoundaries::findWallLinks(lattice, isFluidHost, wallNodesHost, wallLinksHost);
    for (int b = 0; b < (int) wallNodesHost.size(); b++)
        if (decomposition.owns(wallNodesHost[b])) {
            wallNodesLocal.push_back(localIndex(wallNodesHost[b]));
            wallLinksLocal.push_back(wallLinksHost[b]);
        }
    int numWallNodes = wallNodesLocal.size();
    if (numWallNodes > 0) {
        wallNodes.initialize<int>(cc, numWallNodes, "lbmWallNodes");
        wallNodes.upload(wallNodesLocal);
        wallLinks.initialize<int>(cc, numWallNodes, "lbmWallLinks");
        wallLinks.upload(wallLinksLocal);
    }
    solidNodes.initialize<int>(cc, max(1, numSolidEntries), "lbmSolidNodes");
    solidLinks.initialize<int>(cc, max(1, numSolidEntries), "lbmSolidLinks");
    wallExchange.initialize(cc, 3*max(1, numSolidEntries), elementSize, "lbmWallExchange");
    if (numSolidEntries > 0) {
        solidNodes.upload(solidSlotsHost);
        solidLinks.upload(solidLinksHost);
    }
    wallExchange.upload(vector<double>(wallExchange.getSize(), 0.0), true);

    // The exchange of the populations with the domain decomposition (docs/theory.md, section 8).  A population that a
    // fluid node of the block pushes along c_q into a fluid node t of another rank, without crossing an open face, goes
    // to the owner of t, which writes it at the slot (q, t); a population pushed into a solid node stays in the halo,
    // where the bounce-back and the momentum exchange read it.  Both sides list the slots in the order of (t, q), as on
    // the Reference platform.  The frame of the block is made of the fluid nodes that send populations.

    if (decomposed) {
        int numRanks = decomposition.getSize();
        vector<vector<pair<pair<int, int>, int> > > send(numRanks), receive(numRanks);
        vector<int> frameHost, interiorHost;
        for (int n = 0; n < numLocal; n++) {
            int node = globalNode(n);
            if (!isFluidHost[node])
                continue;
            int local[3] = {n%localCount[0], (n/localCount[0])%localCount[1], n/(localCount[0]*localCount[1])};
            int index[3] = {localStart[0]+local[0], localStart[1]+local[1], localStart[2]+local[2]};
            bool sends = false;
            for (int q = 1; q < D3Q19::numVelocities; q++) {
                int c[3] = {D3Q19::cx[q], D3Q19::cy[q], D3Q19::cz[q]}, back[3] = {-c[0], -c[1], -c[2]};
                bool leaves = false, arrives = false;
                for (int a = 0; a < 3; a++) {
                    leaves = leaves || (pad[a] && (local[a]+c[a] < 0 || local[a]+c[a] >= localCount[a]));
                    arrives = arrives || (pad[a] && (local[a]-c[a] < 0 || local[a]-c[a] >= localCount[a]));
                }
                bool crossesFace;
                if (leaves) {
                    int target = globalNeighbor(index, c, crossesFace);
                    if (!crossesFace && isFluidHost[target]) {
                        send[decomposition.ownerOfNode(target)].push_back(make_pair(make_pair(target, q), q*numStored + neighborStorage(local, c)));
                        sends = true;
                    }
                }
                if (arrives) {
                    int source = globalNeighbor(index, back, crossesFace);
                    if (!crossesFace && isFluidHost[source])
                        receive[decomposition.ownerOfNode(source)].push_back(make_pair(make_pair(node, q), q*numStored + storageIndex(n)));
                }
            }
            (sends ? frameHost : interiorHost).push_back(n);
        }
        vector<int> sendSlotsHost, receiveSlotsHost;
        sendCounts.assign(numRanks, 0);
        receiveCounts.assign(numRanks, 0);
        for (int r = 0; r < numRanks; r++) {
            std::sort(send[r].begin(), send[r].end());
            std::sort(receive[r].begin(), receive[r].end());
            for (auto& link : send[r])
                sendSlotsHost.push_back(link.second);
            for (auto& link : receive[r])
                receiveSlotsHost.push_back(link.second);
            sendCounts[r] = send[r].size();
            receiveCounts[r] = receive[r].size();
        }
        sendBytes.resize(numRanks);
        receiveBytes.resize(numRanks);
        sendOffsets.resize(numRanks);
        receiveOffsets.resize(numRanks);
        for (int r = 0; r < numRanks; r++) {
            sendBytes[r] = sendCounts[r]*elementSize;
            receiveBytes[r] = receiveCounts[r]*elementSize;
            sendOffsets[r] = (r == 0 ? 0 : sendOffsets[r-1] + sendBytes[r-1]);
            receiveOffsets[r] = (r == 0 ? 0 : receiveOffsets[r-1] + receiveBytes[r-1]);
        }
        numFrameNodes = frameHost.size();
        numInteriorNodes = interiorHost.size();
        frameNodes.initialize<int>(cc, max(1, numFrameNodes), "lbmFrameNodes");
        interiorNodes.initialize<int>(cc, max(1, numInteriorNodes), "lbmInteriorNodes");
        sendSlots.initialize<int>(cc, max<int>(1, sendSlotsHost.size()), "lbmSendSlots");
        receiveSlots.initialize<int>(cc, max<int>(1, receiveSlotsHost.size()), "lbmReceiveSlots");
        sendBuffer.initialize(cc, max<int>(1, sendSlotsHost.size()), elementSize, "lbmSendBuffer");
        receiveBuffer.initialize(cc, max<int>(1, receiveSlotsHost.size()), elementSize, "lbmReceiveBuffer");
        hostSend.resize(sendBuffer.getSize()*elementSize);
        hostReceive.resize(receiveBuffer.getSize()*elementSize);
        // The populations go from device to device between the ranks of the same node if the MPI library of every rank
        // can read device memory: within a node over NVLink.  Between nodes they go through the host, which was faster
        // on Leonardo (InfiniBand, one port of UCX) for blocks of 128^3 nodes; the transfer from the device was faster
        // only for larger blocks, whose interior collides long enough to hide it.  The library (UCX, under Open MPI)
        // binds its transfers between devices to the CUDA context of the first one, while OpenMM gives every Context its
        // own CUDA context and destroys it with the Context: so only the first Context of the process that exchanges
        // device memory does so, and later ones go through the host (with them the library failed, "context is
        // destroyed").
        static bool deviceMPIClaimed = false;
        bool device = (!deviceMPIClaimed && getDeviceAddress(sendBuffer) != NULL && LBMDecomposition::isDeviceMPIAvailable());
        deviceMPI = (decomposition.maximum(device ? 0.0 : 1.0) == 0.0);
        vector<char> sameNode = decomposition.getRanksOnThisNode();
        if (deviceMPI) {
            deviceMPIClaimed = true;
            packedEvent = cc.createEvent();
        }
        deviceRank.assign(numRanks, 0);
        sendData.resize(numRanks);
        receiveData.resize(numRanks);
        hostSends = hostReceives = false;
        for (int r = 0; r < numRanks; r++) {
            deviceRank[r] = (deviceMPI && sameNode[r]);
            sendData[r] = (deviceRank[r] ? getDeviceAddress(sendBuffer) : hostSend.data()) + sendOffsets[r];
            receiveData[r] = (deviceRank[r] ? getDeviceAddress(receiveBuffer) : hostReceive.data()) + receiveOffsets[r];
            hostSends |= (!deviceRank[r] && sendBytes[r] > 0);
            hostReceives |= (!deviceRank[r] && receiveBytes[r] > 0);
        }
        if (numFrameNodes > 0)
            frameNodes.upload(frameHost);
        if (numInteriorNodes > 0)
            interiorNodes.upload(interiorHost);
        if (!sendSlotsHost.empty())
            sendSlots.upload(sendSlotsHost);
        if (!receiveSlotsHost.empty())
            receiveSlots.upload(receiveSlotsHost);

        // The halo of the rank, as on the Reference platform: the nodes of other ranks among the 26 neighbours of the
        // nodes of the block, across periodic boundaries but not across open faces.
        haloSendNodes.clear();
        haloReceiveNodes.clear();
        if (lattice.densityHaloExchange || lattice.velocityHaloExchange) {
            vector<set<int> > haloSend(numRanks), haloReceive(numRanks);
            for (int n = 0; n < numLocal; n++) {
                int node = globalNode(n);
                int index[3] = {node%lattice.nx, (node/lattice.nx)%lattice.ny, node/(lattice.nx*lattice.ny)};
                for (int dz = -1; dz <= 1; dz++)
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++) {
                            int d[3] = {dx, dy, dz};
                            bool beyondFace;
                            int t = globalNeighbor(index, d, beyondFace);
                            if (beyondFace || decomposition.owns(t))
                                continue;
                            int r = decomposition.ownerOfNode(t);
                            haloSend[r].insert(node);
                            haloReceive[r].insert(t);
                        }
            }
            haloSendNodes.resize(numRanks);
            haloReceiveNodes.resize(numRanks);
            vector<int> storage;
            for (int r = 0; r < numRanks; r++) {
                haloSendNodes[r].assign(haloSend[r].begin(), haloSend[r].end());
                haloReceiveNodes[r].assign(haloReceive[r].begin(), haloReceive[r].end());
                for (int node : haloSendNodes[r]) {
                    int n = localIndex(node);
                    storage.push_back(storageIndex(n));
                }
            }
            haloSendStorage.initialize<int>(cc, max<int>(1, storage.size()), "lbmHaloSendStorage");
            haloBuffer.initialize(cc, 4*max<int>(1, storage.size()), elementSize, "lbmHaloBuffer");
            if (!storage.empty())
                haloSendStorage.upload(storage);
            haloIndex.clear();
            for (int r = 0; r < numRanks; r++)
                for (int node : haloReceiveNodes[r]) {
                    int h = haloIndex.size();
                    haloIndex[node] = h;
                }
            double nan = numeric_limits<double>::quiet_NaN();
            if (lattice.densityHaloExchange)
                haloDensity.assign(haloIndex.size(), nan);
            if (lattice.velocityHaloExchange)
                haloVelocity.assign(3*haloIndex.size(), nan);
        }
    }

    // Coupled particles: their index in the list of the force for every atom, masses in units of the mass of a
    // cell, m_c = rho0 dx^3, and no coupling force before the first step.  The random force uses OpenMM's
    // generator, seeded with the seed of the force.  With the decomposition each rank has its own seed, from the seed
    // of rank 0 as on the Reference platform: the same seed everywhere would give the nodes of every block the same
    // random numbers.

    unsigned int seed = (unsigned int) lattice.randomNumberSeed;
    if (decomposed) {
        int first = (lattice.randomNumberSeed == 0 ? osrngseed() : lattice.randomNumberSeed);
        seed = (uint32_t) decomposition.broadcast(first) + 1000003u*(uint32_t) decomposition.getRank();
        if (seed == 0)
            seed = 1;       // 0 would let OpenMM choose a seed
    }
    int numCoupled = lattice.particles.size();
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    cellReaction.initialize(cc, (numCoupled > 0 ? 3*numStored : 1), elementSize, "lbmCellReaction");
    cellReaction.upload(vector<double>(cellReaction.getSize(), 0.0), true);
    if (numCoupled > 0) {
        vector<int> index(system.getNumParticles(), -1);
        vector<double> mass(numCoupled);
        for (int i = 0; i < numCoupled; i++) {
            index[lattice.particles[i]] = i;
            mass[i] = system.getParticleMass(lattice.particles[i])/cellMass;
        }
        couplingIndex.initialize<int>(cc, index.size(), "lbmCouplingIndex");
        couplingIndex.upload(index);
        particleMass.initialize(cc, numCoupled, elementSize, "lbmParticleMass");
        particleMass.upload(mass, true);
        particleForce.initialize(cc, 3*numCoupled, elementSize, "lbmParticleForce");
        particleForce.upload(vector<double>(3*numCoupled, 0.0), true);
        particleWallMomentum.initialize(cc, 3*numCoupled, elementSize, "lbmParticleWallMomentum");
        particleWallMomentum.upload(vector<double>(3*numCoupled, 0.0), true);
        sortKeys.initialize(cc, numCoupled, sizeof(long long), "lbmSortKeys");
        if (centered) {
            knownVelocity.initialize(cc, 3*numCoupled, elementSize, "lbmKnownVelocity");
            randomForce.initialize(cc, 3*numCoupled, elementSize, "lbmRandomForce");
        }
        noise.initialize<mm_float4>(cc, numCoupled, "lbmNoise");
        noise.upload(vector<mm_float4>(numCoupled, mm_float4(0, 0, 0, 0)));
        sort = cc.createSort(new CouplingSortTrait(), numCoupled, false);
        cc.getIntegrationUtilities().initRandomNumberGenerator(seed);
        if (decomposed && numSolidNodes > 0) {
            globalIsFluid.initialize<int>(cc, numNodes, "lbmGlobalIsFluid");
            globalIsFluid.upload(isFluidHost);
        }
    }

    // A fluctuating fluid (docs/theory.md, section 7) draws four float4 of normal numbers per node and step from
    // OpenMM's generator, after those of the particles, and turns 15 of them into the random part of the
    // populations with the coefficients w_q e_k(c_q)/sqrt(b_k) of the orthogonal basis, k = 4...18.

    if (lattice.fluidFluctuations) {
        cc.getIntegrationUtilities().initRandomNumberGenerator(seed);
        vector<double> basis(D3Q19::numVelocities*15);
        for (int q = 0; q < D3Q19::numVelocities; q++)
            for (int m = 0; m < 15; m++)
                basis[q*15+m] = D3Q19::w[q]*D3Q19::mode(m+4, q)/sqrt(D3Q19::modeNorm[m+4]);
        fluctuationBasis.initialize(cc, basis.size(), elementSize, "lbmFluctuationBasis");
        fluctuationBasis.upload(basis, true);

        // The buffer of the random numbers grows to the numbers of a step now, rather than in the first step.  The
        // checkpoints of OpenMM write the buffer as it is but read it back with the size it has in the Context that
        // loads them (OpenMM 8.3), so a Context created to load a checkpoint must already have the same size.  The
        // boundary nodes draw four float4 more each, after those of all the nodes.
        cc.getIntegrationUtilities().prepareRandomNumbers(4*(numLocal+numBoundaryNodes));
    }

    // Compile the kernels.

    map<string, string> defines;
    defines["NUM_NODES"] = cc.intToString(numLocal);
    defines["NX"] = cc.intToString(localCount[0]);
    defines["NY"] = cc.intToString(localCount[1]);
    defines["NZ"] = cc.intToString(localCount[2]);
    if (decomposed) {
        defines["DOMAIN_DECOMPOSITION"] = "1";
        defines["PAD_X"] = cc.intToString(pad[0]);
        defines["PAD_Y"] = cc.intToString(pad[1]);
        defines["PAD_Z"] = cc.intToString(pad[2]);
        defines["SX"] = cc.intToString(stored[0]);
        defines["SY"] = cc.intToString(stored[1]);
        defines["NUM_STORED"] = cc.intToString(numStored);
    }
    defines["LBM_BLOCK_SIZE"] = cc.intToString(blockSize);
    defines["W0"] = cc.doubleToString(1.0/3.0, true);
    defines["W1"] = cc.doubleToString(1.0/18.0, true);
    defines["W2"] = cc.doubleToString(1.0/36.0, true);
    defines["CS2"] = cc.doubleToString(1.0/3.0, true);
    if (numSolidNodes > 0) {
        defines["HAS_SOLID_NODES"] = "1";
        defines["NUM_SOLID_NODES"] = cc.intToString(numSolidEntries);
    }
    if (numCoupled > 0)
        defines["HAS_COUPLED_PARTICLES"] = "1";
    if (lattice.fluidFluctuations)
        defines["FLUID_FLUCTUATIONS"] = "1";
    if (numWallNodes > 0) {
        defines["BOUNCE_BACK_WALLS"] = "1";
        defines["NUM_WALL_NODES"] = cc.intToString(numWallNodes);
    }
    if (numBoundaryNodes > 0) {
        defines["HAS_BOUNDARY_NODES"] = "1";
        defines["NUM_BOUNDARY_NODES"] = cc.intToString(numBoundaryNodes);
    }
    const char* openAxis[3] = {"OPEN_X", "OPEN_Y", "OPEN_Z"};
    for (int a = 0; a < 3; a++)
        if (lattice.isOpenAxis(a))
            defines[openAxis[a]] = "1";
    ComputeProgram program = cc.compileProgram(CommonLBMKernelSources::lbmFluid, defines);
    computeMomentsKernel = program->createKernel("computeFluidMoments");
    computeMomentsKernel->addArg(populations);
    computeMomentsKernel->addArg(isFluid);
    computeMomentsKernel->addArg(densityDeviation);
    computeMomentsKernel->addArg(momentum);
    computeMomentsKernel->addArg(piNeq);
    sumMomentumKernel = program->createKernel("sumFluidMomentum");
    sumMomentumKernel->addArg(densityDeviation);
    sumMomentumKernel->addArg(momentum);
    sumMomentumKernel->addArg(partialSums);
    sumMomentumKernel->addArg(isFluid);
    centerVelocityKernel = program->createKernel("computeFluidCenterVelocity");
    centerVelocityKernel->addArg(partialSums);
    centerVelocityKernel->addArg(numGroups);
    centerVelocityKernel->addArg(centerVelocity);
    removeMomentumKernel = program->createKernel("removeFluidMomentum");
    removeMomentumKernel->addArg(densityDeviation);
    removeMomentumKernel->addArg(momentum);
    removeMomentumKernel->addArg(centerVelocity);
    removeMomentumKernel->addArg(isFluid);
    collideKernel = program->createKernel("collideAndStream");
    collideKernel->addArg(populations);
    collideKernel->addArg(isFluid);
    collideKernel->addArg(densityDeviation);
    collideKernel->addArg(momentum);
    collideKernel->addArg(piNeq);
    collideKernel->addArg(cellReaction);
    for (int i = 0; i < 4; i++)
        collideKernel->addArg();        // omega and the body acceleration, set by setFluidParameters()
    if (lattice.fluidFluctuations) {
        collideKernel->addArg(cc.getIntegrationUtilities().getRandom());
        collideKernel->addArg(fluctuationBasis);
        collideKernel->addArg();        // mu = kT/cs^2, set by setFluidParameters()
        collideKernel->addArg(0);       // index of the random numbers, set in every step
    }
    if (decomposed) {
        collideKernel->addArg(frameNodes);  // the list of nodes and its size, set in every step
        collideKernel->addArg(0);
        packKernel = program->createKernel("packPopulations");
        packKernel->addArg(populations);
        packKernel->addArg(sendSlots);
        packKernel->addArg(sendBuffer);
        packKernel->addArg((int) sendSlots.getSize());
        unpackKernel = program->createKernel("unpackPopulations");
        unpackKernel->addArg(populations);
        unpackKernel->addArg(receiveSlots);
        unpackKernel->addArg(receiveBuffer);
        unpackKernel->addArg((int) receiveSlots.getSize());
        if (haloSendStorage.isInitialized()) {
            packFieldsKernel = program->createKernel("packFields");
            packFieldsKernel->addArg(densityDeviation);
            packFieldsKernel->addArg(momentum);
            packFieldsKernel->addArg(haloSendStorage);
            packFieldsKernel->addArg(haloBuffer);
            packFieldsKernel->addArg((int) haloSendStorage.getSize());
        }
    }
    if (numWallNodes > 0) {
        bounceBackKernel = program->createKernel("bounceBack");
        bounceBackKernel->addArg(populations);
        bounceBackKernel->addArg(wallNodes);
        bounceBackKernel->addArg(wallLinks);
        wallExchangeKernel = program->createKernel("computeWallExchange");
        wallExchangeKernel->addArg(populations);
        wallExchangeKernel->addArg(solidLinks);
        wallExchangeKernel->addArg(solidNodes);
        wallExchangeKernel->addArg(wallExchange);
    }
    if (numBoundaryNodes > 0) {
        applyBoundariesKernel = program->createKernel("applyBoundaries");
        applyBoundariesKernel->addArg(populations);
        applyBoundariesKernel->addArg(boundaryNodes);
        applyBoundariesKernel->addArg(boundaryUnknown);
        applyBoundariesKernel->addArg(boundarySolid);
        applyBoundariesKernel->addArg(boundaryKindAndFace);
        applyBoundariesKernel->addArg(densityDeviation);
        applyBoundariesKernel->addArg(momentum);
        applyBoundariesKernel->addArg(faceParameters);
        applyBoundariesKernel->addArg(boundaryExchange);
        for (int i = 0; i < 3; i++)
            applyBoundariesKernel->addArg();    // the body acceleration, set by setFluidParameters()
        applyBoundariesKernel->addArg(piNeq);
        applyBoundariesKernel->addArg();        // omega, set by setFluidParameters()
        if (lattice.fluidFluctuations) {
            applyBoundariesKernel->addArg(cc.getIntegrationUtilities().getRandom());
            applyBoundariesKernel->addArg(fluctuationBasis);
            applyBoundariesKernel->addArg();    // mu = kT/cs^2, set by setFluidParameters()
            applyBoundariesKernel->addArg(0);   // index of the random numbers, set in every step
        }
    }
    maxSpeedKernel = program->createKernel("computeMaxFluidSpeed");
    maxSpeedKernel->addArg(populations);
    maxSpeedKernel->addArg(isFluid);
    maxSpeedKernel->addArg(partialMax);

    if (numCoupled > 0) {
        defines["NUM_ATOMS"] = cc.intToString(cc.getNumAtoms());
        defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
        defines["NUM_COUPLED"] = cc.intToString(numCoupled);
        defines["DX"] = cc.doubleToString(lattice.dx, true);
        defines["VELOCITY_SCALE"] = cc.doubleToString(lattice.getVelocityScale(), true);
        defines["FORCE_SCALE"] = cc.doubleToString(cellMass*lattice.dx/(lattice.dt*lattice.dt), true);
        if (decomposed) {
            defines["GNX"] = cc.intToString(lattice.nx);
            defines["GNY"] = cc.intToString(lattice.ny);
            defines["GNZ"] = cc.intToString(lattice.nz);
            defines["OX"] = cc.intToString(localStart[0]);
            defines["OY"] = cc.intToString(localStart[1]);
            defines["OZ"] = cc.intToString(localStart[2]);
            if (decomposition.getRank() != 0)
                defines["SKIP_REFLECTION_MOMENTUM"] = "1";
        }
        if (centered && hasFloatForceBuffers())
            defines["HAS_FLOAT_FORCE_BUFFERS"] = "1";
        ComputeProgram coupling = cc.compileProgram(CommonLBMKernelSources::lbmCoupling, defines);
        if (numSolidNodes > 0) {
            reflectKernel = coupling->createKernel("reflectParticles");
            reflectKernel->addArg(cc.getPosq());
            reflectKernel->addArg(cc.getPosqCorrection());
            reflectKernel->addArg(cc.getVelm());
            reflectKernel->addArg(cc.getAtomIndexArray());
            reflectKernel->addArg(couplingIndex);
            reflectKernel->addArg(decomposed ? globalIsFluid : isFluid);
            reflectKernel->addArg(particleMass);
            reflectKernel->addArg(particleWallMomentum);
        }
        coupleKernel = coupling->createKernel("coupleParticles");
        coupleKernel->addArg(cc.getPosq());
        coupleKernel->addArg(cc.getPosqCorrection());
        coupleKernel->addArg(cc.getVelm());
        coupleKernel->addArg(cc.getAtomIndexArray());
        coupleKernel->addArg(couplingIndex);
        coupleKernel->addArg(particleMass);
        coupleKernel->addArg(densityDeviation);
        coupleKernel->addArg(momentum);
        coupleKernel->addArg(particleForce);
        coupleKernel->addArg(sortKeys);
        coupleKernel->addArg(particleWallMomentum);
        coupleKernel->addArg(cc.getIntegrationUtilities().getRandom());
        coupleKernel->addArg(noise);
        for (int i = 0; i < 5; i++)
            coupleKernel->addArg();     // index of the random numbers, drawNoise, isStep, gamma and kT, set later
        coupleKernel->addArg(isFluid);
        sumReactionsKernel = coupling->createKernel("sumCellReactions");
        sumReactionsKernel->addArg(sortKeys);
        sumReactionsKernel->addArg(particleForce);
        sumReactionsKernel->addArg(cellReaction);
        clearReactionsKernel = coupling->createKernel("clearCellReactions");
        clearReactionsKernel->addArg(sortKeys);
        clearReactionsKernel->addArg(cellReaction);
        applyForcesKernel = coupling->createKernel("applyCouplingForces");
        applyForcesKernel->addArg(cc.getAtomIndexArray());
        applyForcesKernel->addArg(couplingIndex);
        applyForcesKernel->addArg(particleForce);
        applyForcesKernel->addArg(cc.getLongForceBuffer());
        if (centered) {
            prepareCenteredKernel = coupling->createKernel("prepareCenteredDrag");
            prepareCenteredKernel->addArg(cc.getPosq());
            prepareCenteredKernel->addArg(cc.getPosqCorrection());
            prepareCenteredKernel->addArg(cc.getVelm());
            prepareCenteredKernel->addArg(cc.getAtomIndexArray());
            prepareCenteredKernel->addArg(couplingIndex);
            prepareCenteredKernel->addArg(particleMass);
            prepareCenteredKernel->addArg(cc.getLongForceBuffer());
            prepareCenteredKernel->addArg(knownVelocity);
            prepareCenteredKernel->addArg(randomForce);
            prepareCenteredKernel->addArg(sortKeys);
            prepareCenteredKernel->addArg(cc.getIntegrationUtilities().getRandom());
            prepareCenteredKernel->addArg(noise);
            for (int i = 0; i < 4; i++)
                prepareCenteredKernel->addArg();    // index of the random numbers, drawNoise, gamma and kT, set later
            if (hasFloatForceBuffers())
                for (int i = 0; i < 2; i++)
                    prepareCenteredKernel->addArg();    // floating point force buffers and their number, set on first use
            if (decomposed)
                prepareCenteredKernel->addArg(particleForce);
            solveCenteredKernel = coupling->createKernel("solveCenteredDrag");
            solveCenteredKernel->addArg(sortKeys);
            solveCenteredKernel->addArg(particleMass);
            solveCenteredKernel->addArg(knownVelocity);
            solveCenteredKernel->addArg(randomForce);
            solveCenteredKernel->addArg(densityDeviation);
            solveCenteredKernel->addArg(momentum);
            solveCenteredKernel->addArg(particleForce);
            solveCenteredKernel->addArg(cellReaction);
            solveCenteredKernel->addArg(particleWallMomentum);
            for (int i = 0; i < 5; i++)
                solveCenteredKernel->addArg();      // isStep, gamma and the body acceleration, set later
            solveCenteredKernel->addArg(isFluid);
        }
    }
    setFluidParameters();

    // With the Centered drag the work of execute() is done at the end of the force evaluation.  OpenMM owns the
    // post-computation and deletes it with the ComputeContext.

    if (centered)
        cc.addPostComputation(new CenteredDragPostComputation(*this));
    if (packFieldsKernel)
        exchangeHalo();
}

void CommonCalcLBMForceKernel::setFluidParameters() {
    // The relaxation rate, the body acceleration, the friction gamma = friction*dt and the temperature (lattice
    // units) are kernel arguments in the mixed type, so that updateParametersInContext() can change them.
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    double gamma = lattice.friction*lattice.dt;
    double kT = lattice.kT*lattice.dt*lattice.dt/(cellMass*lattice.dx*lattice.dx);
    double values[4] = {lattice.omega, lattice.bodyAcceleration[0], lattice.bodyAcceleration[1], lattice.bodyAcceleration[2]};
    for (int i = 0; i < 4; i++) {
        if (useDouble)
            collideKernel->setArg(6+i, values[i]);
        else
            collideKernel->setArg(6+i, (float) values[i]);
    }
    if (applyBoundariesKernel) {
        // The velocity (lattice units) and the density minus 1 of each face.
        vector<double> faces(24);
        for (int face = 0; face < 6; face++) {
            for (int k = 0; k < 3; k++)
                faces[4*face+k] = lattice.faceVelocity[face][k];
            faces[4*face+3] = lattice.faceDensity[face]-1.0;
        }
        faceParameters.upload(faces, true);
        for (int i = 0; i < 4; i++) {
            int index = (i == 0 ? 13 : 8+i);  // omega, then the body acceleration
            if (useDouble)
                applyBoundariesKernel->setArg(index, values[i]);
            else
                applyBoundariesKernel->setArg(index, (float) values[i]);
        }
    }
    if (lattice.fluidFluctuations) {
        double mu = 3.0*lattice.fluidKT*lattice.dt*lattice.dt/(cellMass*lattice.dx*lattice.dx);
        if (useDouble)
            collideKernel->setArg(12, mu);
        else
            collideKernel->setArg(12, (float) mu);
        if (applyBoundariesKernel) {
            if (useDouble)
                applyBoundariesKernel->setArg(16, mu);
            else
                applyBoundariesKernel->setArg(16, (float) mu);
        }
    }
    if (!lattice.particles.empty()) {
        coupleKernel->setArg(13, 0);       // index of the random numbers, set when they are drawn
        coupleKernel->setArg(14, 0);       // drawNoise
        coupleKernel->setArg(15, 1);       // isStep
        if (useDouble) {
            coupleKernel->setArg(16, gamma);
            coupleKernel->setArg(17, kT);
        }
        else {
            coupleKernel->setArg(16, (float) gamma);
            coupleKernel->setArg(17, (float) kT);
        }
    }
    if (!lattice.particles.empty() && lattice.dragScheme == LBMForce::Centered) {
        prepareCenteredKernel->setArg(12, 0);  // index of the random numbers, set when they are drawn
        prepareCenteredKernel->setArg(13, 0);  // drawNoise
        solveCenteredKernel->setArg(9, 1);     // isStep
        double solveValues[4] = {gamma, lattice.bodyAcceleration[0], lattice.bodyAcceleration[1], lattice.bodyAcceleration[2]};
        if (useDouble) {
            prepareCenteredKernel->setArg(14, gamma);
            prepareCenteredKernel->setArg(15, kT);
            for (int i = 0; i < 4; i++)
                solveCenteredKernel->setArg(10+i, solveValues[i]);
        }
        else {
            prepareCenteredKernel->setArg(14, (float) gamma);
            prepareCenteredKernel->setArg(15, (float) kT);
            for (int i = 0; i < 4; i++)
                solveCenteredKernel->setArg(10+i, (float) solveValues[i]);
        }
    }
}

void CommonCalcLBMForceKernel::beginStep(ContextImpl& context) {
    // As on the Reference platform, the removal of the fluid momentum and the Mach number check are timed by the
    // step count of the Context, which checkpoints save and restore, and coupled particles that move into a wall
    // are reflected here, where OpenMM's AndersenThermostat also changes velocities.
    stepIndex = context.getStepCount();
    stepPending = true;
    if (decomposition.isDecomposed() && !lattice.particles.empty() && lattice.particleCopiesCheck && (!replicasChecked ||
            (lattice.machCheckFrequency > 0 && stepIndex%lattice.machCheckFrequency == 0)))
        checkReplicas(context);
    if (!lattice.particles.empty() && !lattice.solidNodes.empty()) {
        ContextSelector selector(cc);
        reflectKernel->execute(cc.getNumAtoms());
    }
}

void CommonCalcLBMForceKernel::checkReplicas(ContextImpl& context) {
    // As on the Reference platform: with the domain decomposition every rank integrates its own copy of all the
    // particles, and the bits of the positions and velocities are compared over the ranks, through a hash, at the first
    // lattice step and then every machCheckFrequency steps.  The check is collective.
    replicasChecked = true;
    vector<Vec3> positions, velocities;
    context.getPositions(positions);
    context.getVelocities(velocities);
    unsigned long long hash = 14695981039346656037ULL;
    for (const vector<Vec3>* values : {&positions, &velocities})
        for (const Vec3& value : *values)
            for (int k = 0; k < 3; k++) {
                double component = value[k];
                unsigned long long bits;
                memcpy(&bits, &component, sizeof(bits));
                hash = (hash^bits)*1099511628211ULL;
            }
    if (!decomposition.isSameOnAllRanks(hash)) {
        stringstream msg;
        msg << "LBMForce: the positions or velocities of the particles differ between the MPI ranks at lattice step "
            << stepIndex << ". With the domain decomposition every rank holds a copy of all the particles, and the copies "
            << "must be identical: set positions and velocities in the same way on every rank (for example "
            << "setVelocitiesToTemperature() with a fixed random seed)";
        throw OpenMMException(msg.str());
    }
}

void CommonCalcLBMForceKernel::sumParticleForces() {
    // With the domain decomposition each coupling force has been computed by the rank of the nearest node, and is zero
    // on the others: the sum is exact, and the same on every rank.
    vector<double> forces;
    downloadAsDouble(particleForce, forces);
    decomposition.sum(forces.data(), forces.size());
    particleForce.upload(forces, true);
}

double CommonCalcLBMForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // With the Centered drag the post-computation does the work, once OpenMM has computed the other forces.
    if (lattice.dragScheme == LBMForce::Centered)
        return 0.0;
    return evaluate(context.getStepCount(), includeForces);
}

double CommonCalcLBMForceKernel::evaluate(long long stepCount, bool includeForces) {
    // As on the Reference platform: the fluid advances once per integration step, on the first force evaluation
    // after beginStep(), which also computes the coupling forces of the step.  The evaluations between steps do
    // not change the fluid and return the coupling force of the next step, as OpenMM does for every force, so
    // that the kinetic energy of VerletIntegrator is that of the full step.  The random numbers of a step are
    // drawn once.  Before the first step of the Context the coupling force is zero.  The energy is zero.
    // OpenMM repeats all the force evaluations of a step when it has to enlarge its neighbor list
    // (finishComputation() returns valid = false); the repeated evaluation comes before the integrator increments
    // the step count, and it uses the forces of the step again.
    ContextSelector selector(cc);
    bool repeatedStep = (stepForcesCurrent && stepCount == stepIndex-1);
    if (stepPending) {
        stepPending = false;
        advanceFluid();
    }
    else if (includeForces && !lattice.particles.empty() && stepCount > 0 && !repeatedStep)
        computeNextStepForces(stepCount);
    if (includeForces && !lattice.particles.empty())
        applyForcesKernel->execute(cc.getNumAtoms());
    return 0.0;
}

void CommonCalcLBMForceKernel::advanceFluid() {
    computeMomentsKernel->execute(numLocal);
    if (lattice.momentumRemovalFrequency > 0 && stepIndex%lattice.momentumRemovalFrequency == 0)
        removeFluidMomentum();
    int numCoupled = lattice.particles.size();
    if (numCoupled > 0)
        computeCouplingForces(true);
    // The numbers of the fluid are drawn after those of the particles, which the force evaluations between steps
    // may already have drawn: the sequence is the same in both cases.
    if (lattice.fluidFluctuations && lattice.fluidKT > 0) {
        int numBoundaryNodes = (boundaryNodes.isInitialized() ? boundaryNodes.getSize() : 0);
        int randomIndex = cc.getIntegrationUtilities().prepareRandomNumbers(4*(numLocal+numBoundaryNodes));
        collideKernel->setArg(13, randomIndex);
        if (applyBoundariesKernel)
            applyBoundariesKernel->setArg(17, randomIndex);
    }
    collideAndStream();
    if (numCoupled > 0)
        clearReactionsKernel->execute(numCoupled);
    if (bounceBackKernel) {
        bounceBackKernel->execute(wallNodes.getSize());
        wallExchangeKernel->execute(numSolidEntries);
    }
    if (applyBoundariesKernel)
        applyBoundariesKernel->execute(boundaryNodes.getSize());
    if (packFieldsKernel)
        exchangeHalo();
    hasAdvanced = true;
    stepForcesCurrent = true;
    stepIndex++;
    if (lattice.machCheckFrequency > 0 && stepIndex%lattice.machCheckFrequency == 0)
        checkMachNumber();
}

void CommonCalcLBMForceKernel::computeNextStepForces(long long nextStep) {
    // The coupling of the next lattice step computed on the current fluid, positions and velocities, without its
    // reaction on the fluid.  The step recomputes it, with the same random numbers, after any change of the
    // velocities in between (the reflection at the walls, setVelocities()).
    stepForcesCurrent = false;
    computeMomentsKernel->execute(numLocal);
    if (lattice.momentumRemovalFrequency > 0 && nextStep%lattice.momentumRemovalFrequency == 0)
        removeFluidMomentum();
    computeCouplingForces(false);
}

void CommonCalcLBMForceKernel::removeFluidMomentum() {
    // Sums by work group, then over the groups.  With the domain decomposition the second stage writes the sums of
    // rho - 1 and j over the block, and the host adds them over the ranks in rank order, as on the Reference platform.
    sumMomentumKernel->execute(numGroups*blockSize, blockSize);
    centerVelocityKernel->execute(blockSize, blockSize);
    if (decomposition.isDecomposed()) {
        vector<double> sums;
        downloadAsDouble(centerVelocity, sums);
        decomposition.sumInRankOrder(sums.data(), 4);
        double mass = lattice.getNumNodes() + sums[0];
        vector<double> velocity = {sums[1]/mass, sums[2]/mass, sums[3]/mass, 0.0};
        centerVelocity.upload(velocity, true);
    }
    removeMomentumKernel->execute(numLocal);
}

void CommonCalcLBMForceKernel::collideAndStream() {
    if (!decomposition.isDecomposed()) {
        collideKernel->execute(numLocal);
        return;
    }

    // With the domain decomposition: the frame of the block first; then its populations that went into the halo are
    // packed and sent while the interior collides, and those received are unpacked into the block.  Through the host
    // the kernels run in order on the device, and the downloads and uploads wait for them; with an MPI library that
    // reads device memory (deviceMPI) the buffers of the device are sent and received directly.
    int listArgument = (lattice.fluidFluctuations ? 14 : 10);
    if (numFrameNodes > 0) {
        collideKernel->setArg(listArgument, frameNodes);
        collideKernel->setArg(listArgument+1, numFrameNodes);
        collideKernel->execute(numFrameNodes);
    }
    int numSent = accumulate(sendCounts.begin(), sendCounts.end(), 0);
    int numReceived = accumulate(receiveCounts.begin(), receiveCounts.end(), 0);
    if (numSent > 0)
        packKernel->execute(numSent);
    if (hostSends) {
        // Some ranks receive through the host: the download waits for the packing, and the populations for the ranks of
        // the same node, with deviceMPI, are sent from the device meanwhile.
        sendBuffer.download(hostSend.data());
        decomposition.startExchange(sendData, sendBytes, receiveData, receiveBytes);
        runInterior(listArgument);
    }
    else {
        // Only from device to device (or nothing to send): the interior collides while the packing ends.
        if (deviceMPI)
            packedEvent->enqueue();
        runInterior(listArgument);
        if (deviceMPI)
            packedEvent->wait();
        decomposition.startExchange(sendData, sendBytes, receiveData, receiveBytes);
    }
    decomposition.finishExchange();
    if (hostReceives) {
        if (!deviceMPI)
            receiveBuffer.upload(hostReceive.data());
        else
            for (int r = 0; r < (int) deviceRank.size(); r++)
                if (!deviceRank[r] && receiveCounts[r] > 0)
                    receiveBuffer.uploadSubArray(receiveData[r], (int) (receiveOffsets[r]/receiveBuffer.getElementSize()),
                            receiveCounts[r]);
    }
    if (numReceived > 0)
        unpackKernel->execute(numReceived);
}

void CommonCalcLBMForceKernel::runInterior(int listArgument) {
    if (numInteriorNodes > 0) {
        collideKernel->setArg(listArgument, interiorNodes);
        collideKernel->setArg(listArgument+1, numInteriorNodes);
        collideKernel->execute(numInteriorNodes);
    }
}

void CommonCalcLBMForceKernel::uploadPopulations(const vector<double>& values) {
    if (populations.getElementSize() == sizeof(double))
        uploadPopulations(values.data());
    else {
        vector<float> single(values.begin(), values.end());
        uploadPopulations(single.data());
    }
}

void CommonCalcLBMForceKernel::uploadPopulations(const void* data) {
    // OpenMM computes the number of bytes of an upload as an int (ComputeArray::upload() and uploadSubArray() in OpenMM
    // 8.3 to 8.6), which overflows beyond 2 GB: 14.1 million nodes in mixed and double precision.  A larger array is
    // uploaded in parts into a smaller array, and each part copied into its place on the device.
    long long elementSize = populations.getElementSize(), size = populations.getSize();
    if (size*elementSize < (1LL << 31)) {
        populations.upload(data);
        return;
    }
    int partSize = (int) ((1LL << 28)/elementSize);
    if (!copyPartKernel) {
        uploadStaging.initialize(cc, partSize, elementSize, "lbmUploadStaging");
        ComputeProgram program = cc.compileProgram(
            "KERNEL void copyPart(GLOBAL const mixed* RESTRICT part, GLOBAL mixed* RESTRICT target, int start, int count) {\n"
            "    for (int i = GLOBAL_ID; i < count; i += GLOBAL_SIZE)\n"
            "        target[start+i] = part[i];\n"
            "}\n");
        copyPartKernel = program->createKernel("copyPart");
        copyPartKernel->addArg(uploadStaging);
        copyPartKernel->addArg(populations);
        copyPartKernel->addArg();
        copyPartKernel->addArg();
    }
    ComputeEvent copied = cc.createEvent();
    for (long long start = 0; start < size; start += partSize) {
        int count = (int) min((long long) partSize, size-start);
        uploadStaging.uploadSubArray((const char*) data + start*elementSize, 0, count);
        copyPartKernel->setArg(2, (int) start);
        copyPartKernel->setArg(3, count);
        copyPartKernel->execute(count);
        // The next part overwrites the smaller array only after this one is copied.
        copied->enqueue();
        copied->wait();
    }
}

void CommonCalcLBMForceKernel::computeCouplingForces(bool isStep) {
    // The N(0,1) numbers of a step are drawn once, when there is a random force, by the first computation that
    // needs them, and copied into the array noise.  One float4 per padded atom is reserved, although only the
    // first numCoupled are used: this is how the DragOpenMM plugin consumes the generator, so with the same seed
    // both plugins draw the same numbers and their stochastic runs can be compared step by step.  In a lattice
    // step (isStep) the keys are sorted and the reactions of the particles are summed per node.
    int numCoupled = lattice.particles.size();
    bool draw = (lattice.kT > 0 && lattice.friction > 0 && !noiseDrawn);
    int randomIndex = 0;
    if (draw) {
        randomIndex = cc.getIntegrationUtilities().prepareRandomNumbers(cc.getPaddedNumAtoms());
        noiseDrawn = true;
    }
    if (lattice.dragScheme == LBMForce::Centered) {
        // The particles of a node are solved together (docs/theory.md, section 2): the keys are sorted in every
        // evaluation, and the first entry of each node solves its segment.  The other forces are read from
        // OpenMM's fixed point buffer and, where the platform has them, from its floating point buffers; their
        // number is known once OpenMM has created its buffers, after the forces were initialized.
        if (numFloatForceBuffers < 0) {
            numFloatForceBuffers = 0;
            if (hasFloatForceBuffers()) {
                ArrayInterface& buffers = cc.getForceBuffers();
                numFloatForceBuffers = buffers.getSize()/cc.getPaddedNumAtoms();
                prepareCenteredKernel->setArg(16, buffers);
                prepareCenteredKernel->setArg(17, numFloatForceBuffers);
            }
        }
        if (draw)
            prepareCenteredKernel->setArg(12, randomIndex);
        prepareCenteredKernel->setArg(13, draw ? 1 : 0);
        prepareCenteredKernel->execute(cc.getNumAtoms());
        sort->sort(sortKeys);
        solveCenteredKernel->setArg(9, isStep ? 1 : 0);
        solveCenteredKernel->execute(numCoupled);
    }
    else {
        if (draw)
            coupleKernel->setArg(13, randomIndex);
        coupleKernel->setArg(14, draw ? 1 : 0);
        coupleKernel->setArg(15, isStep ? 1 : 0);
        coupleKernel->execute(cc.getNumAtoms());
        if (isStep) {
            sort->sort(sortKeys);
            sumReactionsKernel->execute(numCoupled);
        }
    }
    if (isStep)
        noiseDrawn = false;
    if (decomposition.isDecomposed())
        sumParticleForces();
}

void CommonCalcLBMForceKernel::checkMachNumber() {
    double mach = computeMachNumber();
    if (mach > lattice.machNumberLimit) {
        stringstream msg;
        msg << "LBMForce: the Mach number of the fluid is " << mach << " after " << stepIndex << " lattice steps, "
            << "above the limit " << lattice.machNumberLimit << " (docs/theory.md, section 6). Reduce the forces on the "
            << "fluid, the time step or the friction.";
        throw OpenMMException(msg.str());
    }
#ifdef LBM_DEBUG
    if (mach > 0.1 && !machWarningPrinted) {
        cerr << "Warning: LBMForce: the Mach number of the fluid is " << mach << " after " << stepIndex
             << " lattice steps; the accuracy of the model degrades above 0.1." << endl;
        machWarningPrinted = true;
    }
#endif
}

double CommonCalcLBMForceKernel::computeMachNumber() {
    // Ma = max |j/rho|/c_s, with c_s^2 = 1/3: maxima by work group on the device, then over the groups here, and with
    // the domain decomposition over the ranks (every rank must call it).
    maxSpeedKernel->execute(numGroups*blockSize, blockSize);
    vector<double> maxima;
    downloadAsDouble(partialMax, maxima);
    double maxSpeed2 = *max_element(maxima.begin(), maxima.end());
    return sqrt(3.0*decomposition.maximum(maxSpeed2));
}

void CommonCalcLBMForceKernel::copyParametersToContext(ContextImpl& context, const LBMLatticeParameters& lattice) {
    ContextSelector selector(cc);
    this->lattice = lattice;
    setFluidParameters();
}

void CommonCalcLBMForceKernel::getFluidFields(ContextImpl& context, vector<double>& density, vector<Vec3>& velocity, bool halo) {
    // The velocity of the forced fluid is u = (j + F/2)/rho, with F = rho*g from the body acceleration.  The
    // moments are recomputed from the populations; the next step computes them again before using them.  The nodes of
    // the domain of the rank, and with halo those around it (CalcLBMForceKernel::getFluidFields()): a node of the halo
    // of another rank takes the fields exchanged at the end of the last step, one of the rank (across a periodic
    // boundary, or with one domain) its own fields, if the halo of the field is exchanged.
    ContextSelector selector(cc);
    computeMomentsKernel->execute(numLocal);
    vector<double> dr, j;
    downloadAsDouble(densityDeviation, dr);
    downloadAsDouble(momentum, j);
    double velocityScale = lattice.getVelocityScale();
    double nan = numeric_limits<double>::quiet_NaN();
    bool open[3] = {lattice.isOpenAxis(0), lattice.isOpenAxis(1), lattice.isOpenAxis(2)};
    vector<int> nodes;
    vector<char> inside;
    decomposition.getDomainNodes(halo ? 1 : 0, open, nodes, inside);
    density.assign(nodes.size(), nan);
    velocity.assign(nodes.size(), Vec3(nan, nan, nan));
    for (size_t l = 0; l < nodes.size(); l++) {
        int node = nodes[l];
        bool densityValid = node >= 0 && (inside[l] || lattice.densityHaloExchange);
        bool velocityValid = node >= 0 && (inside[l] || lattice.velocityHaloExchange);
        if (!densityValid && !velocityValid)
            continue;
        if (!decomposition.owns(node)) {
            int h = haloIndex.at(node);
            if (densityValid)
                density[l] = haloDensity[h];
            if (velocityValid)
                velocity[l] = Vec3(haloVelocity[3*h], haloVelocity[3*h+1], haloVelocity[3*h+2]);
            continue;
        }
        double rho = 0;
        Vec3 u;
        if (isFluidHost[node]) {
            int s = storageIndex(localNode(node));
            double r = 1.0 + dr[s];
            rho = r*lattice.density;
            u = (Vec3(j[s], j[numStored+s], j[2*numStored+s])*(1.0/r) + lattice.bodyAcceleration*0.5)*velocityScale;
        }
        if (densityValid)
            density[l] = rho;
        if (velocityValid)
            velocity[l] = u;
    }
}

void CommonCalcLBMForceKernel::exchangeHalo() {
    // The fields of the nodes of the block that are in the halo of other ranks go to those ranks, and the fields of the
    // halo of the rank come from their owners (the lists of initialize()).  Collective.  The device copies rho - 1 and
    // j of the nodes to send from the moments of the current populations, and the host computes the fields as
    // getFluidFields() does, so the copies are identical bit for bit to what the owner returns.
    computeMomentsKernel->execute(numLocal);
    int numRanks = decomposition.getSize(), numSendNodes = 0;
    for (int r = 0; r < numRanks; r++)
        numSendNodes += haloSendNodes[r].size();
    vector<double> moments;
    if (numSendNodes > 0) {
        packFieldsKernel->execute(numSendNodes);
        downloadAsDouble(haloBuffer, moments);
    }
    bool exchangeDensity = !haloDensity.empty(), exchangeVelocity = !haloVelocity.empty();
    double velocityScale = lattice.getVelocityScale();
    vector<vector<double> > send(numRanks), receive(numRanks);
    int k = 0;
    for (int r = 0; r < numRanks; r++) {
        for (int node : haloSendNodes[r]) {
            double density = 0;
            Vec3 velocity;
            if (isFluidHost[node]) {
                double rho = 1.0 + moments[4*k];
                density = rho*lattice.density;
                velocity = (Vec3(moments[4*k+1], moments[4*k+2], moments[4*k+3])*(1.0/rho) + lattice.bodyAcceleration*0.5)*velocityScale;
            }
            k++;
            if (exchangeDensity)
                send[r].push_back(density);
            if (exchangeVelocity)
                for (int a = 0; a < 3; a++)
                    send[r].push_back(velocity[a]);
        }
        receive[r].resize(((exchangeDensity ? 1 : 0) + (exchangeVelocity ? 3 : 0))*haloReceiveNodes[r].size());
    }
    decomposition.exchange(send, receive);
    for (int r = 0; r < numRanks; r++) {
        int i = 0;
        for (int node : haloReceiveNodes[r]) {
            int h = haloIndex[node];
            if (exchangeDensity)
                haloDensity[h] = receive[r][i++];
            if (exchangeVelocity)
                for (int a = 0; a < 3; a++)
                    haloVelocity[3*h+a] = receive[r][i++];
        }
    }
}

Vec3 CommonCalcLBMForceKernel::getWallForce(ContextImpl& context) {
    // The momentum given to the solid nodes in the last step, divided by dt: the part of the deviations f - w,
    // summed over the solid nodes in the order of the list (bounce-back) or over the boundary nodes of the walls
    // in the order of their list (regularized walls), plus the static part of the weights w, plus the reflections
    // and reactions of the coupled particles, summed in particle order.  With the domain decomposition each rank sums
    // the links to its own fluid nodes, and the momenta of the ranks are added in rank order (every rank must call it).
    if (lattice.solidNodes.empty() || !hasAdvanced)
        return Vec3();
    ContextSelector selector(cc);
    vector<double> exchange;
    downloadAsDouble(wallExchange, exchange);
    Vec3 momentum;
    for (int i = 0; i < numSolidEntries; i++)
        momentum += Vec3(exchange[i], exchange[numSolidEntries+i], exchange[2*numSolidEntries+i]);
    momentum += staticWallMomentum;
    if (boundaryExchange.isInitialized()) {
        int numBoundaryNodes = boundaryNodes.getSize();
        downloadAsDouble(boundaryExchange, exchange);
        for (int b = 0; b < numBoundaryNodes; b++)
            momentum += Vec3(exchange[b], exchange[numBoundaryNodes+b], exchange[2*numBoundaryNodes+b]);
        momentum += staticBoundaryMomentum;
    }
    int numCoupled = lattice.particles.size();
    if (numCoupled > 0) {
        vector<double> particles;
        downloadAsDouble(particleWallMomentum, particles);
        for (int i = 0; i < numCoupled; i++)
            momentum += Vec3(particles[i], particles[numCoupled+i], particles[2*numCoupled+i]);
    }
    if (decomposition.isDecomposed()) {
        double sums[3] = {momentum[0], momentum[1], momentum[2]};
        decomposition.sumInRankOrder(sums, 3);
        momentum = Vec3(sums[0], sums[1], sums[2]);
    }
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    return momentum*(cellMass*lattice.dx/lattice.dt)*(1.0/lattice.dt);
}

double CommonCalcLBMForceKernel::getFluidMachNumber(ContextImpl& context) {
    ContextSelector selector(cc);
    return computeMachNumber();
}

void CommonCalcLBMForceKernel::getFluidState(ContextImpl& context, vector<double>& state) {
    // The populations of the domain of the rank, [q*numLocal + n] (with one domain the stored array itself).
    ContextSelector selector(cc);
    if (!decomposition.isDecomposed()) {
        downloadAsDouble(populations, state);
        return;
    }
    vector<double> stored;
    downloadAsDouble(populations, stored);
    state.resize(D3Q19::numVelocities*numLocal);
    for (int n = 0; n < numLocal; n++) {
        int s = storageIndex(n);
        for (int q = 0; q < D3Q19::numVelocities; q++)
            state[q*numLocal+n] = stored[q*numStored+s];
    }
}

void CommonCalcLBMForceKernel::setFluidState(ContextImpl& context, const vector<double>& state) {
    // The populations of the domain of the rank, in the layout of getFluidState().
    ContextSelector selector(cc);
    if (state.size() != D3Q19::numVelocities*numLocal)
        throw OpenMMException("LBMForce: setFluidState() was called with a state of the wrong size");
    if (!decomposition.isDecomposed())
        uploadPopulations(state);
    else {
        vector<double> stored;
        downloadAsDouble(populations, stored);
        for (int n = 0; n < numLocal; n++) {
            int s = storageIndex(n);
            for (int q = 0; q < D3Q19::numVelocities; q++)
                stored[q*numStored+s] = state[q*numLocal+n];
        }
        uploadPopulations(stored);
    }
    stepForcesCurrent = false;
    if (packFieldsKernel)
        exchangeHalo();
}

/** Write the content of an array, as it is on the device, if the array exists. */
static void writeArray(ComputeArray& array, ostream& stream) {
    if (!array.isInitialized())
        return;
    vector<char> buffer(array.getSize()*array.getElementSize());
    array.download(buffer.data());
    stream.write(buffer.data(), buffer.size());
}

/** Read the content of an array written by writeArray(). */
static void readArray(ComputeArray& array, istream& stream) {
    if (!array.isInitialized())
        return;
    vector<char> buffer(array.getSize()*array.getElementSize());
    stream.read(buffer.data(), buffer.size());
    if (stream)
        array.upload(buffer.data());
}

void CommonCalcLBMForceKernel::createCheckpoint(ContextImpl& context, ostream& stream) {
    // The populations, the random numbers drawn for the next step and the momentum given to the walls in the
    // last step, as they are on the device, so that the checkpoint is exact in every precision.  The random
    // number generator belongs to OpenMM and is part of OpenMM checkpoints.
    if (decomposition.isDecomposed())
        throw OpenMMException("LBMForce: with the domain decomposition the checkpoints are written to a file, with "
                "saveCheckpointFile() (openmmlbm.saveCheckpoint() in Python)");
    ContextSelector selector(cc);
    int elementSize = populations.getElementSize();
    stream.write((const char*) &elementSize, sizeof(int));
    int flags[2] = {noiseDrawn ? 1 : 0, hasAdvanced ? 1 : 0};
    stream.write((const char*) flags, sizeof(flags));
    writeArray(populations, stream);
    writeArray(noise, stream);
    writeArray(particleWallMomentum, stream);
    writeArray(wallExchange, stream);
    writeArray(boundaryExchange, stream);
}

void CommonCalcLBMForceKernel::loadCheckpoint(ContextImpl& context, istream& stream) {
    if (decomposition.isDecomposed())
        throw OpenMMException("LBMForce: with the domain decomposition the checkpoints are read from a file, with "
                "loadCheckpointFile() (openmmlbm.loadCheckpoint() in Python)");
    ContextSelector selector(cc);
    int elementSize;
    stream.read((char*) &elementSize, sizeof(int));
    if (!stream || elementSize != populations.getElementSize())
        throw OpenMMException("LBMForce: the checkpoint was written with a different precision");
    int flags[2];
    stream.read((char*) flags, sizeof(flags));
    noiseDrawn = (flags[0] != 0);
    hasAdvanced = (flags[1] != 0);
    stepForcesCurrent = false;
    vector<char> buffer(populations.getSize()*populations.getElementSize());
    stream.read(buffer.data(), buffer.size());
    if (stream)
        uploadPopulations(buffer.data());
    readArray(noise, stream);
    readArray(particleWallMomentum, stream);
    readArray(wallExchange, stream);
    readArray(boundaryExchange, stream);
}

void CommonCalcLBMForceKernel::createRankCheckpoint(ContextImpl& context, ostream& stream) {
    // What createCheckpoint() writes, except the populations.
    ContextSelector selector(cc);
    int elementSize = populations.getElementSize();
    stream.write((const char*) &elementSize, sizeof(int));
    int flags[2] = {noiseDrawn ? 1 : 0, hasAdvanced ? 1 : 0};
    stream.write((const char*) flags, sizeof(flags));
    writeArray(noise, stream);
    writeArray(particleWallMomentum, stream);
    writeArray(wallExchange, stream);
    writeArray(boundaryExchange, stream);
}

void CommonCalcLBMForceKernel::loadRankCheckpoint(ContextImpl& context, istream& stream) {
    ContextSelector selector(cc);
    int elementSize;
    stream.read((char*) &elementSize, sizeof(int));
    if (!stream || elementSize != populations.getElementSize())
        throw OpenMMException("LBMForce: the checkpoint was written with a different precision");
    int flags[2];
    stream.read((char*) flags, sizeof(flags));
    noiseDrawn = (flags[0] != 0);
    hasAdvanced = (flags[1] != 0);
    stepForcesCurrent = false;
    readArray(noise, stream);
    readArray(particleWallMomentum, stream);
    readArray(wallExchange, stream);
    readArray(boundaryExchange, stream);
}

void CommonCalcLBMForceKernel::resetRankState(ContextImpl& context) {
    // Without the momentum of the last step getWallForce() gives zero until the next step, as in a new Context.
    noiseDrawn = false;
    hasAdvanced = false;
    stepForcesCurrent = false;
}

