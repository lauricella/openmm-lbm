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
#include "openmm/OpenMMException.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/IntegrationUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
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

void CommonCalcLBMForceKernel::initialize(const System& system, const LBMForce& force, const LBMLatticeParameters& lattice) {
    ContextSelector selector(cc);
    if (cc.getNumContexts() > 1)
        throw OpenMMException("LBMForce does not support running on multiple devices");
    if (lattice.isDecomposed())
        throw OpenMMException("LBMForce: the domain decomposition (setDomainDecomposition()) is not available yet on "
                "the CUDA, OpenCL and HIP platforms");
    this->lattice = lattice;
    forceGroup = force.getForceGroup();
    bool centered = (lattice.dragScheme == LBMForce::Centered);

    // The fluid is stored in the mixed type: double unless the platform runs in single precision.

    int numNodes = lattice.getNumNodes();
    useDouble = (cc.getUseDoublePrecision() || cc.getUseMixedPrecision());
    int elementSize = (useDouble ? sizeof(double) : sizeof(float));
    blockSize = ComputeContext::ThreadBlockSize;
    numGroups = max(1, min(cc.getNumThreadBlocks(), (numNodes+blockSize-1)/blockSize));
    populations.initialize(cc, D3Q19::numVelocities*numNodes, elementSize, "lbmPopulations");
    densityDeviation.initialize(cc, numNodes, elementSize, "lbmDensityDeviation");
    momentum.initialize(cc, 3*numNodes, elementSize, "lbmMomentum");
    piNeq.initialize(cc, 6*numNodes, elementSize, "lbmPiNeq");
    partialSums.initialize(cc, 4*numGroups, elementSize, "lbmPartialSums");
    partialMax.initialize(cc, numGroups, elementSize, "lbmPartialMax");
    centerVelocity.initialize(cc, 3, elementSize, "lbmCenterVelocity");

    // The fluid starts at equilibrium, with lattice density 1 and the initial velocity.  As on the Reference
    // platform, the populations are stored as deviations from the rest equilibrium, f_q - w_q.

    vector<double> f(D3Q19::numVelocities*numNodes);
    double dfeq[D3Q19::numVelocities];
    D3Q19::equilibriumDeviation(0.0, lattice.initialVelocity[0], lattice.initialVelocity[1], lattice.initialVelocity[2], dfeq);
    for (int q = 0; q < D3Q19::numVelocities; q++)
        for (int node = 0; node < numNodes; node++)
            f[q*numNodes+node] = dfeq[q];

    // Solid nodes hold no fluid: their populations start at zero, that is at the deviation -w_q.  With bounce-back
    // the part w of the populations gives the walls the same momentum in every step (the static pressure): it is
    // computed here once, in double precision, with the links from the solid nodes to the fluid nodes that do not
    // cross an open face.  The same links, seen from the fluid nodes next to the walls (wallNodes), are the bits of
    // wallLinks: along them the bounce-back returns the populations to the fluid.

    int numSolidNodes = lattice.solidNodes.size();
    int size[3] = {lattice.nx, lattice.ny, lattice.nz};
    isFluidHost.assign(numNodes, 1);
    for (int node : lattice.solidNodes)
        isFluidHost[node] = 0;
    staticWallMomentum = Vec3();
    for (int node : lattice.solidNodes) {
        int index[3] = {node%lattice.nx, (node/lattice.nx)%lattice.ny, node/(lattice.nx*lattice.ny)};
        for (int q = 0; q < D3Q19::numVelocities; q++)
            f[q*numNodes+node] = -D3Q19::w[q];
        if (lattice.wallScheme != LBMForce::BounceBack)
            continue;
        for (int q = 1; q < D3Q19::numVelocities; q++) {
            int c[3] = {D3Q19::cx[q], D3Q19::cy[q], D3Q19::cz[q]}, target[3];
            bool crossesFace = false;
            for (int a = 0; a < 3; a++) {
                target[a] = index[a] + c[a];
                crossesFace = crossesFace || (lattice.isOpenAxis(a) && (target[a] < 0 || target[a] >= size[a]));
                target[a] = (target[a] + size[a])%size[a];
            }
            if (!crossesFace && isFluidHost[target[0] + lattice.nx*(target[1] + lattice.ny*target[2])])
                staticWallMomentum -= Vec3(c[0], c[1], c[2])*(2.0*D3Q19::w[q]);
        }
    }

    // Boundary nodes (docs/theory.md, section 1, and internal/LBMBoundaries.h): with regularized walls the fluid
    // nodes next to the solid nodes, and the fluid nodes on the open faces.  They are fluid nodes like the others,
    // except that applyBoundaries rebuilds their unknown populations.  The part w of the populations that a wall node
    // sends into the solid nodes and receives from them gives the walls the same momentum in every step, -2 c_q w_q
    // for each solid direction q.

    LBMBoundaries boundaries;
    boundaries.find(lattice, isFluidHost);
    int numBoundaryNodes = boundaries.nodes.size();
    staticBoundaryMomentum = Vec3();
    vector<int> kindAndFace(numBoundaryNodes);
    for (int b = 0; b < numBoundaryNodes; b++) {
        kindAndFace[b] = boundaries.kind[b] + 4*(boundaries.face[b] + 1);
        for (int q = 1; q < D3Q19::numVelocities; q++)
            if (boundaries.solid[b] & (1<<q))
                staticBoundaryMomentum -= Vec3(D3Q19::cx[q], D3Q19::cy[q], D3Q19::cz[q])*(2.0*D3Q19::w[q]);
    }
    if (numBoundaryNodes > 0) {
        boundaryNodes.initialize<int>(cc, numBoundaryNodes, "lbmBoundaryNodes");
        boundaryNodes.upload(boundaries.nodes);
        boundaryUnknown.initialize<int>(cc, numBoundaryNodes, "lbmBoundaryUnknown");
        boundaryUnknown.upload(boundaries.unknown);
        boundarySolid.initialize<int>(cc, numBoundaryNodes, "lbmBoundarySolid");
        boundarySolid.upload(boundaries.solid);
        boundaryKindAndFace.initialize<int>(cc, numBoundaryNodes, "lbmBoundaryKindAndFace");
        boundaryKindAndFace.upload(kindAndFace);
        boundaryExchange.initialize(cc, 3*numBoundaryNodes, elementSize, "lbmBoundaryExchange");
        boundaryExchange.upload(vector<double>(3*numBoundaryNodes, 0.0), true);
        faceParameters.initialize(cc, 24, elementSize, "lbmFaceParameters");
    }
    populations.upload(f, true);
    isFluid.initialize<int>(cc, numNodes, "lbmIsFluid");
    isFluid.upload(isFluidHost);
    vector<int> wallNodesHost, wallLinksHost;
    LBMBoundaries::findWallLinks(lattice, isFluidHost, wallNodesHost, wallLinksHost);
    int numWallNodes = wallNodesHost.size();
    if (numWallNodes > 0) {
        wallNodes.initialize<int>(cc, numWallNodes, "lbmWallNodes");
        wallNodes.upload(wallNodesHost);
        wallLinks.initialize<int>(cc, numWallNodes, "lbmWallLinks");
        wallLinks.upload(wallLinksHost);
    }
    solidNodes.initialize<int>(cc, max(1, numSolidNodes), "lbmSolidNodes");
    wallExchange.initialize(cc, 3*max(1, numSolidNodes), elementSize, "lbmWallExchange");
    if (numSolidNodes > 0)
        solidNodes.upload(lattice.solidNodes);
    wallExchange.upload(vector<double>(wallExchange.getSize(), 0.0), true);

    // Coupled particles: their index in the list of the force for every atom, masses in units of the mass of a
    // cell, m_c = rho0 dx^3, and no coupling force before the first step.  The random force uses OpenMM's
    // generator, seeded with the seed of the force.

    int numCoupled = lattice.particles.size();
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    cellReaction.initialize(cc, (numCoupled > 0 ? 3*numNodes : 1), elementSize, "lbmCellReaction");
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
        cc.getIntegrationUtilities().initRandomNumberGenerator((unsigned int) lattice.randomNumberSeed);
    }

    // A fluctuating fluid (docs/theory.md, section 7) draws four float4 of normal numbers per node and step from
    // OpenMM's generator, after those of the particles, and turns 15 of them into the random part of the
    // populations with the coefficients w_q e_k(c_q)/sqrt(b_k) of the orthogonal basis, k = 4...18.

    if (lattice.fluidFluctuations) {
        cc.getIntegrationUtilities().initRandomNumberGenerator((unsigned int) lattice.randomNumberSeed);
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
        cc.getIntegrationUtilities().prepareRandomNumbers(4*(numNodes+numBoundaryNodes));
    }

    // Compile the kernels.

    map<string, string> defines;
    defines["NUM_NODES"] = cc.intToString(numNodes);
    defines["NX"] = cc.intToString(lattice.nx);
    defines["NY"] = cc.intToString(lattice.ny);
    defines["NZ"] = cc.intToString(lattice.nz);
    defines["LBM_BLOCK_SIZE"] = cc.intToString(blockSize);
    defines["W0"] = cc.doubleToString(1.0/3.0, true);
    defines["W1"] = cc.doubleToString(1.0/18.0, true);
    defines["W2"] = cc.doubleToString(1.0/36.0, true);
    defines["CS2"] = cc.doubleToString(1.0/3.0, true);
    if (numSolidNodes > 0) {
        defines["HAS_SOLID_NODES"] = "1";
        defines["NUM_SOLID_NODES"] = cc.intToString(numSolidNodes);
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
    if (numWallNodes > 0) {
        bounceBackKernel = program->createKernel("bounceBack");
        bounceBackKernel->addArg(populations);
        bounceBackKernel->addArg(wallNodes);
        bounceBackKernel->addArg(wallLinks);
        wallExchangeKernel = program->createKernel("computeWallExchange");
        wallExchangeKernel->addArg(populations);
        wallExchangeKernel->addArg(isFluid);
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
            reflectKernel->addArg(isFluid);
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
    if (!lattice.particles.empty() && !lattice.solidNodes.empty()) {
        ContextSelector selector(cc);
        reflectKernel->execute(cc.getNumAtoms());
    }
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
    int numNodes = lattice.getNumNodes();
    computeMomentsKernel->execute(numNodes);
    if (lattice.momentumRemovalFrequency > 0 && stepIndex%lattice.momentumRemovalFrequency == 0) {
        sumMomentumKernel->execute(numGroups*blockSize, blockSize);
        centerVelocityKernel->execute(blockSize, blockSize);
        removeMomentumKernel->execute(numNodes);
    }
    int numCoupled = lattice.particles.size();
    if (numCoupled > 0)
        computeCouplingForces(true);
    // The numbers of the fluid are drawn after those of the particles, which the force evaluations between steps
    // may already have drawn: the sequence is the same in both cases.
    if (lattice.fluidFluctuations && lattice.fluidKT > 0) {
        int numBoundaryNodes = (boundaryNodes.isInitialized() ? boundaryNodes.getSize() : 0);
        int randomIndex = cc.getIntegrationUtilities().prepareRandomNumbers(4*(numNodes+numBoundaryNodes));
        collideKernel->setArg(13, randomIndex);
        if (applyBoundariesKernel)
            applyBoundariesKernel->setArg(17, randomIndex);
    }
    collideKernel->execute(numNodes);
    if (numCoupled > 0)
        clearReactionsKernel->execute(numCoupled);
    if (bounceBackKernel) {
        bounceBackKernel->execute(wallNodes.getSize());
        wallExchangeKernel->execute(lattice.solidNodes.size());
    }
    if (applyBoundariesKernel)
        applyBoundariesKernel->execute(boundaryNodes.getSize());
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
    int numNodes = lattice.getNumNodes();
    computeMomentsKernel->execute(numNodes);
    if (lattice.momentumRemovalFrequency > 0 && nextStep%lattice.momentumRemovalFrequency == 0) {
        sumMomentumKernel->execute(numGroups*blockSize, blockSize);
        centerVelocityKernel->execute(blockSize, blockSize);
        removeMomentumKernel->execute(numNodes);
    }
    computeCouplingForces(false);
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
    // Ma = max |j/rho|/c_s, with c_s^2 = 1/3: maxima by work group on the device, then over the groups here.
    maxSpeedKernel->execute(numGroups*blockSize, blockSize);
    vector<double> maxima;
    downloadAsDouble(partialMax, maxima);
    double maxSpeed2 = *max_element(maxima.begin(), maxima.end());
    return sqrt(3.0*maxSpeed2);
}

void CommonCalcLBMForceKernel::copyParametersToContext(ContextImpl& context, const LBMLatticeParameters& lattice) {
    ContextSelector selector(cc);
    this->lattice = lattice;
    setFluidParameters();
}

void CommonCalcLBMForceKernel::getFluidFields(ContextImpl& context, vector<double>& density, vector<Vec3>& velocity) {
    // The velocity of the forced fluid is u = (j + F/2)/rho, with F = rho*g from the body acceleration.  The
    // moments are recomputed from the populations; the next step computes them again before using them.
    ContextSelector selector(cc);
    int numNodes = lattice.getNumNodes();
    computeMomentsKernel->execute(numNodes);
    vector<double> dr, j;
    downloadAsDouble(densityDeviation, dr);
    downloadAsDouble(momentum, j);
    double velocityScale = lattice.getVelocityScale();
    density.resize(numNodes);
    velocity.resize(numNodes);
    for (int node = 0; node < numNodes; node++) {
        if (!isFluidHost[node]) {
            density[node] = 0;
            velocity[node] = Vec3();
            continue;
        }
        double r = 1.0 + dr[node];
        density[node] = r*lattice.density;
        velocity[node] = (Vec3(j[node], j[numNodes+node], j[2*numNodes+node])*(1.0/r) + lattice.bodyAcceleration*0.5)*velocityScale;
    }
}

Vec3 CommonCalcLBMForceKernel::getWallForce(ContextImpl& context) {
    // The momentum given to the solid nodes in the last step, divided by dt: the part of the deviations f - w,
    // summed over the solid nodes in the order of the list (bounce-back) or over the boundary nodes of the walls
    // in the order of their list (regularized walls), plus the static part of the weights w, plus the reflections
    // and reactions of the coupled particles, summed in particle order.
    int numSolidNodes = lattice.solidNodes.size();
    if (numSolidNodes == 0 || !hasAdvanced)
        return Vec3();
    ContextSelector selector(cc);
    vector<double> exchange;
    downloadAsDouble(wallExchange, exchange);
    Vec3 momentum;
    for (int i = 0; i < numSolidNodes; i++)
        momentum += Vec3(exchange[i], exchange[numSolidNodes+i], exchange[2*numSolidNodes+i]);
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
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    return momentum*(cellMass*lattice.dx/lattice.dt)*(1.0/lattice.dt);
}

double CommonCalcLBMForceKernel::getFluidMachNumber(ContextImpl& context) {
    ContextSelector selector(cc);
    return computeMachNumber();
}

void CommonCalcLBMForceKernel::getFluidState(ContextImpl& context, vector<double>& state) {
    ContextSelector selector(cc);
    downloadAsDouble(populations, state);
}

void CommonCalcLBMForceKernel::setFluidState(ContextImpl& context, const vector<double>& state) {
    ContextSelector selector(cc);
    if (state.size() != populations.getSize())
        throw OpenMMException("LBMForce: setFluidState() was called with a state of the wrong size");
    populations.upload(state, true);
    stepForcesCurrent = false;
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
    readArray(populations, stream);
    readArray(noise, stream);
    readArray(particleWallMomentum, stream);
    readArray(wallExchange, stream);
    readArray(boundaryExchange, stream);
}

