#ifndef COMMON_LBM_KERNELS_H_
#define COMMON_LBM_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "LBMKernels.h"
#include "internal/LBMDecomposition.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeKernel.h"
#include <map>

namespace LBMPlugin {

/**
 * The implementation of the lattice Boltzmann fluid shared by the CUDA, OpenCL and HIP platforms.
 * The fluid is stored in the "mixed" type of the platform: float in single precision, double in
 * mixed and double precision.
 *
 * One lattice step has the structure of the Reference platform (ReferenceLBMKernels.h): moments, removal of
 * the fluid momentum when due, coupling of the particles, collision and streaming, bounce-back at the solid
 * nodes, and the Mach number check when due.  The reactions of the particles are summed per node in particle
 * order without atomic operations: the keys node*numCoupled + i are sorted with OpenMM's ComputeSort, and the
 * first entry of each node sums its segment.
 *
 * With the Centered drag scheme the coupling needs the other forces on the particles at the time of the force
 * evaluation, which OpenMM completes only at the end of it: the kernel then does its work in a
 * ForcePostComputation, registered when the Context is created, and execute() does nothing.
 *
 * With the domain decomposition (docs/theory.md, section 8) each rank stores only its block of the lattice, plus a
 * layer of halo nodes along the divided axes (kernels/lbmFluid.cc).  After the collision of the frame of the block
 * (the fluid nodes that push populations into other ranks) the populations pushed into the halo are packed, copied to
 * the host and sent to their owners with non-blocking MPI calls while the interior of the block collides; the
 * populations received are then copied into the block, before the walls and the boundaries.  The lists of slots are
 * built once, in the order of (node, q) of the Reference platform.
 */
class CommonCalcLBMForceKernel : public CalcLBMForceKernel {
public:
    CommonCalcLBMForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system) :
            CalcLBMForceKernel(name, platform), cc(cc), system(system), stepPending(false), stepIndex(0), machWarningPrinted(false),
            hasAdvanced(false), stepForcesCurrent(false), noiseDrawn(false), numFloatForceBuffers(-1), deterministicForces(true),
            replicasChecked(false) {
    }
    /**
     * Whether the platform computes the forces of OpenMM in a deterministic way (the property DeterministicForces of the
     * CUDA and HIP platforms), set by the kernel factory before initialize().  With the domain decomposition and coupled
     * particles every rank must compute the same forces on its copies of the particles.
     */
    void setDeterministicForces(bool deterministic) {
        deterministicForces = deterministic;
    }
    void initialize(const OpenMM::System& system, const LBMForce& force, const LBMLatticeParameters& lattice);
    void beginStep(OpenMM::ContextImpl& context);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const LBMLatticeParameters& lattice);
    void getFluidFields(OpenMM::ContextImpl& context, std::vector<double>& density, std::vector<OpenMM::Vec3>& velocity, bool halo);
    double getFluidMachNumber(OpenMM::ContextImpl& context);
    OpenMM::Vec3 getWallForce(OpenMM::ContextImpl& context);
    void getFluidState(OpenMM::ContextImpl& context, std::vector<double>& state);
    void setFluidState(OpenMM::ContextImpl& context, const std::vector<double>& state);
    void createCheckpoint(OpenMM::ContextImpl& context, std::ostream& stream);
    void loadCheckpoint(OpenMM::ContextImpl& context, std::istream& stream);
    void createRankCheckpoint(OpenMM::ContextImpl& context, std::ostream& stream);
    void loadRankCheckpoint(OpenMM::ContextImpl& context, std::istream& stream);
    void resetRankState(OpenMM::ContextImpl& context);
protected:
    /**
     * True if the platform accumulates part of the forces in the floating point buffers getForceBuffers() of the
     * ComputeContext, which it adds to the fixed point buffer only after the post-computations (OpenCL).
     */
    virtual bool hasFloatForceBuffers() const {
        return false;
    }
private:
    class CenteredDragPostComputation;
    double evaluate(long long stepCount, bool includeForces);
    void advanceFluid();
    void computeNextStepForces(long long nextStep);
    void computeCouplingForces(bool isStep);
    void setFluidParameters();
    void checkMachNumber();
    double computeMachNumber();
    void removeFluidMomentum();
    void collideAndStream();
    /** With the domain decomposition: sum the coupling forces over the ranks, and check that the copies of the
        particles are the same on every rank (collective). */
    void sumParticleForces();
    void checkReplicas(OpenMM::ContextImpl& context);
    /** With the domain decomposition and the exchange of the halo: send the fields of the nodes of the block to the
        ranks whose halo contains them, and receive those of the halo of the rank (collective). */
    void exchangeHalo();
    /** The global index of node n of the block, its storage index (kernels/lbmFluid.cc), and the index in the block of
        a node of the lattice that belongs to the block. */
    int globalNode(int n) const;
    int storageIndex(int n) const;
    int localNode(int node) const;
    OpenMM::ComputeContext& cc;
    const OpenMM::System& system;
    LBMLatticeParameters lattice;
    /** The decomposition of the lattice, the first node and the extent of the block of the rank along each axis, 1
        along the divided axes (which have a layer of halo nodes on each side) and 0 along the others, and the number
        of nodes of the block and of entries per component of the arrays.  Without the decomposition the block is the
        lattice and numStored = numLocal. */
    LBMDecomposition decomposition;
    int localStart[3], localCount[3], pad[3];
    int numLocal, numStored;
    /** The number of entries of solidNodes (the solid nodes, with the decomposition those linked to fluid nodes of the
        block), and of the fluid nodes of the frame and of the interior of the block. */
    int numSolidEntries, numFrameNodes, numInteriorNodes;
    /** See setDeterministicForces(); true once the copies of the particles have been compared (checkReplicas()). */
    bool deterministicForces, replicasChecked;
    /** True between beginStep() and the force evaluation of that integration step. */
    bool stepPending;
    /** Step count of the Context: set by beginStep() at the start of a lattice step, incremented at its end. */
    long long stepIndex;
    /** True once the debug warning about the Mach number has been printed. */
    bool machWarningPrinted;
    /** True once the fluid has advanced by a step: before that the force on the walls is zero. */
    bool hasAdvanced;
    /** True from the lattice step until the coupling forces are recomputed or the state is replaced: the forces
        hold those of the step. */
    bool stepForcesCurrent;
    /** True if noise holds the random numbers of the next lattice step. */
    bool noiseDrawn;
    /** True if the mixed type is double. */
    bool useDouble;
    /** The force group of the force. */
    int forceGroup;
    /** Number of floating point force buffers read by the Centered drag, set on its first use (-1 before). */
    int numFloatForceBuffers;
    /** 1 for fluid nodes and 0 for solid nodes, on the host. */
    std::vector<int> isFluidHost;
    /** Momentum (lattice units) given to the walls in every step by the part w of the populations, the static
        pressure: it depends only on the geometry. */
    OpenMM::Vec3 staticWallMomentum;
    /** The same for the nodes of regularized walls: the part w of the populations that they receive and send. */
    OpenMM::Vec3 staticBoundaryMomentum;
    /** Work group size and number of work groups of the reductions. */
    int blockSize, numGroups;
    /** Deviations of the populations from the rest equilibrium, f_q - w_q, at [q*numNodes + node]. */
    OpenMM::ComputeArray populations;
    /** Moments of the current step: rho - 1, momentum (3 components of numNodes each) and non-equilibrium
        second moment (6 components of numNodes each), in lattice units. */
    OpenMM::ComputeArray densityDeviation, momentum, piNeq;
    /** Partial sums and maxima of the work groups, and the velocity of the centre of mass of the fluid. */
    OpenMM::ComputeArray partialSums, partialMax, centerVelocity;
    /** 1 for fluid nodes and 0 for solid nodes; the list of the solid nodes; the momentum given to each solid node by the deviations f - w in the last step
        (3 components of numSolidNodes each); with bounce-back walls, the fluid nodes next to the solid nodes and the
        bits 1 << q of their directions q towards solid nodes (LBMBoundaries::findWallLinks()). */
    OpenMM::ComputeArray isFluid, solidNodes, solidLinks, wallExchange, wallNodes, wallLinks;
    /** With the domain decomposition and coupled particles, the mask of the fluid nodes of the whole lattice, for the
        reflection of the particles at the walls (every rank reflects all of them). */
    OpenMM::ComputeArray globalIsFluid;
    /** With the domain decomposition: the fluid nodes of the frame and of the interior of the block; the slots of the
        populations to send (in the halo) and to receive (in the block), rank after rank; the buffers of the device; and
        the number of slots for each rank and the buffers of the host. */
    OpenMM::ComputeArray frameNodes, interiorNodes, sendSlots, receiveSlots, sendBuffer, receiveBuffer;
    std::vector<int> sendCounts, receiveCounts;
    std::vector<std::vector<double> > sendBuffers, receiveBuffers;
    /** The halo of the rank (setDensityHaloExchange(), setVelocityHaloExchange()), as on the Reference platform: for each
        rank r, the nodes of the block in the halo of r and the nodes of r in the halo of the block, in index order; the
        storage indices of the first, rank after rank, and the buffer of their moments; and the fields of the nodes of
        the halo in the units of getFluidFields() (density, and velocity with 3 values per node), at the index that
        haloIndex gives each node, empty when not exchanged. */
    std::vector<std::vector<int> > haloSendNodes, haloReceiveNodes;
    OpenMM::ComputeArray haloSendStorage, haloBuffer;
    std::map<int, int> haloIndex;
    std::vector<double> haloDensity, haloVelocity;
    /** Boundary nodes (regularized walls and open faces, internal/LBMBoundaries.h): the nodes, the bits of their
        unknown and solid directions, kind + 4*(face + 1), the momentum given to the wall by each node in the last
        step (deviations f - w, 3 components of numBoundaryNodes each), and the velocity and density minus 1 of the
        six faces (lattice units, 4 per face). */
    OpenMM::ComputeArray boundaryNodes, boundaryUnknown, boundarySolid, boundaryKindAndFace, boundaryExchange, faceParameters;
    /** Coupled particles: index in the list of the force of every atom of the System (-1 if not coupled); masses
        (lattice units); coupling forces (lattice units, 3 components of numCoupled each), those of the step in
        its force evaluation and those of the next step between steps; sort keys node*numCoupled + i; momentum
        given to the walls in the last step by reflections and reactions at solid nodes (lattice units); reaction
        of the particles on every node (3 components of numNodes each); random numbers of the next step, copied
        from OpenMM's generator (one float4 per particle). */
    OpenMM::ComputeArray couplingIndex, particleMass, particleForce, sortKeys, particleWallMomentum, cellReaction;
    OpenMM::ComputeArray noise;
    /** Centered drag: velocity of every coupled particle with half the other forces, v(t - dt/2) + dt Fc/(2m), and
        its random force (lattice units, 3 components of numCoupled each). */
    OpenMM::ComputeArray knownVelocity, randomForce;
    /** Fluctuating fluid: the 19x15 coefficients w_q e_k(c_q)/sqrt(b_k), k = 4...18, that turn the normal numbers of
        a node into the random part of its populations (docs/theory.md, section 7). */
    OpenMM::ComputeArray fluctuationBasis;
    OpenMM::ComputeSort sort;
    OpenMM::ComputeKernel computeMomentsKernel, sumMomentumKernel, centerVelocityKernel, removeMomentumKernel;
    OpenMM::ComputeKernel collideKernel, bounceBackKernel, wallExchangeKernel, applyBoundariesKernel, maxSpeedKernel;
    OpenMM::ComputeKernel reflectKernel, coupleKernel, sumReactionsKernel, clearReactionsKernel, applyForcesKernel;
    OpenMM::ComputeKernel prepareCenteredKernel, solveCenteredKernel, packKernel, unpackKernel, packFieldsKernel;
};

} // namespace LBMPlugin

#endif /*COMMON_LBM_KERNELS_H_*/
