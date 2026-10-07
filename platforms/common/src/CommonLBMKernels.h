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
#include "openmm/common/ComputeArray.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeKernel.h"

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
 */
class CommonCalcLBMForceKernel : public CalcLBMForceKernel {
public:
    CommonCalcLBMForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system) :
            CalcLBMForceKernel(name, platform), cc(cc), system(system), stepPending(false), stepIndex(0), machWarningPrinted(false),
            hasAdvanced(false), stepForcesCurrent(false), noiseDrawn(false), numFloatForceBuffers(-1) {
    }
    void initialize(const OpenMM::System& system, const LBMForce& force, const LBMLatticeParameters& lattice);
    void beginStep(OpenMM::ContextImpl& context);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const LBMLatticeParameters& lattice);
    void getFluidFields(OpenMM::ContextImpl& context, std::vector<double>& density, std::vector<OpenMM::Vec3>& velocity);
    double getFluidMachNumber(OpenMM::ContextImpl& context);
    OpenMM::Vec3 getWallForce(OpenMM::ContextImpl& context);
    void getFluidState(OpenMM::ContextImpl& context, std::vector<double>& state);
    void setFluidState(OpenMM::ContextImpl& context, const std::vector<double>& state);
    void createCheckpoint(OpenMM::ContextImpl& context, std::ostream& stream);
    void loadCheckpoint(OpenMM::ContextImpl& context, std::istream& stream);
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
    OpenMM::ComputeContext& cc;
    const OpenMM::System& system;
    LBMLatticeParameters lattice;
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
    /** Work group size and number of work groups of the reductions. */
    int blockSize, numGroups;
    /** Deviations of the populations from the rest equilibrium, f_q - w_q, at [q*numNodes + node]. */
    OpenMM::ComputeArray populations;
    /** Moments of the current step: rho - 1, momentum (3 components of numNodes each) and non-equilibrium
        second moment (6 components of numNodes each), in lattice units. */
    OpenMM::ComputeArray densityDeviation, momentum, piNeq;
    /** Partial sums and maxima of the work groups, and the velocity of the centre of mass of the fluid. */
    OpenMM::ComputeArray partialSums, partialMax, centerVelocity;
    /** 1 for fluid nodes and 0 for solid nodes; the list of the solid nodes; the momentum given to each solid
        node by the deviations f - w in the last step (3 components of numSolidNodes each). */
    OpenMM::ComputeArray isFluid, solidNodes, wallExchange;
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
    OpenMM::ComputeKernel collideKernel, bounceBackKernel, maxSpeedKernel;
    OpenMM::ComputeKernel reflectKernel, coupleKernel, sumReactionsKernel, clearReactionsKernel, applyForcesKernel;
    OpenMM::ComputeKernel prepareCenteredKernel, solveCenteredKernel;
};

} // namespace LBMPlugin

#endif /*COMMON_LBM_KERNELS_H_*/
