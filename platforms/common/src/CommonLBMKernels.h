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
 * the fluid momentum when due, collision and streaming, and the Mach number check when due.  The particle-fluid
 * coupling and the solid nodes are not implemented on these platforms yet.
 */
class CommonCalcLBMForceKernel : public CalcLBMForceKernel {
public:
    CommonCalcLBMForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system) :
            CalcLBMForceKernel(name, platform), cc(cc), system(system), stepPending(false), stepIndex(0), machWarningPrinted(false) {
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
private:
    void advanceFluid();
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
    /** True if the mixed type is double. */
    bool useDouble;
    /** Work group size and number of work groups of the reductions. */
    int blockSize, numGroups;
    /** Deviations of the populations from the rest equilibrium, f_q - w_q, at [q*numNodes + node]. */
    OpenMM::ComputeArray populations;
    /** Moments of the current step: rho - 1, momentum (3 components of numNodes each) and non-equilibrium
        second moment (6 components of numNodes each), in lattice units. */
    OpenMM::ComputeArray densityDeviation, momentum, piNeq;
    /** Partial sums and maxima of the work groups, and the velocity of the centre of mass of the fluid. */
    OpenMM::ComputeArray partialSums, partialMax, centerVelocity;
    OpenMM::ComputeKernel computeMomentsKernel, sumMomentumKernel, centerVelocityKernel, removeMomentumKernel;
    OpenMM::ComputeKernel collideKernel, maxSpeedKernel;
};

} // namespace LBMPlugin

#endif /*COMMON_LBM_KERNELS_H_*/
