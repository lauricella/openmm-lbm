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
 */
class CommonCalcLBMForceKernel : public CalcLBMForceKernel {
public:
    CommonCalcLBMForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system) :
            CalcLBMForceKernel(name, platform), cc(cc), system(system) {
    }
    void initialize(const OpenMM::System& system, const LBMForce& force, const LBMLatticeParameters& lattice);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const LBMLatticeParameters& lattice);
    void getFluidFields(OpenMM::ContextImpl& context, std::vector<double>& density, std::vector<OpenMM::Vec3>& velocity);
    void getFluidState(OpenMM::ContextImpl& context, std::vector<double>& state);
    void setFluidState(OpenMM::ContextImpl& context, const std::vector<double>& state);
private:
    OpenMM::ComputeContext& cc;
    const OpenMM::System& system;
    LBMLatticeParameters lattice;
    /** Populations f[q*numNodes + node], density and momentum (3 components of numNodes each). */
    OpenMM::ComputeArray populations, density, momentum;
    OpenMM::ComputeKernel computeMomentsKernel;
};

} // namespace LBMPlugin

#endif /*COMMON_LBM_KERNELS_H_*/
