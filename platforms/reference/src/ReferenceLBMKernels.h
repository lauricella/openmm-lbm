#ifndef REFERENCE_LBM_KERNELS_H_
#define REFERENCE_LBM_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "LBMKernels.h"
#include "openmm/Platform.h"
#include <vector>

namespace LBMPlugin {

/**
 * The Reference implementation of the lattice Boltzmann fluid: plain loops in double precision.
 * It is the correctness reference for the other platforms.
 */
class ReferenceCalcLBMForceKernel : public CalcLBMForceKernel {
public:
    ReferenceCalcLBMForceKernel(std::string name, const OpenMM::Platform& platform) : CalcLBMForceKernel(name, platform) {
    }
    void initialize(const OpenMM::System& system, const LBMForce& force, const LBMLatticeParameters& lattice);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const LBMLatticeParameters& lattice);
    void getFluidFields(OpenMM::ContextImpl& context, std::vector<double>& density, std::vector<OpenMM::Vec3>& velocity);
    void getFluidState(OpenMM::ContextImpl& context, std::vector<double>& state);
    void setFluidState(OpenMM::ContextImpl& context, const std::vector<double>& state);
private:
    LBMLatticeParameters lattice;
    /** Populations, stored as f[q*numNodes + node]. */
    std::vector<double> populations;
};

} // namespace LBMPlugin

#endif /*REFERENCE_LBM_KERNELS_H_*/
