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
 *
 * One lattice step (advanceFluid) has the same structure as on the other platforms:
 *  1. moments: density, momentum, non-equilibrium second moment and body force at every node;
 *  2. removal of the fluid momentum, every momentumRemovalFrequency steps;
 *  3. collision and streaming: the populations are rebuilt from the moments of their own node and pushed
 *     to the neighbours.  The collision reads only moments, so a single population array is enough.
 */
class ReferenceCalcLBMForceKernel : public CalcLBMForceKernel {
public:
    ReferenceCalcLBMForceKernel(std::string name, const OpenMM::Platform& platform) : CalcLBMForceKernel(name, platform),
            stepPending(false), stepIndex(0) {
    }
    void initialize(const OpenMM::System& system, const LBMForce& force, const LBMLatticeParameters& lattice);
    void beginStep(OpenMM::ContextImpl& context);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const LBMLatticeParameters& lattice);
    void getFluidFields(OpenMM::ContextImpl& context, std::vector<double>& density, std::vector<OpenMM::Vec3>& velocity);
    void getFluidState(OpenMM::ContextImpl& context, std::vector<double>& state);
    void setFluidState(OpenMM::ContextImpl& context, const std::vector<double>& state);
private:
    void advanceFluid();
    void computeMoments();
    void removeFluidMomentum();
    void collideAndStream();
    LBMLatticeParameters lattice;
    /** Populations, stored as f[q*numNodes + node]. */
    std::vector<double> populations;
    /** Moments of the current step: density, momentum j = rho*u (3 per node), non-equilibrium second moment
        (xx, yy, zz, xy, xz, yz per node) and force density (3 per node), all in lattice units. */
    std::vector<double> rho, momentum, piNeq, forceDensity;
    /** True between beginStep() and the force evaluation of that integration step. */
    bool stepPending;
    /** Number of lattice steps taken so far. */
    long long stepIndex;
};

} // namespace LBMPlugin

#endif /*REFERENCE_LBM_KERNELS_H_*/
