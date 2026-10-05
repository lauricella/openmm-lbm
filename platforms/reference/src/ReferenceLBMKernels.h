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
#include "sfmt/SFMT.h"
#include <vector>

namespace LBMPlugin {

/**
 * The Reference implementation of the lattice Boltzmann fluid and of its coupling to the particles: plain
 * loops in double precision.  It is the correctness reference for the other platforms.
 *
 * One lattice step (advanceFluid) has the same structure as on the other platforms:
 *  1. moments: density, momentum, non-equilibrium second moment and body force at every node;
 *  2. removal of the fluid momentum, in the steps whose index (the step count of the Context) is a multiple
 *     of momentumRemovalFrequency;
 *  3. coupling: each coupled particle feels the explicit Euler-Maruyama drag and random force at its nearest
 *     node, and the node receives the opposite force (docs/theory.md, section 2);
 *  4. collision and streaming: the populations are rebuilt from the moments of their own node and pushed
 *     to the neighbours.  The collision reads only moments, so a single population array is enough;
 *  5. bounce-back at the solid nodes, if any: a population that streamed into a solid node is sent back
 *     to the fluid node it came from.  Solid nodes have no moments and no collision.
 */
class ReferenceCalcLBMForceKernel : public CalcLBMForceKernel {
public:
    ReferenceCalcLBMForceKernel(std::string name, const OpenMM::Platform& platform) : CalcLBMForceKernel(name, platform),
            stepPending(false), stepIndex(0), machWarningPrinted(false), hasStoredGaussian(false), storedGaussian(0) {
    }
    void initialize(const OpenMM::System& system, const LBMForce& force, const LBMLatticeParameters& lattice);
    void beginStep(OpenMM::ContextImpl& context);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const LBMLatticeParameters& lattice);
    void getFluidFields(OpenMM::ContextImpl& context, std::vector<double>& density, std::vector<OpenMM::Vec3>& velocity);
    double getFluidMachNumber(OpenMM::ContextImpl& context);
    void getFluidState(OpenMM::ContextImpl& context, std::vector<double>& state);
    void setFluidState(OpenMM::ContextImpl& context, const std::vector<double>& state);
private:
    void advanceFluid(OpenMM::ContextImpl& context);
    void computeMoments();
    void removeFluidMomentum();
    void coupleParticles(OpenMM::ContextImpl& context);
    int nearestNode(const OpenMM::Vec3& position) const;
    double getGaussianRandom();
    void collideAndStream();
    void bounceBack();
    void checkMachNumber();
    double computeMachNumber() const;
    LBMLatticeParameters lattice;
    /** Populations, stored as f[q*numNodes + node]. */
    std::vector<double> populations;
    /** Moments of the current step: density, momentum j = rho*u (3 per node), non-equilibrium second moment
        (xx, yy, zz, xy, xz, yz per node) and force density (3 per node), all in lattice units. */
    std::vector<double> rho, momentum, piNeq, forceDensity;
    /** 1 for fluid nodes, 0 for solid nodes; empty if there are no solid nodes. */
    std::vector<char> isFluid;
    /** True between beginStep() and the force evaluation of that integration step. */
    bool stepPending;
    /** Step count of the Context: set by beginStep() at the start of a lattice step, incremented at its end. */
    long long stepIndex;
    /** True once the debug warning about the Mach number has been printed. */
    bool machWarningPrinted;
    /** Masses of the coupled particles in lattice units, m/m_c. */
    std::vector<double> particleMass;
    /** Forces on the coupled particles (kJ/mol/nm) computed in the last lattice step.  Every force evaluation
        applies them, so that evaluations outside the integration steps draw no new random numbers. */
    std::vector<OpenMM::Vec3> particleForces;
    /** Reaction of the coupled particles on each node (3 per node), in lattice units. */
    std::vector<double> reaction;
    /** Generator of the random force, owned by the kernel so that its sequence does not depend on other forces. */
    OpenMM_SFMT::SFMT sfmt;
    /** The Box-Muller transform yields two Gaussian numbers: the second is kept for the next call. */
    bool hasStoredGaussian;
    double storedGaussian;
};

} // namespace LBMPlugin

#endif /*REFERENCE_LBM_KERNELS_H_*/
