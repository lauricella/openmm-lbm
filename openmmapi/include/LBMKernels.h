#ifndef LBM_KERNELS_H_
#define LBM_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "LBMForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include <iosfwd>
#include <string>
#include <vector>

namespace LBMPlugin {

/**
 * The parameters of the lattice, in lattice units, together with the factors that convert them
 * from and to OpenMM units.  They are computed once by LBMForceImpl, so that every platform uses
 * the same conversion.  Lattice units: length dx, time dt, mass rho0*dx^3.
 * @private
 */
class LBMLatticeParameters {
public:
    int nx, ny, nz;
    /** Lattice spacing (nm) and lattice time step (ps). */
    double dx, dt;
    /** Mass density of the fluid at rest (Da/nm^3): the lattice density 1 corresponds to it. */
    double density;
    /** BGK relaxation time tau and frequency omega = 1/tau. */
    double tau, omega;
    /** Initial fluid velocity and body acceleration, in lattice units. */
    OpenMM::Vec3 initialVelocity, bodyAcceleration;
    /** Friction (1/ps) and thermal energy kT (kJ/mol) of the coupling.  kT is 0 with the NVE scheme, which has
        no random force. */
    double friction, kT;
    /** Time discretization of the drag, fixed when the Context is created. */
    LBMForce::DragScheme dragScheme;
    int randomNumberSeed, momentumRemovalFrequency;
    /** Frequency (steps) of the Mach number check, 0 to disable, and the largest Mach number allowed. */
    int machCheckFrequency;
    double machNumberLimit;
    /** System indices of the coupled particles. */
    std::vector<int> particles;
    /** Solid nodes, sorted and without repetitions; empty if the whole lattice is fluid. */
    std::vector<int> solidNodes;
    int getNumNodes() const {
        return nx*ny*nz;
    }
    /** Factor that converts a lattice velocity into nm/ps. */
    double getVelocityScale() const {
        return dx/dt;
    }
};

/**
 * This kernel is invoked by LBMForce to advance the fluid and calculate the coupling forces.
 */
class CalcLBMForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcLBMForce";
    }
    CalcLBMForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the LBMForce this kernel will be used for
     * @param lattice    the lattice parameters derived from force, System and integrator
     */
    virtual void initialize(const OpenMM::System& system, const LBMForce& force, const LBMLatticeParameters& lattice) = 0;
    /**
     * Called once at the start of every integration step, before the forces are computed.  The next call to
     * execute() advances the fluid by one lattice step; other force evaluations (getState(), for example)
     * do not.
     *
     * @param context        the context in which to execute this kernel
     */
    virtual void beginStep(OpenMM::ContextImpl& context) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force (always 0: the coupling is dissipative)
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param lattice    the updated lattice parameters
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const LBMLatticeParameters& lattice) = 0;
    /**
     * Get the density (Da/nm^3) and velocity (nm/ps) of the fluid at every node.
     */
    virtual void getFluidFields(OpenMM::ContextImpl& context, std::vector<double>& density, std::vector<OpenMM::Vec3>& velocity) = 0;
    /**
     * Get the largest Mach number of the fluid, max |j/rho|/c_s over the nodes, in lattice units.
     */
    virtual double getFluidMachNumber(OpenMM::ContextImpl& context) = 0;
    /**
     * Get the force (kJ/mol/nm) exerted on the solid nodes during the last lattice step: the momentum given
     * to them by bounce-back (momentum exchange) and by the coupled particles, divided by the time step.
     */
    virtual OpenMM::Vec3 getWallForce(OpenMM::ContextImpl& context) = 0;
    /**
     * Get the populations of all nodes, in lattice units.
     */
    virtual void getFluidState(OpenMM::ContextImpl& context, std::vector<double>& state) = 0;
    /**
     * Set the populations of all nodes, in lattice units.
     */
    virtual void setFluidState(OpenMM::ContextImpl& context, const std::vector<double>& state) = 0;
    /**
     * Write the state of the kernel that OpenMM checkpoints do not contain: the populations, the random
     * numbers already drawn for the next step, the momentum given to the walls in the last step and, where the
     * kernel owns one, the state of its random number generator.
     */
    virtual void createCheckpoint(OpenMM::ContextImpl& context, std::ostream& stream) = 0;
    /**
     * Read a checkpoint written by createCheckpoint() of a kernel of the same platform, precision and System.
     */
    virtual void loadCheckpoint(OpenMM::ContextImpl& context, std::istream& stream) = 0;
};

} // namespace LBMPlugin

#endif /*LBM_KERNELS_H_*/
