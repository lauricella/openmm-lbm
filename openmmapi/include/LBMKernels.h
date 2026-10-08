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
    /** The domains of the lattice along each axis, one per MPI rank (setDomainDecomposition()), resolved when the
        Context is created: 1, 1, 1 without decomposition. */
    int procs[3];
    bool isDecomposed() const {
        return procs[0]*procs[1]*procs[2] > 1;
    }
    /** With the decomposition, whether the copies of the particles are compared over the ranks. */
    bool particleCopiesCheck;
    /** Whether the density and the velocity of the nodes around the domain of the rank (the halo) are exchanged at
        the end of every step (setDensityHaloExchange(), setVelocityHaloExchange()). */
    bool densityHaloExchange, velocityHaloExchange;
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
    /** True if the fluid has thermal fluctuations, fixed when the Context is created, and their thermal energy kT
        (kJ/mol): the temperature of the force with fluctuations, also with the NVE scheme, and 0 without. */
    bool fluidFluctuations;
    double fluidKT;
    int randomNumberSeed, momentumRemovalFrequency;
    /** Frequency (steps) of the Mach number check, 0 to disable, and the largest Mach number allowed. */
    int machCheckFrequency;
    double machNumberLimit;
    /** System indices of the coupled particles. */
    std::vector<int> particles;
    /** Solid nodes, sorted and without repetitions; empty if the whole lattice is fluid. */
    std::vector<int> solidNodes;
    /** Boundary condition at the solid nodes, fixed when the Context is created. */
    LBMForce::WallScheme wallScheme;
    /** Boundary conditions of the faces of the box (fixed when the Context is created), velocities of the Velocity
        faces in lattice units and densities of the Density faces in lattice units (1 is the fluid at rest). */
    LBMForce::BoundaryType faceBoundary[6];
    OpenMM::Vec3 faceVelocity[6];
    double faceDensity[6];
    /** True if the faces perpendicular to the axis (0, 1, 2 for x, y, z) are open. */
    bool isOpenAxis(int axis) const {
        return faceBoundary[2*axis] != LBMForce::Periodic;
    }
    bool hasOpenFaces() const {
        return isOpenAxis(0) || isOpenAxis(1) || isOpenAxis(2);
    }
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
     * Get the density (Da/nm^3) and velocity (nm/ps) of the fluid at the nodes of the domain of the rank (the whole
     * lattice with one domain, LBMDecomposition::getLocalDomain()), in the order of the domain, i fastest: with one
     * domain the order of the node index.  With halo the domain is extended by one node on every side, and the
     * extension holds the fields of the neighbouring nodes, across the periodic boundaries too, for the fields whose
     * halo is exchanged (setDensityHaloExchange(), setVelocityHaloExchange(); with one domain they come from the
     * lattice itself), and NaN for the others and beyond the open faces.  Solid nodes have zero density and velocity.
     */
    virtual void getFluidFields(OpenMM::ContextImpl& context, std::vector<double>& density, std::vector<OpenMM::Vec3>& velocity,
            bool halo) = 0;
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
     * Get the populations of the nodes of the domain of the rank, in lattice units, as [q*numLocal + l] with l the
     * index of the node in the domain (i fastest): with one domain [q*numNodes + node].
     */
    virtual void getFluidState(OpenMM::ContextImpl& context, std::vector<double>& state) = 0;
    /**
     * Set the populations of the nodes of the domain of the rank, in the layout of getFluidState().  With the domain
     * decomposition the call is collective (it exchanges the halo).
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
