#ifndef OPENMM_LBMFORCE_H_
#define OPENMM_LBMFORCE_H_

/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include <vector>
#include "internal/windowsExportLBM.h"

namespace LBMPlugin {

/**
 * This class couples particles to a lattice Boltzmann fluid that fills the periodic box.
 *
 * The fluid is a D3Q19 lattice Boltzmann model with regularized collision and Guo forcing.  Each
 * coupled particle feels a friction force -gamma*m*(v-u) relative to the fluid velocity u at the
 * nearest lattice node, plus a random force that satisfies the fluctuation-dissipation theorem at
 * the given temperature (Euler-Maruyama scheme), and the fluid receives the opposite force.  Drag
 * and noise are part of this force, so the System must be integrated with a VerletIntegrator.
 *
 * The lattice spans the periodic box of the System: the box must be rectangular and the lattice
 * cells must be cubic, that is, the box lengths divided by the grid sizes must be equal.  The
 * lattice time step is the step size of the integrator.
 *
 * All parameters are in OpenMM units: nm, ps, Da (g/mol) and K.
 */

class OPENMM_EXPORT_LBM LBMForce : public OpenMM::Force {
public:
    /**
     * Create an LBMForce.  The grid size must be set with setGridSize() before the force is used.
     */
    LBMForce();
    /**
     * Get the number of lattice nodes along each box vector.
     *
     * @param[out] nx    the number of nodes along x
     * @param[out] ny    the number of nodes along y
     * @param[out] nz    the number of nodes along z
     */
    void getGridSize(int& nx, int& ny, int& nz) const;
    /**
     * Set the number of lattice nodes along each box vector.
     *
     * @param nx    the number of nodes along x
     * @param ny    the number of nodes along y
     * @param nz    the number of nodes along z
     */
    void setGridSize(int nx, int ny, int nz);
    /**
     * Get the mass density of the fluid at rest, measured in Da/nm^3.
     */
    double getFluidDensity() const;
    /**
     * Set the mass density of the fluid at rest, measured in Da/nm^3 (water: 602.2 Da/nm^3).
     */
    void setFluidDensity(double density);
    /**
     * Get the kinematic viscosity of the fluid, measured in nm^2/ps.
     */
    double getKinematicViscosity() const;
    /**
     * Set the kinematic viscosity of the fluid, measured in nm^2/ps.  Together with the lattice
     * spacing dx and the step size dt it fixes the relaxation time tau = 3*viscosity*dt/dx^2 + 1/2,
     * which must be larger than 1/2.
     */
    void setKinematicViscosity(double viscosity);
    /**
     * Get the friction coefficient gamma of the particle-fluid coupling, measured in 1/ps.
     */
    double getFriction() const;
    /**
     * Set the friction coefficient gamma of the particle-fluid coupling, measured in 1/ps.
     */
    void setFriction(double friction);
    /**
     * Get the temperature of the random force on the coupled particles, measured in K.
     */
    double getTemperature() const;
    /**
     * Set the temperature of the random force on the coupled particles, measured in K.
     */
    void setTemperature(double temperature);
    /**
     * Get the random number seed.  See setRandomNumberSeed() for details.
     */
    int getRandomNumberSeed() const;
    /**
     * Set the random number seed.  The precise meaning of this parameter is undefined, and is left up
     * to each Platform to interpret in an appropriate way.  It is guaranteed that if two simulations
     * are run with different random number seeds, the sequence of random forces will be different.
     * On the other hand, no guarantees are made about the behavior of simulations that use the same
     * seed.
     *
     * If seed is set to 0 (which is the default value assigned), a unique seed is chosen when a
     * Context is created from this Force.
     */
    void setRandomNumberSeed(int seed);
    /**
     * Get the uniform acceleration applied to the fluid, measured in nm/ps^2.
     */
    OpenMM::Vec3 getBodyAcceleration() const;
    /**
     * Set a uniform acceleration applied to the fluid (a body force per unit mass), measured in nm/ps^2.
     */
    void setBodyAcceleration(const OpenMM::Vec3& acceleration);
    /**
     * Get the uniform velocity of the fluid when a Context is created, measured in nm/ps.
     */
    OpenMM::Vec3 getInitialFluidVelocity() const;
    /**
     * Set the uniform velocity of the fluid when a Context is created, measured in nm/ps.  The fluid
     * starts at equilibrium with this velocity and the density set by setFluidDensity().
     */
    void setInitialFluidVelocity(const OpenMM::Vec3& velocity);
    /**
     * Get the frequency (in time steps) at which the momentum of the fluid is removed.  0 means the
     * momentum is never removed.
     */
    int getFluidMomentumRemovalFrequency() const;
    /**
     * Set the frequency (in time steps) at which the momentum of the fluid is removed.  0 means the
     * momentum is never removed.
     */
    void setFluidMomentumRemovalFrequency(int frequency);
    /**
     * Get the number of particles coupled to the fluid.
     */
    int getNumParticles() const {
        return particles.size();
    }
    /**
     * Couple a particle to the fluid.  Its mass is taken from the System.
     *
     * @param particle    the index of the particle within the System
     * @return the index of the coupled particle that was added
     */
    int addParticle(int particle);
    /**
     * Get the System index of a coupled particle.
     *
     * @param index    the index of the coupled particle
     * @return the index of the particle within the System
     */
    int getParticle(int index) const;
    /**
     * Set the System index of a coupled particle.
     *
     * @param index       the index of the coupled particle
     * @param particle    the index of the particle within the System
     */
    void setParticle(int index, int particle);
    /**
     * Get the density and velocity of the fluid at every lattice node.  Node (i, j, k) has index
     * i + nx*(j + ny*k).
     *
     * @param context        the Context in which to get the fields
     * @param[out] density   the mass density at each node, measured in Da/nm^3
     * @param[out] velocity  the velocity at each node, measured in nm/ps
     */
    void getFluidFields(OpenMM::Context& context, std::vector<double>& density, std::vector<OpenMM::Vec3>& velocity) const;
    /**
     * Get the complete state of the fluid, so that it can be saved and restored with setFluidState().
     * The state of the fluid is not part of OpenMM checkpoints.  The content of the vector is internal
     * to the plugin (lattice populations in lattice units) and should be treated as opaque.
     *
     * @param context     the Context from which to get the state
     * @param[out] state  the state of the fluid
     */
    void getFluidState(OpenMM::Context& context, std::vector<double>& state) const;
    /**
     * Set the complete state of the fluid, as returned by getFluidState() for a Context with the same
     * grid size.
     *
     * @param context    the Context in which to set the state
     * @param state      the state of the fluid
     */
    void setFluidState(OpenMM::Context& context, const std::vector<double>& state);
    /**
     * Update the friction, temperature and body acceleration in a Context to match those stored in
     * this Force object.  The grid, the fluid density and viscosity and the set of coupled particles
     * cannot be changed this way.
     */
    void updateParametersInContext(OpenMM::Context& context);
    /**
     * Returns true if the force uses periodic boundary conditions and false otherwise.  The fluid
     * always fills the periodic box.
     */
    bool usesPeriodicBoundaryConditions() const {
        return true;
    }
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    int nx, ny, nz, randomNumberSeed, momentumRemovalFrequency;
    double density, viscosity, friction, temperature;
    OpenMM::Vec3 bodyAcceleration, initialVelocity;
    std::vector<int> particles;
};

} // namespace LBMPlugin

#endif /*OPENMM_LBMFORCE_H_*/
