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
#include <iosfwd>
#include <vector>
#include "internal/windowsExportLBM.h"

namespace LBMPlugin {

/**
 * This class couples particles to a lattice Boltzmann fluid that fills the periodic box.
 *
 * The fluid is a D3Q19 lattice Boltzmann model with regularized collision and Guo forcing.  Each
 * coupled particle feels a friction force -gamma*m*(v-u) relative to the fluid velocity u at the
 * nearest lattice node, plus a random force that satisfies the fluctuation-dissipation theorem at
 * the given temperature (Euler-Maruyama scheme), and the fluid receives the opposite force.  The
 * drag is explicit or centred in time (setDragScheme()).  Drag and noise are part of this force, so
 * the System must be integrated with a VerletIntegrator.  The fluid can also have thermal fluctuations of its own at
 * the same temperature (setFluidFluctuations()).
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
     * The scheme of the coupling between the particles and the fluid.
     */
    enum CouplingScheme {
        /**
         * Explicit Euler-Maruyama scheme: friction and random force at the temperature set with setTemperature().
         * This is the default.
         */
        EulerMaruyama = 0,
        /**
         * Friction only, without random force, whatever the temperature: particles and fluid exchange momentum
         * through the drag at zero temperature, with no thermostat.
         */
        NVE = 1
    };
    /**
     * The time discretization of the drag between a particle and the fluid (docs/theory.md, section 2).
     */
    enum DragScheme {
        /**
         * Explicit drag: the velocity of the particle half a step before the force and the velocity of the fluid
         * before the force of the step (Groot and Warren 1997).  The kinetic energy that OpenMM reports
         * (StateDataReporter) has the right temperature.  The velocity of a particle relative to the fluid changes
         * sign at every step for friction*dt > 1 and grows without bound for friction*dt >= 2.  This is the default.
         */
        Explicit = 0,
        /**
         * Centred drag: the velocities of the particle and of the fluid at the time of the force, solved exactly for
         * all the particles of a node (Brunger, Brooks and Karplus 1984).  The velocities of the State, half a step
         * after the force, have the right temperature.  Stable for any friction.  LBMForce must be the last force of
         * the System, and the System must not contain virtual sites.
         */
        Centered = 1
    };
    /**
     * The boundary condition of the fluid at the solid nodes (docs/theory.md, section 1).
     */
    enum WallScheme {
        /**
         * Halfway bounce-back: a population that streams from a fluid node into a solid node returns to the fluid node
         * with the opposite velocity.  The wall lies halfway between the solid node and the fluid node.  Mass is
         * conserved exactly, and with fluid fluctuations the fluid next to the walls is in exact thermal equilibrium.
         * This is the default.
         */
        BounceBack = 0,
        /**
         * Regularized wall: after the streaming, the fluid nodes next to the solid nodes rebuild the populations that
         * come from the solid nodes as those of fluid at rest on the solid node, with the stress of the fluid node and
         * a density that conserves the mass exactly: the thread-safe boundary condition of M. Lauricella et al.,
         * Phys. Fluids 37, 072111 (2025), appendix (docs/theory.md, section 1).  The wall lies on the solid nodes,
         * half a node further out than with the bounce-back.
         */
        Regularized = 1
    };
    /**
     * A face of the box, for the boundary conditions of setFaceBoundary(): XMin is the face x = 0, XMax the face
     * x = box length, and so on.
     */
    enum Face {
        XMin = 0,
        XMax = 1,
        YMin = 2,
        YMax = 3,
        ZMin = 4,
        ZMax = 5
    };
    /**
     * The boundary condition of the fluid on a face of the box (docs/theory.md, section 1).
     */
    enum BoundaryType {
        /**
         * The fluid that leaves through the face enters through the opposite one.  This is the default.
         */
        Periodic = 0,
        /**
         * The fluid beyond the face moves with the velocity set with setFaceVelocity(): an inlet, an outlet or a
         * moving wall.  Its density follows from the mass balance of the fluid that crosses the face.
         */
        Velocity = 1,
        /**
         * The fluid beyond the face has the density, that is the pressure, set with setFaceDensity(), and the
         * velocity of the fluid on the face, filtered in time across the face.
         */
        Density = 2
    };
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
     * Get the decomposition of the lattice into domains, one per MPI rank (setDomainDecomposition()).
     *
     * @param[out] px    the number of domains along x
     * @param[out] py    the number of domains along y
     * @param[out] pz    the number of domains along z
     */
    void getDomainDecomposition(int& px, int& py, int& pz) const;
    /**
     * Set the decomposition of the lattice into px*py*pz domains, one per MPI rank, for runs with several processes
     * (docs/theory.md, Domain decomposition).  The default, 1, 1, 1, is one domain: the whole lattice in one process,
     * without MPI.  A 0 lets MPI choose the number of domains along that axis (MPI_Dims_create).  Every rank runs
     * the same script with the same System; the particles are replicated on every rank, and each rank advances the
     * fluid of its domain.  More than one domain needs a plugin built with MPI (CMake option OPENMM_LBM_MPI), and
     * the product must be the number of MPI ranks.  It is fixed when the Context is created.
     *
     * @param px    the number of domains along x
     * @param py    the number of domains along y
     * @param pz    the number of domains along z
     */
    void setDomainDecomposition(int px, int py, int pz);
    /**
     * Whether the plugin was built with MPI, so that setDomainDecomposition() can ask for more than one domain.
     */
    static bool isMPIAvailable();
    /**
     * The rank of this process in MPI_COMM_WORLD, the number of ranks, and the rank among the processes on the same
     * node (to choose the GPU); 0, 1 and 0 without MPI.  The first call initializes MPI if the program has not done
     * it, and MPI is finalized at exit.
     */
    static int getMPIRank();
    static int getMPISize();
    static int getMPILocalRank();
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
     * Get the temperature of the random force on the coupled particles and, with fluid fluctuations, of the
     * fluid, measured in K.
     */
    double getTemperature() const;
    /**
     * Set the temperature of the random force on the coupled particles and, with fluid fluctuations
     * (setFluidFluctuations()), of the fluid, measured in K.  Particles and fluid share a single heat bath.
     */
    void setTemperature(double temperature);
    /**
     * Get the scheme of the coupling between the particles and the fluid.
     */
    CouplingScheme getCouplingScheme() const;
    /**
     * Set the scheme of the coupling between the particles and the fluid: EulerMaruyama (the default), with
     * friction and random force, or NVE, with friction only and no random force.
     */
    void setCouplingScheme(CouplingScheme scheme);
    /**
     * Get the time discretization of the drag.
     */
    DragScheme getDragScheme() const;
    /**
     * Set the time discretization of the drag: Explicit (the default) or Centered.  It is fixed when a Context is
     * created: updateParametersInContext() cannot change it.
     */
    void setDragScheme(DragScheme scheme);
    /**
     * Get whether the fluid has thermal fluctuations of its own.  See setFluidFluctuations().
     */
    bool getFluidFluctuations() const;
    /**
     * Set whether the fluid has thermal fluctuations of its own (ghost-mode filtered fluctuating lattice Boltzmann
     * model, docs/theory.md, section 7).  With fluctuations, every collision adds to the populations of each node a
     * random part that leaves its density and momentum unchanged and gives the stress and the higher moments the
     * equilibrium fluctuations at the temperature set with setTemperature(), the temperature of the random force on
     * the particles.  The fluid fluctuates with both coupling schemes: with NVE the particles are thermalized only
     * through the fluid.  The random numbers of the fluid are drawn from the random number seed
     * (setRandomNumberSeed()).  The default is false: the fluid has no fluctuations of its own and receives thermal
     * energy only from the random forces on the coupled particles.  It is fixed when a Context is created:
     * updateParametersInContext() cannot change it.
     */
    void setFluidFluctuations(bool fluctuations);
    /**
     * Get the boundary condition of the fluid at the solid nodes.
     */
    WallScheme getWallScheme() const;
    /**
     * Set the boundary condition of the fluid at the solid nodes: BounceBack (the default) or Regularized.  Both are
     * accurate to second order in the lattice spacing; they differ in where the wall lies and in how the fluid next
     * to it fluctuates (docs/theory.md, section 1).  It is fixed when a Context is created: updateParametersInContext()
     * cannot change it.
     */
    void setWallScheme(WallScheme scheme);
    /**
     * Get the boundary condition of the fluid on a face of the box.
     */
    BoundaryType getFaceBoundary(Face face) const;
    /**
     * Set the boundary condition of the fluid on a face of the box: Periodic (the default), Velocity or Density.  The
     * two faces perpendicular to an axis must be both periodic or both open (Velocity or Density, in any
     * combination).  The nodes on an open face (i = 0 for XMin, i = nx - 1 for XMax, and so on) rebuild at every step
     * the populations that come from beyond the face as those of fluid with the stress of the node and the velocity of
     * a Velocity face (with the density from the mass balance) or the density of a Density face (with the velocity
     * of the node, filtered in time across the face), which therefore holds one node beyond the face (docs/theory.md,
     * section 1).  On the nodes shared by several open faces the first Velocity face in the order
     * XMin, XMax, YMin, YMax, ZMin, ZMax gives the velocity; if they are all Density faces, the first one gives the
     * density and the velocity is zero.  A node next to a solid node with the Regularized wall scheme is a wall.
     * With open faces the removal of the fluid momentum must be switched off (setFluidMomentumRemovalFrequency(0)).
     * The particles still see a periodic box.  It is fixed when a Context is created.
     */
    void setFaceBoundary(Face face, BoundaryType type);
    /**
     * Get the velocity of the fluid beyond a Velocity face, measured in nm/ps.
     */
    OpenMM::Vec3 getFaceVelocity(Face face) const;
    /**
     * Set the velocity of the fluid beyond a Velocity face, measured in nm/ps.  It can have any direction: across the
     * face (inlet or outlet) or along it (a moving wall).  The default is zero.  updateParametersInContext() can
     * change it.
     */
    void setFaceVelocity(Face face, const OpenMM::Vec3& velocity);
    /**
     * Get the density of the fluid beyond a Density face, measured in Da/nm^3; 0 means the density of the fluid at rest.
     */
    double getFaceDensity(Face face) const;
    /**
     * Set the density of the fluid beyond a Density face, measured in Da/nm^3.  The pressure of the fluid is
     * p = c_s^2 rho, with c_s^2 = dx^2/(3 dt^2), so a difference of density between two faces drives a flow.  The
     * default, 0, means the density of the fluid at rest (setFluidDensity()).  updateParametersInContext() can
     * change it.
     */
    void setFaceDensity(Face face, double density);
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
     * Context is created from this Force.  The seed also determines the random numbers of the fluid, when it
     * fluctuates (setFluidFluctuations()).
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
     * momentum is never removed.  The momentum is removed in the steps whose index, the step count of the
     * Context at the start of the step, is a multiple of the frequency; checkpoints restore the step
     * count, so a restarted run removes it at the same steps.  The default is 1.
     */
    void setFluidMomentumRemovalFrequency(int frequency);
    /**
     * Get the frequency (in time steps) at which the largest Mach number of the fluid is checked.  0
     * means it is never checked.
     */
    int getMachCheckFrequency() const;
    /**
     * Set the frequency (in time steps) at which the largest Mach number of the fluid, Ma = max |u|/c_s
     * over the lattice nodes, is checked.  If it exceeds the limit set with setMachNumberLimit(), the
     * simulation stops with an exception.  0 means it is never checked.  The default is 100.  Steps are
     * numbered by the step count of the Context, as for the removal of the fluid momentum.
     */
    void setMachCheckFrequency(int frequency);
    /**
     * Get the largest Mach number of the fluid allowed by the check (see setMachCheckFrequency()).
     */
    double getMachNumberLimit() const;
    /**
     * Set the largest Mach number of the fluid allowed by the check (see setMachCheckFrequency()).
     * The default is 0.3, the usual limit of the quasi-incompressible regime of the model.
     */
    void setMachNumberLimit(double limit);
    /**
     * Get the solid nodes of the lattice.
     *
     * @param[out] nodes    the indices i + nx*(j + ny*k) of the solid nodes
     */
    void getSolidNodes(std::vector<int>& nodes) const;
    /**
     * Set the nodes of the lattice that are solid walls.  The fluid does not occupy them.  With the default wall
     * scheme (setWallScheme()) a population that streams into a solid node is sent back to the fluid node it came
     * from (bounce-back), which places the wall halfway between the two nodes; with the Regularized scheme the wall
     * lies on the solid nodes.  Node (i, j, k) has index i + nx*(j + ny*k).  An empty list
     * (the default) means that the whole lattice is fluid.
     *
     * @param nodes    the indices of the solid nodes
     */
    void setSolidNodes(const std::vector<int>& nodes);
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
     * to the plugin and should be treated as opaque: it holds the deviations f_q - w_q of the lattice
     * populations from the rest equilibrium (lattice density 1, at rest), in lattice units, stored as
     * [q*numNodes + node].  Saving and restoring them is exact.
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
     * Write a checkpoint of the part of the state of a Context that belongs to this force and that OpenMM
     * checkpoints (Context::createCheckpoint()) do not contain: the populations of the fluid, the random
     * numbers already drawn for the next step, the force on the walls of the last step and, on the Reference
     * platform, the state of the random number generator of the force.  Together with an OpenMM checkpoint of
     * the same Context, taken at the same step, it continues a run exactly, bit for bit, in a new Context.
     * Like an OpenMM checkpoint, it is specific to the platform, the precision and the System.
     *
     * @param context    the Context of which to write the checkpoint
     * @param stream     the stream to write it to, in binary mode
     */
    void createCheckpoint(OpenMM::Context& context, std::ostream& stream) const;
    /**
     * Load a checkpoint written by createCheckpoint().  Load the OpenMM checkpoint of the same step first:
     * it restores the positions, the velocities, the step count and, on the GPU platforms, the random number
     * generator that the force uses.
     *
     * @param context    the Context in which to load the checkpoint
     * @param stream     the stream to read it from, in binary mode
     */
    void loadCheckpoint(OpenMM::Context& context, std::istream& stream);
    /**
     * Get the largest Mach number of the fluid in a Context, Ma = max |u|/c_s over the lattice nodes,
     * with u = j/rho and c_s = 1/sqrt(3) in lattice units.
     *
     * @param context    the Context in which to compute the Mach number
     */
    double getFluidMachNumber(OpenMM::Context& context) const;
    /**
     * Get the force exerted on the walls during the last lattice step, measured in kJ/mol/nm: the momentum given to
     * the walls by the fluid and by the coupled particles, divided by the time step.  The fluid gives it through
     * bounce-back (momentum exchange method of Ladd) or, with the Regularized wall scheme, as the populations that
     * stream into the solid nodes minus the momentum that the walls put into their fluid nodes when they rebuild
     * them.  The particles give it through their coupling forces at solid nodes and their reflections.  It is zero before the first step and without solid
     * nodes.  With it, the total momentum of particles, fluid and walls is conserved.
     *
     * @param context    the Context in which to get the force
     */
    OpenMM::Vec3 getWallForce(OpenMM::Context& context) const;
    /**
     * Get the parameters of the lattice used in a Context.
     *
     * @param context    the Context for which to get the parameters
     * @param[out] dx    the lattice spacing, measured in nm
     * @param[out] dt    the lattice time step (the step size of the integrator), measured in ps
     * @param[out] tau   the relaxation time, in lattice units
     */
    void getLatticeParametersInContext(const OpenMM::Context& context, double& dx, double& dt, double& tau) const;
    /**
     * Update the parameters of a Context to match those stored in this Force object: the friction,
     * the temperature, the coupling scheme, the body acceleration, the frequency of the removal of the
     * fluid momentum, the frequency and limit of the Mach number check, and the velocities and densities of the
     * faces.  The grid, the fluid density and viscosity, the solid nodes, the wall scheme, the boundary types of the
     * faces, the set of coupled particles, the drag scheme and the fluid fluctuations cannot be changed this way,
     * and an exception is thrown if they differ.  With fluid fluctuations the new
     * temperature applies to the fluid as well.  The initial fluid velocity and the random
     * number seed are used only when a Context is created.  The fluid itself is not modified.
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
    int nx, ny, nz, randomNumberSeed, momentumRemovalFrequency, machCheckFrequency;
    int decomposition[3];
    double density, viscosity, friction, temperature, machNumberLimit;
    CouplingScheme couplingScheme;
    DragScheme dragScheme;
    WallScheme wallScheme;
    BoundaryType faceBoundary[6];
    OpenMM::Vec3 faceVelocity[6];
    double faceDensity[6];
    bool fluidFluctuations;
    OpenMM::Vec3 bodyAcceleration, initialVelocity;
    std::vector<int> particles, solidNodes;
};

} // namespace LBMPlugin

#endif /*OPENMM_LBMFORCE_H_*/
