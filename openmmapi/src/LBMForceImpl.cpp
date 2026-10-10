/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "internal/LBMDecomposition.h"
#include "internal/LBMForceImpl.h"
#include "LBMKernels.h"
#include "openmm/AndersenThermostat.h"
#include "openmm/MonteCarloAnisotropicBarostat.h"
#include "openmm/MonteCarloBarostat.h"
#include "openmm/MonteCarloFlexibleBarostat.h"
#include "openmm/MonteCarloMembraneBarostat.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/State.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

using namespace LBMPlugin;
using namespace OpenMM;
using namespace std;

LBMForceImpl::LBMForceImpl(const LBMForce& owner) : owner(owner) {
}

LBMForceImpl::~LBMForceImpl() {
}

LBMLatticeParameters LBMForceImpl::computeLatticeParameters(const LBMForce& force, const System& system, double stepSize) {
    LBMLatticeParameters lattice;
    force.getGridSize(lattice.nx, lattice.ny, lattice.nz);
    if (lattice.nx <= 0 || lattice.ny <= 0 || lattice.nz <= 0)
        throw OpenMMException("LBMForce: the grid size must be set to positive values with setGridSize()");

    // The domains of the lattice, one per MPI rank (1, 1, 1 without MPI).

    int requested[3];
    force.getDomainDecomposition(requested[0], requested[1], requested[2]);
    LBMDecomposition::resolve(lattice.nx, lattice.ny, lattice.nz, requested, lattice.procs);

    // Every platform indexes the nodes and the populations with 32-bit integers: the lattice must have fewer than 2^31
    // nodes, and the 19 populations of the largest domain, with its halo along the divided axes, fewer than 2^31
    // entries.  Beyond that the sizes wrapped around (on the GPUs an allocation failed with "out of memory").

    const long long maxIndex = numeric_limits<int>::max();
    int gridSize[3] = {lattice.nx, lattice.ny, lattice.nz};
    long long numNodes = 1, domainNodes = 1;
    for (int a = 0; a < 3; a++) {
        numNodes *= gridSize[a];
        domainNodes *= (gridSize[a]+lattice.procs[a]-1)/lattice.procs[a] + (lattice.procs[a] > 1 ? 2 : 0);
    }
    if (numNodes > maxIndex) {
        stringstream msg;
        msg << "LBMForce: the lattice has " << numNodes << " nodes, more than the " << maxIndex << " that the plugin can index";
        throw OpenMMException(msg.str());
    }
    if (19*domainNodes > maxIndex) {
        stringstream msg;
        msg << "LBMForce: a domain of the lattice holds up to " << domainNodes << " nodes (with its halo), more than the "
            << maxIndex/19 << " whose 19 populations the plugin can index: divide the lattice into more domains "
            << "(setDomainDecomposition())";
        throw OpenMMException(msg.str());
    }

    // The lattice spans the periodic box, with cubic cells.

    Vec3 a, b, c;
    system.getDefaultPeriodicBoxVectors(a, b, c);
    if (a[1] != 0 || a[2] != 0 || b[0] != 0 || b[2] != 0 || c[0] != 0 || c[1] != 0)
        throw OpenMMException("LBMForce: the periodic box must be rectangular");
    lattice.dx = a[0]/lattice.nx;
    double dy = b[1]/lattice.ny, dz = c[2]/lattice.nz;
    if (fabs(dy-lattice.dx) > 1e-6*lattice.dx || fabs(dz-lattice.dx) > 1e-6*lattice.dx) {
        stringstream msg;
        msg << "LBMForce: the lattice cells must be cubic, but the box and grid sizes give spacings " << lattice.dx << ", " << dy << ", " << dz << " nm";
        throw OpenMMException(msg.str());
    }

    // Lattice time step and relaxation time.

    if (stepSize <= 0)
        throw OpenMMException("LBMForce: the integrator step size must be positive");
    lattice.dt = stepSize;
    if (force.getFluidDensity() <= 0)
        throw OpenMMException("LBMForce: the fluid density must be positive");
    if (force.getKinematicViscosity() <= 0)
        throw OpenMMException("LBMForce: the kinematic viscosity must be positive");
    lattice.density = force.getFluidDensity();
    lattice.tau = 3.0*force.getKinematicViscosity()*lattice.dt/(lattice.dx*lattice.dx) + 0.5;
    lattice.omega = 1.0/lattice.tau;

    // Velocities and accelerations in lattice units.

    lattice.initialVelocity = force.getInitialFluidVelocity()*(lattice.dt/lattice.dx);
    lattice.bodyAcceleration = force.getBodyAcceleration()*(lattice.dt*lattice.dt/lattice.dx);

    // Coupling.

    if (force.getFriction() < 0)
        throw OpenMMException("LBMForce: the friction must not be negative");
    if (force.getTemperature() < 0)
        throw OpenMMException("LBMForce: the temperature must not be negative");
    if (force.getFluidMomentumRemovalFrequency() < 0)
        throw OpenMMException("LBMForce: the fluid momentum removal frequency must not be negative");
    if (force.getMachCheckFrequency() < 0)
        throw OpenMMException("LBMForce: the Mach number check frequency must not be negative");
    if (force.getMachNumberLimit() <= 0)
        throw OpenMMException("LBMForce: the Mach number limit must be positive");
    lattice.machCheckFrequency = force.getMachCheckFrequency();
    lattice.particleCopiesCheck = force.getParticleCopiesCheck();
    lattice.densityHaloExchange = force.getDensityHaloExchange();
    lattice.velocityHaloExchange = force.getVelocityHaloExchange();
    lattice.machNumberLimit = force.getMachNumberLimit();

    // Solid nodes (walls).

    force.getSolidNodes(lattice.solidNodes);
    sort(lattice.solidNodes.begin(), lattice.solidNodes.end());
    for (int i = 0; i < (int) lattice.solidNodes.size(); i++) {
        if (lattice.solidNodes[i] < 0 || lattice.solidNodes[i] >= lattice.getNumNodes())
            throw OpenMMException("LBMForce: a solid node index is out of range");
        if (i > 0 && lattice.solidNodes[i] == lattice.solidNodes[i-1])
            throw OpenMMException("LBMForce: a solid node is listed more than once");
    }
    if ((int) lattice.solidNodes.size() == lattice.getNumNodes())
        throw OpenMMException("LBMForce: all lattice nodes are solid");
    if (force.getWallScheme() != LBMForce::BounceBack && force.getWallScheme() != LBMForce::Regularized)
        throw OpenMMException("LBMForce: unknown wall scheme");
    lattice.wallScheme = force.getWallScheme();

    // Faces of the box: periodic, or open with a velocity or a density (in lattice units).

    const char* axisName[3] = {"x", "y", "z"};
    int size[3] = {lattice.nx, lattice.ny, lattice.nz};
    for (int face = 0; face < 6; face++) {
        LBMForce::BoundaryType type = force.getFaceBoundary((LBMForce::Face) face);
        if (type != LBMForce::Periodic && type != LBMForce::Velocity && type != LBMForce::Density)
            throw OpenMMException("LBMForce: unknown boundary type of a face");
        lattice.faceBoundary[face] = type;
        lattice.faceVelocity[face] = force.getFaceVelocity((LBMForce::Face) face)*(lattice.dt/lattice.dx);
        double density = force.getFaceDensity((LBMForce::Face) face);
        if (density < 0)
            throw OpenMMException("LBMForce: the density of a face must not be negative");
        lattice.faceDensity[face] = (density == 0 ? 1.0 : density/lattice.density);
    }
    for (int axis = 0; axis < 3; axis++) {
        if ((lattice.faceBoundary[2*axis] == LBMForce::Periodic) != (lattice.faceBoundary[2*axis+1] == LBMForce::Periodic))
            throw OpenMMException(string("LBMForce: the two faces perpendicular to ") + axisName[axis] + " must be both "
                    "periodic or both open (Velocity or Density)");
        if (lattice.isOpenAxis(axis) && size[axis] < 3)
            throw OpenMMException(string("LBMForce: with open faces perpendicular to ") + axisName[axis] + " the grid "
                    "needs at least 3 nodes along " + axisName[axis]);
    }
    if (lattice.hasOpenFaces() && force.getFluidMomentumRemovalFrequency() > 0)
        throw OpenMMException("LBMForce: with open faces the fluid exchanges momentum with the outside, and its momentum "
                "cannot be removed: call setFluidMomentumRemovalFrequency(0)");
    lattice.friction = force.getFriction();
    if (force.getCouplingScheme() != LBMForce::EulerMaruyama && force.getCouplingScheme() != LBMForce::NVE)
        throw OpenMMException("LBMForce: unknown coupling scheme");
    // The NVE scheme has friction only: no random force, whatever the temperature.
    lattice.kT = (force.getCouplingScheme() == LBMForce::NVE ? 0.0 : BOLTZ*force.getTemperature());
    if (force.getDragScheme() != LBMForce::Explicit && force.getDragScheme() != LBMForce::Centered)
        throw OpenMMException("LBMForce: unknown drag scheme");
    lattice.dragScheme = force.getDragScheme();
    if (force.getInterpolationStencil() < LBMForce::NearestNode || force.getInterpolationStencil() > LBMForce::Keys)
        throw OpenMMException("LBMForce: unknown interpolation stencil");
    lattice.interpolationStencil = force.getInterpolationStencil();
    // The fluid fluctuates at the temperature of the force also with the NVE scheme, whose particles are then
    // thermalized only through the fluid.
    lattice.fluidFluctuations = force.getFluidFluctuations();
    lattice.fluidKT = (lattice.fluidFluctuations ? BOLTZ*force.getTemperature() : 0.0);
    lattice.randomNumberSeed = force.getRandomNumberSeed();
    lattice.momentumRemovalFrequency = force.getFluidMomentumRemovalFrequency();
    set<int> seen;
    for (int i = 0; i < force.getNumParticles(); i++) {
        int particle = force.getParticle(i);
        if (particle < 0 || particle >= system.getNumParticles())
            throw OpenMMException("LBMForce: a coupled particle index is out of range");
        if (seen.find(particle) != seen.end())
            throw OpenMMException("LBMForce: a particle is coupled more than once");
        if (system.getParticleMass(particle) <= 0)
            throw OpenMMException("LBMForce: coupled particles must have a positive mass");
        seen.insert(particle);
        lattice.particles.push_back(particle);
    }

    return lattice;
}

void LBMForceImpl::initialize(ContextImpl& context) {
    // Drag and noise are part of the force, so the integrator must not add its own.
    if (dynamic_cast<const VerletIntegrator*>(&context.getIntegrator()) == NULL)
        throw OpenMMException("LBMForce requires a VerletIntegrator: drag and random forces are part of the force");
    lattice = computeLatticeParameters(owner, context.getSystem(), context.getIntegrator().getStepSize());

    // The centred drag needs the other forces on the coupled particles at the time of the step.  OpenMM computes
    // the forces in the order of the System, so on the Reference platform they are complete only when LBMForce
    // comes last; the GPU platforms read them in a post-computation that runs after those of the other forces
    // only when LBMForce comes last.  The forces on virtual sites are moved to the particles that define them
    // after the post-computations, so they would be missing.
    if (lattice.dragScheme == LBMForce::Centered) {
        const System& system = context.getSystem();
        if (&system.getForce(system.getNumForces()-1) != &owner)
            throw OpenMMException("LBMForce: with the Centered drag scheme LBMForce must be the last force of the System; "
                    "add it after all the other forces");
        for (int i = 0; i < system.getNumParticles(); i++)
            if (system.isVirtualSite(i))
                throw OpenMMException("LBMForce: the Centered drag scheme does not support virtual sites");
    }

    // With the domain decomposition every rank integrates all the particles, which must stay identical on every
    // rank (docs/theory.md, section 8).  The integrator is a VerletIntegrator, which draws no random numbers, but an
    // AndersenThermostat or a Monte Carlo barostat would draw different ones on each rank (each rank has its own
    // seed, and a seed 0 is chosen at random by each process).

    if (lattice.isDecomposed()) {
        const System& system = context.getSystem();
        for (int i = 0; i < system.getNumForces(); i++) {
            const Force& f = system.getForce(i);
            if (dynamic_cast<const AndersenThermostat*>(&f) != NULL || dynamic_cast<const MonteCarloBarostat*>(&f) != NULL ||
                    dynamic_cast<const MonteCarloAnisotropicBarostat*>(&f) != NULL ||
                    dynamic_cast<const MonteCarloMembraneBarostat*>(&f) != NULL ||
                    dynamic_cast<const MonteCarloFlexibleBarostat*>(&f) != NULL)
                throw OpenMMException("LBMForce: with the domain decomposition (setDomainDecomposition()) the System "
                        "cannot contain an AndersenThermostat or a Monte Carlo barostat: their random numbers would differ "
                        "between the MPI ranks, which hold copies of the same particles");
        }
    }

    // The warnings are printed once, by rank 0, with the domain decomposition.

    bool warn = (!lattice.isDecomposed() || LBMDecomposition::getWorldRank() == 0);

    // The model is accurate for moderate relaxation times (docs/theory.md, section 6).

    if (warn && (lattice.tau < 0.505 || lattice.tau > 2.0))
        cerr << "Warning: LBMForce: the relaxation time tau = " << lattice.tau << " is outside the range [0.505, 2] "
             << "in which the lattice Boltzmann model is accurate. tau = 3 nu dt/dx^2 + 1/2: change the viscosity, "
             << "the time step or the lattice spacing." << endl;

    // With the explicit drag at the nearest node, the hydrodynamic part of the self-mobility of a coupled particle
    // decreases as tau grows and becomes negative at tau = 1.79 (docs/theory.md, section 2).  With the centred drag
    // it stays positive, and so it does with the three-point stencil (section 9, measured up to tau = 3.51); the
    // trilinear and Keys stencils give the nearest node when the particle is at a node.

    if (warn && !lattice.particles.empty() && lattice.dragScheme == LBMForce::Explicit && lattice.tau > 1.7 &&
            lattice.interpolationStencil != LBMForce::ThreePoint)
        cerr << "Warning: LBMForce: tau = " << lattice.tau << " > 1.7: with the explicit drag at the nearest node "
             << "the hydrodynamic self-mobility of a coupled particle is small, and negative above tau = 1.79, so "
             << "particles move less than they should. Reduce the viscosity or the time step, or use a coarser "
             << "lattice." << endl;

    // The explicit drag multiplies the velocity of a particle relative to the fluid by 1 - gamma*dt in one step
    // (docs/theory.md, section 2).  The centred drag is stable for any friction.

    double gammaDt = lattice.friction*lattice.dt;
    if (warn && !lattice.particles.empty() && lattice.dragScheme == LBMForce::Explicit && gammaDt > 1.0)
        cerr << "Warning: LBMForce: friction*dt = " << gammaDt << " > 1: with the explicit drag the velocity of a "
             << "particle relative to the fluid changes sign at every step" << (gammaDt >= 2.0 ? ", and grows without "
             "bound since friction*dt >= 2" : "") << ". Reduce the friction or the time step." << endl;

    // With the fluctuating fluid the explicit drag makes the coupled particles too hot, by about
    // gamma dt m/(2 m_c (1 + zeta y)), because it misses the response of the cell within the step; the centred drag
    // gives the right temperature (docs/theory.md, section 7).  The warning gives the bound gamma dt m/(2 m_c) for the
    // heaviest coupled particle, times the self weight K = sum_j xi_j^2 of the stencil averaged over a cell, which a
    // particle that moves across the lattice sees (section 9): 1 for the nearest node, (2/3)^3 = 8/27 for the
    // trilinear stencil, 1/8 for the three-point stencil and (57/70)^3 for Keys, the cube of the mean over a period of
    // sum_j phi(r - j)^2 along an axis.

    if (warn && !lattice.particles.empty() && lattice.dragScheme == LBMForce::Explicit && lattice.fluidFluctuations &&
            lattice.kT > 0 && gammaDt > 0) {
        double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
        double maxMass = 0;
        for (int particle : lattice.particles)
            maxMass = max(maxMass, context.getSystem().getParticleMass(particle));
        double selfWeight = 1.0;
        if (lattice.interpolationStencil == LBMForce::Trilinear)
            selfWeight = 8.0/27.0;
        else if (lattice.interpolationStencil == LBMForce::ThreePoint)
            selfWeight = 0.125;
        else if (lattice.interpolationStencil == LBMForce::Keys)
            selfWeight = pow(57.0/70.0, 3);
        cerr << "Warning: LBMForce: with fluid fluctuations the explicit drag makes the coupled particles hotter than "
             << "the set temperature, by about friction*dt*m" << (selfWeight < 1.0 ? "*K" : "") << "/(2 m_c) = "
             << 100.0*gammaDt*maxMass*selfWeight/(2.0*cellMass)
             << "% for the heaviest one (m_c = " << cellMass << " Da is the mass of fluid in a cell";
        if (selfWeight < 1.0)
            cerr << ", K = " << selfWeight << " the self weight of the interpolation stencil averaged over a cell";
        cerr << "). Use the Centered drag scheme with fluid fluctuations." << endl;
    }

    // The size of the fluctuations of the fluid: kT in lattice units, kT/(m_c (dx/dt)^2), and the thermal Mach number
    // sqrt(3 kT), the r.m.s. velocity of a node along one axis over the speed of sound of the lattice, dx/(sqrt(3) dt),
    // and the r.m.s. relative fluctuation of the density of a node.  They depend on the temperature, the density, the
    // lattice spacing and the time step, not on the speed of sound of the real fluid.  The fluctuating lattice
    // Boltzmann method holds only small fluctuations: it was validated up to kT = 1/3000, where on D3Q19 it is
    // unstable for tau <= 0.501 (docs/validation.md, Equilibrium spectra).

    if (warn && lattice.fluidFluctuations && lattice.fluidKT > 0) {
        double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
        double latticeKT = lattice.fluidKT*lattice.dt*lattice.dt/(cellMass*lattice.dx*lattice.dx);
        const double validatedKT = 1.0/3000.0;
        stringstream message;
        message.precision(2);
        message << "LBMForce: fluctuating fluid: kT/(m_c (dx/dt)^2) = " << latticeKT << " in lattice units, thermal "
                << "Mach number sqrt(3 kT) = " << sqrt(3.0*latticeKT) << " (validated up to kT = 1/3000, Mach number "
                << sqrt(3.0*validatedKT) << ")" << endl;
        if (latticeKT > validatedKT*(1.0 + 1e-6))
            message << "Warning: LBMForce: the fluctuations of the fluid, kT = " << latticeKT << " in lattice units, are "
                    << "larger than those validated, kT = 1/3000: the fluctuating fluid may become unstable (at 1/3000 "
                    << "it is for tau <= 0.501). A larger lattice spacing or a smaller time step makes them smaller "
                    << "(kT in lattice units goes as dt^2/dx^5)." << endl;
        cerr << message.str();
    }
#ifdef LBM_DEBUG
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    cerr << "LBMForce lattice: dx = " << lattice.dx << " nm, dt = " << lattice.dt << " ps, m_c = " << cellMass
         << " Da, tau = " << lattice.tau << ", kT/(m_c c_s^2) = "
         << 3.0*lattice.kT*lattice.dt*lattice.dt/(cellMass*lattice.dx*lattice.dx) << endl;
#endif
    decomposition = LBMDecomposition(lattice.nx, lattice.ny, lattice.nz, lattice.procs);
    kernel = context.getPlatform().createKernel(CalcLBMForceKernel::Name(), context);
    kernel.getAs<CalcLBMForceKernel>().initialize(context.getSystem(), owner, lattice);
}

void LBMForceImpl::updateContextState(ContextImpl& context, bool& forcesInvalid) {
    // The coupling forces of a step depend on the velocities, on the fluid and on new random numbers, so the
    // forces of earlier evaluations are never valid for the new step.
    kernel.getAs<CalcLBMForceKernel>().beginStep(context);
    forcesInvalid = true;
}

double LBMForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) == 0)
        return 0.0;
    // The lattice time step is the integrator step size, fixed when the Context was created.
    if (context.getIntegrator().getStepSize() != lattice.dt)
        throw OpenMMException("LBMForce: the integrator step size changed after the Context was created; reinitialize the Context");
    return kernel.getAs<CalcLBMForceKernel>().execute(context, includeForces, includeEnergy);
}

vector<string> LBMForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcLBMForceKernel::Name());
    return names;
}

void LBMForceImpl::updateParametersInContext(ContextImpl& context) {
    LBMLatticeParameters updated = computeLatticeParameters(owner, context.getSystem(), lattice.dt);
    if (updated.nx != lattice.nx || updated.ny != lattice.ny || updated.nz != lattice.nz)
        throw OpenMMException("updateParametersInContext: the grid size cannot be changed");
    if (updated.density != lattice.density || updated.tau != lattice.tau)
        throw OpenMMException("updateParametersInContext: the fluid density and viscosity cannot be changed");
    if (updated.particles != lattice.particles)
        throw OpenMMException("updateParametersInContext: the set of coupled particles cannot be changed");
    if (updated.solidNodes != lattice.solidNodes)
        throw OpenMMException("updateParametersInContext: the solid nodes cannot be changed");
    if (updated.wallScheme != lattice.wallScheme)
        throw OpenMMException("updateParametersInContext: the wall scheme cannot be changed");
    for (int axis = 0; axis < 3; axis++)
        if (updated.procs[axis] != lattice.procs[axis])
            throw OpenMMException("updateParametersInContext: the domain decomposition cannot be changed");
    if (updated.densityHaloExchange != lattice.densityHaloExchange || updated.velocityHaloExchange != lattice.velocityHaloExchange)
        throw OpenMMException("updateParametersInContext: the exchange of the halo cannot be switched on or off");
    for (int face = 0; face < 6; face++)
        if (updated.faceBoundary[face] != lattice.faceBoundary[face])
            throw OpenMMException("updateParametersInContext: the boundary types of the faces cannot be changed");
    if (updated.dragScheme != lattice.dragScheme)
        throw OpenMMException("updateParametersInContext: the drag scheme cannot be changed");
    if (updated.interpolationStencil != lattice.interpolationStencil)
        throw OpenMMException("updateParametersInContext: the interpolation stencil cannot be changed");
    if (updated.fluidFluctuations != lattice.fluidFluctuations)
        throw OpenMMException("updateParametersInContext: the fluid fluctuations cannot be switched on or off");
    lattice = updated;
    kernel.getAs<CalcLBMForceKernel>().copyParametersToContext(context, lattice);
}

void LBMForceImpl::getLocalDomain(int start[3], int count[3]) const {
    decomposition.getLocalDomain(start, count);
}

void LBMForceImpl::getFluidFields(ContextImpl& context, vector<double>& density, vector<Vec3>& velocity, bool gather, bool halo) {
    // The kernels give the fields of the domain of the rank, extended by the halo on request (CalcLBMForceKernel); here
    // they are gathered on rank 0 with gather.  With one domain the domain is the lattice.
    if (gather && halo)
        throw OpenMMException("LBMForce: getFluidFields() cannot gather the fields and add the halo in the same call");
    kernel.getAs<CalcLBMForceKernel>().getFluidFields(context, density, velocity, halo);
    if (!gather || !decomposition.isDecomposed())
        return;
    vector<double> local, all;
    local.reserve(4*density.size());
    for (size_t l = 0; l < density.size(); l++) {
        local.push_back(density[l]);
        for (int a = 0; a < 3; a++)
            local.push_back(velocity[l][a]);
    }
    decomposition.gatherBlocks(local, 4, all);
    density.clear();
    velocity.clear();
    for (size_t node = 0; 4*node < all.size(); node++) {
        density.push_back(all[4*node]);
        velocity.push_back(Vec3(all[4*node+1], all[4*node+2], all[4*node+3]));
    }
}

void LBMForceImpl::getFluidState(ContextImpl& context, vector<double>& state, bool gather) {
    // The state of the domain of the rank, [q*numLocal + l] with l the index of the node in the domain (the state of
    // the kernel), or with gather that of the whole lattice, [q*numNodes + node], on rank 0.  With one domain both are
    // the state of the kernel.
    kernel.getAs<CalcLBMForceKernel>().getFluidState(context, state);
    if (!gather || !decomposition.isDecomposed())
        return;
    const int Q = 19;
    size_t numNodes = lattice.getNumNodes(), numLocal = state.size()/Q;
    vector<double> local(Q*numLocal);
    for (size_t l = 0; l < numLocal; l++)
        for (int q = 0; q < Q; q++)
            local[Q*l + q] = state[q*numLocal + l];
    vector<double> global;
    decomposition.gatherBlocks(local, Q, global);
    state.clear();
    if (global.empty())
        return;
    state.resize(Q*numNodes);
    for (size_t node = 0; node < numNodes; node++)
        for (int q = 0; q < Q; q++)
            state[q*numNodes + node] = global[Q*node + q];
}

void LBMForceImpl::setFluidState(ContextImpl& context, const vector<double>& state, bool scatter) {
    // The reverse of getFluidState(): the state of the domain of the rank, or with scatter that of the whole lattice
    // from rank 0.  With the decomposition the call is collective, and so is the check of the size.
    if (!decomposition.isDecomposed()) {
        kernel.getAs<CalcLBMForceKernel>().setFluidState(context, state);
        return;
    }
    const int Q = 19;
    int start[3], count[3];
    decomposition.getLocalDomain(start, count);
    size_t numNodes = lattice.getNumNodes(), numLocal = count[0]*count[1]*count[2];
    bool wrongSize = (scatter ? decomposition.getRank() == 0 && state.size() != Q*numNodes : state.size() != Q*numLocal);
    decomposition.throwIfAnyError(wrongSize ? "LBMForce: setFluidState() was called with a state of the wrong size (19 "
            "values per node of the lattice with scatter, of the domain of the rank otherwise)" : "");
    if (!scatter) {
        kernel.getAs<CalcLBMForceKernel>().setFluidState(context, state);
        return;
    }
    vector<double> global, local, blockState(Q*numLocal);
    if (decomposition.getRank() == 0) {
        global.resize(Q*numNodes);
        for (size_t node = 0; node < numNodes; node++)
            for (int q = 0; q < Q; q++)
                global[Q*node + q] = state[q*numNodes + node];
    }
    decomposition.scatterBlocks(global, Q, local);
    for (size_t l = 0; l < numLocal; l++)
        for (int q = 0; q < Q; q++)
            blockState[q*numLocal + l] = local[Q*l + q];
    kernel.getAs<CalcLBMForceKernel>().setFluidState(context, blockState);
}

/**
 * A checkpoint starts with a header that identifies it: a tag, the format version, the platform, the grid size,
 * the number of coupled particles, (from version 2) the drag scheme, (from version 3) whether the fluid
 * fluctuates, (from version 4) the wall scheme, (from version 5) the boundary types of the six faces and (from
 * version 6) the interpolation stencil.  The kernel writes the rest.  Version 1 was written before the drag scheme
 * existed, with the explicit drag, versions 1 and 2 before the fluid fluctuations, without them, versions 1 to 3
 * before the wall schemes, with bounce-back, versions 1 to 4 with periodic faces, and versions 1 to 5 with the
 * nearest node.
 */
static const char checkpointTag[8] = {'L', 'B', 'M', 'C', 'K', 'P', 'T', '1'};
static const int checkpointVersion = 6;

void LBMForceImpl::writeCheckpointHeader(ContextImpl& context, ostream& stream) const {
    stream.write(checkpointTag, sizeof(checkpointTag));
    stream.write((const char*) &checkpointVersion, sizeof(int));
    string platform = context.getPlatform().getName();
    int length = platform.size();
    stream.write((const char*) &length, sizeof(int));
    stream.write(platform.c_str(), length);
    int header[14] = {lattice.nx, lattice.ny, lattice.nz, (int) lattice.particles.size(), (int) lattice.dragScheme,
                      (int) lattice.fluidFluctuations, (int) lattice.wallScheme};
    for (int face = 0; face < 6; face++)
        header[7+face] = (int) lattice.faceBoundary[face];
    header[13] = (int) lattice.interpolationStencil;
    stream.write((const char*) header, sizeof(header));
}

void LBMForceImpl::readCheckpointHeader(ContextImpl& context, istream& stream) const {
    char tag[sizeof(checkpointTag)];
    stream.read(tag, sizeof(tag));
    if (!stream || !equal(tag, tag+sizeof(tag), checkpointTag))
        throw OpenMMException("LBMForce: the data are not a checkpoint written by LBMForce::createCheckpoint()");
    int version, length;
    stream.read((char*) &version, sizeof(int));
    if (version < 1 || version > checkpointVersion)
        throw OpenMMException("LBMForce: unsupported checkpoint version");
    stream.read((char*) &length, sizeof(int));
    if (!stream || length < 0 || length > 1000)
        throw OpenMMException("LBMForce: the checkpoint is damaged");
    string platform(length, ' ');
    stream.read(&platform[0], length);
    if (platform != context.getPlatform().getName())
        throw OpenMMException("LBMForce: the checkpoint was written on the platform " + platform + ", not on " +
                context.getPlatform().getName());
    int header[14] = {0, 0, 0, 0, (int) LBMForce::Explicit, 0, (int) LBMForce::BounceBack};
    for (int face = 0; face < 6; face++)
        header[7+face] = (int) LBMForce::Periodic;
    header[13] = (int) LBMForce::NearestNode;
    stream.read((char*) header, (version <= 4 ? version+3 : version == 5 ? 13 : 14)*sizeof(int));
    if (!stream || header[0] != lattice.nx || header[1] != lattice.ny || header[2] != lattice.nz ||
            header[3] != (int) lattice.particles.size())
        throw OpenMMException("LBMForce: the checkpoint was written for a different grid size or number of coupled particles");
    if (header[4] != (int) lattice.dragScheme)
        throw OpenMMException("LBMForce: the checkpoint was written with a different drag scheme");
    if (header[5] != (int) lattice.fluidFluctuations)
        throw OpenMMException(string("LBMForce: the checkpoint was written ") + (header[5] ? "with" : "without") +
                " fluid fluctuations, and this Context has them " + (lattice.fluidFluctuations ? "on" : "off"));
    if (header[6] != (int) lattice.wallScheme)
        throw OpenMMException("LBMForce: the checkpoint was written with a different wall scheme");
    for (int face = 0; face < 6; face++)
        if (header[7+face] != (int) lattice.faceBoundary[face])
            throw OpenMMException("LBMForce: the checkpoint was written with different boundary types of the faces (setFaceBoundary())");
    if (header[13] != (int) lattice.interpolationStencil)
        throw OpenMMException("LBMForce: the checkpoint was written with a different interpolation stencil");
}

void LBMForceImpl::createCheckpoint(ContextImpl& context, ostream& stream) {
    writeCheckpointHeader(context, stream);
    kernel.getAs<CalcLBMForceKernel>().createCheckpoint(context, stream);
    if (!stream)
        throw OpenMMException("LBMForce: error writing the checkpoint");
}

void LBMForceImpl::loadCheckpoint(ContextImpl& context, istream& stream) {
    readCheckpointHeader(context, stream);
    kernel.getAs<CalcLBMForceKernel>().loadCheckpoint(context, stream);
    if (!stream)
        throw OpenMMException("LBMForce: the checkpoint is truncated");
}

/**
 * Checkpoint files.  The file of openmmlbm.saveCheckpoint() with one domain (tag fileTagSingle) holds the OpenMM
 * checkpoint and the checkpoint of the force (createCheckpoint()), each after its length:
 *
 *   tag, int64 length of the OpenMM checkpoint, int64 length of the force checkpoint, the two checkpoints.
 *
 * The file of saveCheckpointFile() (tag fileTagDomains) can be read with any decomposition:
 *
 *   tag, int64 length of the head, head, int64 number of ranks R, R x (int64 offset, int64 length) of the data of
 *   each rank, populations of the whole lattice, data of each rank.
 *
 * The head, written by rank 0, has the header of createCheckpoint(), the precision, the number of blocks along
 * each axis and the particles (time, step count, box, positions and velocities, global parameters), which are the
 * same on every rank.  The populations are doubles, [q*numNodes + node], written and read by every rank for the
 * nodes of its domain.  The data of a rank are its OpenMM checkpoint, after its int64 length, and
 * createRankCheckpoint() of its kernel.  With the same decomposition every rank loads its own data and the run
 * continues bit for bit; with another one the particles are set from the head, and each rank keeps the random
 * number generators of its new Context.
 */
static const char fileTagSingle[] = "OPENMMLBM-CHECKPOINT-1\n";
static const char fileTagDomains[] = "OPENMMLBM-CHECKPOINT-2\n";
static const int fileTagLength = sizeof(fileTagSingle) - 1;

static void writeInt64(string& data, long long value) {
    data.append((const char*) &value, sizeof(value));
}

/** The precision of the Context, as the property "Precision" of the GPU platforms gives it, or "double". */
static string getContextPrecision(ContextImpl& context) {
    const Platform& platform = context.getPlatform();
    const vector<string>& names = platform.getPropertyNames();
    if (find(names.begin(), names.end(), "Precision") == names.end())
        return "double";
    return platform.getPropertyValue(context.getOwner(), "Precision");
}

void LBMForceImpl::saveCheckpointFile(ContextImpl& context, const string& path) {
    // Every rank builds the head, which is the same on all of them, so that every rank knows the offsets.
    CalcLBMForceKernel& lbm = kernel.getAs<CalcLBMForceKernel>();
    const int Q = 19;
    stringstream head;
    writeCheckpointHeader(context, head);
    string precision = getContextPrecision(context);
    int length = precision.size(), blocks[3];
    head.write((const char*) &length, sizeof(int));
    head.write(precision.c_str(), length);
    decomposition.getBlocks(blocks);
    head.write((const char*) blocks, sizeof(blocks));
    State state = context.getOwner().getState(State::Positions | State::Velocities | State::Parameters);
    double time = state.getTime();
    long long stepCount = state.getStepCount(), numParticles = context.getSystem().getNumParticles();
    head.write((const char*) &time, sizeof(double));
    head.write((const char*) &stepCount, sizeof(long long));
    Vec3 box[3];
    state.getPeriodicBoxVectors(box[0], box[1], box[2]);
    head.write((const char*) box, sizeof(box));
    head.write((const char*) &numParticles, sizeof(long long));
    head.write((const char*) state.getPositions().data(), numParticles*sizeof(Vec3));
    head.write((const char*) state.getVelocities().data(), numParticles*sizeof(Vec3));
    const map<string, double>& parameters = state.getParameters();
    int numParameters = parameters.size();
    head.write((const char*) &numParameters, sizeof(int));
    for (auto& parameter : parameters) {
        int nameLength = parameter.first.size();
        head.write((const char*) &nameLength, sizeof(int));
        head.write(parameter.first.c_str(), nameLength);
        head.write((const char*) &parameter.second, sizeof(double));
    }

    // The data of this rank and the table of the data of all the ranks.
    stringstream openmm, rankData;
    context.getOwner().createCheckpoint(openmm);
    string data;
    writeInt64(data, openmm.str().size());
    data += openmm.str();
    lbm.createRankCheckpoint(context, rankData);
    data += rankData.str();
    int numRanks = decomposition.getSize();
    vector<double> lengths(numRanks, 0.0);
    lengths[decomposition.getRank()] = (double) data.size();
    decomposition.sum(lengths.data(), numRanks);
    string start(fileTagDomains, fileTagLength);
    writeInt64(start, head.str().size());
    start += head.str();
    writeInt64(start, numRanks);
    long long fluidOffset = start.size() + 16LL*numRanks;
    long long offset = fluidOffset + ((long long) Q)*lattice.getNumNodes()*sizeof(double), myOffset = 0;
    for (int r = 0; r < numRanks; r++) {
        if (r == decomposition.getRank())
            myOffset = offset;
        writeInt64(start, offset);
        writeInt64(start, (long long) lengths[r]);
        offset += (long long) lengths[r];
    }
    vector<double> populations;
    lbm.getFluidState(context, populations);

    // The file is written under a temporary name and then renamed, so that an interrupted write never replaces a
    // valid checkpoint with a damaged one.
    string temporary = path + ".tmp";
    LBMParallelFile file(decomposition, temporary, true);
    file.writeOnRoot(0, start);
    file.writeDomain(fluidOffset, populations.data(), sizeof(double), Q, true);
    file.writeAt(myOffset, data.data(), data.size());
    file.close();
    string error;
    if (decomposition.getRank() == 0 && rename(temporary.c_str(), path.c_str()) != 0)
        error = "LBMForce: cannot rename " + temporary + " to " + path;
    decomposition.throwIfAnyError(error);
}

/** A reader of the bytes of a head, which throws if they end too soon. */
class HeadReader {
public:
    HeadReader(const string& data, const string& path) : data(data), path(path), position(0) {
    }
    void read(void* value, size_t length) {
        if (position + length > data.size())
            throw OpenMMException("LBMForce: " + path + " is damaged");
        memcpy(value, data.data()+position, length);
        position += length;
    }
    void skip(size_t length) {
        position += length;
    }
    template <class T> T get() {
        T value;
        read(&value, sizeof(T));
        return value;
    }
    string getString() {
        int length = get<int>();
        if (length < 0 || length > 100000)
            throw OpenMMException("LBMForce: " + path + " is damaged");
        string value(length, ' ');
        read(&value[0], length);
        return value;
    }
private:
    const string& data;
    const string& path;
    size_t position;
};

void LBMForceImpl::loadCheckpointFile(ContextImpl& context, const string& path) {
    CalcLBMForceKernel& lbm = kernel.getAs<CalcLBMForceKernel>();
    const int Q = 19;
    LBMParallelFile file(decomposition, path, false);
    long long size = file.getSize();
    string tag(fileTagLength, ' ');
    file.readAt(0, &tag[0], min((long long) fileTagLength, size));
    file.checkErrors();
    if (tag == fileTagSingle) {
        // A file of openmmlbm.saveCheckpoint() with one domain.
        if (decomposition.isDecomposed())
            throw OpenMMException("LBMForce: " + path + " was written with one domain by openmmlbm.saveCheckpoint(), and "
                    "only a Context with one domain can load it; to continue a run with another decomposition write the "
                    "checkpoint with LBMForce.saveCheckpointFile()");
        long long lengths[2];
        file.readAt(fileTagLength, (char*) lengths, sizeof(lengths));
        file.checkErrors();
        if (lengths[0] < 0 || lengths[1] < 0 || fileTagLength + 16 + lengths[0] + lengths[1] != size)
            throw OpenMMException("LBMForce: " + path + " is truncated or damaged");
        string openmm(lengths[0], ' '), force(lengths[1], ' ');
        file.readAt(fileTagLength + 16, &openmm[0], lengths[0]);
        file.readAt(fileTagLength + 16 + lengths[0], &force[0], lengths[1]);
        file.close();
        stringstream openmmStream(openmm), forceStream(force);
        context.getOwner().loadCheckpoint(openmmStream);
        loadCheckpoint(context, forceStream);
        return;
    }
    if (tag != fileTagDomains)
        throw OpenMMException("LBMForce: " + path + " is not a checkpoint written by LBMForce.saveCheckpointFile() or "
                "openmmlbm.saveCheckpoint()");

    // The head, which every rank reads.
    long long headLength = 0, numRanks = 0;
    file.readAt(fileTagLength, (char*) &headLength, sizeof(long long));
    file.checkErrors();
    if (headLength < 0 || fileTagLength + 16 + headLength > size)
        throw OpenMMException("LBMForce: " + path + " is truncated or damaged");
    string head(headLength, ' ');
    file.readAt(fileTagLength + 8, &head[0], headLength);
    file.readAt(fileTagLength + 8 + headLength, (char*) &numRanks, sizeof(long long));
    file.checkErrors();
    stringstream headStream(head);
    readCheckpointHeader(context, headStream);
    HeadReader reader(head, path);
    reader.skip((size_t) headStream.tellg());
    string precision = reader.getString();
    if (precision != getContextPrecision(context))
        throw OpenMMException("LBMForce: " + path + " was written in " + precision + " precision, and this Context uses " +
                getContextPrecision(context));
    int blocks[3], myBlocks[3];
    reader.read(blocks, sizeof(blocks));
    decomposition.getBlocks(myBlocks);
    bool sameDecomposition = (blocks[0] == myBlocks[0] && blocks[1] == myBlocks[1] && blocks[2] == myBlocks[2] &&
                              numRanks == decomposition.getSize());
    if (numRanks < 1 || numRanks != ((long long) blocks[0])*blocks[1]*blocks[2])
        throw OpenMMException("LBMForce: " + path + " is damaged");
    vector<long long> table(2*numRanks);
    long long tableOffset = fileTagLength + 16 + headLength;
    file.readAt(tableOffset, (char*) table.data(), 16*numRanks);
    file.checkErrors();
    long long fluidOffset = tableOffset + 16*numRanks;
    if (fluidOffset + ((long long) Q)*lattice.getNumNodes()*sizeof(double) > size)
        throw OpenMMException("LBMForce: " + path + " is truncated");

    // The particles and the data of the rank.
    if (sameDecomposition) {
        int rank = decomposition.getRank();
        long long offset = table[2*rank], length = table[2*rank+1], openmmLength = -1;
        string data(length >= 8 && offset >= fluidOffset && offset + length <= size ? length : 0, ' ');
        file.readAt(offset, &data[0], data.size());
        int start[3], count[3];
        decomposition.getLocalDomain(start, count);
        vector<double> populations(((size_t) Q)*count[0]*count[1]*count[2]);
        file.readDomain(fluidOffset, populations.data(), sizeof(double), Q, true);
        file.close();
        if (!data.empty())
            memcpy(&openmmLength, data.data(), 8);
        string error;
        if (openmmLength < 0 || 8 + openmmLength > (long long) data.size())
            error = "LBMForce: " + path + " is damaged";
        else {
            // An error of one rank must stop all of them (collective errors).
            try {
                stringstream openmm(data.substr(8, openmmLength));
                context.getOwner().loadCheckpoint(openmm);
            }
            catch (exception& e) {
                error = e.what();
            }
        }
        decomposition.throwIfAnyError(error);
        lbm.setFluidState(context, populations);
        stringstream rankData(data.substr(8 + openmmLength));
        try {
            lbm.loadRankCheckpoint(context, rankData);
            if (!rankData)
                error = "LBMForce: " + path + " is damaged";
        }
        catch (exception& e) {
            error = e.what();
        }
        decomposition.throwIfAnyError(error);
        return;
    }
    double time = reader.get<double>();
    long long stepCount = reader.get<long long>();
    Vec3 box[3];
    reader.read(box, sizeof(box));
    long long numParticles = reader.get<long long>();
    if (numParticles != context.getSystem().getNumParticles())
        throw OpenMMException("LBMForce: " + path + " was written for a System with a different number of particles");
    vector<Vec3> positions(numParticles), velocities(numParticles);
    reader.read(positions.data(), numParticles*sizeof(Vec3));
    reader.read(velocities.data(), numParticles*sizeof(Vec3));
    int numParameters = reader.get<int>();
    map<string, double> parameters;
    for (int i = 0; i < numParameters; i++) {
        string name = reader.getString();
        parameters[name] = reader.get<double>();
    }
    int start[3], count[3];
    decomposition.getLocalDomain(start, count);
    vector<double> populations(((size_t) Q)*count[0]*count[1]*count[2]);
    file.readDomain(fluidOffset, populations.data(), sizeof(double), Q, true);
    file.close();
    Context& owner = context.getOwner();
    owner.setTime(time);
    owner.setStepCount(stepCount);
    owner.setPeriodicBoxVectors(box[0], box[1], box[2]);
    owner.setPositions(positions);
    owner.setVelocities(velocities);
    for (auto& parameter : parameters)
        owner.setParameter(parameter.first, parameter.second);
    lbm.setFluidState(context, populations);
    lbm.resetRankState(context);
}

void LBMForceImpl::writeFluidFile(ContextImpl& context, const string& path, const string& head, const string& tail,
        const string& arrays, bool doublePrecision) {
    // The arrays of the fluid of the whole lattice, each after its length in bytes, between a head and a tail that
    // rank 0 writes: every rank writes the nodes of its domain.
    vector<string> names;
    stringstream list(arrays);
    string name;
    while (list >> name) {
        if (name != "density" && name != "velocity" && name != "solid")
            throw OpenMMException("LBMForce: unknown array of the fluid: " + name);
        names.push_back(name);
    }
    vector<double> density;
    vector<Vec3> velocity;
    kernel.getAs<CalcLBMForceKernel>().getFluidFields(context, density, velocity, false);
    int start[3], count[3];
    decomposition.getLocalDomain(start, count);
    size_t numLocal = density.size();
    long long numNodes = lattice.getNumNodes();
    int realSize = (doublePrecision ? 8 : 4);
    LBMParallelFile file(decomposition, path, true);
    file.writeOnRoot(0, head);
    long long offset = head.size();
    for (const string& array : names) {
        int components = (array == "velocity" ? 3 : 1), elementSize = (array == "solid" ? 1 : realSize);
        vector<char> values(numLocal*components*elementSize, 0);
        if (array == "solid") {
            for (int node : lattice.solidNodes) {
                int i = node%lattice.nx - start[0], j = (node/lattice.nx)%lattice.ny - start[1];
                int k = node/(lattice.nx*lattice.ny) - start[2];
                if (i >= 0 && i < count[0] && j >= 0 && j < count[1] && k >= 0 && k < count[2])
                    values[i + count[0]*(j + count[1]*k)] = 1;
            }
        }
        else
            for (size_t l = 0; l < numLocal; l++)
                for (int c = 0; c < components; c++) {
                    double value = (array == "density" ? density[l] : velocity[l][c]);
                    char* target = &values[(l*components + c)*elementSize];
                    if (doublePrecision)
                        memcpy(target, &value, 8);
                    else {
                        float single = (float) value;
                        memcpy(target, &single, 4);
                    }
                }
        string length;
        writeInt64(length, numNodes*components*elementSize);
        file.writeOnRoot(offset, length);
        file.writeDomain(offset + 8, values.data(), elementSize, components, false);
        offset += 8 + numNodes*components*elementSize;
    }
    file.writeOnRoot(offset, tail);
    file.close();
}

double LBMForceImpl::getFluidMachNumber(ContextImpl& context) {
    return kernel.getAs<CalcLBMForceKernel>().getFluidMachNumber(context);
}

Vec3 LBMForceImpl::getWallForce(ContextImpl& context) {
    return kernel.getAs<CalcLBMForceKernel>().getWallForce(context);
}

void LBMForceImpl::getLatticeParameters(double& dx, double& dt, double& tau) const {
    dx = lattice.dx;
    dt = lattice.dt;
    tau = lattice.tau;
}
