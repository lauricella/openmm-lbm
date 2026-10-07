/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "internal/LBMForceImpl.h"
#include "LBMKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include <algorithm>
#include <cmath>
#include <iostream>
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

    // The model is accurate for moderate relaxation times (docs/theory.md, section 6).

    if (lattice.tau < 0.505 || lattice.tau > 2.0)
        cerr << "Warning: LBMForce: the relaxation time tau = " << lattice.tau << " is outside the range [0.505, 2] "
             << "in which the lattice Boltzmann model is accurate. tau = 3 nu dt/dx^2 + 1/2: change the viscosity, "
             << "the time step or the lattice spacing." << endl;

    // With the explicit drag at the nearest node, the hydrodynamic part of the self-mobility of a coupled particle
    // decreases as tau grows and becomes negative at tau = 1.79 (docs/theory.md, section 2).  With the centred drag
    // it stays positive.

    if (!lattice.particles.empty() && lattice.dragScheme == LBMForce::Explicit && lattice.tau > 1.7)
        cerr << "Warning: LBMForce: tau = " << lattice.tau << " > 1.7: with the explicit drag at the nearest node "
             << "the hydrodynamic self-mobility of a coupled particle is small, and negative above tau = 1.79, so "
             << "particles move less than they should. Reduce the viscosity or the time step, or use a coarser "
             << "lattice." << endl;

    // The explicit drag multiplies the velocity of a particle relative to the fluid by 1 - gamma*dt in one step
    // (docs/theory.md, section 2).  The centred drag is stable for any friction.

    double gammaDt = lattice.friction*lattice.dt;
    if (!lattice.particles.empty() && lattice.dragScheme == LBMForce::Explicit && gammaDt > 1.0)
        cerr << "Warning: LBMForce: friction*dt = " << gammaDt << " > 1: with the explicit drag the velocity of a "
             << "particle relative to the fluid changes sign at every step" << (gammaDt >= 2.0 ? ", and grows without "
             "bound since friction*dt >= 2" : "") << ". Reduce the friction or the time step." << endl;

    // With the fluctuating fluid the explicit drag makes the coupled particles too hot, by about
    // gamma dt m/(2 m_c (1 + zeta y)), because it misses the response of the cell within the step; the centred drag
    // gives the right temperature (docs/theory.md, section 7).  The warning gives the bound gamma dt m/(2 m_c) for the
    // heaviest coupled particle.

    if (!lattice.particles.empty() && lattice.dragScheme == LBMForce::Explicit && lattice.fluidFluctuations &&
            lattice.kT > 0 && gammaDt > 0) {
        double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
        double maxMass = 0;
        for (int particle : lattice.particles)
            maxMass = max(maxMass, context.getSystem().getParticleMass(particle));
        cerr << "Warning: LBMForce: with fluid fluctuations the explicit drag makes the coupled particles hotter than "
             << "the set temperature, by about friction*dt*m/(2 m_c) = " << 100.0*gammaDt*maxMass/(2.0*cellMass)
             << "% for the heaviest one (m_c = " << cellMass << " Da is the mass of fluid in a cell). Use the Centered "
             << "drag scheme with fluid fluctuations." << endl;
    }
#ifdef LBM_DEBUG
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    cerr << "LBMForce lattice: dx = " << lattice.dx << " nm, dt = " << lattice.dt << " ps, m_c = " << cellMass
         << " Da, tau = " << lattice.tau << ", kT/(m_c c_s^2) = "
         << 3.0*lattice.kT*lattice.dt*lattice.dt/(cellMass*lattice.dx*lattice.dx) << endl;
#endif
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
    for (int face = 0; face < 6; face++)
        if (updated.faceBoundary[face] != lattice.faceBoundary[face])
            throw OpenMMException("updateParametersInContext: the boundary types of the faces cannot be changed");
    if (updated.dragScheme != lattice.dragScheme)
        throw OpenMMException("updateParametersInContext: the drag scheme cannot be changed");
    if (updated.fluidFluctuations != lattice.fluidFluctuations)
        throw OpenMMException("updateParametersInContext: the fluid fluctuations cannot be switched on or off");
    lattice = updated;
    kernel.getAs<CalcLBMForceKernel>().copyParametersToContext(context, lattice);
}

void LBMForceImpl::getFluidFields(ContextImpl& context, vector<double>& density, vector<Vec3>& velocity) {
    kernel.getAs<CalcLBMForceKernel>().getFluidFields(context, density, velocity);
}

void LBMForceImpl::getFluidState(ContextImpl& context, vector<double>& state) {
    kernel.getAs<CalcLBMForceKernel>().getFluidState(context, state);
}

void LBMForceImpl::setFluidState(ContextImpl& context, const vector<double>& state) {
    kernel.getAs<CalcLBMForceKernel>().setFluidState(context, state);
}

/**
 * A checkpoint starts with a header that identifies it: a tag, the format version, the platform, the grid size,
 * the number of coupled particles, (from version 2) the drag scheme, (from version 3) whether the fluid
 * fluctuates and (from version 4) the wall scheme.  The kernel writes the rest, which is the same in all versions.
 * Version 1 was written before the drag scheme existed, with the explicit drag, versions 1 and 2 before the fluid
 * fluctuations, without them, and versions 1 to 3 before the wall schemes, with bounce-back.
 */
static const char checkpointTag[8] = {'L', 'B', 'M', 'C', 'K', 'P', 'T', '1'};
static const int checkpointVersion = 4;

void LBMForceImpl::createCheckpoint(ContextImpl& context, ostream& stream) {
    stream.write(checkpointTag, sizeof(checkpointTag));
    stream.write((const char*) &checkpointVersion, sizeof(int));
    string platform = context.getPlatform().getName();
    int length = platform.size();
    stream.write((const char*) &length, sizeof(int));
    stream.write(platform.c_str(), length);
    int header[7] = {lattice.nx, lattice.ny, lattice.nz, (int) lattice.particles.size(), (int) lattice.dragScheme,
                     (int) lattice.fluidFluctuations, (int) lattice.wallScheme};
    stream.write((const char*) header, sizeof(header));
    kernel.getAs<CalcLBMForceKernel>().createCheckpoint(context, stream);
    if (!stream)
        throw OpenMMException("LBMForce: error writing the checkpoint");
}

void LBMForceImpl::loadCheckpoint(ContextImpl& context, istream& stream) {
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
    int header[7] = {0, 0, 0, 0, (int) LBMForce::Explicit, 0, (int) LBMForce::BounceBack};
    stream.read((char*) header, (version+3)*sizeof(int));
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
    kernel.getAs<CalcLBMForceKernel>().loadCheckpoint(context, stream);
    if (!stream)
        throw OpenMMException("LBMForce: the checkpoint is truncated");
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
