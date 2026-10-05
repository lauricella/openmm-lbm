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
#include <cmath>
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
    lattice.friction = force.getFriction();
    lattice.kT = BOLTZ*force.getTemperature();
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
    kernel = context.getPlatform().createKernel(CalcLBMForceKernel::Name(), context);
    kernel.getAs<CalcLBMForceKernel>().initialize(context.getSystem(), owner, lattice);
}

void LBMForceImpl::updateContextState(ContextImpl& context, bool& forcesInvalid) {
    kernel.getAs<CalcLBMForceKernel>().beginStep(context);
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
