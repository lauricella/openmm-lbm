/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

/**
 * Tests shared by all platforms.  Each platform has a Test<Platform>LBMForce.cpp that registers its
 * kernel factories, selects the precision and calls runPlatformTests().
 */

#include "LBMForce.h"
#include "openmm/Context.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/State.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/AssertionUtilities.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace LBMPlugin;
using namespace OpenMM;
using namespace std;

/**
 * Tolerance of a comparison with values stored by the platform: populations and moments are kept in
 * single precision when the platform runs in single precision.
 */
double getStorageTolerance(Platform& platform) {
    if (platform.getName() == "Reference")
        return 1e-12;
    string precision = platform.getPropertyDefaultValue("Precision");
    return (precision == "single" ? 2e-6 : 1e-12);
}

/**
 * Build a periodic System of free particles, all coupled to an LBMForce with an 8x8x8 grid.
 */
System* createSystem(LBMForce*& force, int numParticles=10, double boxSize=4.0) {
    System* system = new System();
    system->setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    force = new LBMForce();
    force->setGridSize(8, 8, 8);
    for (int i = 0; i < numParticles; i++) {
        system->addParticle(100.0);
        force->addParticle(i);
    }
    system->addForce(force);
    return system;
}

vector<Vec3> createPositions(int numParticles, double boxSize=4.0) {
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++)
        positions[i] = Vec3(0.37*i, 0.53*i+0.1, 0.71*i+0.2)*(boxSize/4.0);
    return positions;
}

void testParameters() {
    LBMForce force;
    int nx, ny, nz;
    force.getGridSize(nx, ny, nz);
    ASSERT_EQUAL(0, nx);
    force.setGridSize(10, 12, 14);
    force.getGridSize(nx, ny, nz);
    ASSERT_EQUAL(10, nx);
    ASSERT_EQUAL(12, ny);
    ASSERT_EQUAL(14, nz);
    force.setFluidDensity(500.0);
    force.setKinematicViscosity(2.0);
    force.setFriction(5.0);
    force.setTemperature(310.0);
    force.setRandomNumberSeed(123);
    force.setBodyAcceleration(Vec3(0.1, 0.2, 0.3));
    force.setInitialFluidVelocity(Vec3(-0.1, 0.0, 0.05));
    force.setFluidMomentumRemovalFrequency(10);
    ASSERT_EQUAL(500.0, force.getFluidDensity());
    ASSERT_EQUAL(2.0, force.getKinematicViscosity());
    ASSERT_EQUAL(5.0, force.getFriction());
    ASSERT_EQUAL(310.0, force.getTemperature());
    ASSERT_EQUAL(123, force.getRandomNumberSeed());
    ASSERT_EQUAL_VEC(Vec3(0.1, 0.2, 0.3), force.getBodyAcceleration(), 0.0);
    ASSERT_EQUAL_VEC(Vec3(-0.1, 0.0, 0.05), force.getInitialFluidVelocity(), 0.0);
    ASSERT_EQUAL(10, force.getFluidMomentumRemovalFrequency());
    ASSERT_EQUAL(100, force.getMachCheckFrequency());
    ASSERT_EQUAL(LBMForce::EulerMaruyama, force.getCouplingScheme());
    force.setCouplingScheme(LBMForce::NVE);
    ASSERT_EQUAL(LBMForce::NVE, force.getCouplingScheme());
    ASSERT_EQUAL(0.3, force.getMachNumberLimit());
    force.setMachCheckFrequency(50);
    force.setMachNumberLimit(0.25);
    ASSERT_EQUAL(50, force.getMachCheckFrequency());
    ASSERT_EQUAL(0.25, force.getMachNumberLimit());
    vector<int> solid;
    force.getSolidNodes(solid);
    ASSERT_EQUAL(0, solid.size());
    force.setSolidNodes(vector<int>({5, 2, 9}));
    force.getSolidNodes(solid);
    ASSERT_EQUAL_CONTAINERS(vector<int>({5, 2, 9}), solid);
    ASSERT_EQUAL(0, force.addParticle(3));
    ASSERT_EQUAL(1, force.addParticle(7));
    ASSERT_EQUAL(2, force.getNumParticles());
    force.setParticle(1, 8);
    ASSERT_EQUAL(8, force.getParticle(1));
    ASSERT(force.usesPeriodicBoundaryConditions());
}

/**
 * Before the first step no coupling force has been computed, so the force on the particles is zero; the
 * coupling is dissipative, so the energy is always zero.
 */
void testZeroForce(Platform& platform) {
    LBMForce* force;
    System* system = createSystem(force);
    VerletIntegrator integrator(0.01);
    Context context(*system, integrator, platform);
    context.setPositions(createPositions(system->getNumParticles()));
    State state = context.getState(State::Forces | State::Energy);
    for (int i = 0; i < system->getNumParticles(); i++)
        ASSERT_EQUAL_VEC(Vec3(0, 0, 0), state.getForces()[i], 0.0);
    ASSERT_EQUAL(0.0, state.getPotentialEnergy());
    integrator.step(5);
    delete system;
}

/**
 * The fluid starts at equilibrium with the density and velocity set on the force.
 */
void testInitialFluidFields(Platform& platform) {
    LBMForce* force;
    System* system = createSystem(force);
    Vec3 u0(0.1, -0.05, 0.02);
    force->setFluidDensity(500.0);
    force->setInitialFluidVelocity(u0);
    VerletIntegrator integrator(0.01);
    Context context(*system, integrator, platform);
    context.setPositions(createPositions(system->getNumParticles()));
    vector<double> density;
    vector<Vec3> velocity;
    force->getFluidFields(context, density, velocity);
    ASSERT_EQUAL(8*8*8, density.size());
    ASSERT_EQUAL(8*8*8, velocity.size());
    double tol = getStorageTolerance(platform);
    for (int node = 0; node < (int) density.size(); node++) {
        ASSERT_EQUAL_TOL(500.0, density[node], 5*tol);
        ASSERT_EQUAL_VEC(u0, velocity[node], 1e3*tol);
    }
    delete system;
}

/**
 * The state of the fluid can be read, changed and written back.
 */
void testFluidStateRoundTrip(Platform& platform) {
    LBMForce* force;
    System* system = createSystem(force);
    VerletIntegrator integrator(0.01);
    Context context(*system, integrator, platform);
    context.setPositions(createPositions(system->getNumParticles()));
    vector<double> state;
    force->getFluidState(context, state);
    int numNodes = 8*8*8;
    ASSERT_EQUAL(19*numNodes, state.size());
    double tol = getStorageTolerance(platform);

    // At rest with lattice density 1, the populations are the lattice weights: the state, made of the
    // deviations f - w from the rest equilibrium, is zero.

    for (int i = 0; i < (int) state.size(); i++)
        ASSERT_EQUAL_TOL(0.0, state[i], tol);

    // Move some mass from the rest population to population 1 (+x) at node 5: the density stays the
    // same and the fluid at that node moves along +x.

    double delta = 0.01;
    state[5] -= delta;
    state[numNodes+5] += delta;
    force->setFluidState(context, state);
    vector<double> state2;
    force->getFluidState(context, state2);
    ASSERT_EQUAL(state.size(), state2.size());
    for (int i = 0; i < (int) state.size(); i++)
        ASSERT_EQUAL_TOL(state[i], state2[i], tol);
    vector<double> density;
    vector<Vec3> velocity;
    force->getFluidFields(context, density, velocity);
    double velocityScale = 4.0/8/0.01;
    ASSERT_EQUAL_TOL(force->getFluidDensity(), density[5], 5*tol);
    ASSERT_EQUAL_VEC(Vec3(delta*velocityScale, 0, 0), velocity[5], 1e3*tol);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), velocity[6], 1e3*tol);

    // A state of the wrong size is rejected.

    bool threw = false;
    try {
        state.resize(10);
        force->setFluidState(context, state);
    }
    catch (const OpenMMException& e) {
        threw = true;
    }
    ASSERT(threw);
    delete system;
}

/**
 * Invalid setups are rejected when the Context is created.
 */
void testInvalidSetup(Platform& platform) {
    int numParticles = 10;
    vector<string> cases = {"integrator", "viscosity", "cells", "particle", "duplicate"};
    for (const string& c : cases) {
        LBMForce* force;
        System* system = createSystem(force, numParticles);
        if (c == "viscosity")
            force->setKinematicViscosity(0.0);
        else if (c == "cells")
            force->setGridSize(8, 8, 10);
        else if (c == "particle")
            force->addParticle(numParticles);
        else if (c == "duplicate")
            force->addParticle(0);
        bool threw = false;
        try {
            if (c == "integrator") {
                LangevinMiddleIntegrator integrator(300.0, 1.0, 0.01);
                Context context(*system, integrator, platform);
            }
            else {
                VerletIntegrator integrator(0.01);
                Context context(*system, integrator, platform);
            }
        }
        catch (const OpenMMException& e) {
            threw = true;
        }
        if (!threw)
            throw OpenMMException("The invalid setup '"+c+"' was not rejected");
        delete system;
    }
}

/**
 * Creating a Context on a platform where the fluid update and the coupling are not implemented yet prints a
 * warning; on the Reference platform it does not.
 */
void testPlatformWarning(Platform& platform) {
    LBMForce* force;
    System* system = createSystem(force);
    VerletIntegrator integrator(0.01);
    stringstream captured;
    streambuf* original = cerr.rdbuf(captured.rdbuf());
    try {
        Context context(*system, integrator, platform);
    }
    catch (...) {
        cerr.rdbuf(original);
        throw;
    }
    cerr.rdbuf(original);
    bool warned = (captured.str().find("implemented only on the Reference platform") != string::npos);
    ASSERT(warned == (platform.getName() != "Reference"));
    delete system;
}

void runPlatformTests(Platform& platform) {
    testParameters();
    testPlatformWarning(platform);
    testZeroForce(platform);
    testInitialFluidFields(platform);
    testFluidStateRoundTrip(platform);
    testInvalidSetup(platform);
}
