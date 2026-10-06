/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

/**
 * Tests of the coupling between particles and fluid: drag of the first step, conservation of the total
 * momentum, particles moving with the fluid, partial coupling, force evaluations outside the integration
 * steps and random seeds, reflection at walls, restarts and the warning on the friction.  Include after
 * TestLBMFluid.h and call runCouplingTests().
 */

#include "openmm/reference/SimTKOpenMMRealType.h"

/** Mass (Da) of the particles of the coupling tests. */
const double couplingMass = 100.0;

/**
 * Create a System with numParticles particles of mass couplingMass and a fluid of 8x8x8 nodes (tau = 0.8),
 * without removal of the fluid momentum.  The particles listed in coupled are coupled to the fluid; if the
 * list is empty, all of them are.
 */
System* createCoupledSystem(LBMForce*& force, int numParticles, double friction, double temperature, vector<int> coupled=vector<int>()) {
    int n = 8;
    System* system = new System();
    system->setDefaultPeriodicBoxVectors(Vec3(n*fluidDx, 0, 0), Vec3(0, n*fluidDx, 0), Vec3(0, 0, n*fluidDx));
    force = new LBMForce();
    force->setGridSize(n, n, n);
    force->setFluidDensity(fluidDensity);
    force->setKinematicViscosity((0.8-0.5)/3.0*fluidDx*fluidDx/fluidDt);
    force->setFluidMomentumRemovalFrequency(0);
    force->setFriction(friction);
    force->setTemperature(temperature);
    for (int i = 0; i < numParticles; i++)
        system->addParticle(couplingMass);
    if (coupled.empty())
        for (int i = 0; i < numParticles; i++)
            coupled.push_back(i);
    for (int particle : coupled)
        force->addParticle(particle);
    system->addForce(force);
    return system;
}

/** Total momentum (Da nm/ps) of the particles, at the half step of the leapfrog, and of the fluid. */
Vec3 totalMomentum(Context& context, LBMForce* force) {
    const System& system = context.getSystem();
    State state = context.getState(State::Velocities);
    Vec3 p;
    for (int i = 0; i < system.getNumParticles(); i++)
        p += state.getVelocities()[i]*system.getParticleMass(i);
    vector<double> fluid;
    force->getFluidState(context, fluid);
    double mass;
    Vec3 j;
    totalMoments(fluid, mass, j);
    double cellMass = force->getFluidDensity()*fluidDx*fluidDx*fluidDx;
    return p + j*(cellMass*fluidDx/fluidDt);
}

/**
 * In one step the explicit drag multiplies the velocity of a particle relative to a fluid at rest by
 * 1 - gamma dt.  Before the first step the coupling force is zero; afterwards getState() returns the force of
 * the last step.
 */
void testFirstStepDrag(Platform& platform) {
    LBMForce* force;
    double friction = 5.0;
    System* system = createCoupledSystem(force, 1, friction, 0.0);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(1.1, 1.3, 1.7)));
    Vec3 v0(0.3, -0.2, 0.1);
    context.setVelocities(vector<Vec3>(1, v0));
    ASSERT_EQUAL_VEC(Vec3(), context.getState(State::Forces).getForces()[0], 0.0);
    integrator.step(1);
    State state = context.getState(State::Velocities | State::Forces | State::Energy);
    Vec3 v1 = state.getVelocities()[0];
    ASSERT_EQUAL_VEC(v0*(1.0-friction*fluidDt), v1, 1e-14);
    ASSERT_EQUAL_VEC((v1-v0)*(couplingMass/fluidDt), state.getForces()[0], 1e-12);
    ASSERT_EQUAL(0.0, state.getPotentialEnergy());
    delete system;
}

/**
 * The total momentum of particles and fluid is conserved, with drag and random forces: two particles share a
 * node and one crosses the periodic boundary.  The lattice density of the fluid is rho0.
 */
void testMomentumConservation(Platform& platform, double rho0) {
    LBMForce* force;
    System* system = createCoupledSystem(force, 4, 10.0, 300.0);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    vector<Vec3> positions = {Vec3(1.02, 2.01, 0.98), Vec3(0.97, 1.99, 1.03), Vec3(3.98, 0.02, 3.96), Vec3(2.3, 1.1, 0.6)};
    vector<Vec3> velocities = {Vec3(0.5, -0.2, 0.3), Vec3(-0.4, 0.1, 0.2), Vec3(2.0, -1.5, 1.0), Vec3(0.0, 0.3, -0.6)};
    context.setPositions(positions);
    context.setVelocities(velocities);
    force->setFluidState(context, uniformState(512, rho0, Vec3()));
    double scale = 0;
    for (Vec3 v : velocities)
        scale += couplingMass*sqrt(v.dot(v));
    Vec3 p0 = totalMomentum(context, force);
    integrator.step(50);
    Vec3 p1 = totalMomentum(context, force);
    // The momentum of the fluid is a sum over 19*512 populations, each much larger than the momentum exchanged
    // with the particles: its rounding is about 1e-12 of the momentum of the particles.  An error in the
    // coupling or in the forcing would appear at 1e-2.
    ASSERT_EQUAL_TOL(0.0, sqrt((p1-p0).dot(p1-p0))/scale, 1e-11);
    State state = context.getState(State::Positions | State::Velocities);
    ASSERT(state.getPositions()[2][0] > 4.0);      // crossed the boundary
    Vec3 dv = state.getVelocities()[3]-velocities[3];
    ASSERT(dv.dot(dv) > 0);
    delete system;
}

/**
 * A particle moving with the fluid feels no force: particle and fluid keep the same uniform velocity while the
 * particle crosses two cells, along x, y, z and a diagonal.
 */
void testComoving(Platform& platform) {
    vector<Vec3> flows = {Vec3(1.0, 0, 0), Vec3(0, 1.0, 0), Vec3(0, 0, 1.0), Vec3(0.6, -0.5, 0.6)};
    for (Vec3 u : flows) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 1, 10.0, 0.0);
        force->setInitialFluidVelocity(u);
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        context.setPositions(vector<Vec3>(1, Vec3(1.1, 1.3, 1.7)));
        context.setVelocities(vector<Vec3>(1, u));
        integrator.step(100);
        ASSERT_EQUAL_VEC(u, context.getState(State::Velocities).getVelocities()[0], 1e-13);
        delete system;
    }
}

/**
 * Only the particles added to the force are coupled: the others keep their velocity and feel no force.
 */
void testPartialCoupling(Platform& platform) {
    LBMForce* force;
    System* system = createCoupledSystem(force, 4, 5.0, 0.0, {1, 3});
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions({Vec3(0.3, 0.4, 0.5), Vec3(1.3, 1.4, 1.5), Vec3(2.3, 2.4, 2.5), Vec3(3.3, 3.4, 3.5)});
    Vec3 v0(0.2, 0.1, -0.3);
    context.setVelocities(vector<Vec3>(4, v0));
    integrator.step(10);
    State state = context.getState(State::Velocities | State::Forces);
    for (int i = 0; i < 4; i++) {
        Vec3 v = state.getVelocities()[i];
        if (i%2 == 0) {
            // OpenMM's Reference Verlet recomputes the velocity from the positions: rounding only.
            ASSERT_EQUAL_VEC(v0, v, 1e-12);
            ASSERT_EQUAL_VEC(Vec3(), state.getForces()[i], 0.0);
        }
        else
            ASSERT(v.dot(v) < 0.9*v0.dot(v0));
    }
    delete system;
}

/**
 * Force evaluations outside the integration steps do not change the trajectory: they neither advance the fluid
 * nor draw new random numbers.  Two simulations with the same seed are identical; with seed 0 they differ.
 */
void testForceEvaluationsAndSeeds(Platform& platform) {
    auto run = [&](int seed, bool queries) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 3, 5.0, 300.0);
        force->setRandomNumberSeed(seed);
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        context.setPositions({Vec3(0.3, 0.4, 0.5), Vec3(1.3, 1.4, 1.5), Vec3(2.3, 2.4, 2.5)});
        for (int n = 0; n < 20; n++) {
            if (queries)
                context.getState(State::Forces | State::Energy);
            integrator.step(1);
            if (queries)
                context.getState(State::Forces);
        }
        vector<Vec3> velocities = context.getState(State::Velocities).getVelocities();
        delete system;
        return velocities;
    };
    vector<Vec3> plain = run(1234, false), queried = run(1234, true);
    for (int i = 0; i < 3; i++)
        ASSERT_EQUAL_VEC(plain[i], queried[i], 0.0);
    vector<Vec3> first = run(0, false), second = run(0, false);
    ASSERT(first[0][0] != second[0][0]);
}

/**
 * A coupled particle that reaches a solid node moving into the wall has every component of its velocity
 * reversed and feels the drag of a fluid at rest; an uncoupled particle does not see the wall.
 */
void testWallReflection(Platform& platform) {
    LBMForce* force;
    double friction = 1.0;
    System* system = createCoupledSystem(force, 2, friction, 0.0, {0});
    force->setSolidNodes(wallPlane(8, 8, 8));
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(2, Vec3(1.1, 0.30, 1.7)));
    Vec3 v0(0.5, -10.0, 0.2);
    context.setVelocities(vector<Vec3>(2, v0));
    double decay = 1.0-friction*fluidDt;
    integrator.step(1);         // y = 0.30 - 0.099: the nearest node is now on the plane j = 0
    State state = context.getState(State::Positions | State::Velocities);
    ASSERT_EQUAL_VEC(v0*decay, state.getVelocities()[0], 1e-14);
    ASSERT(state.getPositions()[0][1] < 0.25);
    integrator.step(1);         // reversed, then slowed down by the drag of the wall at rest
    state = context.getState(State::Positions | State::Velocities);
    ASSERT_EQUAL_VEC(-v0*decay*decay, state.getVelocities()[0], 1e-14);
    ASSERT(state.getPositions()[0][1] > 0.25);
    ASSERT_EQUAL_VEC(v0, state.getVelocities()[1], 1e-12);
    delete system;
}

/**
 * A coupled particle at a solid node is reversed only if it moves into the wall: v.n > 0, with n the normal
 * pointing into the wall.  The wall is the single plane j = 0, so it has fluid on both sides: on the side of
 * j = 1 its normal is -y, on the side of j = 7 (y < 0 before wrapping) it is +y.
 */
void testWallReflectionDirection(Platform& platform) {
    LBMForce* force;
    double friction = 1.0;
    System* system = createCoupledSystem(force, 3, friction, 0.0);
    force->setSolidNodes(wallPlane(8, 8, 8));
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    // All three particles have the plane j = 0 as nearest node.
    context.setPositions({Vec3(1.1, 0.20, 1.7), Vec3(2.1, -0.10, 1.7), Vec3(3.1, -0.10, 1.7)});
    vector<Vec3> v0 = {Vec3(0.3, 5.0, 0.1), Vec3(0.3, 5.0, 0.1), Vec3(0.3, -5.0, 0.1)};
    context.setVelocities(v0);
    integrator.step(1);
    vector<Vec3> v1 = context.getState(State::Velocities).getVelocities();
    double decay = 1.0-friction*fluidDt;
    ASSERT_EQUAL_VEC(v0[0]*decay, v1[0], 1e-14);      // side j = 1, moving out: kept
    ASSERT_EQUAL_VEC(-v0[1]*decay, v1[1], 1e-14);     // side j = 7, moving in: reversed
    ASSERT_EQUAL_VEC(v0[2]*decay, v1[2], 1e-14);      // side j = 7, moving out: kept
    delete system;
}

/**
 * With walls the total momentum of particles, fluid and walls is conserved, the momentum given to the walls
 * being the sum of getWallForce() dt over the steps.  The fluid flows against the walls j = 0 and j = 4, one
 * particle is reflected by a wall, and all particles feel drag and random force.
 */
void testWallMomentumBalance(Platform& platform) {
    LBMForce* force;
    System* system = createCoupledSystem(force, 4, 10.0, 300.0);
    vector<int> walls = wallPlane(8, 8, 8);
    for (int node : wallPlane(8, 8, 8))
        walls.push_back(node + 8*4);                   // the plane j = 4
    force->setSolidNodes(walls);
    force->setInitialFluidVelocity(Vec3(0.3, 0.4, -0.2));
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions({Vec3(1.1, 0.30, 1.7), Vec3(2.3, 1.2, 0.6), Vec3(0.7, 3.1, 2.9), Vec3(3.3, 1.6, 3.6)});
    context.setVelocities({Vec3(0.5, -10.0, 0.2), Vec3(0.4, 0.3, -0.2), Vec3(-0.6, 0.2, 0.5), Vec3(0.1, -0.4, 0.3)});
    auto momentum = [&](double& scale) {
        State state = context.getState(State::Velocities);
        Vec3 p;
        scale = 0;
        for (int i = 0; i < 4; i++) {
            Vec3 v = state.getVelocities()[i];
            p += v*couplingMass;
            scale += couplingMass*sqrt(v.dot(v));
        }
        vector<double> density;
        vector<Vec3> velocity;
        force->getFluidFields(context, density, velocity);
        for (int node = 0; node < (int) density.size(); node++) {
            Vec3 pNode = velocity[node]*(density[node]*fluidDx*fluidDx*fluidDx);
            p += pNode;
            scale += sqrt(pNode.dot(pNode));
        }
        return p;
    };
    double scale;
    Vec3 p0 = momentum(scale);
    ASSERT_EQUAL_VEC(Vec3(), force->getWallForce(context), 0.0);
    Vec3 wall;
    for (int n = 0; n < 100; n++) {
        integrator.step(1);
        wall += force->getWallForce(context)*fluidDt;
        if (n == 1)
            ASSERT(context.getState(State::Velocities).getVelocities()[0][1] > 0);     // reflected by the wall j = 0
    }
    Vec3 p1 = momentum(scale);
    ASSERT(wall.dot(wall) > 1.0);
    Vec3 balance = p1 + wall - p0;
    ASSERT_EQUAL_TOL(0.0, sqrt(balance.dot(balance))/scale, 1e-12);
    delete system;
}

/**
 * A run with coupled particles restarted from a checkpoint, with the fluid restored by setFluidState(), is
 * identical to an uninterrupted run.  The temperature is zero, since the state of the random generator is not
 * part of the checkpoint.
 */
void testRestartWithParticles(Platform& platform) {
    int numSteps = 35, split = 13;
    vector<double> uninterruptedFluid, savedFluid, restartedFluid;
    vector<Vec3> uninterrupted, restarted;
    stringstream checkpoint;
    for (int run = 0; run < 3; run++) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 3, 5.0, 0.0);
        force->setFluidMomentumRemovalFrequency(5);
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        context.setPositions({Vec3(0.3, 0.4, 0.5), Vec3(1.3, 1.4, 1.5), Vec3(2.3, 2.4, 2.5)});
        context.setVelocities({Vec3(0.5, -0.2, 0.3), Vec3(-0.4, 0.1, 0.2), Vec3(1.0, -0.5, 0.0)});
        if (run == 0) {
            integrator.step(numSteps);
            uninterrupted = context.getState(State::Velocities).getVelocities();
            force->getFluidState(context, uninterruptedFluid);
        }
        else if (run == 1) {
            integrator.step(split);
            context.createCheckpoint(checkpoint);
            force->getFluidState(context, savedFluid);
        }
        else {
            context.loadCheckpoint(checkpoint);
            force->setFluidState(context, savedFluid);
            integrator.step(numSteps-split);
            restarted = context.getState(State::Velocities).getVelocities();
            force->getFluidState(context, restartedFluid);
        }
        delete system;
    }
    for (int i = 0; i < 3; i++)
        ASSERT_EQUAL_VEC(uninterrupted[i], restarted[i], 0.0);
    for (int i = 0; i < (int) uninterruptedFluid.size(); i++)
        ASSERT_EQUAL(uninterruptedFluid[i], restartedFluid[i]);
}

/**
 * The random force has the amplitude required by the fluctuation-dissipation theorem: free particles reach the
 * temperature T measured from full-step velocities (mean of two consecutive half-step velocities), and
 * T/(1 - gamma dt/2) measured from half-step velocities.  The fluid has no thermal fluctuations of its own and
 * takes part of the momentum of the particles, so both are a few per cent lower (here 100 particles of 100 Da,
 * a fifth of the mass of the fluid): the tolerance of 10% catches errors in the amplitude of the noise, which
 * would appear as factors such as 2 or 1/2.  The seed is fixed, so the test is deterministic.
 */
void testEquipartition(Platform& platform) {
    int numParticles = 100;
    double temperature = 300.0, friction = 10.0;
    LBMForce* force;
    System* system = createCoupledSystem(force, numParticles, friction, temperature);
    force->setRandomNumberSeed(1);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++)
        positions[i] = Vec3(fmod(0.37*i, 4.0), fmod(0.53*i+0.1, 4.0), fmod(0.71*i+0.2, 4.0));
    context.setPositions(positions);
    integrator.step(500);
    vector<Vec3> previous = context.getState(State::Velocities).getVelocities();
    double fullStep = 0, halfStep = 0;
    int numSamples = 2000;
    for (int n = 0; n < numSamples; n++) {
        integrator.step(1);
        vector<Vec3> v = context.getState(State::Velocities).getVelocities();
        for (int i = 0; i < numParticles; i++) {
            Vec3 mid = (v[i]+previous[i])*0.5;
            fullStep += couplingMass*mid.dot(mid);
            halfStep += couplingMass*v[i].dot(v[i]);
        }
        previous = v;
    }
    double scale = 1.0/(3*numParticles*numSamples*BOLTZ);
    ASSERT_EQUAL_TOL(temperature, fullStep*scale, 0.1);
    ASSERT_EQUAL_TOL(temperature/(1-0.5*friction*fluidDt), halfStep*scale, 0.1);
    delete system;
}

/**
 * A warning is printed when tau > 1.7 and particles are coupled: the self-mobility of a particle becomes small,
 * and negative above tau = 1.79.
 */
void testSelfMobilityWarning(Platform& platform) {
    double taus[] = {1.6, 1.75};
    for (double tau : taus)
        for (bool coupled : {false, true}) {
            LBMForce* force;
            System* system;
            if (coupled) {
                system = createCoupledSystem(force, 1, 1.0, 0.0);
                force->setKinematicViscosity((tau-0.5)/3.0*fluidDx*fluidDx/fluidDt);
            }
            else
                system = createFluidSystem(force, 8, 8, 8, tau);     // one particle, not coupled
            VerletIntegrator integrator(fluidDt);
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
            bool warned = (captured.str().find("> 1.7") != string::npos);
            ASSERT(warned == (tau > 1.7 && coupled));
            delete system;
        }
}

/**
 * A warning is printed when friction*dt > 1 and particles are coupled.
 */
void testFrictionWarning(Platform& platform) {
    double frictions[] = {50.0, 150.0};
    for (double friction : frictions) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 1, friction, 0.0);
        VerletIntegrator integrator(fluidDt);
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
        bool warned = (captured.str().find("friction*dt") != string::npos);
        ASSERT(warned == (friction*fluidDt > 1.0));
        delete system;
    }
}

void runCouplingTests(Platform& platform) {
    testFirstStepDrag(platform);
    testMomentumConservation(platform, 1.0);
    testMomentumConservation(platform, 0.98);
    testMomentumConservation(platform, 1.02);
    testComoving(platform);
    testPartialCoupling(platform);
    testForceEvaluationsAndSeeds(platform);
    testWallReflection(platform);
    testWallReflectionDirection(platform);
    testWallMomentumBalance(platform);
    testRestartWithParticles(platform);
    testEquipartition(platform);
    testFrictionWarning(platform);
    testSelfMobilityWarning(platform);
}
