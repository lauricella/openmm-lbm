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
 * A coupled particle that reaches a solid node has every component of its velocity reversed and feels the drag
 * of a fluid at rest; an uncoupled particle does not see the wall.
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
    testRestartWithParticles(platform);
    testFrictionWarning(platform);
}
