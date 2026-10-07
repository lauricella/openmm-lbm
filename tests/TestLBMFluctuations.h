/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

/**
 * Tests of the fluctuating fluid (setFluidFluctuations(), docs/theory.md, section 7): the random part of the
 * collision conserves mass and momentum and gives every moment its equilibrium variance, on a single node and in a
 * fluid at rest; without fluctuations or at zero temperature nothing changes; runs are reproducible and restart
 * exactly; the NVE scheme with a fluctuating fluid; parameters and checkpoints.  The platforms draw different random
 * numbers (the generator of the force on the Reference platform, OpenMM's on the others), so they are compared
 * through the statistics.  Include after TestLBMFluid.h and TestLBMCoupling.h.
 */

/**
 * The polynomials e_k of the orthogonal D3Q19 basis of Lulli et al. (Phys. Rev. E 109, 045304, 2024), k = 0...18,
 * written here independently of the plugin, and their norms b_k = sum_q w_q e_k(c_q)^2.
 */
double fluctuationMode(int k, int x, int y, int z) {
    double x2 = x*x, y2 = y*y, z2 = z*z;
    double values[19] = {1.0, (double) x, (double) y, (double) z, x2-1.0/3.0, y2-1.0/3.0, z2-1.0/3.0,
            (double) x*y, (double) x*z, (double) y*z, y*(x2-1.0/3.0), z*(x2-1.0/3.0), x*(y2-1.0/3.0),
            0.5*x*(y2+2*z2-1), 0.5*y*(x2+2*z2-1), 0.5*z*(x2+2*y2-1),
            x2*y2 - x2/3 - y2/3 + z2/6 + 1.0/18.0,
            2.0/7.0*x2*y2 + x2*z2 - 3.0/7.0*x2 + y2/14 - 2.0/7.0*z2 + 1.0/14.0,
            0.4*x2*y2 + 0.4*x2*z2 + y2*z2 - 0.1*x2 - 0.4*y2 - 0.4*z2 + 0.1};
    return values[k];
}

const double fluctuationModeNorm[19] = {1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 2.0/9.0, 2.0/9.0, 2.0/9.0, 1.0/9.0, 1.0/9.0,
        1.0/9.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 7.0/162.0, 5.0/126.0, 1.0/30.0};

const int fluctuationCx[19] = {0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 0,  0,  0,  0, 1, -1, -1,  1};
const int fluctuationCy[19] = {0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 1, -1,  1, -1, 0,  0,  0,  0};
const int fluctuationCz[19] = {0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1,  1, -1};

/** The 19 moments m_k = sum_q e_k(c_q) (f_q - w_q) of a node of a fluid state (deviations f - w). */
void nodeModes(const vector<double>& state, int node, double modes[19]) {
    int numNodes = state.size()/19;
    for (int k = 0; k < 19; k++) {
        modes[k] = 0;
        for (int q = 0; q < 19; q++)
            modes[k] += fluctuationMode(k, fluctuationCx[q], fluctuationCy[q], fluctuationCz[q])*state[q*numNodes+node];
    }
}

/** Thermal energy kT in the lattice units of the fluid tests (mass rho0 dx^3), at the temperature T (K). */
double latticeKT(double temperature) {
    return BOLTZ*temperature*fluidDt*fluidDt/(fluidDensity*fluidDx*fluidDx*fluidDx*fluidDx*fluidDx);
}

/**
 * The basis of the test is orthogonal, with the norms b_k.  It checks the table above, which the other tests use
 * to measure the moments.
 */
void testFluctuationBasis() {
    double w[19] = {1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
                    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
    for (int k = 0; k < 19; k++)
        for (int l = 0; l < 19; l++) {
            double product = 0;
            for (int q = 0; q < 19; q++)
                product += w[q]*fluctuationMode(k, fluctuationCx[q], fluctuationCy[q], fluctuationCz[q])*
                        fluctuationMode(l, fluctuationCx[q], fluctuationCy[q], fluctuationCz[q]);
            ASSERT_EQUAL_TOL(k == l ? fluctuationModeNorm[k] : 0.0, product, 1e-15);
        }
}

/**
 * A lattice of a single node streams every population back to the node itself, so the state after a step is the
 * post-collision state of the node: f = feq(rho = 1, u = 0) + (1 - omega) fneq + xi.  Mass and momentum are those of
 * the start, exactly, and the moments k = 4...18 have the variance mu b_k, mu = kT/cs^2: with tau = 1 every step
 * draws new, independent values; with tau != 1 the stress modes are an autoregressive sequence with the same
 * stationary variance.  The moments are uncorrelated with each other.
 */
void testFluctuationsSingleNode(Platform& platform, double tau) {
    double temperature = 300.0;
    int numWarmup = 50, numSamples = 20000;
    LBMForce* force;
    System* system = createFluidSystem(force, 1, 1, 1, tau);
    force->setFluidFluctuations(true);
    force->setTemperature(temperature);
    force->setRandomNumberSeed(5);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    double mu = 3.0*latticeKT(temperature);
    vector<double> sum2(19*19, 0.0), state;
    for (int step = 0; step < numWarmup+numSamples; step++) {
        integrator.step(1);
        force->getFluidState(context, state);
        double modes[19];
        nodeModes(state, 0, modes);
        for (int k = 0; k < 4; k++)
            ASSERT_EQUAL_TOL(0.0, modes[k], getFluidTolerance(platform, 1e-15));
        if (step < numWarmup)
            continue;
        for (int k = 4; k < 19; k++)
            for (int l = 4; l < 19; l++)
                sum2[19*k+l] += modes[k]*modes[l];
    }
    for (int k = 4; k < 19; k++)
        for (int l = 4; l < 19; l++) {
            double ratio = sum2[19*k+l]/(numSamples*mu*sqrt(fluctuationModeNorm[k]*fluctuationModeNorm[l]));
            ASSERT_EQUAL_TOL(k == l ? 1.0 : 0.0, ratio, 0.05);
        }
    delete system;
}

/**
 * A fluctuating fluid at rest, without forces, on an 8x8x8 lattice: after a transient, the density, the momentum
 * and every other moment of a node have their equilibrium variances, mu rho, rho kT and mu rho b_k (rho = 1), for a
 * small and a large relaxation time.  The total mass and momentum are conserved.
 */
void testFluctuationsEquilibrium(Platform& platform, double tau) {
    double temperature = 300.0;
    int n = 8, numNodes = n*n*n, numWarmup = 300, numSamples = 300, interval = 5;
    LBMForce* force;
    System* system = createFluidSystem(force, n, n, n, tau);
    force->setFluidFluctuations(true);
    force->setTemperature(temperature);
    force->setRandomNumberSeed(11);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    double kT = latticeKT(temperature), mu = 3.0*kT;
    vector<double> sum2(19, 0.0), state;
    integrator.step(numWarmup);
    for (int sample = 0; sample < numSamples; sample++) {
        integrator.step(interval);
        force->getFluidState(context, state);
        for (int node = 0; node < numNodes; node++) {
            double modes[19];
            nodeModes(state, node, modes);
            for (int k = 0; k < 19; k++)
                sum2[k] += modes[k]*modes[k];
        }
    }
    double mass;
    Vec3 momentum;
    totalMoments(state, mass, momentum);
    ASSERT_EQUAL_TOL((double) numNodes, mass, getFluidTolerance(platform, 1e-13, 1e-5));
    ASSERT_EQUAL_VEC(Vec3(), momentum, getFluidTolerance(platform, 1e-13, 1e-5));
    double count = (double) numSamples*numNodes;
    ASSERT_EQUAL_TOL(1.0, sum2[0]/(count*mu), 0.05);
    for (int k = 1; k < 4; k++)
        ASSERT_EQUAL_TOL(1.0, sum2[k]/(count*kT), 0.05);
    for (int k = 4; k < 19; k++)
        ASSERT_EQUAL_TOL(1.0, sum2[k]/(count*mu*fluctuationModeNorm[k]), 0.05);
    delete system;
}

/**
 * With the fluctuations switched on at zero temperature the run is identical, bit for bit, to the run without
 * them: walls, body force and coupled particles with the NVE scheme.
 */
void testFluctuationsAtZeroTemperature(Platform& platform) {
    vector<double> fluid[2];
    vector<Vec3> velocities[2];
    for (int run = 0; run < 2; run++) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 3, 5.0, 0.0);
        force->setCouplingScheme(LBMForce::NVE);
        force->setBodyAcceleration(Vec3(0.3, 0.0, 0.1));
        vector<int> wall;
        for (int k = 0; k < 8; k++)
            for (int i = 0; i < 8; i++)
                wall.push_back(i+8*8*k);
        force->setSolidNodes(wall);
        force->setFluidFluctuations(run == 1);
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        context.setPositions({Vec3(0.3, 0.8, 0.5), Vec3(1.3, 1.4, 1.5), Vec3(2.3, 2.4, 2.5)});
        context.setVelocities({Vec3(0.1, -0.2, 0.3), Vec3(-0.3, 0.2, 0.1), Vec3(0.2, 0.2, -0.2)});
        integrator.step(30);
        force->getFluidState(context, fluid[run]);
        velocities[run] = context.getState(State::Velocities).getVelocities();
        delete system;
    }
    for (int i = 0; i < (int) fluid[0].size(); i++)
        ASSERT_EQUAL(fluid[0][i], fluid[1][i]);
    for (int i = 0; i < (int) velocities[0].size(); i++)
        ASSERT_EQUAL_VEC(velocities[0][i], velocities[1][i], 0.0);
}

/**
 * A fluctuating fluid with coupled particles at a temperature: two runs with the same seed are identical, also when
 * one of them evaluates the forces between steps, and a different seed gives a different run.
 */
void testFluctuationsReproducible(Platform& platform) {
    vector<double> fluid[3];
    vector<Vec3> velocities[3];
    for (int run = 0; run < 3; run++) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 3, 5.0, 300.0);
        force->setFluidFluctuations(true);
        force->setRandomNumberSeed(run == 2 ? 8 : 7);
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        context.setPositions({Vec3(0.3, 0.8, 0.5), Vec3(1.3, 1.4, 1.5), Vec3(2.3, 2.4, 2.5)});
        for (int step = 0; step < 20; step++) {
            integrator.step(1);
            if (run == 1)
                context.getState(State::Forces | State::Energy);
        }
        force->getFluidState(context, fluid[run]);
        velocities[run] = context.getState(State::Velocities).getVelocities();
        delete system;
    }
    double difference = 0;
    for (int i = 0; i < (int) fluid[0].size(); i++) {
        ASSERT_EQUAL(fluid[0][i], fluid[1][i]);
        difference = max(difference, fabs(fluid[0][i]-fluid[2][i]));
    }
    for (int i = 0; i < (int) velocities[0].size(); i++)
        ASSERT_EQUAL_VEC(velocities[0][i], velocities[1][i], 0.0);
    ASSERT(difference > 1e-6);
}

/**
 * A run with a fluctuating fluid and coupled particles restarted from the checkpoints of OpenMM and of the force is
 * identical, bit for bit, to the uninterrupted run.  The seed is 0, so the two Contexts choose different seeds:
 * the checkpoint restores the generator.
 */
void testFluctuationsRestart(Platform& platform) {
    int numSteps = 23, split = 9;
    vector<double> fluid[2];
    vector<Vec3> velocities[2];
    stringstream openmmCheckpoint, forceCheckpoint;
    for (int run = 0; run < 2; run++) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 3, 5.0, 300.0);
        force->setFluidFluctuations(true);
        force->setRandomNumberSeed(0);
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        if (run == 0) {
            context.setPositions({Vec3(0.3, 0.8, 0.5), Vec3(1.3, 1.4, 1.5), Vec3(2.3, 2.4, 2.5)});
            integrator.step(split);
            context.getState(State::Forces);
            context.createCheckpoint(openmmCheckpoint);
            force->createCheckpoint(context, forceCheckpoint);
        }
        else {
            context.loadCheckpoint(openmmCheckpoint);
            force->loadCheckpoint(context, forceCheckpoint);
        }
        integrator.step(numSteps-split);
        force->getFluidState(context, fluid[run]);
        velocities[run] = context.getState(State::Velocities).getVelocities();
        delete system;
    }
    for (int i = 0; i < (int) fluid[0].size(); i++)
        ASSERT_EQUAL(fluid[0][i], fluid[1][i]);
    for (int i = 0; i < (int) velocities[0].size(); i++)
        ASSERT_EQUAL_VEC(velocities[0][i], velocities[1][i], 0.0);
}

/**
 * With the NVE scheme the particles have no random force, but a fluctuating fluid sets particles at rest in motion;
 * without fluid fluctuations they stay at rest.
 */
void testFluctuationsWithNVE(Platform& platform) {
    for (int fluctuations = 0; fluctuations < 2; fluctuations++) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 2, 5.0, 300.0);
        force->setCouplingScheme(LBMForce::NVE);
        force->setFluidFluctuations(fluctuations == 1);
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        context.setPositions({Vec3(0.3, 0.8, 0.5), Vec3(1.3, 1.4, 1.5)});
        integrator.step(20);
        vector<Vec3> v = context.getState(State::Velocities).getVelocities();
        double speed2 = v[0].dot(v[0]) + v[1].dot(v[1]);
        if (fluctuations) {
            ASSERT(speed2 > 0);
        }
        else {
            ASSERT_EQUAL(0.0, speed2);
        }
        delete system;
    }
}

/**
 * The fluctuations cannot be switched on or off with updateParametersInContext(), which changes their
 * temperature, and a checkpoint is refused by a Context with the fluctuations switched differently.
 */
void testFluctuationsParameters(Platform& platform) {
    LBMForce* force;
    System* system = createCoupledSystem(force, 2, 5.0, 300.0);
    force->setFluidFluctuations(true);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions({Vec3(0.3, 0.8, 0.5), Vec3(1.3, 1.4, 1.5)});
    integrator.step(2);
    force->setTemperature(310.0);
    force->updateParametersInContext(context);
    integrator.step(1);
    force->setFluidFluctuations(false);
    bool thrown = false;
    try {
        force->updateParametersInContext(context);
    }
    catch (OpenMMException& e) {
        thrown = true;
    }
    ASSERT(thrown);
    stringstream checkpoint;
    force->setFluidFluctuations(true);
    force->createCheckpoint(context, checkpoint);
    LBMForce* force2;
    System* system2 = createCoupledSystem(force2, 2, 5.0, 300.0);
    VerletIntegrator integrator2(fluidDt);
    Context context2(*system2, integrator2, platform);
    thrown = false;
    try {
        force2->loadCheckpoint(context2, checkpoint);
    }
    catch (OpenMMException& e) {
        thrown = true;
    }
    ASSERT(thrown);
    delete system;
    delete system2;
}

/**
 * With the fluid fluctuations a warning is printed when the drag is explicit, particles are coupled with the EM scheme
 * at T > 0 and the friction is not zero (docs/theory.md, section 7): the explicit drag makes them too hot.
 */
void testFluctuationDragWarning(Platform& platform) {
    for (bool fluctuations : {false, true})
        for (LBMForce::DragScheme drag : {LBMForce::Explicit, LBMForce::Centered})
            for (LBMForce::CouplingScheme scheme : {LBMForce::EulerMaruyama, LBMForce::NVE}) {
                LBMForce* force;
                System* system = createCoupledSystem(force, 2, 10.0, 300.0);
                force->setFluidFluctuations(fluctuations);
                force->setDragScheme(drag);
                force->setCouplingScheme(scheme);
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
                bool warned = (captured.str().find("with fluid fluctuations the explicit drag") != string::npos);
                ASSERT(warned == (fluctuations && drag == LBMForce::Explicit && scheme == LBMForce::EulerMaruyama));
                if (warned)
                    ASSERT(captured.str().find("= 6.64") != string::npos);
                delete system;
            }
}

void runFluctuationTests(Platform& platform) {
    testFluctuationBasis();
    testFluctuationsSingleNode(platform, 1.0);
    testFluctuationsSingleNode(platform, 0.8);
    testFluctuationsEquilibrium(platform, 0.8);
    testFluctuationsEquilibrium(platform, 2.5);
    testFluctuationsAtZeroTemperature(platform);
    testFluctuationsReproducible(platform);
    testFluctuationsRestart(platform);
    testFluctuationsWithNVE(platform);
    testFluctuationsParameters(platform);
    testFluctuationDragWarning(platform);
}
