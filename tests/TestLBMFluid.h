/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

/**
 * Tests of the lattice Boltzmann fluid on its own (no coupled particles): steady uniform flow,
 * conservation, body force, removal of the fluid momentum, viscosity from the decay of a shear wave,
 * the rule that the fluid advances only in integration steps, and restarts from checkpoints
 * (runFluidTests()); solid nodes and walls (runWallTests()).  Include after TestLBMForce.h.
 */

#include "openmm/internal/AssertionUtilities.h"
#include <cmath>
#include <sstream>
#include <vector>

/** Lattice spacing (nm), time step (ps) and density (Da/nm^3) of the fluid tests. */
const double fluidDx = 0.5, fluidDt = 0.01, fluidDensity = 602.2;

/**
 * Tolerance of a fluid test: the given tolerance in double and mixed precision, where the fluid is stored in
 * double precision, and the resolution of single precision (getStorageTolerance()) otherwise.
 */
double getFluidTolerance(Platform& platform, double tolerance) {
    double storage = getStorageTolerance(platform);
    return (storage > 1e-12 ? storage : tolerance);
}

/**
 * Create a System with a fluid of nx*ny*nz nodes and relaxation time tau, without removal of the fluid
 * momentum.  The System has one particle, which is not coupled to the fluid.
 */
System* createFluidSystem(LBMForce*& force, int nx, int ny, int nz, double tau) {
    System* system = new System();
    system->setDefaultPeriodicBoxVectors(Vec3(nx*fluidDx, 0, 0), Vec3(0, ny*fluidDx, 0), Vec3(0, 0, nz*fluidDx));
    system->addParticle(1.0);
    force = new LBMForce();
    force->setGridSize(nx, ny, nz);
    force->setFluidDensity(fluidDensity);
    force->setKinematicViscosity((tau-0.5)/3.0*fluidDx*fluidDx/fluidDt);
    force->setFluidMomentumRemovalFrequency(0);     // the default (every step) is switched on only where tested
    system->addForce(force);
    return system;
}

/**
 * Equilibrium state with the given lattice density and velocity at every node, in the form of the fluid state:
 * the deviations f_q - w_q of the populations from the rest equilibrium.
 */
vector<double> uniformState(int numNodes, double rho, Vec3 u) {
    vector<double> state(19*numNodes);
    double w[19] = {1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
                    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
    int cx[19] = {0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 0,  0,  0,  0, 1, -1, -1,  1};
    int cy[19] = {0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 1, -1,  1, -1, 0,  0,  0,  0};
    int cz[19] = {0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1,  1, -1};
    for (int q = 0; q < 19; q++) {
        double cu = cx[q]*u[0] + cy[q]*u[1] + cz[q]*u[2];
        double dfeq = w[q]*((rho-1.0) + rho*(3.0*cu + 4.5*cu*cu - 1.5*u.dot(u)));
        for (int node = 0; node < numNodes; node++)
            state[q*numNodes+node] = dfeq;
    }
    return state;
}

/** Total mass and momentum of a fluid state (deviations f - w), in lattice units: mass = numNodes + sum (f - w). */
void totalMoments(const vector<double>& state, double& mass, Vec3& momentum) {
    int cx[19] = {0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 0,  0,  0,  0, 1, -1, -1,  1};
    int cy[19] = {0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 1, -1,  1, -1, 0,  0,  0,  0};
    int cz[19] = {0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1,  1, -1};
    int numNodes = state.size()/19;
    mass = numNodes;
    momentum = Vec3();
    for (int q = 0; q < 19; q++)
        for (int node = 0; node < numNodes; node++) {
            double f = state[q*numNodes+node];
            mass += f;
            momentum += Vec3(cx[q], cy[q], cz[q])*f;
        }
}

/**
 * A uniform flow is an exact steady state of the lattice update.
 */
void testUniformFlowIsSteady(Platform& platform) {
    LBMForce* force;
    System* system = createFluidSystem(force, 6, 5, 4, 0.8);
    force->setInitialFluidVelocity(Vec3(0.03, -0.02, 0.01)*(fluidDx/fluidDt));
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    vector<double> before, after;
    force->getFluidState(context, before);
    integrator.step(50);
    force->getFluidState(context, after);
    double tol = getFluidTolerance(platform, 1e-13);
    for (int i = 0; i < (int) before.size(); i++)
        ASSERT_EQUAL_TOL(before[i], after[i], tol);
    delete system;
}

/**
 * Mass and momentum of a perturbed fluid are conserved without body force and momentum removal.
 */
void testFluidConservation(Platform& platform) {
    LBMForce* force;
    System* system = createFluidSystem(force, 6, 5, 4, 0.7);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    vector<double> state;
    force->getFluidState(context, state);
    for (int i = 0; i < (int) state.size(); i++)
        state[i] += 1e-3*sin(1.3*i + 0.7*(i%11));
    force->setFluidState(context, state);
    double mass0, mass1;
    Vec3 p0, p1;
    totalMoments(state, mass0, p0);
    integrator.step(100);
    force->getFluidState(context, state);
    totalMoments(state, mass1, p1);
    double tol = getFluidTolerance(platform, 1e-13);
    ASSERT_EQUAL_TOL(mass0, mass1, tol);
    ASSERT(p0.dot(p0) > 1e-10);
    ASSERT_EQUAL_VEC(p0, p1, tol*mass0);
    delete system;
}

/**
 * A body acceleration g adds the momentum rho*g per node and step (lattice units), also when the lattice
 * density differs from 1, and the fluid velocity after n steps is (n + 1/2) g dt.
 */
void testBodyForce(Platform& platform, double rho0) {
    LBMForce* force;
    System* system = createFluidSystem(force, 4, 3, 5, 0.9);
    Vec3 g(0.5, -0.3, 0.2);
    force->setBodyAcceleration(g);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    int numNodes = 4*3*5;
    force->setFluidState(context, uniformState(numNodes, rho0, Vec3()));
    int numSteps = 20;
    integrator.step(numSteps);
    vector<double> state, density;
    vector<Vec3> velocity;
    force->getFluidState(context, state);
    double mass;
    Vec3 p;
    totalMoments(state, mass, p);
    Vec3 gLattice = g*(fluidDt*fluidDt/fluidDx);
    double tol = getFluidTolerance(platform, 1e-13);
    ASSERT_EQUAL_TOL(rho0*numNodes, mass, tol);
    ASSERT_EQUAL_VEC(gLattice*(numSteps*rho0*numNodes), p, tol);
    force->getFluidFields(context, density, velocity);
    for (int node = 0; node < numNodes; node++) {
        ASSERT_EQUAL_TOL(rho0*fluidDensity, density[node], tol);
        ASSERT_EQUAL_VEC(g*((numSteps+0.5)*fluidDt), velocity[node], getFluidTolerance(platform, 1e-12));
    }
    delete system;
}

/**
 * The momentum of the fluid is removed in the steps whose index (the step count of the Context, from 0) is
 * a multiple of the removal frequency, before the collision: with a body force F per node and frequency 3 the total
 * momentum after n steps is ((n-1)%3 + 1) F per node.
 */
void testFluidMomentumRemoval(Platform& platform) {
    LBMForce* force;
    System* system = createFluidSystem(force, 4, 4, 4, 1.0);
    Vec3 g(0.5, 0.0, 0.0);
    force->setBodyAcceleration(g);
    force->setInitialFluidVelocity(Vec3(0.01, 0.02, -0.01)*(fluidDx/fluidDt));
    force->setFluidMomentumRemovalFrequency(3);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    int numNodes = 64;
    double gLattice = g[0]*fluidDt*fluidDt/fluidDx;
    for (int n = 1; n <= 7; n++) {
        integrator.step(1);
        vector<double> state;
        force->getFluidState(context, state);
        double mass;
        Vec3 p;
        totalMoments(state, mass, p);
        // The total momentum is a sum of 19*numNodes populations of order 0.05 with cancellations: rounding ~1e-14.
        ASSERT_EQUAL_VEC(Vec3(((n-1)%3 + 1)*gLattice*numNodes, 0, 0), p, getFluidTolerance(platform, 1e-13));
    }
    delete system;
}

/**
 * A transverse shear wave u_x = U sin(k y) decays as exp(-nu k^2 t), with the kinematic viscosity
 * nu = (tau - 1/2)/3 of the lattice.
 */
void testShearWaveViscosity(Platform& platform, double tau) {
    int nx = 2, ny = 64, nz = 2;
    LBMForce* force;
    System* system = createFluidSystem(force, nx, ny, nz, tau);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    int numNodes = nx*ny*nz;
    double k = 2*M_PI/ny, amplitude = 1e-3;
    vector<double> state(19*numNodes);
    for (int j = 0; j < ny; j++) {
        vector<double> column = uniformState(1, 1.0, Vec3(amplitude*sin(k*j), 0, 0));
        for (int kk = 0; kk < nz; kk++)
            for (int i = 0; i < nx; i++)
                for (int q = 0; q < 19; q++)
                    state[q*numNodes + i + nx*(j + ny*kk)] = column[q];
    }
    force->setFluidState(context, state);

    // Amplitude of the sine mode of u_x, in lattice units.

    auto measure = [&]() {
        vector<double> density;
        vector<Vec3> velocity;
        force->getFluidFields(context, density, velocity);
        double sum = 0;
        for (int node = 0; node < numNodes; node++) {
            int j = (node/nx)%ny;
            sum += velocity[node][0]*sin(k*j);
        }
        return 2*sum/numNodes/(fluidDx/fluidDt);
    };
    int t1 = 200, t2 = 1200;
    integrator.step(t1);
    double a1 = measure();
    integrator.step(t2-t1);
    double a2 = measure();
    double nuMeasured = -log(a2/a1)/((t2-t1)*k*k);
    double nu = (tau-0.5)/3.0;
    // ASSERT_EQUAL_TOL is absolute for values below 1: compare the relative difference explicitly.
    if (fabs(nuMeasured/nu - 1.0) > 2e-3) {
        stringstream msg;
        msg << "shear wave at tau = " << tau << ": measured viscosity " << nuMeasured << ", expected " << nu;
        throwException(__FILE__, __LINE__, msg.str());
    }
    delete system;
}

/**
 * Requests for forces or energy outside an integration step, and setVelocitiesToTemperature(), do not
 * advance the fluid.
 */
void testQueriesDoNotAdvanceFluid(Platform& platform) {
    vector<double> reference, queried;
    for (int run = 0; run < 2; run++) {
        LBMForce* force;
        System* system = createFluidSystem(force, 4, 4, 4, 0.8);
        force->setBodyAcceleration(Vec3(0.5, -0.2, 0.1));
        force->setInitialFluidVelocity(Vec3(0.01, 0.0, 0.0)*(fluidDx/fluidDt));
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
        for (int n = 0; n < 10; n++) {
            integrator.step(1);
            if (run == 1) {
                context.getState(State::Forces);
                context.getState(State::Forces | State::Energy);
                context.setVelocitiesToTemperature(300.0, 1234);
            }
        }
        force->getFluidState(context, run == 0 ? reference : queried);
        delete system;
    }
    for (int i = 0; i < (int) reference.size(); i++)
        ASSERT_EQUAL(reference[i], queried[i]);
}

/**
 * The removal of the fluid momentum follows the step count of the Context, which checkpoints save and
 * restore: a run restarted from a checkpoint, with the fluid restored by setFluidState(), is identical to an
 * uninterrupted run, also when the checkpoint is not at a multiple of the removal frequency.
 */
void testRestartFromCheckpoint(Platform& platform) {
    int numSteps = 35, split = 13;
    vector<double> uninterrupted, saved, restarted;
    stringstream checkpoint;
    for (int run = 0; run < 3; run++) {
        LBMForce* force;
        System* system = createFluidSystem(force, 4, 4, 4, 0.8);
        force->setBodyAcceleration(Vec3(0.5, -0.2, 0.1));
        force->setInitialFluidVelocity(Vec3(0.01, 0.0, 0.0)*(fluidDx/fluidDt));
        force->setFluidMomentumRemovalFrequency(5);
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
        if (run == 0) {
            integrator.step(numSteps);
            force->getFluidState(context, uninterrupted);
        }
        else if (run == 1) {
            integrator.step(split);
            context.createCheckpoint(checkpoint);
            force->getFluidState(context, saved);
        }
        else {
            context.loadCheckpoint(checkpoint);
            force->setFluidState(context, saved);
            integrator.step(numSteps-split);
            ASSERT_EQUAL(numSteps, context.getStepCount());
            force->getFluidState(context, restarted);
        }
        delete system;
    }
    for (int i = 0; i < (int) uninterrupted.size(); i++)
        ASSERT_EQUAL(uninterrupted[i], restarted[i]);
}

/**
 * updateParametersInContext() changes the body acceleration, the removal of the fluid momentum and the Mach
 * number check of an existing Context, and rejects a change of the viscosity.
 */
void testUpdateParameters(Platform& platform) {
    LBMForce* force;
    System* system = createFluidSystem(force, 4, 4, 4, 0.8);
    force->setBodyAcceleration(Vec3(0.5, 0, 0));
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    vector<double> state;
    double mass;
    Vec3 p;
    integrator.step(5);
    force->getFluidState(context, state);
    totalMoments(state, mass, p);
    ASSERT(p[0] > 1e-6);

    // No acceleration and removal at every step: the fluid stops.
    force->setBodyAcceleration(Vec3());
    force->setFluidMomentumRemovalFrequency(1);
    force->updateParametersInContext(context);
    integrator.step(1);
    force->getFluidState(context, state);
    totalMoments(state, mass, p);
    ASSERT_EQUAL_VEC(Vec3(), p, getFluidTolerance(platform, 1e-13));

    // A Mach number limit below the current Mach number, checked at every step.
    force->setBodyAcceleration(Vec3(0.5, 0, 0));
    force->setFluidMomentumRemovalFrequency(0);
    force->updateParametersInContext(context);
    integrator.step(5);
    force->setMachCheckFrequency(1);
    force->setMachNumberLimit(0.5*force->getFluidMachNumber(context));
    force->updateParametersInContext(context);
    bool thrown = false;
    try {
        integrator.step(1);
    }
    catch (const OpenMMException& e) {
        thrown = (string(e.what()).find("Mach number") != string::npos);
    }
    ASSERT(thrown);

    // The viscosity is fixed when the Context is created.
    force->setKinematicViscosity(2.0*force->getKinematicViscosity());
    thrown = false;
    try {
        force->updateParametersInContext(context);
    }
    catch (const OpenMMException& e) {
        thrown = (string(e.what()).find("cannot be changed") != string::npos);
    }
    ASSERT(thrown);
    delete system;
}

/**
 * The Mach number of the fluid is checked every N steps: above the limit the step throws an exception.
 */
void testMachNumberCheck(Platform& platform) {
    double mach = 0.35, speed = mach/sqrt(3.0);      // lattice units
    for (int variant = 0; variant < 3; variant++) {
        LBMForce* force;
        System* system = createFluidSystem(force, 4, 4, 4, 0.8);
        force->setInitialFluidVelocity(Vec3(speed, 0, 0)*(fluidDx/fluidDt));
        force->setMachCheckFrequency(variant == 1 ? 0 : 10);
        if (variant == 2)
            force->setMachNumberLimit(0.5);
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
        ASSERT_EQUAL_TOL(mach, force->getFluidMachNumber(context), getFluidTolerance(platform, 1e-12));
        integrator.step(9);
        bool thrown = false;
        try {
            integrator.step(1);
        }
        catch (const OpenMMException& e) {
            thrown = (string(e.what()).find("Mach number") != string::npos);
        }
        ASSERT(thrown == (variant == 0));
        delete system;
    }
}

/**
 * getLatticeParametersInContext() returns the lattice spacing, the time step and the relaxation time.
 */
void testLatticeParameters(Platform& platform) {
    LBMForce* force;
    System* system = createFluidSystem(force, 4, 6, 8, 0.9);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    double dx, dt, tau;
    force->getLatticeParametersInContext(context, dx, dt, tau);
    ASSERT_EQUAL_TOL(fluidDx, dx, 1e-14);
    ASSERT_EQUAL_TOL(fluidDt, dt, 1e-14);
    ASSERT_EQUAL_TOL(0.9, tau, 1e-12);
    delete system;
}

/**
 * A warning is printed on stderr when a Context is created with tau outside [0.505, 2].
 */
void testRelaxationTimeWarning(Platform& platform) {
    double taus[] = {0.503, 1.0, 2.2};
    for (double tau : taus) {
        LBMForce* force;
        System* system = createFluidSystem(force, 4, 4, 4, tau);
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
        bool warned = (captured.str().find("relaxation time") != string::npos);
        ASSERT(warned == (tau < 0.505 || tau > 2.0));
        delete system;
    }
}

/** Indices of the nodes of the plane j = 0. */
vector<int> wallPlane(int nx, int ny, int nz) {
    vector<int> nodes;
    for (int k = 0; k < nz; k++)
        for (int i = 0; i < nx; i++)
            nodes.push_back(i + nx*ny*k);
    return nodes;
}

/**
 * Invalid lists of solid nodes are rejected when the Context is created.
 */
void testSolidNodeChecks(Platform& platform) {
    vector<vector<int> > invalid = {{-1}, {64}, {3, 5, 3}};
    vector<int> all(64);
    for (int i = 0; i < 64; i++)
        all[i] = i;
    invalid.push_back(all);
    for (const vector<int>& nodes : invalid) {
        LBMForce* force;
        System* system = createFluidSystem(force, 4, 4, 4, 0.8);
        force->setSolidNodes(nodes);
        VerletIntegrator integrator(fluidDt);
        bool thrown = false;
        try {
            Context context(*system, integrator, platform);
        }
        catch (const OpenMMException& e) {
            thrown = true;
        }
        ASSERT(thrown);
        delete system;
    }
}

/**
 * Poiseuille flow between the walls of the solid plane j = 0 (the lattice is periodic, so the plane bounds
 * the channel on both sides), driven by a body force along x.  With halfway bounce-back the steady profile
 * of the scheme is, in lattice units, exactly
 *   u(y) = g/(2 nu) (y - 1/2)(ny - 1/2 - y) + g (16 Lambda - 3)/(24 nu),   Lambda = (tau - 1/2)/2,
 * the solution of the two-relaxation-time scheme with Lambda = (tau - 1/2)(tau_odd - 1/2), since the
 * regularized collision relaxes the odd non-hydrodynamic moments with tau_odd = 1.  The slip term vanishes
 * at tau = 7/8.
 */
void testPoiseuille(Platform& platform, double tau) {
    int nx = 2, ny = 12, nz = 2;
    LBMForce* force;
    System* system = createFluidSystem(force, nx, ny, nz, tau);
    double g = 1e-5;                             // lattice units
    force->setBodyAcceleration(Vec3(g*fluidDx/(fluidDt*fluidDt), 0, 0));
    force->setSolidNodes(wallPlane(nx, ny, nz));
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    double nu = (tau-0.5)/3.0, lambda = (tau-0.5)/2.0, h = ny-1;
    integrator.step((int) (4*h*h/nu));
    vector<double> density;
    vector<Vec3> velocity;
    force->getFluidFields(context, density, velocity);
    double umax = g/(2*nu)*(0.5*h)*(0.5*h) + g*(16*lambda-3)/(24*nu);
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                int node = i + nx*(j + ny*k);
                Vec3 u = velocity[node]*(fluidDt/fluidDx);
                if (j == 0) {
                    ASSERT_EQUAL(0.0, density[node]);
                    ASSERT_EQUAL_VEC(Vec3(), u, 0.0);
                    continue;
                }
                double exact = g/(2*nu)*(j-0.5)*(ny-0.5-j) + g*(16*lambda-3)/(24*nu);
                ASSERT_EQUAL_VEC(Vec3(exact/umax, 0, 0), u*(1.0/umax), 1e-9);
            }

    // In the steady state the walls carry the whole body force: g times the mass of the fluid.
    double mass = 0;
    for (double d : density)
        mass += d*fluidDx*fluidDx*fluidDx;
    Vec3 bodyForce(g*fluidDx/(fluidDt*fluidDt)*mass, 0, 0);
    ASSERT_EQUAL_VEC(bodyForce, force->getWallForce(context), 1e-8);
    delete system;
}

/**
 * With solid nodes the mass of the fluid is conserved, and the removal of the momentum sums only fluid
 * nodes.
 */
void testWallConservation(Platform& platform) {
    int nx = 6, ny = 5, nz = 4, numNodes = nx*ny*nz;
    vector<int> block;
    for (int k = 1; k < 3; k++)
        for (int j = 1; j < 3; j++)
            for (int i = 2; i < 4; i++)
                block.push_back(i + nx*(j + ny*k));
    for (int removal = 0; removal < 2; removal++) {
        LBMForce* force;
        System* system = createFluidSystem(force, nx, ny, nz, 0.7);
        force->setSolidNodes(block);
        force->setInitialFluidVelocity(Vec3(0.02, -0.01, 0.0)*(fluidDx/fluidDt));
        force->setFluidMomentumRemovalFrequency(removal);
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
        vector<double> density;
        vector<Vec3> velocity;
        auto fluidMass = [&]() {
            force->getFluidFields(context, density, velocity);
            double mass = 0;
            for (double d : density)
                mass += d/fluidDensity;
            return mass;
        };
        double mass0 = fluidMass();
        ASSERT_EQUAL_TOL((double) (numNodes - block.size()), mass0, 1e-13);
        integrator.step(removal == 0 ? 100 : 1);
        ASSERT_EQUAL_TOL(mass0, fluidMass(), 1e-13);
        if (removal == 1) {
            Vec3 p;
            for (int node = 0; node < numNodes; node++)
                p += velocity[node]*(density[node]/fluidDensity*fluidDt/fluidDx);
            ASSERT_EQUAL_VEC(Vec3(), p, 1e-13);
        }
        delete system;
    }
}

void runFluidTests(Platform& platform) {
    testUniformFlowIsSteady(platform);
    testFluidConservation(platform);
    testBodyForce(platform, 1.0);
    testBodyForce(platform, 0.98);
    testBodyForce(platform, 1.02);
    testFluidMomentumRemoval(platform);
    testShearWaveViscosity(platform, 0.6);
    testShearWaveViscosity(platform, 1.0);
    testShearWaveViscosity(platform, 1.5);
    testQueriesDoNotAdvanceFluid(platform);
    testRestartFromCheckpoint(platform);
    testUpdateParameters(platform);
    testMachNumberCheck(platform);
    testLatticeParameters(platform);
    testRelaxationTimeWarning(platform);
}

void runWallTests(Platform& platform) {
    testSolidNodeChecks(platform);
    testPoiseuille(platform, 0.7);
    testPoiseuille(platform, 0.875);
    testPoiseuille(platform, 1.2);
    testWallConservation(platform);
}
