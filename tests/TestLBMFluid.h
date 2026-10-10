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
#include "internal/LBMBoundaries.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

/** Lattice spacing (nm), time step (ps) and density (Da/nm^3) of the fluid tests. */
const double fluidDx = 0.5, fluidDt = 0.01, fluidDensity = 602.2;

/**
 * Tolerance of a fluid test: the given tolerance in double and mixed precision, where the fluid is stored in
 * double precision; in single precision the resolution of the type (getStorageTolerance()), or the tolerance
 * singleTolerance of the test if it is given.
 */
double getFluidTolerance(Platform& platform, double tolerance, double singleTolerance=0) {
    double storage = getStorageTolerance(platform);
    if (storage <= 1e-12)
        return tolerance;
    return (singleTolerance > 0 ? singleTolerance : storage);
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
    // In single precision the steady state is the result of thousands of steps rounded in float: on an A100 the
    // profile and the force on the walls differ from the exact values by up to 1.1e-5 and 8e-6.
    double tol = getFluidTolerance(platform, 1e-9, 5e-5);
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
                ASSERT_EQUAL_VEC(Vec3(exact/umax, 0, 0), u*(1.0/umax), tol);
            }

    // In the steady state the walls carry the whole body force: g times the mass of the fluid.
    double mass = 0;
    for (double d : density)
        mass += d*fluidDx*fluidDx*fluidDx;
    Vec3 bodyForce(g*fluidDx/(fluidDt*fluidDt)*mass, 0, 0);
    ASSERT_EQUAL_VEC(bodyForce, force->getWallForce(context), getFluidTolerance(platform, 1e-8, 5e-5));
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
        double tol = getFluidTolerance(platform, 1e-13);
        double mass0 = fluidMass();
        ASSERT_EQUAL_TOL((double) (numNodes - block.size()), mass0, tol);
        integrator.step(removal == 0 ? 100 : 1);
        ASSERT_EQUAL_TOL(mass0, fluidMass(), tol);
        if (removal == 1) {
            Vec3 p;
            for (int node = 0; node < numNodes; node++)
                p += velocity[node]*(density[node]/fluidDensity*fluidDt/fluidDx);
            ASSERT_EQUAL_VEC(Vec3(), p, tol);
        }
        delete system;
    }
}

/**
 * Poiseuille flow between regularized walls (setWallScheme(Regularized)): the solid plane j = 0, with the periodic
 * boundary, puts the walls on the solid nodes j = 0 and j = ny, so the channel has the width H = ny.  The steady
 * profile of the scheme is, in lattice units, exactly
 *   u(y) = g/(2 nu) y (ny - y) + 3 g (tau - 1)/(tau - 1/2)
 * at every fluid node: the parabola that vanishes on the solid nodes, displaced by a slip that vanishes at tau = 1
 * (docs/theory.md, section 1).  The density stays that of the fluid at rest, and in the steady state the walls carry
 * the whole body force.
 */
void testRegularizedPoiseuille(Platform& platform, double tau) {
    int nx = 2, ny = 12, nz = 2;
    LBMForce* force;
    System* system = createFluidSystem(force, nx, ny, nz, tau);
    double g = 1e-5;                             // lattice units
    force->setBodyAcceleration(Vec3(g*fluidDx/(fluidDt*fluidDt), 0, 0));
    force->setSolidNodes(wallPlane(nx, ny, nz));
    force->setWallScheme(LBMForce::Regularized);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    double nu = (tau-0.5)/3.0, h = ny;
    integrator.step((int) (4*h*h/nu));
    vector<double> density;
    vector<Vec3> velocity;
    force->getFluidFields(context, density, velocity);
    double umax = g/(2*nu)*(0.5*h)*(0.5*h);
    double tol = getFluidTolerance(platform, 1e-9, 5e-5);
    double mass = 0;
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                int node = i + nx*(j + ny*k);
                Vec3 u = velocity[node]*(fluidDt/fluidDx);
                mass += density[node]*fluidDx*fluidDx*fluidDx;
                if (j == 0) {
                    ASSERT_EQUAL(0.0, density[node]);
                    ASSERT_EQUAL_VEC(Vec3(), u, 0.0);
                    continue;
                }
                ASSERT_EQUAL_TOL(1.0, density[node]/fluidDensity, tol);
                double exact = g/(2*nu)*j*(ny-j) + 3*g*(tau-1)/(tau-0.5);
                ASSERT_EQUAL_VEC(Vec3(exact/umax, 0, 0), u*(1.0/umax), tol);
            }
    Vec3 bodyForce(g*fluidDx/(fluidDt*fluidDt)*mass, 0, 0);
    ASSERT_EQUAL_VEC(bodyForce, force->getWallForce(context), getFluidTolerance(platform, 1e-8, 5e-5));
    delete system;
}

/**
 * Solid nodes of the conservation tests: the plane i = 0, one node thick (with fluid on both sides through the
 * periodic boundary), and a block of 2x2x2 nodes, so that the fluid nodes i = 1 lie between two walls.
 */
vector<int> wallTestNodes(int nx, int ny, int nz) {
    vector<int> nodes;
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                if (i == 0 || (i >= 2 && i < 4 && j >= 1 && j < 3 && k >= 1 && k < 3))
                    nodes.push_back(i + nx*(j + ny*k));
    return nodes;
}

/**
 * With both wall schemes the mass of the fluid is conserved exactly, and the momentum of the fluid changes by the
 * body force on the fluid minus the momentum given to the walls (getWallForce()): P(t + dt) - P(t) =
 * (M g - F_wall) dt, at every step.  The fluid starts with a uniform velocity against the walls.
 */
void testWallBalance(Platform& platform, LBMForce::WallScheme scheme) {
    int nx = 6, ny = 5, nz = 4;
    LBMForce* force;
    System* system = createFluidSystem(force, nx, ny, nz, 0.7);
    force->setSolidNodes(wallTestNodes(nx, ny, nz));
    force->setWallScheme(scheme);
    force->setInitialFluidVelocity(Vec3(0.02, -0.01, 0.005)*(fluidDx/fluidDt));
    Vec3 g = Vec3(1e-5, 2e-5, -1e-5)*(fluidDx/(fluidDt*fluidDt));
    force->setBodyAcceleration(g);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    double cellVolume = fluidDx*fluidDx*fluidDx;
    auto moments = [&](double& mass, Vec3& p) {
        vector<double> density;
        vector<Vec3> velocity;
        force->getFluidFields(context, density, velocity);
        mass = 0;
        p = Vec3();
        for (int node = 0; node < (int) density.size(); node++) {
            mass += density[node]*cellVolume;
            p += velocity[node]*(density[node]*cellVolume);
        }
    };
    double mass0, mass;
    Vec3 p0, p;
    moments(mass0, p0);
    double scale = mass0*0.02*fluidDx/fluidDt;
    for (int step = 0; step < 30; step++) {
        integrator.step(1);
        moments(mass, p);
        ASSERT_EQUAL_TOL(mass0, mass, getFluidTolerance(platform, 1e-13));
        Vec3 balance = p - p0 - (g*mass - force->getWallForce(context))*fluidDt;
        ASSERT_EQUAL_TOL(0.0, sqrt(balance.dot(balance))/scale, getFluidTolerance(platform, 1e-12));
        p0 = p;
    }
    delete system;
}

/**
 * Couette flow between two faces that set the velocity (setFaceBoundary()): Velocity faces, or DensityVelocity faces
 * at the density of the fluid at rest.  The faces perpendicular to x and y are periodic, the face ZMin is at rest and
 * the face ZMax moves with the velocity U along x.  The steady profile is exactly linear, u_x(z) = U (z + 1)/(nz + 1),
 * with the walls on the nodes beyond the faces, z = -1 and z = nz (docs/theory.md, section 1), and the density stays
 * uniform.
 */
void testCouette(Platform& platform, double tau, LBMForce::BoundaryType type=LBMForce::Velocity) {
    int nx = 2, ny = 2, nz = 10;
    LBMForce* force;
    System* system = createFluidSystem(force, nx, ny, nz, tau);
    double U = 0.01;                             // lattice units
    force->setFaceBoundary(LBMForce::ZMin, type);
    force->setFaceBoundary(LBMForce::ZMax, type);
    force->setFaceVelocity(LBMForce::ZMax, Vec3(U*fluidDx/fluidDt, 0, 0));
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    double nu = (tau-0.5)/3.0, h = nz+1;
    integrator.step((int) (6*h*h/nu));
    vector<double> density;
    vector<Vec3> velocity;
    force->getFluidFields(context, density, velocity);
    double tol = getFluidTolerance(platform, 1e-9, 5e-5);
    for (int node = 0; node < nx*ny*nz; node++) {
        int k = node/(nx*ny);
        ASSERT_EQUAL_VEC(Vec3((k+1)/h, 0, 0), velocity[node]*(fluidDt/(fluidDx*U)), tol);
        ASSERT_EQUAL_TOL(1.0, density[node]/fluidDensity, tol);
    }
    delete system;
}

/**
 * A uniform flow from an inlet YMin (a Velocity or DensityVelocity face) to an outlet YMax (a Density or
 * DensityVelocity face), with the density of the fluid at rest and the velocity of the flow on both, is steady: the
 * faces leave it unchanged.  The faces perpendicular to x and z are periodic.
 */
void testUniformFlowThroughFaces(Platform& platform, LBMForce::BoundaryType inlet=LBMForce::Velocity,
        LBMForce::BoundaryType outlet=LBMForce::Density) {
    int nx = 3, ny = 8, nz = 3, numNodes = nx*ny*nz;
    LBMForce* force;
    System* system = createFluidSystem(force, nx, ny, nz, 0.8);
    Vec3 u = Vec3(0.0, 0.02, 0.0)*(fluidDx/fluidDt);
    force->setFaceBoundary(LBMForce::YMin, inlet);
    force->setFaceBoundary(LBMForce::YMax, outlet);
    force->setFaceVelocity(LBMForce::YMin, u);
    force->setFaceVelocity(LBMForce::YMax, u);
    force->setInitialFluidVelocity(u);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    integrator.step(50);
    vector<double> density;
    vector<Vec3> velocity;
    force->getFluidFields(context, density, velocity);
    double tol = getFluidTolerance(platform, 1e-13);
    for (int node = 0; node < numNodes; node++) {
        ASSERT_EQUAL_VEC(u*(fluidDt/fluidDx), velocity[node]*(fluidDt/fluidDx), tol);
        ASSERT_EQUAL_TOL(1.0, density[node]/fluidDensity, tol);
    }
    delete system;
}

/**
 * Flow in a square duct driven by a difference of density (pressure) between two Density faces: no-slip walls
 * perpendicular to x and z (the solid planes i = 0 and k = 0 of the periodic box, with bounce-back: a duct of width
 * H = n - 1 between the walls at 1/2 and n - 1/2), and the faces YMin and YMax at the densities 1.01 and 1 of the fluid
 * at rest.  At the node ny/2 the density lies on the straight line between those of the faces, held one node beyond
 * them (y = -1 and y = ny), and the velocity is
 * that of the incompressible flow in a rectangular duct, u = sum over odd m of 4 K H^2/(m pi)^3 (-1)^((m-1)/2)
 * (1 - cosh(m pi z/H)/cosh(m pi/2)) cos(m pi x/H), with K = c_s^2 |drho/dy|/(rho nu) from the gradient of the density
 * in the middle, within the error of the walls (second order) and the compressibility of the fluid (1 %).  The
 * gradient is steeper than (rho_in - rho_out)/(ny + 1), the distance between the nodes beyond the faces: part of the
 * difference of density goes into the regions next to the faces, where the flow enters and leaves (docs/theory.md,
 * section 1).  The flow is steady: the staggered mode (-1)^(y+t) j_y, which the bounce-back walls conserve, is damped
 * by the Density faces.
 */
void testPressureDrivenDuct(Platform& platform) {
    int n = 8, ny = 16;
    double tau = 1.0;
    LBMForce* force;
    System* system = createFluidSystem(force, n, ny, n, tau);
    vector<int> walls;
    for (int k = 0; k < n; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < n; i++)
                if (i == 0 || k == 0)
                    walls.push_back(i + n*(j + ny*k));
    force->setSolidNodes(walls);
    force->setFaceBoundary(LBMForce::YMin, LBMForce::Density);
    force->setFaceBoundary(LBMForce::YMax, LBMForce::Density);
    force->setFaceDensity(LBMForce::YMin, 1.01*fluidDensity);
    force->setFaceDensity(LBMForce::YMax, fluidDensity);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    double nu = (tau-0.5)/3.0;
    integrator.step((int) (3*ny*ny/nu));
    vector<double> density[2];
    vector<Vec3> velocity[2];
    for (int run = 0; run < 2; run++) {
        force->getFluidFields(context, density[run], velocity[run]);
        integrator.step(1);
    }
    int middle = ny/2;
    double H = n-1, rho = density[0][n/2 + n*(middle + ny*(n/2))]/fluidDensity;
    ASSERT_EQUAL_TOL(1.0 + 0.01*(ny-middle)/(ny+1), rho, 2e-4);
    double gradient = (density[0][n/2 + n*(middle-2 + ny*(n/2))] - density[0][n/2 + n*(middle+2 + ny*(n/2))])/(4*fluidDensity);
    ASSERT(gradient > 0.01/(ny+1) && gradient < 0.01/(ny-3));
    double K = (gradient/3.0)/(rho*nu), umax = 0, error = 0, change = 0;
    for (int k = 1; k < n; k++)
        for (int i = 1; i < n; i++) {
            int node = i + n*(middle + ny*k);
            double x = i - 0.5*n, z = k - 0.5*n, exact = 0;
            for (int m = 1; m < 200; m += 2) {
                double a = m*M_PI/H;
                exact += 4*K*H*H/pow(m*M_PI, 3)*(m%4 == 1 ? 1 : -1)*(1 - cosh(a*z)/cosh(a*H/2))*cos(a*x);
            }
            double u = velocity[0][node][1]*fluidDt/fluidDx;
            umax = max(umax, exact);
            error = max(error, fabs(u-exact));
            change = max(change, fabs(velocity[1][node][1]-velocity[0][node][1])*fluidDt/fluidDx);
        }
    ASSERT(error < 0.02*umax);
    ASSERT(change < getFluidTolerance(platform, 1e-9, 1e-4)*umax);
    delete system;
}

/**
 * A fluid at rest between two DensityVelocity faces YMin and YMax with the same density and velocity, different from
 * those of the fluid at rest and with a component along the faces, reaches the uniform flow with that density and
 * velocity (docs/theory.md, section 1).  The faces perpendicular to x and z are periodic.
 */
void testDensityVelocityReservoir(Platform& platform) {
    int nx = 3, ny = 12, nz = 3, numNodes = nx*ny*nz;
    LBMForce* force;
    System* system = createFluidSystem(force, nx, ny, nz, 1.0);
    double rho = 1.01;
    Vec3 u(0.01, 0.02, 0.0);                     // lattice units
    for (LBMForce::Face face : {LBMForce::YMin, LBMForce::YMax}) {
        force->setFaceBoundary(face, LBMForce::DensityVelocity);
        force->setFaceDensity(face, rho*fluidDensity);
        force->setFaceVelocity(face, u*(fluidDx/fluidDt));
    }
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    integrator.step(2500);
    vector<double> density;
    vector<Vec3> velocity;
    force->getFluidFields(context, density, velocity);
    double tol = getFluidTolerance(platform, 1e-9, 1e-5);
    for (int node = 0; node < numNodes; node++) {
        ASSERT_EQUAL_VEC(u, velocity[node]*(fluidDt/fluidDx), tol);
        ASSERT_EQUAL_TOL(rho, density[node]/fluidDensity, tol);
    }
    delete system;
}

/**
 * A DensityVelocity inlet YMin whose density, 1.01, differs from that of a Density outlet YMax, 1: the face sets the
 * populations that enter the fluid, not the density and the velocity of the node, so the steady flow is uniform with
 * the density of the outlet and the velocity u at which those populations, sum over c_y = 1 of feq_q(rho, u) =
 * rho (1 + 3u + 3u^2)/6, are those of the inlet, 1.01 (1 + 3 u_b + 3 u_b^2)/6 (docs/theory.md, section 1).  Faces
 * perpendicular to x and z periodic.
 */
void testDensityVelocityInlet(Platform& platform) {
    int nx = 3, ny = 16, nz = 3, numNodes = nx*ny*nz;
    LBMForce* force;
    System* system = createFluidSystem(force, nx, ny, nz, 0.8);
    double rhoIn = 1.01, ub = 0.01;
    force->setFaceBoundary(LBMForce::YMin, LBMForce::DensityVelocity);
    force->setFaceBoundary(LBMForce::YMax, LBMForce::Density);
    force->setFaceDensity(LBMForce::YMin, rhoIn*fluidDensity);
    force->setFaceVelocity(LBMForce::YMin, Vec3(0.0, ub, 0.0)*(fluidDx/fluidDt));
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 0.2, 0.3)));
    integrator.step(4000);
    vector<double> density;
    vector<Vec3> velocity;
    force->getFluidFields(context, density, velocity);
    double a = rhoIn*(1.0 + 3.0*ub + 3.0*ub*ub) - 1.0;
    double u = (-3.0 + sqrt(9.0 + 12.0*a))/6.0;
    double tol = getFluidTolerance(platform, 1e-10, 1e-5);
    for (int node = 0; node < numNodes; node++) {
        ASSERT_EQUAL_VEC(Vec3(0.0, u, 0.0), velocity[node]*(fluidDt/fluidDx), tol);
        ASSERT_EQUAL_TOL(1.0, density[node]/fluidDensity, tol);
    }
    delete system;
}

/**
 * The kinds of the nodes on the open faces (internal/LBMBoundaries.h): on the nodes shared by several faces the first
 * face that sets the velocity (Velocity or DensityVelocity) in the order XMin ... ZMax gives the velocity, and the
 * density too if it is a DensityVelocity face; the nodes of Density faces only take the density of their face, or
 * of the first one with zero velocity if they lie on more than one.
 */
void testBoundaryKinds() {
    LBMLatticeParameters lattice;
    lattice.nx = lattice.ny = lattice.nz = 4;
    lattice.wallScheme = LBMForce::BounceBack;
    LBMForce::BoundaryType types[6] = {LBMForce::Density, LBMForce::Velocity, LBMForce::DensityVelocity,
                                       LBMForce::Density, LBMForce::Density, LBMForce::DensityVelocity};
    for (int face = 0; face < 6; face++)
        lattice.faceBoundary[face] = types[face];
    LBMBoundaries boundaries;
    boundaries.find(lattice, vector<int>());
    ASSERT_EQUAL(4*4*4 - 2*2*2, (int) boundaries.nodes.size());
    struct Expected {int i, j, k, kind, face;};
    vector<Expected> expected = {
        {0, 1, 1, LBMBoundaries::Density, 0},           // XMin only
        {3, 1, 1, LBMBoundaries::Velocity, 1},          // XMax only
        {1, 0, 1, LBMBoundaries::DensityVelocity, 2},   // YMin only
        {0, 0, 1, LBMBoundaries::DensityVelocity, 2},   // XMin (Density) and YMin
        {3, 0, 1, LBMBoundaries::Velocity, 1},          // XMax comes before YMin
        {1, 3, 0, LBMBoundaries::DensityAtRest, 3},     // YMax and ZMin, both Density
        {0, 3, 0, LBMBoundaries::DensityAtRest, 0},     // XMin, YMax and ZMin, all Density
        {0, 3, 3, LBMBoundaries::DensityVelocity, 5},   // XMin and YMax (Density) and ZMax
        {3, 0, 3, LBMBoundaries::Velocity, 1}           // XMax, YMin and ZMax
    };
    for (const Expected& e : expected) {
        int node = e.i + 4*(e.j + 4*e.k);
        int b = find(boundaries.nodes.begin(), boundaries.nodes.end(), node) - boundaries.nodes.begin();
        ASSERT(b < (int) boundaries.nodes.size());
        ASSERT_EQUAL(e.kind, boundaries.kind[b]);
        ASSERT_EQUAL(e.face, boundaries.face[b]);
    }
}

/**
 * Invalid open faces are rejected when the Context is created: only one face of an axis open, open faces with the
 * removal of the fluid momentum, and fewer than 3 nodes along an open axis.
 */
void testFaceChecks(Platform& platform) {
    for (int test = 0; test < 3; test++) {
        LBMForce* force;
        System* system = createFluidSystem(force, 4, (test == 2 ? 2 : 4), 4, 0.8);
        force->setFaceBoundary(LBMForce::YMin, LBMForce::Velocity);
        if (test != 0)
            force->setFaceBoundary(LBMForce::YMax, LBMForce::Density);
        if (test == 1)
            force->setFluidMomentumRemovalFrequency(1);
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

/**
 * A checkpoint records the wall scheme and the boundary types of the faces: it is loaded by a Context with the same
 * ones, and refused by a Context with another wall scheme or other faces.
 */
void testCheckpointBoundaries(Platform& platform) {
    auto create = [&](LBMForce*& force, LBMForce::WallScheme scheme, LBMForce::BoundaryType yFaces) {
        System* system = createFluidSystem(force, 4, 5, 4, 0.8);
        force->setSolidNodes(wallPlane(4, 5, 4));
        force->setWallScheme(scheme);
        force->setFaceBoundary(LBMForce::YMin, yFaces);
        force->setFaceBoundary(LBMForce::YMax, yFaces);
        return system;
    };
    LBMForce* force;
    System* system = create(force, LBMForce::Regularized, LBMForce::Density);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.1, 1.2, 0.3)));
    integrator.step(3);
    stringstream checkpoint;
    force->createCheckpoint(context, checkpoint);
    string data = checkpoint.str();
    const char* expected[3] = {NULL, "wall scheme", "boundary types of the faces"};
    LBMForce::WallScheme schemes[3] = {LBMForce::Regularized, LBMForce::BounceBack, LBMForce::Regularized};
    LBMForce::BoundaryType faces[3] = {LBMForce::Density, LBMForce::Density, LBMForce::Velocity};
    for (int i = 0; i < 3; i++) {
        LBMForce* force2;
        System* system2 = create(force2, schemes[i], faces[i]);
        VerletIntegrator integrator2(fluidDt);
        Context context2(*system2, integrator2, platform);
        stringstream stream(data);
        string message;
        try {
            force2->loadCheckpoint(context2, stream);
        }
        catch (const OpenMMException& e) {
            message = e.what();
        }
        if (expected[i] == NULL) {
            ASSERT(message.empty());
        }
        else {
            ASSERT(message.find(expected[i]) != string::npos);
        }
        delete system2;
    }
    delete system;
}

/**
 * With walls and open faces (a Velocity inlet and a Density outlet, whose time filter reads the velocity of the
 * previous step from the populations) and a body force, a run restarted from getFluidState() and setFluidState()
 * is identical, bit for bit, to an uninterrupted run: the state of the fluid holds everything.
 */
void testFluidStateRestartWithBoundaries(Platform& platform, LBMForce::WallScheme scheme) {
    int numSteps = 30, split = 11;
    vector<double> saved, fluid[2];
    for (int run = 0; run < 2; run++) {
        LBMForce* force;
        System* system = createFluidSystem(force, 6, 5, 4, 0.8);
        force->setSolidNodes(wallPlane(6, 5, 4));
        force->setWallScheme(scheme);
        force->setFaceBoundary(LBMForce::XMin, LBMForce::Velocity);
        force->setFaceBoundary(LBMForce::XMax, LBMForce::Density);
        force->setFaceVelocity(LBMForce::XMin, Vec3(0.4, 0.1, 0.0));
        force->setFaceDensity(LBMForce::XMax, 1.003*fluidDensity);
        force->setBodyAcceleration(Vec3(0.0, 0.0, 2.0));
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        context.setPositions(vector<Vec3>(1, Vec3(0.1, 1.2, 0.3)));
        if (run == 0) {
            integrator.step(split);
            force->getFluidState(context, saved);
        }
        else
            force->setFluidState(context, saved);
        integrator.step(numSteps-split);
        force->getFluidState(context, fluid[run]);
        delete system;
    }
    for (int i = 0; i < (int) fluid[0].size(); i++)
        ASSERT_EQUAL(fluid[0][i], fluid[1][i]);
}

void runWallTests(Platform& platform) {
    testSolidNodeChecks(platform);
    testPoiseuille(platform, 0.7);
    testPoiseuille(platform, 0.875);
    testPoiseuille(platform, 1.2);
    testWallConservation(platform);
    testWallBalance(platform, LBMForce::BounceBack);
    testRegularizedPoiseuille(platform, 0.7);
    testRegularizedPoiseuille(platform, 1.0);
    testRegularizedPoiseuille(platform, 1.5);
    testWallBalance(platform, LBMForce::Regularized);
    testCouette(platform, 0.6);
    testCouette(platform, 1.0);
    testCouette(platform, 1.5);
    testCouette(platform, 0.6, LBMForce::DensityVelocity);
    testCouette(platform, 1.5, LBMForce::DensityVelocity);
    testUniformFlowThroughFaces(platform);
    testUniformFlowThroughFaces(platform, LBMForce::DensityVelocity, LBMForce::Density);
    testUniformFlowThroughFaces(platform, LBMForce::DensityVelocity, LBMForce::DensityVelocity);
    testDensityVelocityReservoir(platform);
    testDensityVelocityInlet(platform);
    testPressureDrivenDuct(platform);
    testFaceChecks(platform);
    testCheckpointBoundaries(platform);
    testFluidStateRestartWithBoundaries(platform, LBMForce::BounceBack);
    testFluidStateRestartWithBoundaries(platform, LBMForce::Regularized);
    if (platform.getName() == "Reference")
        testBoundaryKinds();
}
