/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

/**
 * Tests of the centred drag (LBMForce::Centered, docs/theory.md, section 2): the closed form of the first step,
 * the velocity of the fluid in the drag equal to the one that the collision puts in the equilibrium, several
 * particles at a node against the direct solution of the linear system, a particle at a solid node, stability at
 * a large friction and the requirements on the System.  The coupling tests of TestLBMCoupling.h that do not
 * depend on the drag are run again with the centred drag.  Include after TestLBMCoupling.h and call
 * runCenteredTests().
 */

#include "openmm/CustomExternalForce.h"
#include "openmm/VirtualSite.h"

/**
 * Create a System like createCoupledSystem(), with the centred drag, particles of the given masses, all coupled,
 * and a constant external force on each of them, added to the System before the LBMForce.
 */
System* createCenteredSystem(LBMForce*& force, const vector<double>& masses, const vector<Vec3>& externalForces, double friction) {
    int n = 8;
    System* system = new System();
    system->setDefaultPeriodicBoxVectors(Vec3(n*fluidDx, 0, 0), Vec3(0, n*fluidDx, 0), Vec3(0, 0, n*fluidDx));
    CustomExternalForce* external = new CustomExternalForce("-fx*x-fy*y-fz*z");
    external->addPerParticleParameter("fx");
    external->addPerParticleParameter("fy");
    external->addPerParticleParameter("fz");
    for (int i = 0; i < (int) masses.size(); i++) {
        system->addParticle(masses[i]);
        Vec3 f = externalForces[i];
        external->addParticle(i, {f[0], f[1], f[2]});
    }
    system->addForce(external);
    force = new LBMForce();
    force->setGridSize(n, n, n);
    force->setFluidDensity(fluidDensity);
    force->setKinematicViscosity((0.8-0.5)/3.0*fluidDx*fluidDx/fluidDt);
    force->setFluidMomentumRemovalFrequency(0);
    force->setFriction(friction);
    force->setTemperature(0.0);
    force->setDragScheme(LBMForce::Centered);
    for (int i = 0; i < (int) masses.size(); i++)
        force->addParticle(i);
    system->addForce(force);
    return system;
}

/** Total momentum (Da nm/ps) of the fluid. */
Vec3 fluidMomentum(Context& context, LBMForce* force) {
    vector<double> fluid;
    force->getFluidState(context, fluid);
    double mass;
    Vec3 j;
    totalMoments(fluid, mass, j);
    double cellMass = force->getFluidDensity()*fluidDx*fluidDx*fluidDx;
    return j*(cellMass*fluidDx/fluidDt);
}

/**
 * In one step the centred drag multiplies the velocity of a particle in a fluid at rest by
 * (1 - a + a m/m_c)/(1 + a + a m/m_c), with a = gamma dt/2 and m_c the mass of the fluid in a cell.  Between
 * steps getState() returns the force of the next step.
 */
void testCenteredFirstStep(Platform& platform) {
    LBMForce* force;
    double friction = 50.0;
    System* system = createCoupledSystem(force, 1, friction, 0.0);
    force->setDragScheme(LBMForce::Centered);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(1.1, 1.3, 1.7)));
    Vec3 v0(0.3, -0.2, 0.1);
    context.setVelocities(vector<Vec3>(1, v0));
    ASSERT_EQUAL_VEC(Vec3(), context.getState(State::Forces).getForces()[0], 0.0);
    integrator.step(1);
    State state = context.getState(State::Velocities | State::Forces);
    Vec3 v1 = state.getVelocities()[0];
    double a = 0.5*friction*fluidDt, b = a*couplingMass/(fluidDensity*fluidDx*fluidDx*fluidDx);
    ASSERT_EQUAL_VEC(v0*((1-a+b)/(1+a+b)), v1, getCouplingTolerance(platform, 1e-14));
    integrator.step(1);
    Vec3 v2 = context.getState(State::Velocities).getVelocities()[0];
    ASSERT_EQUAL_VEC((v2-v1)*(couplingMass/fluidDt), state.getForces()[0], getCouplingTolerance(platform, 1e-12));
    delete system;
}

/**
 * Solve, for each Cartesian component, the linear system of the centred drag for the particles of one node,
 * (1 + a) F_k + (a m_k/m_c) sum_l F_l = -gamma m_k (v~_k - u~), by Gaussian elimination.
 */
vector<Vec3> solveCenteredDrag(const vector<double>& masses, const vector<Vec3>& knownVelocities, Vec3 knownFluidVelocity,
        double friction, double cellMass) {
    int n = masses.size();
    double a = 0.5*friction*fluidDt;
    vector<Vec3> forces(n);
    for (int c = 0; c < 3; c++) {
        vector<vector<double> > matrix(n, vector<double>(n+1));
        for (int k = 0; k < n; k++) {
            for (int l = 0; l < n; l++)
                matrix[k][l] = (k == l ? 1+a : 0) + a*masses[k]/cellMass;
            matrix[k][n] = -friction*masses[k]*(knownVelocities[k][c]-knownFluidVelocity[c]);
        }
        for (int k = 0; k < n; k++)
            for (int l = k+1; l < n; l++) {
                double factor = matrix[l][k]/matrix[k][k];
                for (int m = k; m <= n; m++)
                    matrix[l][m] -= factor*matrix[k][m];
            }
        for (int k = n-1; k >= 0; k--) {
            double sum = matrix[k][n];
            for (int l = k+1; l < n; l++)
                sum -= matrix[k][l]*forces[l][c];
            forces[k][c] = sum/matrix[k][k];
        }
    }
    return forces;
}

/**
 * The centred drag uses the velocities of particles and fluid at the time of the force.  For the fluid it is the
 * velocity that the collision puts in the equilibrium, u_c(t) = (j + G/2)/rho with G the total force on the node
 * (Guo forcing), here with a uniform flow u0 and a body acceleration g: u~ = u0 + g dt/2 before the reaction of the
 * particles.  For the particles it includes half the kick of the other forces, v~ = v0 + Fc dt/(2m).  Three
 * particles of different masses share a node and a fourth is alone: the velocities after the first step agree
 * with the direct solution of the linear system, and the fluid receives the opposite of the coupling forces.
 */
void testCenteredSharedNode(Platform& platform) {
    vector<double> masses = {80.0, 120.0, 150.0, 100.0};
    vector<Vec3> external = {Vec3(100, -50, 30), Vec3(-20, 40, 0), Vec3(0, 0, -80), Vec3(60, 10, -10)};
    double friction = 30.0;
    LBMForce* force;
    System* system = createCenteredSystem(force, masses, external, friction);
    Vec3 u0(0.4, -0.3, 0.2), g(20.0, 10.0, -30.0);
    force->setInitialFluidVelocity(u0);
    force->setBodyAcceleration(g);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    // The first three particles have node (2, 2, 2) as nearest node, the fourth node (6, 5, 1).
    context.setPositions({Vec3(1.02, 0.98, 1.05), Vec3(0.95, 1.1, 0.97), Vec3(1.1, 1.04, 0.92), Vec3(3.1, 2.6, 0.4)});
    vector<Vec3> v0 = {Vec3(0.5, -0.2, 0.3), Vec3(-0.4, 0.1, 0.2), Vec3(0.2, 0.6, -0.1), Vec3(-0.3, 0.2, 0.4)};
    context.setVelocities(v0);
    Vec3 p0 = fluidMomentum(context, force);
    integrator.step(1);
    vector<Vec3> v1 = context.getState(State::Velocities).getVelocities();
    double cellMass = fluidDensity*fluidDx*fluidDx*fluidDx;
    vector<Vec3> known(4);
    for (int i = 0; i < 4; i++)
        known[i] = v0[i] + external[i]*(0.5*fluidDt/masses[i]);
    Vec3 knownFluid = u0 + g*(0.5*fluidDt);
    vector<Vec3> shared = solveCenteredDrag({masses[0], masses[1], masses[2]}, {known[0], known[1], known[2]}, knownFluid, friction, cellMass);
    vector<Vec3> alone = solveCenteredDrag({masses[3]}, {known[3]}, knownFluid, friction, cellMass);
    vector<Vec3> drag = {shared[0], shared[1], shared[2], alone[0]};
    Vec3 total;
    for (int i = 0; i < 4; i++) {
        ASSERT_EQUAL_VEC(v0[i] + (external[i]+drag[i])*(fluidDt/masses[i]), v1[i], getCouplingTolerance(platform, 1e-12));
        total += drag[i];
    }
    // After the collision every node has the momentum j + G: the fluid gains the momentum of the body force and
    // loses that of the coupling forces.
    Vec3 p1 = fluidMomentum(context, force);
    Vec3 expected = p0 + g*(512*cellMass*fluidDt) - total*fluidDt;
    ASSERT_EQUAL_VEC(expected, p1, getCouplingTolerance(platform, 1e-11));
    delete system;
}

/**
 * A solid node is a wall at rest of infinite mass: a particle at a solid node that moves out of the wall (so it
 * is not reflected) is slowed down by the factor (1 - a)/(1 + a), with a = gamma dt/2, and the wall receives the
 * opposite of its coupling force.
 */
void testCenteredWall(Platform& platform) {
    LBMForce* force;
    double friction = 40.0;
    System* system = createCoupledSystem(force, 1, friction, 0.0);
    force->setDragScheme(LBMForce::Centered);
    force->setSolidNodes(wallPlane(8, 8, 8));
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(1.1, 0.20, 1.7)));      // nearest node on the plane j = 0
    Vec3 v0(0.3, 5.0, 0.1);                                            // moving out of the wall, towards j = 1
    context.setVelocities(vector<Vec3>(1, v0));
    integrator.step(1);
    Vec3 v1 = context.getState(State::Velocities).getVelocities()[0];
    double a = 0.5*friction*fluidDt;
    ASSERT_EQUAL_VEC(v0*((1-a)/(1+a)), v1, getCouplingTolerance(platform, 1e-14));
    ASSERT_EQUAL_VEC((v0-v1)*(couplingMass/fluidDt), force->getWallForce(context), getCouplingTolerance(platform, 1e-12));
    delete system;
}

/**
 * The centred drag is stable for any friction: with gamma dt = 3 and four particles at the same node, where the
 * explicit drag diverges, the velocities decay towards that of the whole system, and the total momentum is
 * conserved.
 */
void testCenteredLargeFriction(Platform& platform) {
    LBMForce* force;
    double friction = 300.0;
    System* system = createCoupledSystem(force, 4, friction, 0.0);
    force->setDragScheme(LBMForce::Centered);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions({Vec3(1.02, 0.98, 1.05), Vec3(0.95, 1.1, 0.97), Vec3(1.1, 1.04, 0.92), Vec3(0.98, 1.03, 1.01)});
    vector<Vec3> v0 = {Vec3(1.0, -0.5, 0.3), Vec3(-0.8, 0.4, 0.6), Vec3(0.2, 1.0, -0.9), Vec3(0.5, -0.7, 0.1)};
    context.setVelocities(v0);
    double scale = 0;
    for (Vec3 v : v0)
        scale += couplingMass*sqrt(v.dot(v));
    Vec3 p0 = totalMomentum(context, force);
    integrator.step(200);
    vector<Vec3> v = context.getState(State::Velocities).getVelocities();
    for (int i = 0; i < 4; i++)
        ASSERT(sqrt(v[i].dot(v[i])) < 0.1);
    Vec3 p1 = totalMomentum(context, force);
    ASSERT_EQUAL_TOL(0.0, sqrt((p1-p0).dot(p1-p0))/scale, getCouplingTolerance(platform, 1e-11));
    delete system;
}

/**
 * With the centred drag LBMForce must be the last force of the System, the System must not contain virtual sites,
 * the drag scheme cannot be changed by updateParametersInContext(), and a checkpoint is refused by a Context with
 * the other drag scheme.
 */
void testCenteredRequirements(Platform& platform) {
    auto contextFails = [&](System& system) {
        VerletIntegrator integrator(fluidDt);
        try {
            Context context(system, integrator, platform);
        }
        catch (OpenMMException& e) {
            return true;
        }
        return false;
    };

    // LBMForce not last: refused with the centred drag only.
    LBMForce* force;
    System* system = createCoupledSystem(force, 2, 5.0, 0.0);
    system->addForce(new CustomExternalForce("0"));
    ASSERT(!contextFails(*system));
    force->setDragScheme(LBMForce::Centered);
    ASSERT(contextFails(*system));
    delete system;

    // A virtual site.
    system = createCoupledSystem(force, 3, 5.0, 0.0, {0, 1});
    system->setParticleMass(2, 0.0);
    system->setVirtualSite(2, new TwoParticleAverageSite(0, 1, 0.5, 0.5));
    ASSERT(!contextFails(*system));
    force->setDragScheme(LBMForce::Centered);
    ASSERT(contextFails(*system));
    delete system;

    // Changing the drag scheme in a Context.
    system = createCoupledSystem(force, 2, 5.0, 0.0);
    force->setDragScheme(LBMForce::Centered);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions({Vec3(0.3, 0.4, 0.5), Vec3(1.3, 1.4, 1.5)});
    force->setDragScheme(LBMForce::Explicit);
    bool thrown = false;
    try {
        force->updateParametersInContext(context);
    }
    catch (OpenMMException& e) {
        thrown = true;
    }
    ASSERT(thrown);

    // A checkpoint of the centred drag loaded with the explicit one.
    stringstream checkpoint;
    force->createCheckpoint(context, checkpoint);
    LBMForce* explicitForce;
    System* explicitSystem = createCoupledSystem(explicitForce, 2, 5.0, 0.0);
    VerletIntegrator explicitIntegrator(fluidDt);
    Context explicitContext(*explicitSystem, explicitIntegrator, platform);
    thrown = false;
    try {
        explicitForce->loadCheckpoint(explicitContext, checkpoint);
    }
    catch (OpenMMException& e) {
        thrown = true;
    }
    ASSERT(thrown);
    delete system;
    delete explicitSystem;
}

void runCenteredTests(Platform& platform) {
    testCenteredFirstStep(platform);
    testCenteredSharedNode(platform);
    testCenteredWall(platform);
    testCenteredLargeFriction(platform);
    testCenteredRequirements(platform);
    LBMForce::DragScheme drag = LBMForce::Centered;
    testFullStepKineticEnergy(platform, drag);
    testMomentumConservation(platform, 1.0, drag);
    testMomentumConservation(platform, 0.98, drag);
    testMomentumConservation(platform, 1.02, drag);
    testComoving(platform, drag);
    testPartialCoupling(platform, drag);
    testForceEvaluationsAndSeeds(platform, drag);
    testWallMomentumBalance(platform, drag);
    testRestartWithParticles(platform, drag);
    testCheckpointWithRandomForce(platform, drag);
    testEquipartition(platform, drag);
}
