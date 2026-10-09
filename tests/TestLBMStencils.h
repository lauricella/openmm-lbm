/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

/**
 * Tests of the interpolation stencils of the coupling (docs/theory.md, section 9): the weights of the kernels, a
 * linear velocity field interpolated exactly, the nearest node recovered at a node, the conservation of momentum with
 * overlapping stencils and walls, the invariance under a translation by one node, and the errors of what is not
 * available yet.  Include after TestLBMCoupling.h and call runStencilTests().
 */

#include "internal/LBMStencils.h"
#include <functional>

const LBMForce::InterpolationStencil allStencils[] = {LBMForce::Trilinear, LBMForce::ThreePoint, LBMForce::Keys};

/**
 * Along an axis the weights sum to one and their first moment vanishes at every position; the second moment is
 * 0 to 1/4 for the trilinear kernel, 1/4 to 1/3 for the three-point kernel and zero for Keys; the sum of the squares
 * is 1/2 at every position for the three-point kernel.  The trilinear kernel and Keys interpolate.
 */
void testStencilWeights() {
    for (LBMForce::InterpolationStencil stencil : allStencils) {
        int w = LBMStencils::width(stencil);
        for (int i = 0; i <= 1000; i++) {
            double s = 5.0 + i/1000.0;
            int index[LBMStencils::maxWidth];
            double weight[LBMStencils::maxWidth];
            LBMStencils::axisWeights(stencil, s, 8, index, weight);
            double sum = 0, first = 0, second = 0, squares = 0;
            for (int n = 0; n < w; n++) {
                // index is wrapped into [0, 8); the node lies at index + 8 near s = 5...6.
                double x = (index[n] < 2 ? index[n] + 8 : index[n]);
                sum += weight[n];
                first += weight[n]*(s-x);
                second += weight[n]*(s-x)*(s-x);
                squares += weight[n]*weight[n];
            }
            ASSERT_EQUAL_TOL(1.0, sum, 1e-14);
            ASSERT_EQUAL_TOL(0.0, first, 1e-14);
            if (stencil == LBMForce::Trilinear)
                ASSERT(second > -1e-14 && second < 0.25+1e-14);
            if (stencil == LBMForce::ThreePoint) {
                ASSERT(second > 0.25-1e-14 && second < 1.0/3.0+1e-14);
                ASSERT_EQUAL_TOL(0.5, squares, 1e-14);
            }
            if (stencil == LBMForce::Keys)
                ASSERT_EQUAL_TOL(0.0, second, 1e-13);
        }
        if (stencil != LBMForce::ThreePoint) {
            ASSERT_EQUAL(1.0, LBMStencils::kernel(stencil, 0.0));
            ASSERT_EQUAL(0.0, LBMStencils::kernel(stencil, 1.0));
            ASSERT_EQUAL(0.0, LBMStencils::kernel(stencil, -2.0));
        }
        else
            ASSERT_EQUAL_TOL(2.0/3.0, LBMStencils::kernel(stencil, 0.0), 1e-15);
    }
}

/**
 * Fluid at equilibrium with a velocity and a density that vary linearly in space (away from the periodic wrap): in the
 * first step a particle at rest feels F = gamma m u(X) with u(X) the linear field at the particle, whatever the
 * density, with every stencil.
 */
void testStencilLinearField(Platform& platform) {
    int n = 8;
    Vec3 u0(0.01, -0.02, 0.015), x0(2.0, 2.0, 2.0);
    Vec3 grad[3] = {Vec3(0.002, -0.001, 0.0005), Vec3(0.0015, 0.001, -0.002), Vec3(-0.001, 0.0005, 0.002)};
    auto linearVelocity = [&](Vec3 x) {
        Vec3 d = x-x0;
        return u0 + grad[0]*d[0] + grad[1]*d[1] + grad[2]*d[2];
    };
    vector<double> state(19*n*n*n);
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++) {
                Vec3 x(i*fluidDx, j*fluidDx, k*fluidDx);
                double rho = 1.0 + 0.01*(i-4) - 0.005*(j-4) + 0.008*(k-4);
                vector<double> node = uniformState(1, rho, linearVelocity(x)*(fluidDt/fluidDx));
                for (int q = 0; q < 19; q++)
                    state[q*n*n*n + i + n*(j + n*k)] = node[q];
            }
    double friction = 5.0;
    Vec3 position(2.13, 1.87, 2.21);
    for (LBMForce::InterpolationStencil stencil : allStencils) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 1, friction, 0.0);
        force->setInterpolationStencil(stencil);
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        context.setPositions(vector<Vec3>(1, position));
        force->setFluidState(context, state);
        integrator.step(1);
        Vec3 v1 = context.getState(State::Velocities).getVelocities()[0];
        ASSERT_EQUAL_VEC(linearVelocity(position)*(friction*fluidDt), v1, getCouplingTolerance(platform, 1e-12));
        delete system;
    }
}

/**
 * The trilinear kernel and Keys weigh only the node on which a particle lies: there the first step is the same, bit
 * for bit, as with the nearest node, for the particle and for the fluid.
 */
void testStencilAtNode(Platform& platform) {
    int n = 8;
    vector<double> state(19*n*n*n);
    for (int node = 0; node < n*n*n; node++) {
        vector<double> one = uniformState(1, 1.0 + 0.001*(node%7), Vec3(0.01*sin(node), 0.01*cos(node), 0.005*sin(3.0*node)));
        for (int q = 0; q < 19; q++)
            state[q*n*n*n + node] = one[q];
    }
    vector<double> referenceState;
    Vec3 referenceVelocity;
    for (LBMForce::InterpolationStencil stencil : {LBMForce::NearestNode, LBMForce::Trilinear, LBMForce::Keys}) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 1, 5.0, 0.0);
        force->setInterpolationStencil(stencil);
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        context.setPositions(vector<Vec3>(1, Vec3(1.0, 1.5, 2.0)));
        context.setVelocities(vector<Vec3>(1, Vec3(0.3, -0.2, 0.1)));
        force->setFluidState(context, state);
        integrator.step(1);
        Vec3 v1 = context.getState(State::Velocities).getVelocities()[0];
        vector<double> fluid;
        force->getFluidState(context, fluid);
        if (stencil == LBMForce::NearestNode) {
            referenceState = fluid;
            referenceVelocity = v1;
        }
        else {
            ASSERT_EQUAL_VEC(referenceVelocity, v1, 0.0);
            for (int i = 0; i < (int) fluid.size(); i++)
                ASSERT_EQUAL(referenceState[i], fluid[i]);
        }
        delete system;
    }
}

/**
 * With every stencil the total momentum of particles and fluid is conserved, with the random force, overlapping
 * stencils and a particle crossing the periodic boundary; with walls, the momentum given to the solid nodes of the
 * stencils (a wall at rest) closes the balance.
 */
void testStencilMomentumConservation(Platform& platform) {
    for (LBMForce::InterpolationStencil stencil : allStencils) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 4, 10.0, 300.0);
        force->setInterpolationStencil(stencil);
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        vector<Vec3> positions = {Vec3(1.02, 2.01, 0.98), Vec3(1.27, 1.79, 1.13), Vec3(3.98, 0.02, 3.96), Vec3(2.3, 1.1, 0.6)};
        vector<Vec3> velocities = {Vec3(0.5, -0.2, 0.3), Vec3(-0.4, 0.1, 0.2), Vec3(2.0, -1.5, 1.0), Vec3(0.0, 0.3, -0.6)};
        context.setPositions(positions);
        context.setVelocities(velocities);
        force->setFluidState(context, uniformState(512, 1.01, Vec3()));
        double scale = 0;
        for (Vec3 v : velocities)
            scale += couplingMass*sqrt(v.dot(v));
        Vec3 p0 = totalMomentum(context, force);
        integrator.step(50);
        Vec3 p1 = totalMomentum(context, force);
        ASSERT_EQUAL_TOL(0.0, sqrt((p1-p0).dot(p1-p0))/scale, getCouplingTolerance(platform, 1e-11));
        ASSERT(context.getState(State::Positions).getPositions()[2][0] > 4.0);      // crossed the boundary
        delete system;
    }
    for (LBMForce::InterpolationStencil stencil : allStencils) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 3, 10.0, 300.0);
        force->setInterpolationStencil(stencil);
        force->setSolidNodes(wallPlane(8, 8, 8));
        VerletIntegrator integrator(fluidDt);
        Context context(*system, integrator, platform);
        // Particles near the wall j = 0, whose stencils cover solid nodes.
        context.setPositions({Vec3(1.1, 0.40, 1.7), Vec3(2.3, 0.61, 0.6), Vec3(0.7, 3.55, 2.9)});
        context.setVelocities({Vec3(0.5, -1.0, 0.2), Vec3(0.4, 0.3, -0.2), Vec3(-0.6, 0.2, 0.5)});
        // The momentum of the fluid from its fields, which leave out the solid nodes (as in testWallMomentumBalance()).
        auto momentum = [&](double& scale) {
            Vec3 p;
            scale = 0;
            State state = context.getState(State::Velocities);
            for (Vec3 v : state.getVelocities()) {
                p += v*couplingMass;
                scale += couplingMass*sqrt(v.dot(v));
            }
            vector<double> density;
            vector<Vec3> velocity;
            force->getFluidFields(context, density, velocity);
            for (int node = 0; node < (int) density.size(); node++)
                p += velocity[node]*(density[node]*fluidDx*fluidDx*fluidDx);
            return p;
        };
        double scale;
        Vec3 p0 = momentum(scale), wall;
        for (int step = 0; step < 50; step++) {
            integrator.step(1);
            wall += force->getWallForce(context)*fluidDt;
        }
        Vec3 balance = momentum(scale) + wall - p0;
        ASSERT(wall.dot(wall) > 0);
        ASSERT_EQUAL_TOL(0.0, sqrt(balance.dot(balance))/scale, getCouplingTolerance(platform, 1e-11));
        delete system;
    }
}

/**
 * Moving particles, fluid and solid nodes by one node along each axis moves the whole run by one node.
 */
void testStencilTranslation(Platform& platform) {
    int n = 8;
    vector<double> state(19*n*n*n);
    for (int node = 0; node < n*n*n; node++) {
        vector<double> one = uniformState(1, 1.0, Vec3(0.01*sin(node), 0.01*cos(node), 0.005*sin(3.0*node)));
        for (int q = 0; q < 19; q++)
            state[q*n*n*n + node] = one[q];
    }
    auto shifted = [&](const vector<double>& s) {
        vector<double> out(s.size());
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    for (int q = 0; q < 19; q++)
                        out[q*n*n*n + (i+1)%n + n*((j+1)%n + n*((k+1)%n))] = s[q*n*n*n + i + n*(j + n*k)];
        return out;
    };
    vector<Vec3> positions = {Vec3(1.13, 2.07, 0.91), Vec3(1.31, 1.88, 1.17), Vec3(2.6, 3.2, 0.4)};
    vector<Vec3> velocities = {Vec3(0.5, -0.2, 0.3), Vec3(-0.4, 0.1, 0.2), Vec3(0.2, -0.5, 0.1)};
    for (LBMForce::InterpolationStencil stencil : allStencils) {
        vector<vector<double> > fluid(2);
        vector<vector<Vec3> > x(2);
        for (int shift = 0; shift < 2; shift++) {
            LBMForce* force;
            System* system = createCoupledSystem(force, 3, 10.0, 0.0);
            force->setInterpolationStencil(stencil);
            VerletIntegrator integrator(fluidDt);
            Context context(*system, integrator, platform);
            vector<Vec3> p = positions;
            for (Vec3& r : p)
                r += Vec3(fluidDx, fluidDx, fluidDx)*shift;
            context.setPositions(p);
            context.setVelocities(velocities);
            force->setFluidState(context, shift ? shifted(state) : state);
            integrator.step(30);
            force->getFluidState(context, fluid[shift]);
            x[shift] = context.getState(State::Positions).getPositions();
            delete system;
        }
        vector<double> moved = shifted(fluid[0]);
        for (int i = 0; i < (int) moved.size(); i++)
            ASSERT_EQUAL_TOL(moved[i], fluid[1][i], 1e-12);
        for (int i = 0; i < 3; i++)
            ASSERT_EQUAL_VEC(x[0][i] + Vec3(fluidDx, fluidDx, fluidDx), x[1][i], 1e-12);
    }
}

/**
 * The stencil is fixed when a Context is created and must match the checkpoints; what is not available yet stops with
 * an error: the centred drag, open faces and, on the GPU platforms, every stencil other than NearestNode.
 */
void testStencilErrors(Platform& platform) {
    auto fails = [&](function<void(LBMForce*)> setup) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 1, 5.0, 0.0);
        force->setInterpolationStencil(LBMForce::Trilinear);
        setup(force);
        VerletIntegrator integrator(fluidDt);
        bool thrown = false;
        try {
            Context context(*system, integrator, platform);
        }
        catch (OpenMMException& e) {
            thrown = true;
        }
        delete system;
        return thrown;
    };
    if (platform.getName() != "Reference") {
        ASSERT(fails([](LBMForce* f) {}));
        return;
    }
    ASSERT(!fails([](LBMForce* f) {}));
    ASSERT(fails([](LBMForce* f) {f->setDragScheme(LBMForce::Centered);}));
    ASSERT(fails([](LBMForce* f) {
        f->setFaceBoundary(LBMForce::XMin, LBMForce::Velocity);
        f->setFaceBoundary(LBMForce::XMax, LBMForce::Velocity);
    }));
    ASSERT(fails([](LBMForce* f) {f->setInterpolationStencil((LBMForce::InterpolationStencil) 7);}));

    LBMForce* force;
    System* system = createCoupledSystem(force, 1, 5.0, 0.0);
    force->setInterpolationStencil(LBMForce::ThreePoint);
    VerletIntegrator integrator(fluidDt);
    Context context(*system, integrator, platform);
    context.setPositions(vector<Vec3>(1, Vec3(0.3, 0.4, 0.5)));
    force->setInterpolationStencil(LBMForce::Keys);
    bool thrown = false;
    try {
        force->updateParametersInContext(context);
    }
    catch (OpenMMException& e) {
        thrown = true;
    }
    ASSERT(thrown);
    force->setInterpolationStencil(LBMForce::ThreePoint);
    stringstream checkpoint;
    force->createCheckpoint(context, checkpoint);
    LBMForce* force2;
    System* system2 = createCoupledSystem(force2, 1, 5.0, 0.0);
    VerletIntegrator integrator2(fluidDt);
    Context context2(*system2, integrator2, platform);
    thrown = false;
    try {
        force2->loadCheckpoint(context2, checkpoint);
    }
    catch (OpenMMException& e) {
        thrown = (string(e.what()).find("interpolation stencil") != string::npos);
    }
    ASSERT(thrown);
    delete system;
    delete system2;
}

void runStencilTests(Platform& platform) {
    testStencilErrors(platform);
    if (platform.getName() != "Reference")
        return;
    testStencilWeights();
    testStencilLinearField(platform);
    testStencilAtNode(platform);
    testStencilMomentumConservation(platform);
    testStencilTranslation(platform);
}
