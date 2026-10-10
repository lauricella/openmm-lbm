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
void testStencilMomentumConservation(Platform& platform, LBMForce::DragScheme drag) {
    for (LBMForce::InterpolationStencil stencil : allStencils) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 4, 10.0, 300.0);
        force->setInterpolationStencil(stencil);
        force->setDragScheme(drag);
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
        force->setDragScheme(drag);
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
 * The centred drag with a stencil solves a linear system for the particles whose stencils overlap (docs/theory.md,
 * section 9).  The forces of the first step are compared with the solution computed here by Gaussian elimination from
 * the fluid state, the positions, the velocities and the other forces: three particles with overlapping stencils, one
 * of them near a wall (the plane j = 0), an isolated one, different masses, a field acting on the particles.
 */
void testCenteredStencilSolve(Platform& platform) {
    int n = 8, numNodes = n*n*n, numParticles = 4;
    double friction = 20.0, h = 0.5;
    double masses[] = {100.0, 300.0, 1000.0, 150.0};
    vector<Vec3> positions = {Vec3(1.13, 0.62, 1.71), Vec3(1.37, 0.81, 1.52), Vec3(1.02, 1.04, 1.93), Vec3(3.1, 2.9, 0.4)};
    vector<Vec3> velocities = {Vec3(0.5, -0.2, 0.3), Vec3(-0.4, 0.1, 0.2), Vec3(0.2, -0.5, 0.1), Vec3(0.3, 0.3, -0.6)};
    vector<double> state(19*numNodes);
    for (int node = 0; node < numNodes; node++) {
        vector<double> one = uniformState(1, 1.0 + 0.002*(node%5), Vec3(0.01*sin(node), 0.01*cos(node), 0.005*sin(3.0*node)));
        for (int q = 0; q < 19; q++)
            state[q*numNodes + node] = one[q];
    }
    vector<int> solid = wallPlane(n, n, n);
    for (LBMForce::InterpolationStencil stencil : allStencils) {
        System system;
        system.setDefaultPeriodicBoxVectors(Vec3(n*fluidDx, 0, 0), Vec3(0, n*fluidDx, 0), Vec3(0, 0, n*fluidDx));
        CustomExternalForce* field = new CustomExternalForce("-3*x+2*y-z");
        system.addForce(field);
        LBMForce* force = new LBMForce();
        force->setGridSize(n, n, n);
        force->setFluidDensity(fluidDensity);
        force->setKinematicViscosity((0.8-0.5)/3.0*fluidDx*fluidDx/fluidDt);
        force->setFluidMomentumRemovalFrequency(0);
        force->setFriction(friction);
        force->setTemperature(0.0);
        force->setDragScheme(LBMForce::Centered);
        force->setInterpolationStencil(stencil);
        force->setSolidNodes(solid);
        force->setForceGroup(1);
        for (int i = 0; i < numParticles; i++) {
            system.addParticle(masses[i]);
            field->addParticle(i);
            force->addParticle(i);
        }
        system.addForce(force);
        VerletIntegrator integrator(fluidDt);
        Context context(system, integrator, platform);
        context.setPositions(positions);
        context.setVelocities(velocities);
        force->setFluidState(context, state);

        // The fluid before the step, in lattice units, and the system of the centred drag.

        int cx[19] = {0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 0,  0,  0,  0, 1, -1, -1,  1};
        int cy[19] = {0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 1, -1,  1, -1, 0,  0,  0,  0};
        int cz[19] = {0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1,  1, -1};
        vector<double> rho(numNodes, 0.0);
        vector<Vec3> u(numNodes);
        vector<char> isSolid(numNodes, 0);
        for (int node : solid)
            isSolid[node] = 1;
        for (int node = 0; node < numNodes; node++) {
            if (isSolid[node])
                continue;
            double r = 1.0;
            Vec3 j;
            for (int q = 0; q < 19; q++) {
                double f = state[q*numNodes + node];
                r += f;
                j += Vec3(cx[q], cy[q], cz[q])*f;
            }
            rho[node] = r;
            u[node] = j*(1.0/r);
        }
        double cellMass = fluidDensity*fluidDx*fluidDx*fluidDx;
        double forceScale = cellMass*fluidDx/(fluidDt*fluidDt), gamma = friction*fluidDt, a = gamma*h;
        vector<vector<double> > weight(numParticles, vector<double>(numNodes, 0.0));
        int w = LBMStencils::width(stencil);
        for (int i = 0; i < numParticles; i++) {
            int index[3][LBMStencils::maxWidth];
            double axis[3][LBMStencils::maxWidth];
            for (int k = 0; k < 3; k++)
                LBMStencils::axisWeights(stencil, positions[i][k]/fluidDx, n, index[k], axis[k]);
            for (int c = 0; c < w; c++)
                for (int b = 0; b < w; b++)
                    for (int aa = 0; aa < w; aa++)
                        weight[i][index[0][aa] + n*(index[1][b] + n*index[2][c])] += axis[0][aa]*axis[1][b]*axis[2][c];
        }
        State before = context.getState(State::Forces, false, 1);        // the field alone (force group 0 is the field)
        vector<double> matrix(numParticles*numParticles);
        vector<Vec3> rhs(numParticles);
        for (int i = 0; i < numParticles; i++) {
            double m = masses[i]/cellMass;
            Vec3 Fc = before.getForces()[i]*(1.0/forceScale);
            Vec3 known = velocities[i]*(fluidDt/fluidDx) + Fc*(h/m);
            Vec3 U;
            for (int node = 0; node < numNodes; node++)
                if (rho[node] > 0)
                    U += u[node]*weight[i][node];
            rhs[i] = (known-U)*(-gamma*m);
            for (int l = 0; l < numParticles; l++) {
                double K = 0;
                for (int node = 0; node < numNodes; node++)
                    if (rho[node] > 0)
                        K += weight[i][node]*weight[l][node]/rho[node];
                matrix[i*numParticles + l] = (i == l ? 1.0+a : 0.0) + a*m*K;
            }
        }
        // Gaussian elimination with partial pivoting, the three components at once.
        vector<int> order(numParticles);
        for (int i = 0; i < numParticles; i++)
            order[i] = i;
        for (int col = 0; col < numParticles; col++) {
            int pivot = col;
            for (int row = col+1; row < numParticles; row++)
                if (fabs(matrix[row*numParticles+col]) > fabs(matrix[pivot*numParticles+col]))
                    pivot = row;
            for (int k = 0; k < numParticles; k++)
                swap(matrix[col*numParticles+k], matrix[pivot*numParticles+k]);
            swap(rhs[col], rhs[pivot]);
            for (int row = col+1; row < numParticles; row++) {
                double factor = matrix[row*numParticles+col]/matrix[col*numParticles+col];
                for (int k = col; k < numParticles; k++)
                    matrix[row*numParticles+k] -= factor*matrix[col*numParticles+k];
                rhs[row] -= rhs[col]*factor;
            }
        }
        vector<Vec3> expected(numParticles);
        for (int row = numParticles-1; row >= 0; row--) {
            Vec3 sum = rhs[row];
            for (int k = row+1; k < numParticles; k++)
                sum -= expected[k]*matrix[row*numParticles+k];
            expected[row] = sum*(1.0/matrix[row*numParticles+row]);
        }

        // The coupling force of the step, from the change of the velocities.

        integrator.step(1);
        State after = context.getState(State::Velocities);
        for (int i = 0; i < numParticles; i++) {
            Vec3 coupling = (after.getVelocities()[i]-velocities[i])*(masses[i]/fluidDt) - before.getForces()[i];
            Vec3 F = expected[i]*forceScale;
            ASSERT_EQUAL_VEC(F, coupling, 1e-10*sqrt(F.dot(F)));
        }
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
 * an error: open faces and, on the GPU platforms, every stencil other than NearestNode.
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
    ASSERT(!fails([](LBMForce* f) {f->setDragScheme(LBMForce::Centered);}));
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

/**
 * With the explicit drag the warning for tau > 1.7 (self-mobility small or negative) is printed with the trilinear and
 * Keys stencils, which give the nearest node at a node, and not with the three-point stencil, whose self-mobility stays
 * positive (docs/theory.md, section 9).  With the fluctuating fluid the warning on the heating of the explicit drag
 * gives friction*dt*m*K/(2 m_c) with the self weight K averaged over a cell: 8/27, 1/8 and (57/70)^3 of the 6.64% of
 * the nearest node for a particle of 100 Da with friction*dt = 0.1.
 */
void testStencilWarnings(Platform& platform) {
    const char* heating[] = {"= 1.968", "= 0.830", "= 3.586"};
    for (int k = 0; k < 3; k++) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 1, 10.0, 300.0);
        force->setFluidFluctuations(true);
        force->setInterpolationStencil(allStencils[k]);
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
        ASSERT(captured.str().find("the explicit drag makes the coupled particles hotter") != string::npos);
        ASSERT(captured.str().find(heating[k]) != string::npos);
        ASSERT(captured.str().find("averaged over a cell") != string::npos);
        delete system;
    }
    for (LBMForce::InterpolationStencil stencil : allStencils) {
        LBMForce* force;
        System* system = createCoupledSystem(force, 1, 1.0, 0.0);
        force->setKinematicViscosity((1.75-0.5)/3.0*fluidDx*fluidDx/fluidDt);
        force->setInterpolationStencil(stencil);
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
        ASSERT(warned == (stencil != LBMForce::ThreePoint));
        delete system;
    }
}

/**
 * With the centred drag and the fluctuating fluid the canonical distribution is stationary with every stencil
 * (docs/theory.md, section 9): two particles of 100 and 1000 Da with overlapping stencils, held at their positions on a
 * 4^3 lattice (the position is set again before every step, the velocity evolves), have at the half step the
 * temperature of the canonical ensemble with the total momentum fixed, m <v^2>/3 = kT (1 - m/M), M being the mass of
 * the particles and the fluid.  The statistical error is about 1.3%; with the explicit drag, or with the fluid without
 * fluctuations, the temperatures differ by 9 to 48%.
 */
void testStencilCanonicalTemperature(Platform& platform) {
    int n = 4, steps = 40000, every = 4;
    double temperature = 300.0, kT = BOLTZ*temperature;
    const LBMForce::InterpolationStencil stencils[] = {LBMForce::NearestNode, LBMForce::Trilinear, LBMForce::ThreePoint,
                                                       LBMForce::Keys};
    vector<double> masses = {100.0, 1000.0};
    vector<Vec3> positions = {Vec3(1.5, 2.0, 2.0)*fluidDx, Vec3(2.7, 2.2, 2.1)*fluidDx};
    double totalMass = masses[0] + masses[1] + n*n*n*fluidDensity*fluidDx*fluidDx*fluidDx;
    for (LBMForce::InterpolationStencil stencil : stencils) {
        System system;
        system.setDefaultPeriodicBoxVectors(Vec3(n*fluidDx, 0, 0), Vec3(0, n*fluidDx, 0), Vec3(0, 0, n*fluidDx));
        LBMForce* force = new LBMForce();
        force->setGridSize(n, n, n);
        force->setFluidDensity(fluidDensity);
        force->setKinematicViscosity((1.1-0.5)/3.0*fluidDx*fluidDx/fluidDt);
        force->setFluidMomentumRemovalFrequency(0);
        force->setFriction(10.0);
        force->setTemperature(temperature);
        force->setFluidFluctuations(true);
        force->setDragScheme(LBMForce::Centered);
        force->setInterpolationStencil(stencil);
        force->setRandomNumberSeed(5);
        for (int i = 0; i < 2; i++) {
            system.addParticle(masses[i]);
            force->addParticle(i);
        }
        system.addForce(force);
        VerletIntegrator integrator(fluidDt);
        Context context(system, integrator, platform);
        context.setPositions(positions);
        for (int step = 0; step < steps/10; step++) {
            context.setPositions(positions);
            integrator.step(1);
        }
        vector<double> sum(2, 0.0);
        for (int step = 0; step < steps; step++) {
            context.setPositions(positions);
            integrator.step(1);
            if (step%every == 0) {
                State state = context.getState(State::Velocities);
                for (int i = 0; i < 2; i++)
                    sum[i] += masses[i]*state.getVelocities()[i].dot(state.getVelocities()[i])/3.0;
            }
        }
        for (int i = 0; i < 2; i++)
            ASSERT_EQUAL_TOL(kT*(1.0-masses[i]/totalMass), sum[i]/(steps/every), 0.05);
    }
}

void runStencilTests(Platform& platform) {
    testStencilErrors(platform);
    if (platform.getName() != "Reference")
        return;
    testStencilWeights();
    testStencilLinearField(platform);
    testStencilAtNode(platform);
    testStencilMomentumConservation(platform, LBMForce::Explicit);
    testStencilMomentumConservation(platform, LBMForce::Centered);
    testCenteredStencilSolve(platform);
    testStencilTranslation(platform);
    testStencilWarnings(platform);
    testStencilCanonicalTemperature(platform);
}
