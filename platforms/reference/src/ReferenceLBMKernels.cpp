/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "ReferenceLBMKernels.h"
#include "internal/D3Q19.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <iostream>
#include <sstream>

using namespace LBMPlugin;
using namespace OpenMM;
using namespace std;

void ReferenceCalcLBMForceKernel::initialize(const System& system, const LBMForce& force, const LBMLatticeParameters& lattice) {
    this->lattice = lattice;

    // The fluid starts at equilibrium, with lattice density 1 and the initial velocity.

    int numNodes = lattice.getNumNodes();
    populations.resize(D3Q19::numVelocities*numNodes);
    double feq[D3Q19::numVelocities];
    D3Q19::equilibrium(1.0, lattice.initialVelocity[0], lattice.initialVelocity[1], lattice.initialVelocity[2], feq);
    for (int q = 0; q < D3Q19::numVelocities; q++)
        for (int node = 0; node < numNodes; node++)
            populations[q*numNodes+node] = feq[q];
    rho.resize(numNodes);
    momentum.resize(3*numNodes);
    piNeq.resize(6*numNodes);
    forceDensity.resize(3*numNodes);

    // Solid nodes hold no fluid: their populations start at zero.

    isFluid.clear();
    if (!lattice.solidNodes.empty()) {
        isFluid.resize(numNodes, 1);
        for (int node : lattice.solidNodes) {
            isFluid[node] = 0;
            for (int q = 0; q < D3Q19::numVelocities; q++)
                populations[q*numNodes+node] = 0.0;
        }
    }
}

void ReferenceCalcLBMForceKernel::beginStep(ContextImpl& context) {
    // The removal of the fluid momentum and the Mach number check are timed by the step count of the
    // Context, which checkpoints save and restore: a run restarted from a checkpoint repeats them at the
    // same steps as an uninterrupted run.
    stepIndex = context.getStepCount();
    stepPending = true;
}

double ReferenceCalcLBMForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // The fluid advances once per integration step, on the first force evaluation after beginStep().
    // The particle-fluid coupling is not implemented yet: no force is applied to the particles.
    if (stepPending) {
        stepPending = false;
        advanceFluid();
    }
    return 0.0;
}

void ReferenceCalcLBMForceKernel::advanceFluid() {
    computeMoments();
    if (lattice.momentumRemovalFrequency > 0 && stepIndex%lattice.momentumRemovalFrequency == 0)
        removeFluidMomentum();
    collideAndStream();
    if (!isFluid.empty())
        bounceBack();
    stepIndex++;
    if (lattice.machCheckFrequency > 0 && stepIndex%lattice.machCheckFrequency == 0)
        checkMachNumber();
}

void ReferenceCalcLBMForceKernel::checkMachNumber() {
    double mach = computeMachNumber();
    if (mach > lattice.machNumberLimit) {
        stringstream msg;
        msg << "LBMForce: the Mach number of the fluid is " << mach << " after " << stepIndex << " lattice steps, "
            << "above the limit " << lattice.machNumberLimit << " (docs/theory.md, section 6). Reduce the forces on the "
            << "fluid, the time step or the friction.";
        throw OpenMMException(msg.str());
    }
#ifdef LBM_DEBUG
    if (mach > 0.1 && !machWarningPrinted) {
        cerr << "Warning: LBMForce: the Mach number of the fluid is " << mach << " after " << stepIndex
             << " lattice steps; the accuracy of the model degrades above 0.1." << endl;
        machWarningPrinted = true;
    }
#endif
}

void ReferenceCalcLBMForceKernel::computeMoments() {
    int numNodes = lattice.getNumNodes();
    double f[D3Q19::numVelocities];
    for (int node = 0; node < numNodes; node++) {
        if (!isFluid.empty() && !isFluid[node]) {
            // No fluid: zero density and momentum, so the node does not enter the momentum removal.
            rho[node] = 0;
            for (int k = 0; k < 3; k++)
                momentum[3*node+k] = forceDensity[3*node+k] = 0;
            for (int k = 0; k < 6; k++)
                piNeq[6*node+k] = 0;
            continue;
        }
        double r = 0, jx = 0, jy = 0, jz = 0;
        for (int q = 0; q < D3Q19::numVelocities; q++) {
            f[q] = populations[q*numNodes+node];
            r += f[q];
            jx += D3Q19::cx[q]*f[q];
            jy += D3Q19::cy[q]*f[q];
            jz += D3Q19::cz[q]*f[q];
        }
        rho[node] = r;
        momentum[3*node] = jx;
        momentum[3*node+1] = jy;
        momentum[3*node+2] = jz;
        D3Q19::nonEquilibriumMoment(f, r, jx, jy, jz, &piNeq[6*node]);

        // A body acceleration g acts on the fluid as the force density rho*g.

        forceDensity[3*node] = r*lattice.bodyAcceleration[0];
        forceDensity[3*node+1] = r*lattice.bodyAcceleration[1];
        forceDensity[3*node+2] = r*lattice.bodyAcceleration[2];
    }
}

void ReferenceCalcLBMForceKernel::removeFluidMomentum() {
    // Subtract the velocity of the centre of mass of the fluid, u_cm = sum(j)/sum(rho), from every node:
    // j <- j - rho*u_cm.  The non-equilibrium moments are left as they are.
    int numNodes = lattice.getNumNodes();
    double mass = 0, px = 0, py = 0, pz = 0;
    for (int node = 0; node < numNodes; node++) {
        mass += rho[node];
        px += momentum[3*node];
        py += momentum[3*node+1];
        pz += momentum[3*node+2];
    }
    double ux = px/mass, uy = py/mass, uz = pz/mass;
    for (int node = 0; node < numNodes; node++) {
        momentum[3*node] -= rho[node]*ux;
        momentum[3*node+1] -= rho[node]*uy;
        momentum[3*node+2] -= rho[node]*uz;
    }
}

void ReferenceCalcLBMForceKernel::collideAndStream() {
    // Regularized collision with Guo forcing,
    //   f_q(x + c_q) = feq_q(rho, u) + (1 - omega) fneq_q(Pi_neq) + S_q(u, F)/2,  u = (j + F/2)/rho,
    // written in push form: each population is computed from the moments of its own node only.
    int nx = lattice.nx, ny = lattice.ny, nz = lattice.nz;
    int numNodes = lattice.getNumNodes();
    double omega = lattice.omega;
    double feq[D3Q19::numVelocities], fneq[D3Q19::numVelocities], s[D3Q19::numVelocities];
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                int node = i + nx*(j + ny*k);
                if (!isFluid.empty() && !isFluid[node])
                    continue;
                double r = rho[node];
                const double* F = &forceDensity[3*node];
                double ux = (momentum[3*node] + 0.5*F[0])/r;
                double uy = (momentum[3*node+1] + 0.5*F[1])/r;
                double uz = (momentum[3*node+2] + 0.5*F[2])/r;
                D3Q19::equilibrium(r, ux, uy, uz, feq);
                D3Q19::regularizedNonEquilibrium(&piNeq[6*node], fneq);
                D3Q19::guoForcing(ux, uy, uz, F[0], F[1], F[2], s);
                for (int q = 0; q < D3Q19::numVelocities; q++) {
                    int di = (i + D3Q19::cx[q] + nx)%nx;
                    int dj = (j + D3Q19::cy[q] + ny)%ny;
                    int dk = (k + D3Q19::cz[q] + nz)%nz;
                    populations[q*numNodes + di + nx*(dj + ny*dk)] = feq[q] + (1.0-omega)*fneq[q] + 0.5*s[q];
                }
            }
}

void ReferenceCalcLBMForceKernel::bounceBack() {
    // Halfway bounce-back: the population that streamed from the fluid node s + c_q into the solid node s,
    // moving along -c_q, returns to s + c_q moving along c_q.  The wall lies halfway between the two nodes.
    int nx = lattice.nx, ny = lattice.ny, nz = lattice.nz;
    int numNodes = lattice.getNumNodes();
    for (int node : lattice.solidNodes) {
        int i = node%nx, j = (node/nx)%ny, k = node/(nx*ny);
        for (int q = 1; q < D3Q19::numVelocities; q++) {
            int di = (i + D3Q19::cx[q] + nx)%nx;
            int dj = (j + D3Q19::cy[q] + ny)%ny;
            int dk = (k + D3Q19::cz[q] + nz)%nz;
            populations[q*numNodes + di + nx*(dj + ny*dk)] = populations[D3Q19::opposite[q]*numNodes + node];
        }
    }
}

void ReferenceCalcLBMForceKernel::copyParametersToContext(ContextImpl& context, const LBMLatticeParameters& lattice) {
    this->lattice = lattice;
}

void ReferenceCalcLBMForceKernel::getFluidFields(ContextImpl& context, vector<double>& density, vector<Vec3>& velocity) {
    // The velocity of the forced fluid is u = (j + F/2)/rho, with F = rho*g from the body acceleration.
    int numNodes = lattice.getNumNodes();
    double velocityScale = lattice.getVelocityScale();
    density.resize(numNodes);
    velocity.resize(numNodes);
    for (int node = 0; node < numNodes; node++) {
        if (!isFluid.empty() && !isFluid[node]) {
            density[node] = 0;
            velocity[node] = Vec3();
            continue;
        }
        double r = 0, jx = 0, jy = 0, jz = 0;
        for (int q = 0; q < D3Q19::numVelocities; q++) {
            double f = populations[q*numNodes+node];
            r += f;
            jx += D3Q19::cx[q]*f;
            jy += D3Q19::cy[q]*f;
            jz += D3Q19::cz[q]*f;
        }
        density[node] = r*lattice.density;
        velocity[node] = (Vec3(jx, jy, jz)*(1.0/r) + lattice.bodyAcceleration*0.5)*velocityScale;
    }
}

double ReferenceCalcLBMForceKernel::getFluidMachNumber(ContextImpl& context) {
    return computeMachNumber();
}

double ReferenceCalcLBMForceKernel::computeMachNumber() const {
    // Ma = max |j/rho|/c_s, with c_s^2 = 1/3.
    int numNodes = lattice.getNumNodes();
    double maxSpeed2 = 0;
    for (int node = 0; node < numNodes; node++) {
        if (!isFluid.empty() && !isFluid[node])
            continue;
        double r = 0, jx = 0, jy = 0, jz = 0;
        for (int q = 0; q < D3Q19::numVelocities; q++) {
            double f = populations[q*numNodes+node];
            r += f;
            jx += D3Q19::cx[q]*f;
            jy += D3Q19::cy[q]*f;
            jz += D3Q19::cz[q]*f;
        }
        maxSpeed2 = max(maxSpeed2, (jx*jx + jy*jy + jz*jz)/(r*r));
    }
    return sqrt(3.0*maxSpeed2);
}

void ReferenceCalcLBMForceKernel::getFluidState(ContextImpl& context, vector<double>& state) {
    state = populations;
}

void ReferenceCalcLBMForceKernel::setFluidState(ContextImpl& context, const vector<double>& state) {
    if (state.size() != populations.size())
        throw OpenMMException("LBMForce: setFluidState() was called with a state of the wrong size");
    populations = state;
}
