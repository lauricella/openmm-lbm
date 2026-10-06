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
#include "openmm/internal/OSRngSeed.h"
#include "openmm/reference/ReferencePlatform.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

using namespace LBMPlugin;
using namespace OpenMM;
using namespace std;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->positions;
}

static vector<Vec3>& extractVelocities(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->velocities;
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->forces;
}

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

    // Coupled particles: masses in units of the mass of a cell, m_c = rho0 dx^3.  No coupling force exists
    // before the first step.

    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    int numParticles = lattice.particles.size();
    particleMass.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        particleMass[i] = system.getParticleMass(lattice.particles[i])/cellMass;
    particleForces.assign(numParticles, Vec3());
    reaction.resize(3*numNodes);
    int seed = lattice.randomNumberSeed;
    if (seed == 0)
        seed = osrngseed();
    OpenMM_SFMT::init_gen_rand((uint32_t) seed, sfmt);
    hasStoredGaussian = false;
}

void ReferenceCalcLBMForceKernel::beginStep(ContextImpl& context) {
    // The removal of the fluid momentum and the Mach number check are timed by the step count of the
    // Context, which checkpoints save and restore: a run restarted from a checkpoint repeats them at the
    // same steps as an uninterrupted run.
    stepIndex = context.getStepCount();
    stepPending = true;

    // A coupled particle whose nearest node is solid and that moves into the wall, v.n > 0 with n the normal
    // pointing into the wall, has every component of its velocity reversed, as for a no-slip wall.  A particle
    // that already moves out of the wall keeps its velocity.  This is done here, at the start of the step,
    // where OpenMM's AndersenThermostat also changes velocities, so that the coupling and the integrator see
    // the new values.
    if (!isFluid.empty()) {
        vector<Vec3>& positions = extractPositions(context);
        vector<Vec3>& velocities = extractVelocities(context);
        for (int particle : lattice.particles)
            if (!isFluid[nearestNode(positions[particle])] && velocities[particle].dot(wallNormal(positions[particle])) > 0)
                velocities[particle] = -velocities[particle];
    }
}

double ReferenceCalcLBMForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // The fluid advances, and the coupling forces are computed, once per integration step: on the first force
    // evaluation after beginStep().  Every force evaluation applies the coupling forces of the last step, so
    // that evaluations outside the integration steps (getState(), for example) neither advance the fluid nor
    // draw new random forces.  The coupling is dissipative: its energy is zero.
    if (stepPending) {
        stepPending = false;
        advanceFluid(context);
    }
    if (includeForces) {
        vector<Vec3>& forces = extractForces(context);
        for (int i = 0; i < (int) lattice.particles.size(); i++)
            forces[lattice.particles[i]] += particleForces[i];
    }
    return 0.0;
}

void ReferenceCalcLBMForceKernel::advanceFluid(ContextImpl& context) {
    computeMoments();
    if (lattice.momentumRemovalFrequency > 0 && stepIndex%lattice.momentumRemovalFrequency == 0)
        removeFluidMomentum();
    if (!lattice.particles.empty())
        coupleParticles(context);
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

void ReferenceCalcLBMForceKernel::coupleParticles(ContextImpl& context) {
    // Explicit Euler-Maruyama coupling at the nearest node, in lattice units (time step 1):
    //   F = -gamma m (v - j/rho) + sqrt(2 gamma m kT) xi,
    // with xi three independent N(0,1) numbers.  v is the velocity of OpenMM's leapfrog, v(t - dt/2), and j is
    // the momentum of the fluid before the force of this step, j(t - dt/2).  The particle receives F and the
    // node -F: the reactions of the particles of a node are summed in particle order, then added to the force
    // density of the node.  A solid node has rho = 0 and is at rest; the reaction it receives leaves the fluid.
    int numNodes = lattice.getNumNodes();
    vector<Vec3>& positions = extractPositions(context);
    vector<Vec3>& velocities = extractVelocities(context);
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    double gamma = lattice.friction*lattice.dt;
    double kT = lattice.kT*lattice.dt*lattice.dt/(cellMass*lattice.dx*lattice.dx);
    double velocityScale = lattice.getVelocityScale();
    double forceScale = cellMass*lattice.dx/(lattice.dt*lattice.dt);
    fill(reaction.begin(), reaction.end(), 0.0);
    for (int i = 0; i < (int) lattice.particles.size(); i++) {
        int particle = lattice.particles[i];
        int node = nearestNode(positions[particle]);
        Vec3 u;
        if (rho[node] > 0)
            u = Vec3(momentum[3*node], momentum[3*node+1], momentum[3*node+2])*(1.0/rho[node]);
        Vec3 v = velocities[particle]*(1.0/velocityScale);
        double m = particleMass[i];
        Vec3 f = (v-u)*(-gamma*m);
        if (kT > 0 && gamma > 0) {
            double sigma = sqrt(2.0*gamma*m*kT);
            double xi0 = getGaussianRandom(), xi1 = getGaussianRandom(), xi2 = getGaussianRandom();
            f += Vec3(xi0, xi1, xi2)*sigma;
        }
        particleForces[i] = f*forceScale;
        for (int k = 0; k < 3; k++)
            reaction[3*node+k] -= f[k];
    }
    for (int k = 0; k < 3*numNodes; k++)
        forceDensity[k] += reaction[k];
}

int ReferenceCalcLBMForceKernel::nearestNode(const Vec3& position) const {
    // After wrapping the position into the box, node i owns the interval [(i - 1/2) dx, (i + 1/2) dx).
    int size[3] = {lattice.nx, lattice.ny, lattice.nz};
    int index[3];
    for (int k = 0; k < 3; k++) {
        double s = position[k]/lattice.dx;
        s -= floor(s/size[k])*size[k];
        index[k] = ((int) floor(s + 0.5))%size[k];
    }
    return index[0] + lattice.nx*(index[1] + lattice.ny*index[2]);
}

Vec3 ReferenceCalcLBMForceKernel::wallNormal(const Vec3& position) const {
    // Gradient of the solid indicator (1 at solid nodes, 0 at fluid nodes), interpolated trilinearly between
    // the eight nodes of the lattice cell that contains the position.  It points from the fluid into the wall,
    // also for a wall one node thick, whose side is given by the cell of the position.  It is zero if the
    // eight nodes are all solid or all fluid.
    int size[3] = {lattice.nx, lattice.ny, lattice.nz};
    int index[3][2];
    double weight[3][2];
    for (int k = 0; k < 3; k++) {
        double s = position[k]/lattice.dx;
        s -= floor(s/size[k])*size[k];
        double lower = floor(s);
        index[k][0] = ((int) lower)%size[k];
        index[k][1] = (index[k][0]+1)%size[k];
        weight[k][1] = s-lower;
        weight[k][0] = 1.0-weight[k][1];
    }
    double solid[2][2][2];
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++)
            for (int c = 0; c < 2; c++)
                solid[a][b][c] = (isFluid[index[0][a] + lattice.nx*(index[1][b] + lattice.ny*index[2][c])] ? 0.0 : 1.0);
    Vec3 gradient;
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++) {
            gradient[0] += weight[1][a]*weight[2][b]*(solid[1][a][b]-solid[0][a][b]);
            gradient[1] += weight[0][a]*weight[2][b]*(solid[a][1][b]-solid[a][0][b]);
            gradient[2] += weight[0][a]*weight[1][b]*(solid[a][b][1]-solid[a][b][0]);
        }
    return gradient;
}

double ReferenceCalcLBMForceKernel::getGaussianRandom() {
    // Box-Muller transform of two uniform numbers, the first in (0, 1] so that its logarithm is finite.
    if (hasStoredGaussian) {
        hasStoredGaussian = false;
        return storedGaussian;
    }
    const double twoPi = 6.283185307179586476925287;
    double r = sqrt(-2.0*log(1.0 - OpenMM_SFMT::genrand_real2(sfmt)));
    double angle = twoPi*OpenMM_SFMT::genrand_real2(sfmt);
    storedGaussian = r*sin(angle);
    hasStoredGaussian = true;
    return r*cos(angle);
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
