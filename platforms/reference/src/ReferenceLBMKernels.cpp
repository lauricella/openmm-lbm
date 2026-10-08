/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "ReferenceLBMKernels.h"
#include "internal/D3Q19.h"
#include "internal/LBMBoundaries.h"
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

    // The fluid starts at equilibrium, with lattice density 1 and the initial velocity.  The populations are
    // stored as deviations from the rest equilibrium, f_q - w_q.

    int numNodes = lattice.getNumNodes();
    populations.resize(D3Q19::numVelocities*numNodes);
    double dfeq[D3Q19::numVelocities];
    D3Q19::equilibriumDeviation(0.0, lattice.initialVelocity[0], lattice.initialVelocity[1], lattice.initialVelocity[2], dfeq);
    for (int q = 0; q < D3Q19::numVelocities; q++)
        for (int node = 0; node < numNodes; node++)
            populations[q*numNodes+node] = dfeq[q];
    rho.resize(numNodes);
    densityDeviation.resize(numNodes);
    momentum.resize(3*numNodes);
    piNeq.resize(6*numNodes);
    forceDensity.resize(3*numNodes);

    // Solid nodes hold no fluid: their populations start at zero, that is at the deviation -w_q.

    isFluid.clear();
    if (!lattice.solidNodes.empty()) {
        isFluid.resize(numNodes, 1);
        for (int node : lattice.solidNodes) {
            isFluid[node] = 0;
            for (int q = 0; q < D3Q19::numVelocities; q++)
                populations[q*numNodes+node] = -D3Q19::w[q];
        }
    }
    LBMBoundaries::findWallLinks(lattice, isFluid, wallNodes, wallLinks);

    // Boundary nodes (docs/theory.md, section 1): with regularized walls the fluid nodes next to the solid nodes, and
    // the fluid nodes on the open faces of the box.  For each of them, the directions q whose source node x - c_q is
    // solid, or lies beyond an open face, are unknown after the streaming.  A node next to a solid node has the wall
    // at rest; on the faces, the first Velocity face of the node in the order XMin ... ZMax gives the velocity, and
    // otherwise its Density faces give the density.  They are fluid nodes like the others in everything else.

    LBMBoundaries boundaries;
    boundaries.find(lattice, isFluid);
    boundaryNodes = boundaries.nodes;
    unknownDirections = boundaries.unknown;
    solidDirections = boundaries.solid;
    boundaryKind = boundaries.kind;
    boundaryFace = boundaries.face;

    // Coupled particles: masses in units of the mass of a cell, m_c = rho0 dx^3.  The coupling force is zero
    // before the first step.

    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    int numParticles = lattice.particles.size();
    particleMass.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        particleMass[i] = system.getParticleMass(lattice.particles[i])/cellMass;
    particleForces.assign(numParticles, Vec3());
    noise.assign(numParticles, Vec3());
    noiseDrawn = false;
    reaction.resize(3*numNodes);
    wallMomentum = Vec3();
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
    // the new values.  The wall receives the momentum 2 m v taken from the particle.
    wallMomentum = Vec3();
    if (!isFluid.empty()) {
        vector<Vec3>& positions = extractPositions(context);
        vector<Vec3>& velocities = extractVelocities(context);
        double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
        for (int i = 0; i < (int) lattice.particles.size(); i++) {
            int particle = lattice.particles[i];
            Vec3 v = velocities[particle];
            if (!isFluid[nearestNode(positions[particle])] && v.dot(wallNormal(positions[particle])) > 0) {
                wallMomentum += v*(2.0*particleMass[i]*cellMass);
                velocities[particle] = -v;
            }
        }
    }
}

double ReferenceCalcLBMForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // The fluid advances once per integration step, on the first force evaluation after beginStep(), which also
    // computes the coupling forces of the step.  The force evaluations between steps (getState(), for example)
    // do not change the fluid: they return the coupling force of the next step, as OpenMM does for every force,
    // whose value at time t is the one the next step uses.  With VerletIntegrator the kinetic energy that OpenMM
    // computes from v(t - dt/2) + dt F(t)/(2m) is then that of the full step.  The random numbers of a step are
    // drawn once, by the first evaluation that needs them, so extra evaluations do not change the run.  Before
    // the first step of the Context the coupling force is zero.  The coupling is dissipative: its energy is zero.
    // The GPU platforms repeat the force evaluations of a step when they enlarge their neighbor list; a repeated
    // evaluation comes before the integrator increments the step count, and it uses the forces of the step again.
    // The Reference platform never repeats them, but follows the same rule.
    bool repeatedStep = (stepForcesCurrent && context.getStepCount() == stepIndex-1);
    if (stepPending) {
        stepPending = false;
        advanceFluid(context);
    }
    else if (includeForces && !lattice.particles.empty() && context.getStepCount() > 0 && !repeatedStep)
        computeNextStepForces(context);
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
        couple(context, true);
    collideAndStream();
    if (!wallNodes.empty()) {
        bounceBack();
        computeWallMomentum();
    }
    if (!boundaryNodes.empty())
        applyBoundaries();
    stepForcesCurrent = true;
    stepIndex++;
    if (lattice.machCheckFrequency > 0 && stepIndex%lattice.machCheckFrequency == 0)
        checkMachNumber();
}

void ReferenceCalcLBMForceKernel::computeNextStepForces(ContextImpl& context) {
    // The coupling of the next lattice step computed on the current fluid, positions and velocities, without
    // its reaction on the fluid.  The step recomputes it, with the same random numbers, after any change of
    // the velocities in between (the reflection at the walls, setVelocities()).
    stepForcesCurrent = false;
    computeMoments();
    long long nextStep = context.getStepCount();
    if (lattice.momentumRemovalFrequency > 0 && nextStep%lattice.momentumRemovalFrequency == 0)
        removeFluidMomentum();
    couple(context, false);
}

void ReferenceCalcLBMForceKernel::couple(ContextImpl& context, bool isStep) {
    if (lattice.dragScheme == LBMForce::Centered)
        coupleParticlesCentered(context, isStep);
    else
        coupleParticles(context, isStep);
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
            rho[node] = densityDeviation[node] = 0;
            for (int k = 0; k < 3; k++)
                momentum[3*node+k] = forceDensity[3*node+k] = 0;
            for (int k = 0; k < 6; k++)
                piNeq[6*node+k] = 0;
            continue;
        }
        // From the deviations df_q = f_q - w_q: rho = 1 + sum df_q and j = sum c_q df_q, since sum w_q = 1 and
        // sum w_q c_q = 0.
        double dr = 0, jx = 0, jy = 0, jz = 0;
        for (int q = 0; q < D3Q19::numVelocities; q++) {
            f[q] = populations[q*numNodes+node];
            dr += f[q];
            jx += D3Q19::cx[q]*f[q];
            jy += D3Q19::cy[q]*f[q];
            jz += D3Q19::cz[q]*f[q];
        }
        double r = 1.0 + dr;
        rho[node] = r;
        densityDeviation[node] = dr;
        momentum[3*node] = jx;
        momentum[3*node+1] = jy;
        momentum[3*node+2] = jz;
        D3Q19::nonEquilibriumMomentFromDeviation(f, dr, jx, jy, jz, &piNeq[6*node]);

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

void ReferenceCalcLBMForceKernel::coupleParticles(ContextImpl& context, bool isStep) {
    // Explicit Euler-Maruyama coupling at the nearest node, in lattice units (time step 1):
    //   F = -gamma m (v - j/rho) + sqrt(2 gamma m kT) xi,
    // with xi three independent N(0,1) numbers.  v is the velocity of OpenMM's leapfrog, v(t - dt/2), and j is
    // the momentum of the fluid before the force of this step, j(t - dt/2).  In a lattice step (isStep) the
    // particle receives F and the node -F: the reactions of the particles of a node are summed in particle
    // order, then added to the force density of the node.  A solid node has rho = 0 and is at rest; the
    // reaction it receives leaves the fluid.  The random numbers are drawn once per step, in particle order.
    vector<Vec3>& positions = extractPositions(context);
    vector<Vec3>& velocities = extractVelocities(context);
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    double gamma = lattice.friction*lattice.dt;
    double kT = lattice.kT*lattice.dt*lattice.dt/(cellMass*lattice.dx*lattice.dx);
    double velocityScale = lattice.getVelocityScale();
    double forceScale = cellMass*lattice.dx/(lattice.dt*lattice.dt);
    bool randomForce = (kT > 0 && gamma > 0);
    if (randomForce)
        drawNoise();
    if (isStep)
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
        if (randomForce)
            f += noise[i]*sqrt(2.0*gamma*m*kT);
        particleForces[i] = f*forceScale;
        if (isStep)
            for (int k = 0; k < 3; k++)
                reaction[3*node+k] -= f[k];
    }
    if (isStep)
        applyReaction();
}

void ReferenceCalcLBMForceKernel::coupleParticlesCentered(ContextImpl& context, bool isStep) {
    // Centred drag at the nearest node (docs/theory.md, section 2), in lattice units (time step 1, h = 1/2):
    //   F_k = -gamma m_k [v_k(t) - u_c(t)] + sqrt(2 gamma m_k kT) xi_k,
    // with v_k(t) = v_k(t - h) + h (Fc_k + F_k)/m_k the velocity of particle k at the time of the force, Fc_k the
    // other forces on it, and u_c(t) = (j_c + h G_c)/rho_c the velocity of its node c that the collision puts in
    // the equilibrium, G_c being the total force on the node.  The system is linear and couples only the
    // particles of the same node.  With a = gamma h, v~_k = v_k(t - h) + h Fc_k/m_k, u~ = (j_c + h Fbody_c)/rho_c,
    // m_c = rho_c, and M, P~ and R the sums of m_k, m_k v~_k and of the random forces over the particles of the
    // node, the sum S of their forces and the velocity of the node are
    //   S = [-gamma (P~ - M u~) + R]/(1 + a + a M/m_c),   u_c(t) = u~ - h S/m_c,
    // and then F_k = [-gamma m_k (v~_k - u_c(t)) + R_k]/(1 + a).  The node receives -S, so that G_c = Fbody_c - S.
    // A solid node is a wall at rest of infinite mass: u_c(t) = 0; the boundary nodes of regularized walls and open
    // faces are fluid nodes like the others.  Fc_k is read from the forces of OpenMM, to which
    // the other forces of the System have been added since LBMForce is the last one.  The particles of a node are
    // summed in particle order, as on the GPU platforms (keys node*N + k, sorted).
    int numParticles = lattice.particles.size();
    vector<Vec3>& positions = extractPositions(context);
    vector<Vec3>& velocities = extractVelocities(context);
    vector<Vec3>& forces = extractForces(context);
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    double gamma = lattice.friction*lattice.dt;
    double kT = lattice.kT*lattice.dt*lattice.dt/(cellMass*lattice.dx*lattice.dx);
    double velocityScale = lattice.getVelocityScale();
    double forceScale = cellMass*lattice.dx/(lattice.dt*lattice.dt);
    double h = 0.5, a = gamma*h;
    bool randomForce = (kT > 0 && gamma > 0);
    if (randomForce)
        drawNoise();
    vector<long long> keys(numParticles);
    vector<Vec3> knownVelocity(numParticles), randomForces(numParticles);
    for (int i = 0; i < numParticles; i++) {
        int particle = lattice.particles[i];
        double m = particleMass[i];
        knownVelocity[i] = velocities[particle]*(1.0/velocityScale) + forces[particle]*(h/(m*forceScale));
        if (randomForce)
            randomForces[i] = noise[i]*sqrt(2.0*gamma*m*kT);
        keys[i] = nearestNode(positions[particle])*(long long) numParticles + i;
    }
    sort(keys.begin(), keys.end());
    if (isStep)
        fill(reaction.begin(), reaction.end(), 0.0);
    for (int first = 0; first < numParticles; ) {
        int node = keys[first]/numParticles;
        int end = first;
        double mass = 0;
        Vec3 particleMomentum, random;
        for (; end < numParticles && keys[end]/numParticles == node; end++) {
            int i = keys[end]%numParticles;
            mass += particleMass[i];
            particleMomentum += knownVelocity[i]*particleMass[i];
            random += randomForces[i];
        }
        Vec3 u, sum;
        if (rho[node] > 0) {
            const double* j = &momentum[3*node];
            const double* F = &forceDensity[3*node];
            Vec3 known = Vec3(j[0] + h*F[0], j[1] + h*F[1], j[2] + h*F[2])*(1.0/rho[node]);
            sum = ((particleMomentum - known*mass)*(-gamma) + random)*(1.0/(1.0 + a + a*mass/rho[node]));
            u = known - sum*(h/rho[node]);
        }
        else
            sum = ((particleMomentum - u*mass)*(-gamma) + random)*(1.0/(1.0 + a));
        for (int k = first; k < end; k++) {
            int i = keys[k]%numParticles;
            Vec3 f = ((knownVelocity[i] - u)*(-gamma*particleMass[i]) + randomForces[i])*(1.0/(1.0 + a));
            particleForces[i] = f*forceScale;
        }
        if (isStep)
            for (int k = 0; k < 3; k++)
                reaction[3*node+k] = -sum[k];
        first = end;
    }
    if (isStep)
        applyReaction();
}

void ReferenceCalcLBMForceKernel::drawNoise() {
    // The random numbers of a step are drawn once, three per coupled particle in particle order, by the first
    // evaluation that needs them.
    if (noiseDrawn)
        return;
    for (int i = 0; i < (int) lattice.particles.size(); i++) {
        double xi0 = getGaussianRandom();
        double xi1 = getGaussianRandom();
        double xi2 = getGaussianRandom();
        noise[i] = Vec3(xi0, xi1, xi2);
    }
    noiseDrawn = true;
}

void ReferenceCalcLBMForceKernel::applyReaction() {
    // At the end of the coupling of a lattice step: the random numbers have been used, and the reaction of the
    // particles is added to the force density of the nodes.  The reaction on a solid node goes to the wall.
    int numNodes = lattice.getNumNodes();
    noiseDrawn = false;
    for (int node = 0; node < numNodes; node++)
        for (int k = 0; k < 3; k++)
            forceDensity[3*node+k] += reaction[3*node+k];
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    double momentumScale = cellMass*lattice.dx/lattice.dt;
    for (int node : lattice.solidNodes)
        wallMomentum += Vec3(reaction[3*node], reaction[3*node+1], reaction[3*node+2])*momentumScale;
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
    // written in push form: each population is computed from the moments of its own node only.  The stored
    // deviation f_q - w_q uses the deviation of the equilibrium, feq_q - w_q.
    // A fluctuating fluid (docs/theory.md, section 7) adds the random part xi_q of D3Q19::fluctuation(), with the
    // amplitudes sqrt(mu rho omega (2 - omega)) on the stress modes and sqrt(mu rho) on the ghost modes, which
    // relax with rate 1, and mu = kT/cs^2 in lattice units.  The 15 normal numbers of a node are drawn from the
    // generator of the force, node after node in index order.
    int nx = lattice.nx, ny = lattice.ny, nz = lattice.nz;
    int numNodes = lattice.getNumNodes();
    double omega = lattice.omega;
    double dfeq[D3Q19::numVelocities], fneq[D3Q19::numVelocities], s[D3Q19::numVelocities];
    double xi[D3Q19::numVelocities], normal[15];
    bool fluctuate = (lattice.fluidFluctuations && lattice.fluidKT > 0);
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    double mu = 3.0*lattice.fluidKT*lattice.dt*lattice.dt/(cellMass*lattice.dx*lattice.dx);
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
                D3Q19::equilibriumDeviation(densityDeviation[node], ux, uy, uz, dfeq);
                D3Q19::regularizedNonEquilibrium(&piNeq[6*node], fneq);
                D3Q19::guoForcing(ux, uy, uz, F[0], F[1], F[2], s);
                if (fluctuate) {
                    for (int m = 0; m < 15; m++)
                        normal[m] = getGaussianRandom();
                    D3Q19::fluctuation(normal, sqrt(mu*r*omega*(2.0-omega)), sqrt(mu*r), xi);
                }
                for (int q = 0; q < D3Q19::numVelocities; q++) {
                    int di = (i + D3Q19::cx[q] + nx)%nx;
                    int dj = (j + D3Q19::cy[q] + ny)%ny;
                    int dk = (k + D3Q19::cz[q] + nz)%nz;
                    double value = dfeq[q] + (1.0-omega)*fneq[q] + 0.5*s[q];
                    if (fluctuate)
                        value += xi[q];
                    populations[q*numNodes + di + nx*(dj + ny*dk)] = value;
                }
            }
}

void ReferenceCalcLBMForceKernel::bounceBack() {
    // Halfway bounce-back (wall scheme BounceBack), done by the fluid nodes next to the walls after the streaming.  For
    // a direction q whose node x + c_q is solid, the population that x built in its collision for q from its own
    // moments rho, j and Pi_neq, force and random part streamed into the solid node; x takes it back as its population
    // along -c_q, the one that arrives from the wall.  The wall lies halfway between the two nodes.  The stored
    // deviations f - w are copied as they are, since opposite directions have the same weight.
    int nx = lattice.nx, ny = lattice.ny, nz = lattice.nz;
    int numNodes = lattice.getNumNodes();
    for (int b = 0; b < (int) wallNodes.size(); b++) {
        int node = wallNodes[b];
        int i = node%nx, j = (node/nx)%ny, k = node/(nx*ny);
        for (int q = 1; q < D3Q19::numVelocities; q++) {
            if (!(wallLinks[b] & (1<<q)))
                continue;
            int solid = (i + D3Q19::cx[q] + nx)%nx + nx*((j + D3Q19::cy[q] + ny)%ny + ny*((k + D3Q19::cz[q] + nz)%nz));
            populations[D3Q19::opposite[q]*numNodes + node] = populations[q*numNodes + solid];
        }
    }
}

void ReferenceCalcLBMForceKernel::computeWallMomentum() {
    // Momentum exchange of the halfway bounce-back (Ladd 1994): the population that streamed from the fluid node
    // s + c_q into the solid node s, moving along -c_q, went back to s + c_q moving along c_q, so on each such link
    // the wall at rest receives the momentum f (-c_q) - f c_q.  It uses the full population f = (f - w) + w, whose
    // part w carries the static pressure.  Only links to fluid nodes that do not cross an open face count: between two
    // solid nodes nothing streams.  The sum runs over the solid nodes in the order of the list, as on the GPU
    // platforms.
    int nx = lattice.nx, ny = lattice.ny, nz = lattice.nz;
    int numNodes = lattice.getNumNodes();
    Vec3 exchanged;
    for (int node : lattice.solidNodes) {
        int i = node%nx, j = (node/nx)%ny, k = node/(nx*ny);
        for (int q = 1; q < D3Q19::numVelocities; q++) {
            int di = i + D3Q19::cx[q], dj = j + D3Q19::cy[q], dk = k + D3Q19::cz[q];
            if ((lattice.isOpenAxis(0) && (di < 0 || di >= nx)) || (lattice.isOpenAxis(1) && (dj < 0 || dj >= ny)) ||
                    (lattice.isOpenAxis(2) && (dk < 0 || dk >= nz)))
                continue;       // the link crosses an open face
            di = (di + nx)%nx;
            dj = (dj + ny)%ny;
            dk = (dk + nz)%nz;
            int target = di + nx*(dj + ny*dk);
            if (!isFluid[target])
                continue;
            double df = populations[D3Q19::opposite[q]*numNodes + node];
            exchanged -= Vec3(D3Q19::cx[q], D3Q19::cy[q], D3Q19::cz[q])*(2.0*(df + D3Q19::w[q]));
        }
    }
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    wallMomentum += exchanged*(cellMass*lattice.dx/lattice.dt);
}

void ReferenceCalcLBMForceKernel::applyBoundaries() {
    // Regularized boundaries (docs/theory.md, section 1), after the streaming, at the boundary nodes: the fluid nodes
    // on open faces and, with regularized walls, the fluid nodes next to solid nodes.  At a boundary node x the
    // populations of the directions q whose source x - c_q lies beyond an open face, or is solid, are unknown.  Each
    // of them is rebuilt as the population that a node at x - c_q would send, a node with the moments of x except the
    // imposed one: feq_q(rho_b, u_b) + (1 - omega) fneq_q(Pi_neq of x) + S_q(u_b, rho_b g)/2, plus, with a
    // fluctuating fluid, a random part of its own, drawn for x after those of the collision.  The known populations
    // of x are not changed, and the boundary lies on the nodes x - c_q.
    //  - Next to solid nodes (walls, which prevail over the faces): u_b = 0 and rho_b from the mass balance (below).
    //  - Velocity face: u_b the velocity of the face, rho_b from the mass balance with the inflow (below).
    //  - Density face: rho_b the density of the face; the velocity along the face is that of x, and the velocity
    //    across it the mean of the velocity that gives x the density of the face, from the populations that have
    //    arrived (the unknown ones replaced by the bounce-back of their opposites), and the velocity of x at the start
    //    of the step: a filter in time, without memory, that damps the staggered mode and does not change steady
    //    flows.
    //  - Nodes shared by several Density faces (and no Velocity face): rho_b of the first face and u_b = 0.
    // On walls the solid nodes receive the momentum of the populations that x sent into them and give that of the
    // rebuilt populations.  Each node reads only its own populations and moments and the solid slots that it wrote
    // itself.
    int nx = lattice.nx, ny = lattice.ny, nz = lattice.nz;
    int numNodes = lattice.getNumNodes();
    Vec3 g = lattice.bodyAcceleration;
    double omega = lattice.omega;
    bool fluctuate = (lattice.fluidFluctuations && lattice.fluidKT > 0);
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    double mu = 3.0*lattice.fluidKT*lattice.dt*lattice.dt/(cellMass*lattice.dx*lattice.dx);
    double dfeq[D3Q19::numVelocities], fneq[D3Q19::numVelocities], s[D3Q19::numVelocities];
    double xi[D3Q19::numVelocities], normal[15];
    Vec3 exchanged;
    for (int b = 0; b < (int) boundaryNodes.size(); b++) {
        int node = boundaryNodes[b], unknown = unknownDirections[b], solid = solidDirections[b];
        int kind = boundaryKind[b], face = boundaryFace[b];
        int i = node%nx, j = (node/nx)%ny, k = node/(nx*ny);
        const double* F = &forceDensity[3*node];
        double rb = rho[node], dr = densityDeviation[node];
        Vec3 ub((momentum[3*node] + 0.5*F[0])/rb, (momentum[3*node+1] + 0.5*F[1])/rb, (momentum[3*node+2] + 0.5*F[2])/rb);
        if (kind == LBMBoundaries::Wall || kind == LBMBoundaries::DensityAtRest)
            ub = Vec3();
        if (kind == LBMBoundaries::Velocity)
            ub = lattice.faceVelocity[face];
        if (kind == LBMBoundaries::Density || kind == LBMBoundaries::DensityAtRest) {
            dr = lattice.faceDensity[face]-1.0;
            rb = lattice.faceDensity[face];
        }
        if (kind == LBMBoundaries::Density) {
            // Time filter of the velocity across the face: the mean of the velocity that gives the node the density of
            // the face, from the populations that have arrived (the unknown ones replaced by the bounce-back of their
            // opposites), and the velocity of the node at the start of the step.
            int axis = face/2;
            double inward = (face%2 == 0 ? 1.0 : -1.0), sum = 0;
            for (int q = 0; q < D3Q19::numVelocities; q++)
                sum += populations[(unknown & (1<<q) ? D3Q19::opposite[q] : q)*numNodes+node];
            ub[axis] = 0.5*(inward*(dr-sum)/rb + ub[axis]);
        }
        Vec3 Fb = g*rb;
        D3Q19::equilibriumDeviation(dr, ub[0], ub[1], ub[2], dfeq);
        D3Q19::regularizedNonEquilibrium(&piNeq[6*node], fneq);
        D3Q19::guoForcing(ub[0], ub[1], ub[2], Fb[0], Fb[1], Fb[2], s);
        if (fluctuate) {
            for (int m = 0; m < 15; m++)
                normal[m] = getGaussianRandom();
            D3Q19::fluctuation(normal, sqrt(mu*rb*omega*(2.0-omega)), sqrt(mu*rb), xi);
        }
        if (kind == LBMBoundaries::Wall || kind == LBMBoundaries::Velocity) {
            // Mass balance: the rebuilt populations carry the mass that arrives on their links, that is the mass that
            // x sent into the solid nodes (exactly the mass that leaves the fluid there) and, across a face, the
            // population that arrived at x moving out of the face plus the inflow 6 w_q rho_b c_q.u_b (Zou and He;
            // eq. 5.3 of Latt).  Links whose opposite direction is unknown too do not count.  Linear in
            // rho_b = 1 + dr, written for dr so that a small deviation keeps its precision.
            double e0[D3Q19::numVelocities], sg[D3Q19::numVelocities];
            D3Q19::equilibriumDeviation(0.0, ub[0], ub[1], ub[2], e0);
            D3Q19::guoForcing(ub[0], ub[1], ub[2], g[0], g[1], g[2], sg);
            double numerator = 0, denominator = 0;
            for (int q = 1; q < D3Q19::numVelocities; q++) {
                if (!(unknown & (1<<q)))
                    continue;
                double arriving, flux = 0;
                if (solid & (1<<q)) {
                    int source = (i - D3Q19::cx[q] + nx)%nx + nx*((j - D3Q19::cy[q] + ny)%ny + ny*((k - D3Q19::cz[q] + nz)%nz));
                    arriving = populations[D3Q19::opposite[q]*numNodes + source];
                }
                else if (!(unknown & (1<<D3Q19::opposite[q]))) {
                    arriving = populations[D3Q19::opposite[q]*numNodes + node];
                    flux = 6.0*D3Q19::w[q]*(D3Q19::cx[q]*ub[0] + D3Q19::cy[q]*ub[1] + D3Q19::cz[q]*ub[2]);
                }
                else
                    continue;
                double rest = (1.0-omega)*fneq[q] + (fluctuate ? xi[q] : 0.0);
                numerator += arriving - rest - e0[q] - 0.5*sg[q] + flux;
                denominator += D3Q19::w[q] + e0[q] + 0.5*sg[q] - flux;
            }
            if (denominator != 0) {
                dr = numerator/denominator;
                rb = 1.0 + dr;
                Fb = g*rb;
                D3Q19::equilibriumDeviation(dr, ub[0], ub[1], ub[2], dfeq);
                D3Q19::guoForcing(ub[0], ub[1], ub[2], Fb[0], Fb[1], Fb[2], s);
            }
        }
        for (int q = 1; q < D3Q19::numVelocities; q++) {
            if (!(unknown & (1<<q)))
                continue;
            double value = dfeq[q] + (1.0-omega)*fneq[q] + 0.5*s[q];
            if (fluctuate)
                value += xi[q];
            if (solid & (1<<q)) {
                // The wall receives the momentum of the population that x sent into the solid node x - c_q, along
                // -c_q, and gives that of the population that comes back along c_q (full populations f = (f - w) + w).
                int source = (i - D3Q19::cx[q] + nx)%nx + nx*((j - D3Q19::cy[q] + ny)%ny + ny*((k - D3Q19::cz[q] + nz)%nz));
                double sent = populations[D3Q19::opposite[q]*numNodes + source];
                exchanged -= Vec3(D3Q19::cx[q], D3Q19::cy[q], D3Q19::cz[q])*(sent + value + 2.0*D3Q19::w[q]);
            }
            populations[q*numNodes+node] = value;
        }
    }
    wallMomentum += exchanged*(cellMass*lattice.dx/lattice.dt);
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
        double r = 1.0, jx = 0, jy = 0, jz = 0;            // from the deviations f - w
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

Vec3 ReferenceCalcLBMForceKernel::getWallForce(ContextImpl& context) {
    return wallMomentum*(1.0/lattice.dt);
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
        double r = 1.0, jx = 0, jy = 0, jz = 0;            // from the deviations f - w
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
    // The state is the stored deviations f_q - w_q: saving and restoring them is exact.
    state = populations;
}

void ReferenceCalcLBMForceKernel::setFluidState(ContextImpl& context, const vector<double>& state) {
    if (state.size() != populations.size())
        throw OpenMMException("LBMForce: setFluidState() was called with a state of the wrong size");
    populations = state;
    stepForcesCurrent = false;
}

void ReferenceCalcLBMForceKernel::createCheckpoint(ContextImpl& context, ostream& stream) {
    // The populations, the random numbers drawn for the next step, the state of the generator of the force
    // (not part of OpenMM checkpoints on this platform) and the momentum given to the walls in the last step.
    stream.write((const char*) populations.data(), sizeof(double)*populations.size());
    int drawn = noiseDrawn;
    stream.write((const char*) &drawn, sizeof(int));
    for (const Vec3& xi : noise)
        for (int k = 0; k < 3; k++) {
            double value = xi[k];
            stream.write((const char*) &value, sizeof(double));
        }
    sfmt.createCheckpoint(stream);
    int stored = hasStoredGaussian;
    stream.write((const char*) &stored, sizeof(int));
    stream.write((const char*) &storedGaussian, sizeof(double));
    for (int k = 0; k < 3; k++) {
        double value = wallMomentum[k];
        stream.write((const char*) &value, sizeof(double));
    }
}

void ReferenceCalcLBMForceKernel::loadCheckpoint(ContextImpl& context, istream& stream) {
    stream.read((char*) populations.data(), sizeof(double)*populations.size());
    stepForcesCurrent = false;
    int drawn;
    stream.read((char*) &drawn, sizeof(int));
    noiseDrawn = (drawn != 0);
    for (Vec3& xi : noise)
        for (int k = 0; k < 3; k++)
            stream.read((char*) &xi[k], sizeof(double));
    sfmt.loadCheckpoint(stream);
    int stored;
    stream.read((char*) &stored, sizeof(int));
    hasStoredGaussian = (stored != 0);
    stream.read((char*) &storedGaussian, sizeof(double));
    for (int k = 0; k < 3; k++)
        stream.read((char*) &wallMomentum[k], sizeof(double));
}
