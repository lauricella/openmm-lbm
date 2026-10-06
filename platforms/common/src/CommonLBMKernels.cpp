/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "CommonLBMKernels.h"
#include "CommonLBMKernelSources.h"
#include "internal/D3Q19.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>

using namespace LBMPlugin;
using namespace OpenMM;
using namespace std;

/**
 * Download an array of floats or doubles into a vector of doubles.  ComputeArray::download() converts
 * types only from OpenMM 8.4, and the plugin supports OpenMM 8.3.
 */
static void downloadAsDouble(const ComputeArray& array, vector<double>& data) {
    data.resize(array.getSize());
    if (array.getElementSize() == sizeof(double))
        array.download(data.data());
    else {
        vector<float> values(array.getSize());
        array.download(values.data());
        for (size_t i = 0; i < values.size(); i++)
            data[i] = values[i];
    }
}

void CommonCalcLBMForceKernel::initialize(const System& system, const LBMForce& force, const LBMLatticeParameters& lattice) {
    ContextSelector selector(cc);
    if (cc.getNumContexts() > 1)
        throw OpenMMException("LBMForce does not support running on multiple devices");
    this->lattice = lattice;

    // Until the coupling is ported, coupled particles on this platform would silently feel no force: say so at
    // every Context creation.
    if (!lattice.particles.empty())
        cerr << "Warning: LBMForce: the particle-fluid coupling is implemented only on the Reference platform in "
             << "this version. On this platform no force acts on the coupled particles, and the fluid feels no "
             << "reaction." << endl;

    // The fluid is stored in the mixed type: double unless the platform runs in single precision.

    int numNodes = lattice.getNumNodes();
    useDouble = (cc.getUseDoublePrecision() || cc.getUseMixedPrecision());
    int elementSize = (useDouble ? sizeof(double) : sizeof(float));
    blockSize = ComputeContext::ThreadBlockSize;
    numGroups = max(1, min(cc.getNumThreadBlocks(), (numNodes+blockSize-1)/blockSize));
    populations.initialize(cc, D3Q19::numVelocities*numNodes, elementSize, "lbmPopulations");
    densityDeviation.initialize(cc, numNodes, elementSize, "lbmDensityDeviation");
    momentum.initialize(cc, 3*numNodes, elementSize, "lbmMomentum");
    piNeq.initialize(cc, 6*numNodes, elementSize, "lbmPiNeq");
    partialSums.initialize(cc, 4*numGroups, elementSize, "lbmPartialSums");
    partialMax.initialize(cc, numGroups, elementSize, "lbmPartialMax");
    centerVelocity.initialize(cc, 3, elementSize, "lbmCenterVelocity");

    // The fluid starts at equilibrium, with lattice density 1 and the initial velocity.  As on the Reference
    // platform, the populations are stored as deviations from the rest equilibrium, f_q - w_q.

    vector<double> f(D3Q19::numVelocities*numNodes);
    double dfeq[D3Q19::numVelocities];
    D3Q19::equilibriumDeviation(0.0, lattice.initialVelocity[0], lattice.initialVelocity[1], lattice.initialVelocity[2], dfeq);
    for (int q = 0; q < D3Q19::numVelocities; q++)
        for (int node = 0; node < numNodes; node++)
            f[q*numNodes+node] = dfeq[q];

    // Solid nodes hold no fluid: their populations start at zero, that is at the deviation -w_q.  The part w of
    // the populations gives the walls the same momentum in every step (the static pressure): it is computed
    // here once, in double precision, with the links from the solid nodes to the fluid nodes.

    int numSolidNodes = lattice.solidNodes.size();
    isFluidHost.assign(numNodes, 1);
    for (int node : lattice.solidNodes)
        isFluidHost[node] = 0;
    staticWallMomentum = Vec3();
    for (int node : lattice.solidNodes) {
        int i = node%lattice.nx, j = (node/lattice.nx)%lattice.ny, k = node/(lattice.nx*lattice.ny);
        for (int q = 0; q < D3Q19::numVelocities; q++)
            f[q*numNodes+node] = -D3Q19::w[q];
        for (int q = 1; q < D3Q19::numVelocities; q++) {
            int di = (i + D3Q19::cx[q] + lattice.nx)%lattice.nx;
            int dj = (j + D3Q19::cy[q] + lattice.ny)%lattice.ny;
            int dk = (k + D3Q19::cz[q] + lattice.nz)%lattice.nz;
            if (isFluidHost[di + lattice.nx*(dj + lattice.ny*dk)])
                staticWallMomentum -= Vec3(D3Q19::cx[q], D3Q19::cy[q], D3Q19::cz[q])*(2.0*D3Q19::w[q]);
        }
    }
    populations.upload(f, true);
    isFluid.initialize<int>(cc, numNodes, "lbmIsFluid");
    isFluid.upload(isFluidHost);
    solidNodes.initialize<int>(cc, max(1, numSolidNodes), "lbmSolidNodes");
    wallExchange.initialize(cc, 3*max(1, numSolidNodes), elementSize, "lbmWallExchange");
    if (numSolidNodes > 0)
        solidNodes.upload(lattice.solidNodes);
    wallExchange.upload(vector<double>(wallExchange.getSize(), 0.0), true);

    // Compile the kernels.

    map<string, string> defines;
    defines["NUM_NODES"] = cc.intToString(numNodes);
    defines["NX"] = cc.intToString(lattice.nx);
    defines["NY"] = cc.intToString(lattice.ny);
    defines["NZ"] = cc.intToString(lattice.nz);
    defines["LBM_BLOCK_SIZE"] = cc.intToString(blockSize);
    defines["W0"] = cc.doubleToString(1.0/3.0, true);
    defines["W1"] = cc.doubleToString(1.0/18.0, true);
    defines["W2"] = cc.doubleToString(1.0/36.0, true);
    defines["CS2"] = cc.doubleToString(1.0/3.0, true);
    if (numSolidNodes > 0) {
        defines["HAS_SOLID_NODES"] = "1";
        defines["NUM_SOLID_NODES"] = cc.intToString(numSolidNodes);
    }
    ComputeProgram program = cc.compileProgram(CommonLBMKernelSources::lbmFluid, defines);
    computeMomentsKernel = program->createKernel("computeFluidMoments");
    computeMomentsKernel->addArg(populations);
    computeMomentsKernel->addArg(isFluid);
    computeMomentsKernel->addArg(densityDeviation);
    computeMomentsKernel->addArg(momentum);
    computeMomentsKernel->addArg(piNeq);
    sumMomentumKernel = program->createKernel("sumFluidMomentum");
    sumMomentumKernel->addArg(densityDeviation);
    sumMomentumKernel->addArg(momentum);
    sumMomentumKernel->addArg(partialSums);
    centerVelocityKernel = program->createKernel("computeFluidCenterVelocity");
    centerVelocityKernel->addArg(partialSums);
    centerVelocityKernel->addArg(numGroups);
    centerVelocityKernel->addArg(centerVelocity);
    removeMomentumKernel = program->createKernel("removeFluidMomentum");
    removeMomentumKernel->addArg(densityDeviation);
    removeMomentumKernel->addArg(momentum);
    removeMomentumKernel->addArg(centerVelocity);
    collideKernel = program->createKernel("collideAndStream");
    collideKernel->addArg(populations);
    collideKernel->addArg(isFluid);
    collideKernel->addArg(densityDeviation);
    collideKernel->addArg(momentum);
    collideKernel->addArg(piNeq);
    for (int i = 0; i < 4; i++)
        collideKernel->addArg();        // omega and the body acceleration, set by setFluidParameters()
    if (numSolidNodes > 0) {
        bounceBackKernel = program->createKernel("bounceBack");
        bounceBackKernel->addArg(populations);
        bounceBackKernel->addArg(isFluid);
        bounceBackKernel->addArg(solidNodes);
        bounceBackKernel->addArg(wallExchange);
    }
    maxSpeedKernel = program->createKernel("computeMaxFluidSpeed");
    maxSpeedKernel->addArg(populations);
    maxSpeedKernel->addArg(isFluid);
    maxSpeedKernel->addArg(partialMax);
    setFluidParameters();
}

void CommonCalcLBMForceKernel::setFluidParameters() {
    // The relaxation rate and the body acceleration (lattice units) are arguments of the collision, in the mixed type.
    double values[4] = {lattice.omega, lattice.bodyAcceleration[0], lattice.bodyAcceleration[1], lattice.bodyAcceleration[2]};
    for (int i = 0; i < 4; i++) {
        if (useDouble)
            collideKernel->setArg(5+i, values[i]);
        else
            collideKernel->setArg(5+i, (float) values[i]);
    }
}

void CommonCalcLBMForceKernel::beginStep(ContextImpl& context) {
    // As on the Reference platform, the removal of the fluid momentum and the Mach number check are timed by the
    // step count of the Context, which checkpoints save and restore.
    stepIndex = context.getStepCount();
    stepPending = true;
}

double CommonCalcLBMForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // The fluid advances once per integration step: on the first force evaluation after beginStep().  Other force
    // evaluations (getState(), for example) do not advance it.  The particle-fluid coupling is not implemented on
    // this platform yet: no force is applied.
    if (stepPending) {
        stepPending = false;
        ContextSelector selector(cc);
        advanceFluid();
    }
    return 0.0;
}

void CommonCalcLBMForceKernel::advanceFluid() {
    int numNodes = lattice.getNumNodes();
    computeMomentsKernel->execute(numNodes);
    if (lattice.momentumRemovalFrequency > 0 && stepIndex%lattice.momentumRemovalFrequency == 0) {
        sumMomentumKernel->execute(numGroups*blockSize, blockSize);
        centerVelocityKernel->execute(blockSize, blockSize);
        removeMomentumKernel->execute(numNodes);
    }
    collideKernel->execute(numNodes);
    if (!lattice.solidNodes.empty())
        bounceBackKernel->execute(lattice.solidNodes.size());
    hasAdvanced = true;
    stepIndex++;
    if (lattice.machCheckFrequency > 0 && stepIndex%lattice.machCheckFrequency == 0)
        checkMachNumber();
}

void CommonCalcLBMForceKernel::checkMachNumber() {
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

double CommonCalcLBMForceKernel::computeMachNumber() {
    // Ma = max |j/rho|/c_s, with c_s^2 = 1/3: maxima by work group on the device, then over the groups here.
    maxSpeedKernel->execute(numGroups*blockSize, blockSize);
    vector<double> maxima;
    downloadAsDouble(partialMax, maxima);
    double maxSpeed2 = *max_element(maxima.begin(), maxima.end());
    return sqrt(3.0*maxSpeed2);
}

void CommonCalcLBMForceKernel::copyParametersToContext(ContextImpl& context, const LBMLatticeParameters& lattice) {
    ContextSelector selector(cc);
    this->lattice = lattice;
    setFluidParameters();
}

void CommonCalcLBMForceKernel::getFluidFields(ContextImpl& context, vector<double>& density, vector<Vec3>& velocity) {
    // The velocity of the forced fluid is u = (j + F/2)/rho, with F = rho*g from the body acceleration.  The
    // moments are recomputed from the populations; the next step computes them again before using them.
    ContextSelector selector(cc);
    int numNodes = lattice.getNumNodes();
    computeMomentsKernel->execute(numNodes);
    vector<double> dr, j;
    downloadAsDouble(densityDeviation, dr);
    downloadAsDouble(momentum, j);
    double velocityScale = lattice.getVelocityScale();
    density.resize(numNodes);
    velocity.resize(numNodes);
    for (int node = 0; node < numNodes; node++) {
        if (!isFluidHost[node]) {
            density[node] = 0;
            velocity[node] = Vec3();
            continue;
        }
        double r = 1.0 + dr[node];
        density[node] = r*lattice.density;
        velocity[node] = (Vec3(j[node], j[numNodes+node], j[2*numNodes+node])*(1.0/r) + lattice.bodyAcceleration*0.5)*velocityScale;
    }
}

Vec3 CommonCalcLBMForceKernel::getWallForce(ContextImpl& context) {
    // The momentum given to the solid nodes in the last step, divided by dt: the part of the deviations f - w,
    // summed over the solid nodes in the order of the list, plus the static part of the weights w.
    int numSolidNodes = lattice.solidNodes.size();
    if (numSolidNodes == 0 || !hasAdvanced)
        return Vec3();
    ContextSelector selector(cc);
    vector<double> exchange;
    downloadAsDouble(wallExchange, exchange);
    Vec3 momentum;
    for (int i = 0; i < numSolidNodes; i++)
        momentum += Vec3(exchange[i], exchange[numSolidNodes+i], exchange[2*numSolidNodes+i]);
    momentum += staticWallMomentum;
    double cellMass = lattice.density*lattice.dx*lattice.dx*lattice.dx;
    return momentum*(cellMass*lattice.dx/lattice.dt)*(1.0/lattice.dt);
}

double CommonCalcLBMForceKernel::getFluidMachNumber(ContextImpl& context) {
    ContextSelector selector(cc);
    return computeMachNumber();
}

void CommonCalcLBMForceKernel::getFluidState(ContextImpl& context, vector<double>& state) {
    ContextSelector selector(cc);
    downloadAsDouble(populations, state);
}

void CommonCalcLBMForceKernel::setFluidState(ContextImpl& context, const vector<double>& state) {
    ContextSelector selector(cc);
    if (state.size() != populations.getSize())
        throw OpenMMException("LBMForce: setFluidState() was called with a state of the wrong size");
    populations.upload(state, true);
}
