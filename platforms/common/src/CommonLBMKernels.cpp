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
#include <map>

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

    // The fluid is stored in the mixed type: double unless the platform runs in single precision.

    int numNodes = lattice.getNumNodes();
    bool useDouble = (cc.getUseDoublePrecision() || cc.getUseMixedPrecision());
    int elementSize = (useDouble ? sizeof(double) : sizeof(float));
    populations.initialize(cc, D3Q19::numVelocities*numNodes, elementSize, "lbmPopulations");
    density.initialize(cc, numNodes, elementSize, "lbmDensity");
    momentum.initialize(cc, 3*numNodes, elementSize, "lbmMomentum");

    // The fluid starts at equilibrium, with lattice density 1 and the initial velocity.

    vector<double> f(D3Q19::numVelocities*numNodes);
    double feq[D3Q19::numVelocities];
    D3Q19::equilibrium(1.0, lattice.initialVelocity[0], lattice.initialVelocity[1], lattice.initialVelocity[2], feq);
    for (int q = 0; q < D3Q19::numVelocities; q++)
        for (int node = 0; node < numNodes; node++)
            f[q*numNodes+node] = feq[q];
    populations.upload(f, true);

    // Compile the kernels.

    map<string, string> defines;
    defines["NUM_NODES"] = cc.intToString(numNodes);
    ComputeProgram program = cc.compileProgram(CommonLBMKernelSources::lbmFluid, defines);
    computeMomentsKernel = program->createKernel("computeFluidMoments");
    computeMomentsKernel->addArg(populations);
    computeMomentsKernel->addArg(density);
    computeMomentsKernel->addArg(momentum);
}

void CommonCalcLBMForceKernel::beginStep(ContextImpl& context) {
    // The fluid does not advance on this platform yet: the lattice update is ported from the Reference
    // platform in the next phase.
}

double CommonCalcLBMForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // The fluid update and the particle-fluid coupling are not implemented yet: no force is applied.
    return 0.0;
}

void CommonCalcLBMForceKernel::copyParametersToContext(ContextImpl& context, const LBMLatticeParameters& lattice) {
    this->lattice = lattice;
}

void CommonCalcLBMForceKernel::getFluidFields(ContextImpl& context, vector<double>& density, vector<Vec3>& velocity) {
    ContextSelector selector(cc);
    int numNodes = lattice.getNumNodes();
    computeMomentsKernel->execute(numNodes);
    vector<double> rho, j;
    downloadAsDouble(this->density, rho);
    downloadAsDouble(momentum, j);
    double velocityScale = lattice.getVelocityScale();
    density.resize(numNodes);
    velocity.resize(numNodes);
    for (int node = 0; node < numNodes; node++) {
        density[node] = rho[node]*lattice.density;
        velocity[node] = Vec3(j[node], j[numNodes+node], j[2*numNodes+node])*(velocityScale/rho[node]);
    }
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
