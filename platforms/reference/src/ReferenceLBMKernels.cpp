/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "ReferenceLBMKernels.h"
#include "internal/D3Q19.h"
#include "openmm/OpenMMException.h"

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
}

double ReferenceCalcLBMForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // The fluid update and the particle-fluid coupling are not implemented yet: no force is applied.
    return 0.0;
}

void ReferenceCalcLBMForceKernel::copyParametersToContext(ContextImpl& context, const LBMLatticeParameters& lattice) {
    this->lattice = lattice;
}

void ReferenceCalcLBMForceKernel::getFluidFields(ContextImpl& context, vector<double>& density, vector<Vec3>& velocity) {
    int numNodes = lattice.getNumNodes();
    double velocityScale = lattice.getVelocityScale();
    density.resize(numNodes);
    velocity.resize(numNodes);
    for (int node = 0; node < numNodes; node++) {
        double rho = 0, jx = 0, jy = 0, jz = 0;
        for (int q = 0; q < D3Q19::numVelocities; q++) {
            double f = populations[q*numNodes+node];
            rho += f;
            jx += D3Q19::cx[q]*f;
            jy += D3Q19::cy[q]*f;
            jz += D3Q19::cz[q]*f;
        }
        density[node] = rho*lattice.density;
        velocity[node] = Vec3(jx, jy, jz)*(velocityScale/rho);
    }
}

void ReferenceCalcLBMForceKernel::getFluidState(ContextImpl& context, vector<double>& state) {
    state = populations;
}

void ReferenceCalcLBMForceKernel::setFluidState(ContextImpl& context, const vector<double>& state) {
    if (state.size() != populations.size())
        throw OpenMMException("LBMForce: setFluidState() was called with a state of the wrong size");
    populations = state;
}
