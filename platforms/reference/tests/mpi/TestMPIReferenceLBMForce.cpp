/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

/**
 * The domain decomposition (docs/theory.md, section 8) on the Reference platform, built only with the CMake option
 * OPENMM_LBM_MPI and run by CTest under mpiexec with 2 ranks.  Every rank runs the same System twice, with one domain
 * and with two domains along x, and compares: the populations of the fluid nodes of its domain, the state gathered
 * on rank 0, the fields of its domain with the exchanged halo, and the positions and velocities of the coupled
 * particles, which must all be identical bit for bit without the removal of the fluid momentum.  The run with two
 * domains also writes a checkpoint file after half of the steps (LBMForce::saveCheckpointFile(), with MPI-IO), from
 * which the run is continued with two domains and with one domain on each rank: both must give the same result.  The
 * more complete script python/tests/mpi_decomposition.py runs the same comparisons on every platform and
 * decomposition.
 */

#include "LBMForce.h"
#include "openmm/Context.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/AssertionUtilities.h"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace LBMPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerLBMReferenceKernelFactories();

const int NX = 8, NY = 6, NZ = 6;

struct Result {
    vector<double> state, gathered, density;
    vector<Vec3> velocity, positions, velocities;
    int start[3], count[3];
};

static Result run(int px, const string& saveFile="", const string& loadFile="") {
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3));
    LBMForce* force = new LBMForce();
    force->setGridSize(NX, NY, NZ);
    force->setKinematicViscosity((0.8-0.5)/3*0.25/0.01);
    force->setBodyAcceleration(Vec3(0.5, -0.2, 0.1));
    force->setInitialFluidVelocity(Vec3(0.5, 0.2, -0.3));
    force->setFluidMomentumRemovalFrequency(0);
    force->setTemperature(0.0);
    force->setFriction(10.0);
    vector<int> solid;
    for (int k = 0; k < NZ; k++)
        for (int i = 0; i < NX; i++)
            solid.push_back(i + NX*NY*k);       // the plane j = 0, with bounce-back
    force->setSolidNodes(solid);
    // Coupled particles near the border of the two blocks, across the periodic boundary, and one moving into the wall
    // (reflected at the first step).
    CustomExternalForce* field = new CustomExternalForce("-3*x+2*y-z");
    double positions[4][3] = {{1.95, 1.2, 1.3}, {2.05, 1.6, 0.7}, {0.05, 2.9, 2.95}, {3.2, 0.1, 1.3}};
    double velocities[4][3] = {{1.5, 0.3, -0.4}, {-1.0, 0.5, 0.2}, {-0.8, 1.0, 0.6}, {0.4, -0.9, 0.5}};
    for (int i = 0; i < 4; i++) {
        system.addParticle(50.0);
        field->addParticle(i);
        force->addParticle(i);
    }
    system.addForce(field);
    force->setDomainDecomposition(px, 1, 1);
    force->setDensityHaloExchange(true);
    force->setVelocityHaloExchange(true);
    system.addForce(force);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, Platform::getPlatformByName("Reference"));
    vector<Vec3> x, v;
    for (int i = 0; i < 4; i++) {
        x.push_back(Vec3(positions[i][0], positions[i][1], positions[i][2]));
        v.push_back(Vec3(velocities[i][0], velocities[i][1], velocities[i][2]));
    }
    if (loadFile.empty()) {
        context.setPositions(x);
        context.setVelocities(v);
        integrator.step(20);
    }
    else
        force->loadCheckpointFile(context, loadFile);
    if (!saveFile.empty())
        force->saveCheckpointFile(context, saveFile);
    integrator.step(20);
    Result result;
    force->getFluidState(context, result.state);
    force->getFluidState(context, result.gathered, true);
    force->getFluidFields(context, result.density, result.velocity, false, true);
    force->getLocalDomain(context, result.start[0], result.start[1], result.start[2], result.count[0], result.count[1], result.count[2]);
    State state = context.getState(State::Positions | State::Velocities);
    result.positions = state.getPositions();
    result.velocities = state.getVelocities();
    return result;
}

int main(int argc, char* argv[]) {
    try {
        registerLBMReferenceKernelFactories();
        int rank = LBMForce::getMPIRank(), size = LBMForce::getMPISize();
        if (size != 2)
            throw OpenMMException("this test runs with 2 MPI ranks (mpiexec -n 2)");
        string file = "TestMPIReferenceLBMForce.chk";
        Result single = run(1), split = run(2, file), restartedSplit = run(2, "", file), restartedSingle = run(1, "", file);
        // The continued runs are the uninterrupted ones.
        ASSERT(restartedSplit.state == split.state && restartedSplit.positions == split.positions &&
               restartedSplit.velocities == split.velocities);
        for (int node = 0; node < NX*NY*NZ; node++)
            if ((node/NX)%NY != 0)
                for (int q = 0; q < 19; q++)
                    ASSERT(restartedSingle.state[q*NX*NY*NZ + node] == single.state[q*NX*NY*NZ + node]);
        ASSERT(restartedSingle.positions == single.positions && restartedSingle.velocities == single.velocities);
        const int Q = 19;
        int numNodes = NX*NY*NZ, numLocal = split.count[0]*split.count[1]*split.count[2];
        ASSERT_EQUAL(NX/2*NY*NZ, numLocal);
        ASSERT_EQUAL((size_t) Q*numLocal, split.state.size());
        // The fluid nodes of the domain (the slots of the solid nodes are not part of the state of the fluid).
        for (int l = 0; l < numLocal; l++) {
            int i = split.start[0] + l%split.count[0], j = split.start[1] + (l/split.count[0])%split.count[1];
            int k = split.start[2] + l/(split.count[0]*split.count[1]);
            if (j == 0)
                continue;
            int node = i + NX*(j + NY*k);
            for (int q = 0; q < Q; q++)
                ASSERT(split.state[q*numLocal + l] == single.state[q*numNodes + node]);
        }
        if (rank == 0) {
            ASSERT_EQUAL((size_t) Q*numNodes, split.gathered.size());
            for (int node = 0; node < numNodes; node++)
                if ((node/NX)%NY != 0)
                    for (int q = 0; q < Q; q++)
                        ASSERT(split.gathered[q*numNodes + node] == single.state[q*numNodes + node]);
        }
        else
            ASSERT(split.gathered.empty());
        // The domain with the halo against the fields of one domain, across the periodic boundaries.
        int size3[3] = {split.count[0]+2, split.count[1]+2, split.count[2]+2};
        for (int c = 0; c < size3[2]; c++)
            for (int b = 0; b < size3[1]; b++)
                for (int a = 0; a < size3[0]; a++) {
                    int l = a + size3[0]*(b + size3[1]*c);
                    int i = (split.start[0] + a - 1 + NX)%NX, j = (split.start[1] + b - 1 + NY)%NY;
                    int k = (split.start[2] + c - 1 + NZ)%NZ;
                    int l1 = (i + 1) + (NX + 2)*((j + 1) + (NY + 2)*(k + 1));
                    ASSERT(split.density[l] == single.density[l1]);
                    for (int d = 0; d < 3; d++)
                        ASSERT(split.velocity[l][d] == single.velocity[l1][d]);
                }
        for (int i = 0; i < 4; i++)
            for (int d = 0; d < 3; d++) {
                ASSERT(split.positions[i][d] == single.positions[i][d]);
                ASSERT(split.velocities[i][d] == single.velocities[i][d]);
            }
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    if (LBMForce::getMPIRank() == 0)
        cout << "Done" << endl;
    return 0;
}
