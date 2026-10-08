/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "internal/LBMDecomposition.h"
#include "openmm/OpenMMException.h"
#include <algorithm>
#include <cstdlib>
#include <sstream>
#ifdef OPENMM_LBM_MPI
#include <mpi.h>
#endif

using namespace LBMPlugin;
using namespace OpenMM;
using namespace std;

#ifdef OPENMM_LBM_MPI
static void finalizeMPI() {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized)
        MPI_Finalize();
}

// Initialize MPI on first use, unless the program (or mpi4py) has done it; then finalize it at exit.
static void ensureMPI() {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
        atexit(finalizeMPI);
    }
}
#endif

bool LBMDecomposition::isMPIAvailable() {
#ifdef OPENMM_LBM_MPI
    return true;
#else
    return false;
#endif
}

int LBMDecomposition::getWorldRank() {
#ifdef OPENMM_LBM_MPI
    ensureMPI();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
#else
    return 0;
#endif
}

int LBMDecomposition::getWorldSize() {
#ifdef OPENMM_LBM_MPI
    ensureMPI();
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
#else
    return 1;
#endif
}

int LBMDecomposition::getLocalRank() {
#ifdef OPENMM_LBM_MPI
    ensureMPI();
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int rank;
    MPI_Comm_rank(local, &rank);
    MPI_Comm_free(&local);
    return rank;
#else
    return 0;
#endif
}

void LBMDecomposition::resolve(int nx, int ny, int nz, const int requested[3], int procs[3]) {
    for (int a = 0; a < 3; a++) {
        if (requested[a] < 0)
            throw OpenMMException("LBMForce: the domain decomposition (setDomainDecomposition()) cannot be negative");
        procs[a] = requested[a];
    }
    if (procs[0] == 1 && procs[1] == 1 && procs[2] == 1)
        return;
    if (!isMPIAvailable())
        throw OpenMMException("LBMForce: setDomainDecomposition() asks for more than one domain, but the plugin was "
                "built without MPI (CMake option OPENMM_LBM_MPI)");
#ifdef OPENMM_LBM_MPI
    int size = getWorldSize();
    if (procs[0] == 0 || procs[1] == 0 || procs[2] == 0)
        MPI_Dims_create(size, 3, procs);
    int n[3] = {nx, ny, nz};
    if (procs[0]*procs[1]*procs[2] != size) {
        stringstream message;
        message << "LBMForce: the domain decomposition " << procs[0] << " x " << procs[1] << " x " << procs[2]
                << " (setDomainDecomposition()) does not match the " << size << " MPI ranks";
        throw OpenMMException(message.str());
    }
    for (int a = 0; a < 3; a++)
        if (procs[a] > n[a]) {
            stringstream message;
            message << "LBMForce: the domain decomposition has " << procs[a] << " domains along axis " << a
                    << ", more than the " << n[a] << " nodes of the lattice";
            throw OpenMMException(message.str());
        }
#endif
}

LBMDecomposition::LBMDecomposition() : rank(0), size(1) {
    for (int a = 0; a < 3; a++) {
        n[a] = 1;
        procs[a] = 1;
        coords[a] = 0;
    }
}

LBMDecomposition::LBMDecomposition(int nx, int ny, int nz, const int procs[3]) {
    n[0] = nx;
    n[1] = ny;
    n[2] = nz;
    size = 1;
    for (int a = 0; a < 3; a++) {
        this->procs[a] = procs[a];
        size *= procs[a];
    }
    rank = (size > 1 ? getWorldRank() : 0);
    coords[0] = rank%procs[0];
    coords[1] = (rank/procs[0])%procs[1];
    coords[2] = rank/(procs[0]*procs[1]);
}

void LBMDecomposition::getLocalDomain(int start[3], int count[3]) const {
    for (int a = 0; a < 3; a++) {
        start[a] = (int) (((long long) n[a])*coords[a]/procs[a]);
        count[a] = (int) (((long long) n[a])*(coords[a] + 1)/procs[a]) - start[a];
    }
}

void LBMDecomposition::sumInRankOrder(double* values, int count) const {
    if (size == 1)
        return;
#ifdef OPENMM_LBM_MPI
    // MPI_Allreduce may add in an order that depends on the implementation: gather the partial sums and add them
    // in rank order instead, so that the result is the same on every rank and in every run.
    vector<double> all(count*size);
    MPI_Allgather(values, count, MPI_DOUBLE, all.data(), count, MPI_DOUBLE, MPI_COMM_WORLD);
    for (int k = 0; k < count; k++) {
        double sum = 0;
        for (int r = 0; r < size; r++)
            sum += all[r*count + k];
        values[k] = sum;
    }
#endif
}

double LBMDecomposition::maximum(double value) const {
    if (size == 1)
        return value;
#ifdef OPENMM_LBM_MPI
    double result;
    MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
#else
    return value;
#endif
}

int LBMDecomposition::broadcast(int value) const {
#ifdef OPENMM_LBM_MPI
    if (size > 1)
        MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    return value;
}

void LBMDecomposition::exchange(const vector<vector<double> >& send, vector<vector<double> >& receive) const {
    if (size == 1)
        return;
#ifdef OPENMM_LBM_MPI
    vector<MPI_Request> requests;
    for (int r = 0; r < size; r++)
        if (!receive[r].empty()) {
            requests.push_back(MPI_Request());
            MPI_Irecv(receive[r].data(), (int) receive[r].size(), MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &requests.back());
        }
    for (int r = 0; r < size; r++)
        if (!send[r].empty()) {
            requests.push_back(MPI_Request());
            MPI_Isend(send[r].data(), (int) send[r].size(), MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &requests.back());
        }
    MPI_Waitall((int) requests.size(), requests.data(), MPI_STATUSES_IGNORE);
#endif
}
