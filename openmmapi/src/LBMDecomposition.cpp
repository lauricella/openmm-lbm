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

void LBMDecomposition::abortIfParallel(int errorCode) {
#ifdef OPENMM_LBM_MPI
    int initialized, finalized;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized && !finalized) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size > 1)
            MPI_Abort(MPI_COMM_WORLD, errorCode);
    }
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

void LBMDecomposition::getDomainNodes(int pad, const bool open[3], vector<int>& node, vector<char>& inside) const {
    int start[3], count[3];
    getLocalDomain(start, count);
    node.clear();
    inside.clear();
    for (int k = -pad; k < count[2] + pad; k++)
        for (int j = -pad; j < count[1] + pad; j++)
            for (int i = -pad; i < count[0] + pad; i++) {
                int local[3] = {i, j, k}, global[3];
                bool isInside = true, beyondFace = false;
                for (int a = 0; a < 3; a++) {
                    isInside = isInside && local[a] >= 0 && local[a] < count[a];
                    global[a] = start[a] + local[a];
                    if (global[a] < 0 || global[a] >= n[a]) {
                        beyondFace = beyondFace || open[a];
                        global[a] = (global[a] + n[a])%n[a];
                    }
                }
                node.push_back(beyondFace ? -1 : global[0] + n[0]*(global[1] + n[1]*global[2]));
                inside.push_back(isInside ? 1 : 0);
            }
}

void LBMDecomposition::getDomainOfRank(int r, int start[3], int count[3]) const {
    int c[3] = {r%procs[0], (r/procs[0])%procs[1], r/(procs[0]*procs[1])};
    for (int a = 0; a < 3; a++) {
        start[a] = (int) (((long long) n[a])*c[a]/procs[a]);
        count[a] = (int) (((long long) n[a])*(c[a] + 1)/procs[a]) - start[a];
    }
}

void LBMDecomposition::gatherBlocks(const vector<double>& local, int valuesPerNode, vector<double>& global) const {
    if (size == 1) {
        global = local;
        return;
    }
#ifdef OPENMM_LBM_MPI
    // One element of the messages is a node (valuesPerNode doubles), so that the counts stay within an int.
    MPI_Datatype nodeType;
    MPI_Type_contiguous(valuesPerNode, MPI_DOUBLE, &nodeType);
    MPI_Type_commit(&nodeType);
    vector<int> counts(size), displacements(size);
    long long total = 0;
    for (int r = 0; r < size; r++) {
        int start[3], count[3];
        getDomainOfRank(r, start, count);
        counts[r] = count[0]*count[1]*count[2];
        displacements[r] = (int) total;
        total += counts[r];
    }
    vector<double> buffer(rank == 0 ? total*valuesPerNode : 0);
    MPI_Gatherv(local.data(), counts[rank], nodeType, buffer.data(), counts.data(), displacements.data(), nodeType, 0,
            MPI_COMM_WORLD);
    MPI_Type_free(&nodeType);
    global.clear();
    if (rank != 0)
        return;
    global.resize(((size_t) n[0])*n[1]*n[2]*valuesPerNode);
    for (int r = 0; r < size; r++) {
        int start[3], count[3];
        getDomainOfRank(r, start, count);
        size_t l = displacements[r];
        for (int k = 0; k < count[2]; k++)
            for (int j = 0; j < count[1]; j++)
                for (int i = 0; i < count[0]; i++, l++) {
                    size_t node = (start[0] + i) + ((size_t) n[0])*((start[1] + j) + ((size_t) n[1])*(start[2] + k));
                    for (int v = 0; v < valuesPerNode; v++)
                        global[node*valuesPerNode + v] = buffer[l*valuesPerNode + v];
                }
    }
#endif
}

void LBMDecomposition::scatterBlocks(const vector<double>& global, int valuesPerNode, vector<double>& local) const {
    if (size == 1) {
        local = global;
        return;
    }
#ifdef OPENMM_LBM_MPI
    MPI_Datatype nodeType;
    MPI_Type_contiguous(valuesPerNode, MPI_DOUBLE, &nodeType);
    MPI_Type_commit(&nodeType);
    vector<int> counts(size), displacements(size);
    long long total = 0;
    for (int r = 0; r < size; r++) {
        int start[3], count[3];
        getDomainOfRank(r, start, count);
        counts[r] = count[0]*count[1]*count[2];
        displacements[r] = (int) total;
        total += counts[r];
    }
    vector<double> buffer;
    if (rank == 0) {
        buffer.resize(total*valuesPerNode);
        for (int r = 0; r < size; r++) {
            int start[3], count[3];
            getDomainOfRank(r, start, count);
            size_t l = displacements[r];
            for (int k = 0; k < count[2]; k++)
                for (int j = 0; j < count[1]; j++)
                    for (int i = 0; i < count[0]; i++, l++) {
                        size_t node = (start[0] + i) + ((size_t) n[0])*((start[1] + j) + ((size_t) n[1])*(start[2] + k));
                        for (int v = 0; v < valuesPerNode; v++)
                            buffer[l*valuesPerNode + v] = global[node*valuesPerNode + v];
                    }
        }
    }
    local.resize(((size_t) counts[rank])*valuesPerNode);
    MPI_Scatterv(buffer.data(), counts.data(), displacements.data(), nodeType, local.data(), counts[rank], nodeType, 0,
            MPI_COMM_WORLD);
    MPI_Type_free(&nodeType);
#endif
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

void LBMDecomposition::sum(double* values, int count) const {
    if (size == 1)
        return;
#ifdef OPENMM_LBM_MPI
    MPI_Allreduce(MPI_IN_PLACE, values, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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

string LBMDecomposition::broadcast(const string& value) const {
    if (size == 1)
        return value;
#ifdef OPENMM_LBM_MPI
    int length = broadcast((int) value.size());
    vector<char> buffer(value.begin(), value.end());
    buffer.resize(length);
    MPI_Bcast(buffer.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
    return string(buffer.begin(), buffer.end());
#else
    return value;
#endif
}

bool LBMDecomposition::isSameOnAllRanks(unsigned long long value) const {
    if (size == 1)
        return true;
#ifdef OPENMM_LBM_MPI
    // The largest of the values and of their complements, that is the largest and the smallest value, in one call.
    unsigned long long local[2] = {value, ~value}, result[2];
    MPI_Allreduce(local, result, 2, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    return result[0] == ~result[1];
#else
    return true;
#endif
}

void LBMDecomposition::throwIfAnyError(const string& error) const {
    if (size == 1) {
        if (!error.empty())
            throw OpenMMException(error);
        return;
    }
#ifdef OPENMM_LBM_MPI
    int local = (error.empty() ? size : rank), first;
    MPI_Allreduce(&local, &first, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (first == size)
        return;
    int length = (int) error.size();
    MPI_Bcast(&length, 1, MPI_INT, first, MPI_COMM_WORLD);
    vector<char> message(error.begin(), error.end());
    message.resize(length);
    MPI_Bcast(message.data(), length, MPI_CHAR, first, MPI_COMM_WORLD);
    throw OpenMMException(string(message.begin(), message.end()));
#endif
}

void LBMDecomposition::requireSameOnAllRanks(const string& value, const string& what) const {
    string first = broadcast(value);
    string error;
    if (value != first) {
        stringstream message;
        message << "LBMForce: with the domain decomposition every MPI rank must use the same " << what << ": rank "
                << rank << " has " << value << ", rank 0 has " << first;
        error = message.str();
    }
    throwIfAnyError(error);
}

void LBMDecomposition::startExchange(const vector<vector<double> >& send, vector<vector<double> >& receive) {
    numRequests = 0;
    if (size == 1)
        return;
#ifdef OPENMM_LBM_MPI
    requests.resize(2*size*sizeof(MPI_Request));
    MPI_Request* request = (MPI_Request*) requests.data();
    for (int r = 0; r < size; r++)
        if (!receive[r].empty())
            MPI_Irecv(receive[r].data(), (int) receive[r].size(), MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &request[numRequests++]);
    for (int r = 0; r < size; r++)
        if (!send[r].empty())
            MPI_Isend(send[r].data(), (int) send[r].size(), MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &request[numRequests++]);
#endif
}

void LBMDecomposition::finishExchange() {
#ifdef OPENMM_LBM_MPI
    if (numRequests > 0)
        MPI_Waitall(numRequests, (MPI_Request*) requests.data(), MPI_STATUSES_IGNORE);
#endif
    numRequests = 0;
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
