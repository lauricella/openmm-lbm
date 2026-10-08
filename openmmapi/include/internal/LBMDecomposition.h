#ifndef OPENMM_LBM_DECOMPOSITION_H_
#define OPENMM_LBM_DECOMPOSITION_H_

/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "internal/windowsExportLBM.h"
#include <string>
#include <vector>

namespace LBMPlugin {

/**
 * The decomposition of the lattice into blocks, one per MPI rank of MPI_COMM_WORLD (setDomainDecomposition(),
 * docs/theory.md).  The px*py*pz blocks are numbered with x fastest, rank = cx + px*(cy + py*cz); along each axis
 * block c holds the nodes [n*c/p, n*(c + 1)/p), so blocks differ by at most one node.  Without MPI, or with one
 * block, every node is owned and the collective operations do nothing.  The MPI calls are all in
 * LBMDecomposition.cpp, which is the only file compiled with MPI (CMake option OPENMM_LBM_MPI): this header does
 * not include mpi.h.
 * @private
 */
class OPENMM_EXPORT_LBM LBMDecomposition {
public:
    /** True if the plugin was built with MPI. */
    static bool isMPIAvailable();
    /** The rank in MPI_COMM_WORLD, the number of ranks and the rank among those on the same node; 0, 1 and 0 without
        MPI.  The first call initializes MPI if nobody has done it yet, and MPI is finalized at exit. */
    static int getWorldRank();
    static int getWorldSize();
    static int getLocalRank();
    /** MPI_Abort on MPI_COMM_WORLD with the error code if MPI is running with more than one rank; otherwise nothing. */
    static void abortIfParallel(int errorCode);
    /**
     * Resolve a requested decomposition for a lattice of size nx, ny, nz: the zeros are chosen with MPI_Dims_create,
     * and the product must be the number of ranks.  1, 1, 1 needs no MPI.  Throws an OpenMMException otherwise.
     */
    static void resolve(int nx, int ny, int nz, const int requested[3], int procs[3]);
    /** One block: the whole lattice. */
    LBMDecomposition();
    /** The decomposition of an nx*ny*nz lattice into procs[0]*procs[1]*procs[2] blocks, for this rank. */
    LBMDecomposition(int nx, int ny, int nz, const int procs[3]);
    /** True if there is more than one block. */
    bool isDecomposed() const {
        return size > 1;
    }
    int getRank() const {
        return rank;
    }
    int getSize() const {
        return size;
    }
    /** The block of the rank that owns the node index i along axis a. */
    int blockOf(int axis, int i) const {
        return (procs[axis]*(i + 1) - 1)/n[axis];
    }
    /** The rank that owns node (i, j, k). */
    int ownerOf(int i, int j, int k) const {
        return blockOf(0, i) + procs[0]*(blockOf(1, j) + procs[1]*blockOf(2, k));
    }
    /** The rank that owns the node with index i + nx*(j + ny*k). */
    int ownerOfNode(int node) const {
        return ownerOf(node%n[0], (node/n[0])%n[1], node/(n[0]*n[1]));
    }
    bool owns(int node) const {
        return size == 1 || ownerOfNode(node) == rank;
    }
    /** The first node and the number of nodes of this rank along each axis. */
    void getLocalDomain(int start[3], int count[3]) const;
    /** Replace each of the n values by its sum over the ranks, added in rank order (the same on every rank and for
        every run with the same decomposition). */
    void sumInRankOrder(double* values, int n) const;
    /** Replace each of the n values by its sum over the ranks (MPI_Allreduce).  Exact, and the same on every rank,
        when at most one rank has a value different from zero, as for the coupling forces of the particles. */
    void sum(double* values, int n) const;
    /** The largest value over the ranks. */
    double maximum(double value) const;
    /** The value of rank 0, on every rank. */
    int broadcast(int value) const;
    std::string broadcast(const std::string& value) const;
    /** True on every rank if the value is the same on every rank. */
    bool isSameOnAllRanks(unsigned long long value) const;
    /**
     * Collective error check: if the error of some rank is not empty, every rank throws an OpenMMException with the
     * message of the first such rank; otherwise it returns.  An error found by one rank only must reach all of them,
     * or the others would wait forever in the next communication.
     */
    void throwIfAnyError(const std::string& error) const;
    /** Collective check that a setting (for example the platform and its precision) is the same on every rank: if
        not, every rank throws an OpenMMException that names it. */
    void requireSameOnAllRanks(const std::string& value, const std::string& what) const;
    /** Exchange values with other ranks: send[r] goes to rank r, and receive[r], already sized, comes from rank r.
        Empty vectors are not sent. */
    void exchange(const std::vector<std::vector<double> >& send, std::vector<std::vector<double> >& receive) const;
    /** The same exchange without blocking: it starts the sends and the receives and returns at once, so that the
        caller can compute meanwhile; finishExchange() waits for them.  The vectors must not change in between. */
    void startExchange(const std::vector<std::vector<double> >& send, std::vector<std::vector<double> >& receive);
    void finishExchange();
private:
    int n[3], procs[3], coords[3], rank, size;
    std::vector<char> requests;         // the MPI_Request of the pending exchange (mpi.h is not included here)
    int numRequests = 0;
};

} // namespace LBMPlugin

#endif /*OPENMM_LBM_DECOMPOSITION_H_*/
