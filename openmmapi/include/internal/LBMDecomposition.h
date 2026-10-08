#ifndef OPENMM_LBM_DECOMPOSITION_H_
#define OPENMM_LBM_DECOMPOSITION_H_

/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "internal/windowsExportLBM.h"
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
    /** The largest value over the ranks. */
    double maximum(double value) const;
    /** The value of rank 0, on every rank. */
    int broadcast(int value) const;
    /** Exchange values with other ranks: send[r] goes to rank r, and receive[r], already sized, comes from rank r.
        Empty vectors are not sent. */
    void exchange(const std::vector<std::vector<double> >& send, std::vector<std::vector<double> >& receive) const;
private:
    int n[3], procs[3], coords[3], rank, size;
};

} // namespace LBMPlugin

#endif /*OPENMM_LBM_DECOMPOSITION_H_*/
