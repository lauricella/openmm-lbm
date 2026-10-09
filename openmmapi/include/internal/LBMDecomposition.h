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
    /** The size of the lattice and the number of blocks along each axis. */
    void getGridSize(int nodes[3]) const {
        for (int a = 0; a < 3; a++)
            nodes[a] = n[a];
    }
    void getBlocks(int blocks[3]) const {
        for (int a = 0; a < 3; a++)
            blocks[a] = procs[a];
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
    /** The same for rank r. */
    void getDomainOfRank(int r, int start[3], int count[3]) const;
    /**
     * The nodes of the domain of this rank extended by pad nodes on every side, in the order of the domain (i
     * fastest): node[l] is the index of the node of the lattice at position l, across the periodic boundaries, or -1
     * beyond an open face (open[a] is true for the axes with open faces), and inside[l] is 1 for the nodes of the
     * domain itself.
     */
    void getDomainNodes(int pad, const bool open[3], std::vector<int>& node, std::vector<char>& inside) const;
    /**
     * Gather on rank 0 the values of the nodes of every rank: local holds valuesPerNode values for each node of this
     * rank, in the order of the block (i fastest); on rank 0, global receives valuesPerNode values for each node of
     * the lattice, in the order of the node index i + nx*(j + ny*k).  Collective; global is left empty on the other
     * ranks.
     */
    void gatherBlocks(const std::vector<double>& local, int valuesPerNode, std::vector<double>& global) const;
    /** The reverse of gatherBlocks(): rank 0 sends to every rank the values of its nodes.  Collective; global is
        read only on rank 0. */
    void scatterBlocks(const std::vector<double>& global, int valuesPerNode, std::vector<double>& local) const;
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

/**
 * A file that the ranks of a decomposition write or read together, for the checkpoints and the VTK files of the
 * fluid: with more than one block through MPI-IO, so that each rank writes or reads its own part of one file, and
 * with one block through the standard library.  The arrays of the lattice are stored in the order of the node index
 * i + nx*(j + ny*k) whatever the decomposition, so that a file written with one decomposition can be read with any
 * other.  The methods are collective unless stated otherwise; an error on any rank is thrown on every rank, by the
 * collective method in which it happens or by close().
 * @private
 */
class OPENMM_EXPORT_LBM LBMParallelFile {
public:
    /** Open the file to write it (created, or emptied if it exists) or to read it. */
    LBMParallelFile(const LBMDecomposition& decomposition, const std::string& path, bool write);
    ~LBMParallelFile();
    /** Rank 0 writes the bytes at the offset; the other ranks write nothing.  Not collective. */
    void writeOnRoot(long long offset, const std::string& data);
    /** Every rank writes its own bytes at its own offset.  Not collective. */
    void writeAt(long long offset, const char* data, long long length);
    /** Every rank reads the bytes at the offset.  Not collective; a read past the end of the file is an error. */
    void readAt(long long offset, char* data, long long length);
    /**
     * Write an array of the whole lattice that starts at the offset, with valuesPerNode values of elementSize bytes
     * for each node: node by node ([node*valuesPerNode + v]) or, with valueMajor, value by value
     * ([v*numNodes + node]).  Every rank gives the values of the nodes of its domain, in the same layout over the
     * domain (numNodes and the node index replaced by those of the domain, getLocalDomain()).
     */
    void writeDomain(long long offset, const void* data, int elementSize, int valuesPerNode, bool valueMajor);
    /** Read the part of such an array that belongs to the domain of this rank. */
    void readDomain(long long offset, void* data, int elementSize, int valuesPerNode, bool valueMajor);
    /** The size of the file in bytes. */
    long long getSize();
    /** Throw on every rank the first error of any rank, if there is one. */
    void checkErrors();
    /** Close the file, and throw on every rank the first error of any rank. */
    void close();
private:
    void transferDomain(long long offset, void* data, int elementSize, int valuesPerNode, bool valueMajor, bool write);
    const LBMDecomposition& decomposition;
    std::string path, error;
    bool parallel, isOpen;
    std::vector<char> handle;           // the MPI_File with more than one rank
    void* stream;                       // the std::fstream with one rank
};

} // namespace LBMPlugin

#endif /*OPENMM_LBM_DECOMPOSITION_H_*/
