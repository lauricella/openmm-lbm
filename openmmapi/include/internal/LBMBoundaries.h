#ifndef OPENMM_LBM_BOUNDARIES_H_
#define OPENMM_LBM_BOUNDARIES_H_

/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "LBMKernels.h"
#include "internal/D3Q19.h"
#include <vector>

namespace LBMPlugin {

/**
 * The boundary nodes of the fluid (docs/theory.md, section 1), found in the same way on every platform: with
 * regularized walls the fluid nodes next to the solid nodes, which lie on the walls, and the fluid nodes on the
 * open faces of the box.  For each of them, the directions q whose source node x - c_q is solid, or lies beyond an
 * open face, are unknown after the streaming.  A node next to a solid node is a wall at rest; on the faces, the
 * first Velocity face of the node in the order XMin ... ZMax gives the velocity, and otherwise its Density faces
 * give the density.
 * @private
 */
class LBMBoundaries {
public:
    /** What a boundary node imposes: a wall at rest, the velocity of a face, the density of a face (with the velocity
        along the face zero), or the density of a face with zero velocity (nodes shared by several Density faces). */
    enum Kind {Wall = 0, Velocity = 1, Density = 2, DensityAtRest = 3};
    /** The boundary nodes and, for each of them, the bits 1 << q of the directions q whose populations are unknown
        after the streaming (unknown) and of those among them whose source node x - c_q is solid (solid), its Kind
        and the face that gives its velocity or density (-1 on walls). */
    std::vector<int> nodes, unknown, solid, kind, face;
    /** For every node: 1 for the nodes of regularized walls, 2 for the other boundary nodes (on open faces), 0
        elsewhere; empty if there are no boundary nodes. */
    std::vector<char> type;
    /**
     * Find the boundary nodes of a lattice.  isFluid is nonzero for the fluid nodes and zero for the solid ones,
     * or empty if there are no solid nodes.
     */
    template <class T>
    void find(const LBMLatticeParameters& lattice, const std::vector<T>& isFluid) {
        nodes.clear();
        unknown.clear();
        solid.clear();
        kind.clear();
        face.clear();
        type.clear();
        bool regularized = (!lattice.solidNodes.empty() && lattice.wallScheme == LBMForce::Regularized);
        if (!regularized && !lattice.hasOpenFaces())
            return;
        int size[3] = {lattice.nx, lattice.ny, lattice.nz};
        int nx = size[0], ny = size[1];
        int numNodes = lattice.getNumNodes();
        type.resize(numNodes, 0);
        for (int node = 0; node < numNodes; node++) {
            if (!isFluid.empty() && !isFluid[node])
                continue;
            int index[3] = {node%nx, (node/nx)%ny, node/(nx*ny)};
            int solidBits = 0, outsideBits = 0;
            for (int q = 1; q < D3Q19::numVelocities; q++) {
                int c[3] = {D3Q19::cx[q], D3Q19::cy[q], D3Q19::cz[q]}, source[3];
                bool beyondFace = false;
                for (int a = 0; a < 3; a++) {
                    source[a] = index[a] - c[a];
                    if (source[a] < 0 || source[a] >= size[a]) {
                        beyondFace = beyondFace || lattice.isOpenAxis(a);
                        source[a] = (source[a] + size[a])%size[a];
                    }
                }
                if (beyondFace)
                    outsideBits |= 1<<q;
                else if (regularized && !isFluid[source[0] + nx*(source[1] + ny*source[2])])
                    solidBits |= 1<<q;
            }
            if (solidBits == 0 && outsideBits == 0)
                continue;
            int nodeKind = Wall, nodeFace = -1;
            if (solidBits == 0) {
                int velocityFace = -1, densityFace = -1, numDensityFaces = 0;
                for (int f = 0; f < 6; f++) {
                    int a = f/2;
                    if (!lattice.isOpenAxis(a) || index[a] != (f%2 == 0 ? 0 : size[a]-1))
                        continue;
                    if (lattice.faceBoundary[f] == LBMForce::Velocity && velocityFace < 0)
                        velocityFace = f;
                    if (lattice.faceBoundary[f] == LBMForce::Density && numDensityFaces++ == 0)
                        densityFace = f;
                }
                if (velocityFace >= 0) {
                    nodeKind = Velocity;
                    nodeFace = velocityFace;
                }
                else {
                    nodeKind = (numDensityFaces == 1 ? Density : DensityAtRest);
                    nodeFace = densityFace;
                }
            }
            nodes.push_back(node);
            unknown.push_back(solidBits | outsideBits);
            solid.push_back(solidBits);
            kind.push_back(nodeKind);
            face.push_back(nodeFace);
            type[node] = (nodeKind == Wall ? 1 : 2);
        }
    }
};

} // namespace LBMPlugin

#endif /*OPENMM_LBM_BOUNDARIES_H_*/
