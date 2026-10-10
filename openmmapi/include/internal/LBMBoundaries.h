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
 * regularized walls the fluid nodes next to the solid nodes, and the fluid nodes on the open faces of the box.  For each of them, the directions q whose source node x - c_q is solid, or lies beyond an
 * open face, are unknown after the streaming.  A node next to a solid node is a wall at rest; on the faces, the
 * first face of the node that sets the velocity (Velocity or DensityVelocity) in the order XMin ... ZMax gives the
 * velocity, and the density too if it is a DensityVelocity face, and otherwise its Density faces give the density.
 * @private
 */
class LBMBoundaries {
public:
    /** What a boundary node imposes on the populations that it rebuilds: a wall at rest, the velocity of a face, the
        density of a face, the density of a face with zero velocity (nodes shared by several Density faces), or both
        the density and the velocity of a face. */
    enum Kind {Wall = 0, Velocity = 1, Density = 2, DensityAtRest = 3, DensityVelocity = 4};
    /** The boundary nodes and, for each of them, the bits 1 << q of the directions q whose populations are unknown
        after the streaming (unknown) and of those among them whose source node x - c_q is solid (solid), its Kind
        and the face that gives its velocity or density (-1 on walls). */
    std::vector<int> nodes, unknown, solid, kind, face;
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
        bool regularized = (!lattice.solidNodes.empty() && lattice.wallScheme == LBMForce::Regularized);
        if (!regularized && !lattice.hasOpenFaces())
            return;
        int size[3] = {lattice.nx, lattice.ny, lattice.nz};
        int nx = size[0], ny = size[1];
        int numNodes = lattice.getNumNodes();
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
                    LBMForce::BoundaryType type = lattice.faceBoundary[f];
                    if ((type == LBMForce::Velocity || type == LBMForce::DensityVelocity) && velocityFace < 0)
                        velocityFace = f;
                    if (type == LBMForce::Density && numDensityFaces++ == 0)
                        densityFace = f;
                }
                if (velocityFace >= 0) {
                    bool both = (lattice.faceBoundary[velocityFace] == LBMForce::DensityVelocity);
                    nodeKind = (both ? DensityVelocity : Velocity);
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
        }
    }
    /**
     * The fluid nodes next to the walls with the halfway bounce-back (wall scheme BounceBack, docs/theory.md, section
     * 1) and, for each of them, the bits 1 << q of the directions q whose node x + c_q is solid, without the links
     * that cross an open face.  After the streaming each of these nodes takes back, along -c_q, the population that it
     * built for the direction q and sent into the solid node.  isFluid is nonzero for the fluid nodes and zero for the
     * solid ones; the lists are empty with regularized walls or without solid nodes.
     */
    template <class T>
    static void findWallLinks(const LBMLatticeParameters& lattice, const std::vector<T>& isFluid,
            std::vector<int>& wallNodes, std::vector<int>& wallLinks) {
        wallNodes.clear();
        wallLinks.clear();
        if (lattice.solidNodes.empty() || lattice.wallScheme != LBMForce::BounceBack)
            return;
        int size[3] = {lattice.nx, lattice.ny, lattice.nz};
        int nx = size[0], ny = size[1];
        int numNodes = lattice.getNumNodes();
        for (int node = 0; node < numNodes; node++) {
            if (!isFluid[node])
                continue;
            int index[3] = {node%nx, (node/nx)%ny, node/(nx*ny)};
            int links = 0;
            for (int q = 1; q < D3Q19::numVelocities; q++) {
                int c[3] = {D3Q19::cx[q], D3Q19::cy[q], D3Q19::cz[q]}, target[3];
                bool crossesFace = false;
                for (int a = 0; a < 3; a++) {
                    target[a] = index[a] + c[a];
                    crossesFace = crossesFace || (lattice.isOpenAxis(a) && (target[a] < 0 || target[a] >= size[a]));
                    target[a] = (target[a] + size[a])%size[a];
                }
                if (!crossesFace && !isFluid[target[0] + nx*(target[1] + ny*target[2])])
                    links |= 1<<q;
            }
            if (links != 0) {
                wallNodes.push_back(node);
                wallLinks.push_back(links);
            }
        }
    }
};

} // namespace LBMPlugin

#endif /*OPENMM_LBM_BOUNDARIES_H_*/
