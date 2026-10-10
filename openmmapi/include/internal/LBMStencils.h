#ifndef OPENMM_LBM_STENCILS_H_
#define OPENMM_LBM_STENCILS_H_

/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "LBMForce.h"
#include <algorithm>
#include <cmath>

namespace LBMPlugin {

/**
 * The interpolation stencils of the coupling (docs/theory.md, section 9), the same on every platform.  The weight of
 * node j for a particle at X is the product over the three axes of phi((X - x_j)/dx), with the kernels of the trilinear
 * interpolation (Birdsall and Fuss 1969), of Roma, Peskin and Berger (1999) and of Keys (1981), written from the
 * formulas of the papers.
 * @private
 */
class LBMStencils {
public:
    /** The largest number of nodes of a stencil along an axis. */
    static const int maxWidth = 4;
    /** The number of nodes of the stencil along an axis: 1, 2, 3 or 4. */
    static int width(LBMForce::InterpolationStencil stencil) {
        switch (stencil) {
            case LBMForce::Trilinear:
                return 2;
            case LBMForce::ThreePoint:
                return 3;
            case LBMForce::Keys:
                return 4;
            default:
                return 1;
        }
    }
    /** The kernel phi(r) of the stencil, r being the distance from the node in lattice spacings. */
    static double kernel(LBMForce::InterpolationStencil stencil, double r) {
        double s = std::fabs(r);
        switch (stencil) {
            case LBMForce::Trilinear:
                return (s < 1.0 ? 1.0-s : 0.0);
            case LBMForce::ThreePoint:
                if (s <= 0.5)
                    return (1.0 + std::sqrt(1.0-3.0*s*s))/3.0;
                if (s < 1.5)
                    return (5.0 - 3.0*s - std::sqrt(std::max(0.0, 1.0-3.0*(1.0-s)*(1.0-s))))/6.0;
                return 0.0;
            case LBMForce::Keys:
                if (s <= 1.0)
                    return 1.0 - 2.5*s*s + 1.5*s*s*s;
                if (s < 2.0)
                    return 2.0 - 4.0*s + 2.5*s*s - 0.5*s*s*s;
                return 0.0;
            default:
                return (r >= -0.5 && r < 0.5 ? 1.0 : 0.0);
        }
    }
    /**
     * The nodes of the stencil along one axis and their weights, for a coordinate s in lattice spacings, already
     * wrapped into [0, n): width(stencil) nodes in increasing order, starting from floor(s) for the trilinear
     * kernel, round(s) - 1 for the three-point kernel and floor(s) - 1 for Keys, wrapped periodically into [0, n),
     * or along an axis with open faces (open) moved onto the nearest node of the lattice, 0 or n - 1, so that a stencil
     * never reaches across an open face.  Nodes outside the support of the kernel have weight 0 (the trilinear kernel
     * at a node).  A node that appears in more than one slot (an axis shorter than the stencil, or the nodes beyond an
     * open face) gets the sum of their weights, added in slot order, in its first slot, and the other slots weight 0,
     * so that the sum of the squares of the weights of the stencil is that of its distinct nodes.
     */
    static void axisWeights(LBMForce::InterpolationStencil stencil, double s, int n, bool open, int* index, double* weight) {
        int w = width(stencil);
        int first;
        if (stencil == LBMForce::ThreePoint)
            first = (int) std::floor(s + 0.5) - 1;
        else if (stencil == LBMForce::Keys)
            first = (int) std::floor(s) - 1;
        else if (stencil == LBMForce::Trilinear)
            first = (int) std::floor(s);
        else
            first = (int) std::floor(s + 0.5);
        for (int i = 0; i < w; i++) {
            weight[i] = kernel(stencil, s - (first + i));
            index[i] = (open ? std::min(std::max(first + i, 0), n - 1) : ((first + i)%n + n)%n);
        }
        for (int i = 1; i < w; i++)
            for (int j = 0; j < i; j++)
                if (index[j] == index[i]) {
                    weight[j] += weight[i];
                    weight[i] = 0.0;
                    break;
                }
    }
    /**
     * The node that owns a particle with a stencil along one axis, for a coordinate s wrapped into [0, n): the nearest
     * node, wrapped periodically, or along an axis with open faces the nearest node of the lattice, so that the
     * particles between node n - 1 and the end of the box belong to node n - 1, like their stencils.
     */
    static int ownerIndex(double s, int n, bool open) {
        int i = (int) std::floor(s + 0.5);
        return (open ? std::min(i, n - 1) : i%n);
    }
};

} // namespace LBMPlugin

#endif /*OPENMM_LBM_STENCILS_H_*/
