#ifndef OPENMM_LBM_D3Q19_H_
#define OPENMM_LBM_D3Q19_H_

/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

namespace LBMPlugin {

/**
 * The D3Q19 velocity set in lattice units, with the ordering of the populations used everywhere in
 * the plugin.  Populations are stored as f[q*numNodes + node], and node (i, j, k) has index
 * i + nx*(j + ny*k).
 *
 *   q = 0: rest;  1-6: (+-1, 0, 0), (0, +-1, 0), (0, 0, +-1);  7-18: the twelve diagonals.
 *
 * The equilibrium is the second-order Hermite expansion of the weakly compressible model:
 * the zeroth moment is the density rho and the first moment is the momentum j = rho*u.
 * @private
 */
class D3Q19 {
public:
    static const int numVelocities = 19;
    static constexpr int cx[19] = {0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 0,  0,  0,  0, 1, -1, -1,  1};
    static constexpr int cy[19] = {0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 1, -1,  1, -1, 0,  0,  0,  0};
    static constexpr int cz[19] = {0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1,  1, -1};
    static constexpr double w[19] = {1.0/3.0,
            1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
            1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
            1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
    /**
     * Compute the equilibrium populations f_q = w_q rho [1 + c.u/cs^2 + (c.u)^2/(2 cs^4) - u.u/(2 cs^2)]
     * with cs^2 = 1/3, for density rho and velocity (ux, uy, uz) in lattice units.
     */
    static void equilibrium(double rho, double ux, double uy, double uz, double feq[19]) {
        double uu = ux*ux + uy*uy + uz*uz;
        for (int q = 0; q < 19; q++) {
            double cu = cx[q]*ux + cy[q]*uy + cz[q]*uz;
            feq[q] = w[q]*rho*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*uu);
        }
    }
};

} // namespace LBMPlugin

#endif /*OPENMM_LBM_D3Q19_H_*/
