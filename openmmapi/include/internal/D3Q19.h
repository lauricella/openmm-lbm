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
    /** Index of the opposite velocity, c[opposite[q]] = -c[q]. */
    static constexpr int opposite[19] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
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
    /**
     * Compute the deviation of the equilibrium from the rest equilibrium at lattice density 1,
     *   feq_q - w_q = w_q [drho + rho (c.u/cs^2 + (c.u)^2/(2 cs^4) - u.u/(2 cs^2))],
     * for the density rho = 1 + drho and the velocity (ux, uy, uz).  The plugin stores the populations as
     * deviations f_q - w_q from the rest equilibrium, so that small signals keep their full precision.
     */
    static void equilibriumDeviation(double drho, double ux, double uy, double uz, double dfeq[19]) {
        double rho = 1.0 + drho;
        double uu = ux*ux + uy*uy + uz*uz;
        for (int q = 0; q < 19; q++) {
            double cu = cx[q]*ux + cy[q]*uy + cz[q]*uz;
            dfeq[q] = w[q]*(drho + rho*(3.0*cu + 4.5*cu*cu - 1.5*uu));
        }
    }
    /**
     * Second-order Hermite polynomial H2(c_q) = c_q c_q - cs^2 I, as the six components xx, yy, zz, xy,
     * xz, yz.
     */
    static void hermite2(int q, double h[6]) {
        h[0] = cx[q]*cx[q] - 1.0/3.0;
        h[1] = cy[q]*cy[q] - 1.0/3.0;
        h[2] = cz[q]*cz[q] - 1.0/3.0;
        h[3] = cx[q]*cy[q];
        h[4] = cx[q]*cz[q];
        h[5] = cy[q]*cz[q];
    }
    /**
     * Non-equilibrium second moment Pi_neq = sum_q H2(c_q) (f_q - feq_q) of the populations f at one node,
     * with the equilibrium at the velocity j/rho of the populations themselves (no forcing shift).
     * Since feq has the same density as f, sum_q (f_q - feq_q) = 0 and the -cs^2 I part of H2 does not
     * contribute: Pi_neq equals the plain second moment of f - feq.  Components xx, yy, zz, xy, xz, yz.
     */
    static void nonEquilibriumMoment(const double f[19], double rho, double jx, double jy, double jz, double pi[6]) {
        double feq[19], h[6];
        equilibrium(rho, jx/rho, jy/rho, jz/rho, feq);
        for (int k = 0; k < 6; k++)
            pi[k] = 0.0;
        for (int q = 0; q < 19; q++) {
            hermite2(q, h);
            double fneq = f[q]-feq[q];
            for (int k = 0; k < 6; k++)
                pi[k] += h[k]*fneq;
        }
    }
    /**
     * The same moment computed from the deviations df_q = f_q - w_q of the populations from the rest
     * equilibrium, with drho = sum_q df_q and j = sum_q c_q df_q.  The rest equilibrium cancels in f - feq, so
     * the difference is taken between small numbers.
     */
    static void nonEquilibriumMomentFromDeviation(const double df[19], double drho, double jx, double jy, double jz, double pi[6]) {
        double rho = 1.0 + drho;
        double dfeq[19], h[6];
        equilibriumDeviation(drho, jx/rho, jy/rho, jz/rho, dfeq);
        for (int k = 0; k < 6; k++)
            pi[k] = 0.0;
        for (int q = 0; q < 19; q++) {
            hermite2(q, h);
            double fneq = df[q]-dfeq[q];
            for (int k = 0; k < 6; k++)
                pi[k] += h[k]*fneq;
        }
    }
    /**
     * Regularized non-equilibrium populations fneq_q = w_q/(2 cs^4) H2(c_q):Pi_neq, rebuilt from the second
     * moment alone, so that all higher (ghost) moments are zero.  The -cs^2 I part of H2 is needed here: it
     * gives the term -cs^2 tr(Pi_neq), without which fneq would carry mass.
     */
    static void regularizedNonEquilibrium(const double pi[6], double fneq[19]) {
        double h[6];
        for (int q = 0; q < 19; q++) {
            hermite2(q, h);
            fneq[q] = 4.5*w[q]*(h[0]*pi[0] + h[1]*pi[1] + h[2]*pi[2] + 2.0*(h[3]*pi[3] + h[4]*pi[4] + h[5]*pi[5]));
        }
    }
    /**
     * Guo forcing term S_q = w_q [(c_q - u).F/cs^2 + (c_q.u)(c_q.F)/cs^4] for the force density F and the
     * velocity u.  Its zeroth moment is 0 and its first moment is F.
     */
    static void guoForcing(double ux, double uy, double uz, double fx, double fy, double fz, double s[19]) {
        double uf = ux*fx + uy*fy + uz*fz;
        for (int q = 0; q < 19; q++) {
            double cu = cx[q]*ux + cy[q]*uy + cz[q]*uz;
            double cf = cx[q]*fx + cy[q]*fy + cz[q]*fz;
            s[q] = w[q]*(3.0*(cf - uf) + 9.0*cu*cf);
        }
    }
};

} // namespace LBMPlugin

#endif /*OPENMM_LBM_D3Q19_H_*/
