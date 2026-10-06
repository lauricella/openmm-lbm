/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

/**
 * Compute the density (moment 0) and the momentum (moment 1) of the populations at every node.
 * Populations are stored as deviations from the rest equilibrium, df[q*NUM_NODES + node] = f_q - w_q, with
 * the D3Q19 ordering of internal/D3Q19.h, so rho = 1 + sum df and j = sum c df; the momentum is stored as
 * j[node], j[NUM_NODES + node], j[2*NUM_NODES + node].
 */
KERNEL void computeFluidMoments(GLOBAL const mixed* RESTRICT f, GLOBAL mixed* RESTRICT density, GLOBAL mixed* RESTRICT momentum) {
    for (int node = GLOBAL_ID; node < NUM_NODES; node += GLOBAL_SIZE) {
        mixed fq[19];
        for (int q = 0; q < 19; q++)
            fq[q] = f[q*NUM_NODES+node];
        mixed drho = 0;
        for (int q = 0; q < 19; q++)
            drho += fq[q];
        density[node] = 1 + drho;
        momentum[node] = (fq[1]+fq[7]+fq[9]+fq[15]+fq[18]) - (fq[2]+fq[8]+fq[10]+fq[16]+fq[17]);
        momentum[NUM_NODES+node] = (fq[3]+fq[7]+fq[10]+fq[11]+fq[13]) - (fq[4]+fq[8]+fq[9]+fq[12]+fq[14]);
        momentum[2*NUM_NODES+node] = (fq[5]+fq[11]+fq[14]+fq[15]+fq[17]) - (fq[6]+fq[12]+fq[13]+fq[16]+fq[18]);
    }
}
