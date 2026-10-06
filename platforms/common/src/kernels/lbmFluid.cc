/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

/**
 * The lattice update of the fluid, with the arithmetic of the Reference platform (ReferenceLBMKernels.cpp and
 * internal/D3Q19.h), so that the two agree to rounding.
 *
 * Populations are stored as deviations from the rest equilibrium, df[q*NUM_NODES + node] = f_q - w_q, with the
 * D3Q19 ordering of internal/D3Q19.h; node (i, j, k) has index i + NX*(j + NY*k).  The moments of a step are
 * stored component by component: the density as rho - 1, the momentum j as [k*NUM_NODES + node] (k = x, y, z)
 * and the non-equilibrium second moment as [k*NUM_NODES + node] (k = xx, yy, zz, xy, xz, yz).
 *
 * Defines: NUM_NODES, NX, NY, NZ, LBM_BLOCK_SIZE (the work group size of the reductions, a power of 2), the
 * weights W0, W1, W2 (1/3, 1/18, 1/36) and CS2 (1/3) in the mixed type.  No atomic operations are used: sums and
 * maxima are reduced by work group, then over the work groups in a fixed order, so that runs are reproducible.
 */

#define DECLARE_D3Q19_VELOCITIES \
    const int cx[19] = {0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 0,  0,  0,  0, 1, -1, -1,  1}; \
    const int cy[19] = {0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 1, -1,  1, -1, 0,  0,  0,  0}; \
    const int cz[19] = {0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1,  1, -1};

DEVICE mixed latticeWeight(int q) {
    return (q == 0 ? W0 : (q < 7 ? W1 : W2));
}

/**
 * Compute the moments of every node: rho - 1 = sum df, j = sum c df, and the non-equilibrium second moment
 * Pi_neq = sum H2(c) (df - dfeq), with the equilibrium at the velocity j/rho of the populations themselves.
 */
KERNEL void computeFluidMoments(GLOBAL const mixed* RESTRICT f, GLOBAL mixed* RESTRICT densityDeviation,
        GLOBAL mixed* RESTRICT momentum, GLOBAL mixed* RESTRICT piNeq) {
    DECLARE_D3Q19_VELOCITIES
    for (int node = GLOBAL_ID; node < NUM_NODES; node += GLOBAL_SIZE) {
        mixed df[19];
        mixed dr = 0, jx = 0, jy = 0, jz = 0;
        for (int q = 0; q < 19; q++) {
            df[q] = f[q*NUM_NODES+node];
            dr += df[q];
            jx += cx[q]*df[q];
            jy += cy[q]*df[q];
            jz += cz[q]*df[q];
        }
        densityDeviation[node] = dr;
        momentum[node] = jx;
        momentum[NUM_NODES+node] = jy;
        momentum[2*NUM_NODES+node] = jz;
        mixed rho = 1 + dr;
        mixed ux = jx/rho, uy = jy/rho, uz = jz/rho;
        mixed uu = ux*ux + uy*uy + uz*uz;
        mixed pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
        for (int q = 0; q < 19; q++) {
            mixed cu = cx[q]*ux + cy[q]*uy + cz[q]*uz;
            mixed dfeq = latticeWeight(q)*(dr + rho*(3.0f*cu + 4.5f*cu*cu - 1.5f*uu));
            mixed fneq = df[q]-dfeq;
            pxx += ((mixed) (cx[q]*cx[q]) - CS2)*fneq;
            pyy += ((mixed) (cy[q]*cy[q]) - CS2)*fneq;
            pzz += ((mixed) (cz[q]*cz[q]) - CS2)*fneq;
            pxy += (cx[q]*cy[q])*fneq;
            pxz += (cx[q]*cz[q])*fneq;
            pyz += (cy[q]*cz[q])*fneq;
        }
        piNeq[node] = pxx;
        piNeq[NUM_NODES+node] = pyy;
        piNeq[2*NUM_NODES+node] = pzz;
        piNeq[3*NUM_NODES+node] = pxy;
        piNeq[4*NUM_NODES+node] = pxz;
        piNeq[5*NUM_NODES+node] = pyz;
    }
}

/**
 * First stage of the removal of the fluid momentum: each work group sums rho - 1 and j over its nodes and writes
 * the four sums to partialSums[4*group + k].
 */
KERNEL void sumFluidMomentum(GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL const mixed* RESTRICT momentum,
        GLOBAL mixed* RESTRICT partialSums) {
    LOCAL mixed sums[4*LBM_BLOCK_SIZE];
    mixed dr = 0, px = 0, py = 0, pz = 0;
    for (int node = GLOBAL_ID; node < NUM_NODES; node += GLOBAL_SIZE) {
        dr += densityDeviation[node];
        px += momentum[node];
        py += momentum[NUM_NODES+node];
        pz += momentum[2*NUM_NODES+node];
    }
    sums[LOCAL_ID] = dr;
    sums[LBM_BLOCK_SIZE+LOCAL_ID] = px;
    sums[2*LBM_BLOCK_SIZE+LOCAL_ID] = py;
    sums[3*LBM_BLOCK_SIZE+LOCAL_ID] = pz;
    for (int offset = LBM_BLOCK_SIZE/2; offset > 0; offset /= 2) {
        SYNC_THREADS;
        if (LOCAL_ID < offset)
            for (int k = 0; k < 4; k++)
                sums[k*LBM_BLOCK_SIZE+LOCAL_ID] += sums[k*LBM_BLOCK_SIZE+LOCAL_ID+offset];
    }
    if (LOCAL_ID == 0)
        for (int k = 0; k < 4; k++)
            partialSums[4*GROUP_ID+k] = sums[k*LBM_BLOCK_SIZE];
}

/**
 * Second stage, run by a single work group: sum the partial sums of the groups and compute the velocity of the
 * centre of mass of the fluid, u_cm = sum(j)/sum(rho), with sum(rho) = NUM_NODES + sum(rho - 1).
 */
KERNEL void computeFluidCenterVelocity(GLOBAL const mixed* RESTRICT partialSums, int numPartialSums,
        GLOBAL mixed* RESTRICT centerVelocity) {
    LOCAL mixed sums[4*LBM_BLOCK_SIZE];
    mixed s[4] = {0, 0, 0, 0};
    for (int i = LOCAL_ID; i < numPartialSums; i += LOCAL_SIZE)
        for (int k = 0; k < 4; k++)
            s[k] += partialSums[4*i+k];
    for (int k = 0; k < 4; k++)
        sums[k*LBM_BLOCK_SIZE+LOCAL_ID] = s[k];
    for (int offset = LBM_BLOCK_SIZE/2; offset > 0; offset /= 2) {
        SYNC_THREADS;
        if (LOCAL_ID < offset)
            for (int k = 0; k < 4; k++)
                sums[k*LBM_BLOCK_SIZE+LOCAL_ID] += sums[k*LBM_BLOCK_SIZE+LOCAL_ID+offset];
    }
    if (LOCAL_ID == 0) {
        mixed mass = NUM_NODES + sums[0];
        centerVelocity[0] = sums[LBM_BLOCK_SIZE]/mass;
        centerVelocity[1] = sums[2*LBM_BLOCK_SIZE]/mass;
        centerVelocity[2] = sums[3*LBM_BLOCK_SIZE]/mass;
    }
}

/**
 * Third stage: subtract the velocity of the centre of mass from every node, j <- j - rho*u_cm.  The
 * non-equilibrium moments are left as they are.
 */
KERNEL void removeFluidMomentum(GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL mixed* RESTRICT momentum,
        GLOBAL const mixed* RESTRICT centerVelocity) {
    mixed ux = centerVelocity[0], uy = centerVelocity[1], uz = centerVelocity[2];
    for (int node = GLOBAL_ID; node < NUM_NODES; node += GLOBAL_SIZE) {
        mixed rho = 1 + densityDeviation[node];
        momentum[node] -= rho*ux;
        momentum[NUM_NODES+node] -= rho*uy;
        momentum[2*NUM_NODES+node] -= rho*uz;
    }
}

/**
 * Regularized collision with Guo forcing and push streaming,
 *   f_q(x + c_q) = feq_q(rho, u) + (1 - omega) fneq_q(Pi_neq) + S_q(u, F)/2,  u = (j + F/2)/rho,  F = rho*g,
 * stored as the deviation f_q - w_q.  Each population is computed from the moments of its own node only, so the
 * populations can be overwritten in place: every (q, target node) is written by exactly one thread.
 */
KERNEL void collideAndStream(GLOBAL mixed* RESTRICT f, GLOBAL const mixed* RESTRICT densityDeviation,
        GLOBAL const mixed* RESTRICT momentum, GLOBAL const mixed* RESTRICT piNeq, mixed omega, mixed gx, mixed gy, mixed gz) {
    DECLARE_D3Q19_VELOCITIES
    for (int node = GLOBAL_ID; node < NUM_NODES; node += GLOBAL_SIZE) {
        int i = node%NX, j = (node/NX)%NY, k = node/(NX*NY);
        mixed dr = densityDeviation[node];
        mixed rho = 1 + dr;
        mixed fx = rho*gx, fy = rho*gy, fz = rho*gz;
        mixed ux = (momentum[node] + 0.5f*fx)/rho;
        mixed uy = (momentum[NUM_NODES+node] + 0.5f*fy)/rho;
        mixed uz = (momentum[2*NUM_NODES+node] + 0.5f*fz)/rho;
        mixed uu = ux*ux + uy*uy + uz*uz;
        mixed uf = ux*fx + uy*fy + uz*fz;
        mixed pxx = piNeq[node], pyy = piNeq[NUM_NODES+node], pzz = piNeq[2*NUM_NODES+node];
        mixed pxy = piNeq[3*NUM_NODES+node], pxz = piNeq[4*NUM_NODES+node], pyz = piNeq[5*NUM_NODES+node];
        for (int q = 0; q < 19; q++) {
            mixed w = latticeWeight(q);
            mixed cu = cx[q]*ux + cy[q]*uy + cz[q]*uz;
            mixed dfeq = w*(dr + rho*(3.0f*cu + 4.5f*cu*cu - 1.5f*uu));
            mixed hxx = (mixed) (cx[q]*cx[q]) - CS2, hyy = (mixed) (cy[q]*cy[q]) - CS2, hzz = (mixed) (cz[q]*cz[q]) - CS2;
            mixed fneq = 4.5f*w*(hxx*pxx + hyy*pyy + hzz*pzz + 2.0f*((cx[q]*cy[q])*pxy + (cx[q]*cz[q])*pxz + (cy[q]*cz[q])*pyz));
            mixed cf = cx[q]*fx + cy[q]*fy + cz[q]*fz;
            mixed s = w*(3.0f*(cf - uf) + 9.0f*cu*cf);
            int target = (i+cx[q]+NX)%NX + NX*((j+cy[q]+NY)%NY + NY*((k+cz[q]+NZ)%NZ));
            f[q*NUM_NODES+target] = dfeq + (1-omega)*fneq + 0.5f*s;
        }
    }
}

/**
 * Maximum over the nodes of |j/rho|^2, computed from the populations: each work group writes the maximum over
 * its nodes to partialMax[group].
 */
KERNEL void computeMaxFluidSpeed(GLOBAL const mixed* RESTRICT f, GLOBAL mixed* RESTRICT partialMax) {
    LOCAL mixed maxima[LBM_BLOCK_SIZE];
    DECLARE_D3Q19_VELOCITIES
    mixed maxSpeed2 = 0;
    for (int node = GLOBAL_ID; node < NUM_NODES; node += GLOBAL_SIZE) {
        mixed dr = 0, jx = 0, jy = 0, jz = 0;
        for (int q = 0; q < 19; q++) {
            mixed df = f[q*NUM_NODES+node];
            dr += df;
            jx += cx[q]*df;
            jy += cy[q]*df;
            jz += cz[q]*df;
        }
        mixed rho = 1 + dr;
        maxSpeed2 = max(maxSpeed2, (jx*jx + jy*jy + jz*jz)/(rho*rho));
    }
    maxima[LOCAL_ID] = maxSpeed2;
    for (int offset = LBM_BLOCK_SIZE/2; offset > 0; offset /= 2) {
        SYNC_THREADS;
        if (LOCAL_ID < offset)
            maxima[LOCAL_ID] = max(maxima[LOCAL_ID], maxima[LOCAL_ID+offset]);
    }
    if (LOCAL_ID == 0)
        partialMax[GROUP_ID] = maxima[0];
}
