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
 *
 * With solid nodes, HAS_SOLID_NODES and NUM_SOLID_NODES are defined, and isFluid[node] is 0 at the solid nodes.
 * A solid node holds no fluid: its density is 0 (rho - 1 = -1, so that it does not enter the removal of the
 * momentum), its momentum and stress are 0, and it neither collides nor enters the Mach number.
 *
 * With boundary nodes (regularized walls or open faces, internal/LBMBoundaries.h), HAS_BOUNDARY_NODES and
 * NUM_BOUNDARY_NODES are defined, and isFluid[node] is 2 at the nodes of regularized walls and 3 at the nodes of
 * open faces.  Their velocity is imposed: they are left out of the removal of the momentum, and the reaction of the
 * particles does not act on them.  OPEN_X, OPEN_Y and OPEN_Z are defined for the axes with open faces.
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
KERNEL void computeFluidMoments(GLOBAL const mixed* RESTRICT f, GLOBAL const int* RESTRICT isFluid,
        GLOBAL mixed* RESTRICT densityDeviation, GLOBAL mixed* RESTRICT momentum, GLOBAL mixed* RESTRICT piNeq) {
    DECLARE_D3Q19_VELOCITIES
    for (int node = GLOBAL_ID; node < NUM_NODES; node += GLOBAL_SIZE) {
#ifdef HAS_SOLID_NODES
        if (!isFluid[node]) {
            densityDeviation[node] = -1;
            for (int k = 0; k < 3; k++)
                momentum[k*NUM_NODES+node] = 0;
            for (int k = 0; k < 6; k++)
                piNeq[k*NUM_NODES+node] = 0;
            continue;
        }
#endif
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
 * the four sums to partialSums[4*group + k].  The boundary nodes are left out.
 */
KERNEL void sumFluidMomentum(GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL const mixed* RESTRICT momentum,
        GLOBAL mixed* RESTRICT partialSums, GLOBAL const int* RESTRICT isFluid) {
    LOCAL mixed sums[4*LBM_BLOCK_SIZE];
    mixed dr = 0, px = 0, py = 0, pz = 0;
    for (int node = GLOBAL_ID; node < NUM_NODES; node += GLOBAL_SIZE) {
#ifdef HAS_BOUNDARY_NODES
        if (isFluid[node] > 1) {
            dr -= 1;        // a boundary node counts as a node without fluid
            continue;
        }
#endif
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
 * non-equilibrium moments are left as they are.  The boundary nodes keep their velocity.
 */
KERNEL void removeFluidMomentum(GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL mixed* RESTRICT momentum,
        GLOBAL const mixed* RESTRICT centerVelocity, GLOBAL const int* RESTRICT isFluid) {
    mixed ux = centerVelocity[0], uy = centerVelocity[1], uz = centerVelocity[2];
    for (int node = GLOBAL_ID; node < NUM_NODES; node += GLOBAL_SIZE) {
#ifdef HAS_BOUNDARY_NODES
        if (isFluid[node] > 1)
            continue;
#endif
        mixed rho = 1 + densityDeviation[node];
        momentum[node] -= rho*ux;
        momentum[NUM_NODES+node] -= rho*uy;
        momentum[2*NUM_NODES+node] -= rho*uz;
    }
}

/**
 * Regularized collision with Guo forcing and push streaming,
 *   f_q(x + c_q) = feq_q(rho, u) + (1 - omega) fneq_q(Pi_neq) + S_q(u, F)/2,  u = (j + F/2)/rho,
 * with F = rho*g plus, with coupled particles (HAS_COUPLED_PARTICLES), the reaction of the particles of the node
 * (except on the boundary nodes, whose velocity is imposed), stored as the deviation f_q - w_q.  A fluctuating fluid (FLUID_FLUCTUATIONS) adds a random part that conserves
 * the mass and momentum of the node.  Each population is computed from the moments of its own node only, so the
 * populations can be overwritten in place: every (q, target node) is written by exactly one thread.
 */
KERNEL void collideAndStream(GLOBAL mixed* RESTRICT f, GLOBAL const int* RESTRICT isFluid, GLOBAL const mixed* RESTRICT densityDeviation,
        GLOBAL const mixed* RESTRICT momentum, GLOBAL const mixed* RESTRICT piNeq, GLOBAL const mixed* RESTRICT cellReaction,
        mixed omega, mixed gx, mixed gy, mixed gz
#ifdef FLUID_FLUCTUATIONS
        , GLOBAL const float4* RESTRICT random, GLOBAL const mixed* RESTRICT fluctuationBasis, mixed mu, int randomIndex
#endif
        ) {
    DECLARE_D3Q19_VELOCITIES
    for (int node = GLOBAL_ID; node < NUM_NODES; node += GLOBAL_SIZE) {
#ifdef HAS_SOLID_NODES
        if (!isFluid[node])
            continue;
#endif
        int i = node%NX, j = (node/NX)%NY, k = node/(NX*NY);
        mixed dr = densityDeviation[node];
        mixed rho = 1 + dr;
        mixed fx = rho*gx, fy = rho*gy, fz = rho*gz;
#ifdef HAS_COUPLED_PARTICLES
#ifdef HAS_BOUNDARY_NODES
        if (isFluid[node] == 1)
#endif
        {
            fx += cellReaction[node];
            fy += cellReaction[NUM_NODES+node];
            fz += cellReaction[2*NUM_NODES+node];
        }
#endif
        mixed ux = (momentum[node] + 0.5f*fx)/rho;
        mixed uy = (momentum[NUM_NODES+node] + 0.5f*fy)/rho;
        mixed uz = (momentum[2*NUM_NODES+node] + 0.5f*fz)/rho;
        mixed uu = ux*ux + uy*uy + uz*uz;
        mixed uf = ux*fx + uy*fy + uz*fz;
        mixed pxx = piNeq[node], pyy = piNeq[NUM_NODES+node], pzz = piNeq[2*NUM_NODES+node];
        mixed pxy = piNeq[3*NUM_NODES+node], pxz = piNeq[4*NUM_NODES+node], pyz = piNeq[5*NUM_NODES+node];
#ifdef FLUID_FLUCTUATIONS
        // Random part xi_q = sum_m fluctuationBasis[q*15 + m] a_m r_m (docs/theory.md, section 7): the normal numbers
        // r_m of the node are the 16 components of random[randomIndex + 4*node ... + 3], of which the first 6 go to
        // the stress modes, with amplitude sqrt(mu rho omega (2 - omega)), and the next 9 to the ghost modes, with
        // amplitude sqrt(mu rho).  At zero temperature (mu = 0) nothing is added and no numbers are drawn.
        bool fluctuate = (mu > 0);
        mixed xi[19];
        for (int q = 0; q < 19; q++)
            xi[q] = 0;
        if (fluctuate) {
            mixed stressAmplitude = sqrt(mu*rho*omega*(2-omega)), ghostAmplitude = sqrt(mu*rho);
            for (int b = 0; b < 4; b++) {
                float4 r4 = random[randomIndex + 4*node + b];
                float r[4] = {r4.x, r4.y, r4.z, r4.w};
                for (int c = 0; c < 4; c++) {
                    int m = 4*b + c;
                    if (m < 15) {
                        mixed a = (m < 6 ? stressAmplitude : ghostAmplitude)*r[c];
                        for (int q = 0; q < 19; q++)
                            xi[q] += fluctuationBasis[q*15+m]*a;
                    }
                }
            }
        }
#endif
        for (int q = 0; q < 19; q++) {
            mixed w = latticeWeight(q);
            mixed cu = cx[q]*ux + cy[q]*uy + cz[q]*uz;
            mixed dfeq = w*(dr + rho*(3.0f*cu + 4.5f*cu*cu - 1.5f*uu));
            mixed hxx = (mixed) (cx[q]*cx[q]) - CS2, hyy = (mixed) (cy[q]*cy[q]) - CS2, hzz = (mixed) (cz[q]*cz[q]) - CS2;
            mixed fneq = 4.5f*w*(hxx*pxx + hyy*pyy + hzz*pzz + 2.0f*((cx[q]*cy[q])*pxy + (cx[q]*cz[q])*pxz + (cy[q]*cz[q])*pyz));
            mixed cf = cx[q]*fx + cy[q]*fy + cz[q]*fz;
            mixed s = w*(3.0f*(cf - uf) + 9.0f*cu*cf);
            int target = (i+cx[q]+NX)%NX + NX*((j+cy[q]+NY)%NY + NY*((k+cz[q]+NZ)%NZ));
            mixed value = dfeq + (1-omega)*fneq + 0.5f*s;
#ifdef FLUID_FLUCTUATIONS
            if (fluctuate)
                value += xi[q];
#endif
            f[q*NUM_NODES+target] = value;
        }
    }
}

/**
 * Maximum over the nodes of |j/rho|^2, computed from the populations: each work group writes the maximum over
 * its nodes to partialMax[group].
 */
KERNEL void computeMaxFluidSpeed(GLOBAL const mixed* RESTRICT f, GLOBAL const int* RESTRICT isFluid, GLOBAL mixed* RESTRICT partialMax) {
    LOCAL mixed maxima[LBM_BLOCK_SIZE];
    DECLARE_D3Q19_VELOCITIES
    mixed maxSpeed2 = 0;
    for (int node = GLOBAL_ID; node < NUM_NODES; node += GLOBAL_SIZE) {
#ifdef HAS_SOLID_NODES
        if (!isFluid[node])
            continue;
#endif
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

#ifdef BOUNCE_BACK_WALLS
/**
 * Halfway bounce-back (wall scheme BounceBack), one thread per fluid node next to the walls, after collideAndStream.
 * For a direction q whose node x + c_q is solid (bit q of wallLinks), the population that x built in its collision
 * for q from its own moments rho, j and Pi_neq, force and random part streamed into the solid node; x takes it back
 * as its population along -c_q, the one that arrives from the wall.  A thread reads the solid slots that its node
 * wrote in the streaming and writes populations of its own node whose source is solid, which no other thread
 * touches.  The deviations f - w are copied as they are, since opposite directions have the same weight.
 */
KERNEL void bounceBack(GLOBAL mixed* RESTRICT f, GLOBAL const int* RESTRICT wallNodes, GLOBAL const int* RESTRICT wallLinks) {
    DECLARE_D3Q19_VELOCITIES
    for (int b = GLOBAL_ID; b < NUM_WALL_NODES; b += GLOBAL_SIZE) {
        int node = wallNodes[b], links = wallLinks[b];
        int x = node%NX, y = (node/NX)%NY, z = node/(NX*NY);
        for (int q = 1; q < 19; q++) {
            if (!(links & (1<<q)))
                continue;
            int solid = (x+cx[q]+NX)%NX + NX*((y+cy[q]+NY)%NY + NY*((z+cz[q]+NZ)%NZ));
            f[(q%2 == 1 ? q+1 : q-1)*NUM_NODES+node] = f[q*NUM_NODES+solid];
        }
    }
}

/**
 * Momentum exchange of the halfway bounce-back (Ladd 1994), one thread per solid node: the population that streamed
 * from the fluid node s + c_q into the solid node s, moving along -c_q, went back to s + c_q moving along c_q, so the
 * wall at rest receives -2 f c_q on each link, with the full population f = (f - w) + w.  Only links to fluid nodes
 * that do not cross an open face count, and a thread reads only populations of its own solid node.  Each thread
 * writes the part of f - w, -2 sum c_q (f - w), to wallExchange[k*NUM_SOLID_NODES + i]; the host sums it over the
 * solid nodes in the order of the list and adds the part of w, the static pressure, which depends only on the
 * geometry and is computed once in double precision.  Kept apart, the static pressure does not hide the
 * hydrodynamic part in single precision.
 */
KERNEL void computeWallExchange(GLOBAL const mixed* RESTRICT f, GLOBAL const int* RESTRICT isFluid, GLOBAL const int* RESTRICT solidNodes,
        GLOBAL mixed* RESTRICT wallExchange) {
    DECLARE_D3Q19_VELOCITIES
    for (int i = GLOBAL_ID; i < NUM_SOLID_NODES; i += GLOBAL_SIZE) {
        int node = solidNodes[i];
        int x = node%NX, y = (node/NX)%NY, z = node/(NX*NY);
        mixed px = 0, py = 0, pz = 0;
        for (int q = 1; q < 19; q++) {
#ifdef OPEN_X
            if (x+cx[q] < 0 || x+cx[q] >= NX)
                continue;       // the link crosses an open face
#endif
#ifdef OPEN_Y
            if (y+cy[q] < 0 || y+cy[q] >= NY)
                continue;
#endif
#ifdef OPEN_Z
            if (z+cz[q] < 0 || z+cz[q] >= NZ)
                continue;
#endif
            int target = (x+cx[q]+NX)%NX + NX*((y+cy[q]+NY)%NY + NY*((z+cz[q]+NZ)%NZ));
            if (!isFluid[target])
                continue;
            int opposite = (q%2 == 1 ? q+1 : q-1);
            mixed df = f[opposite*NUM_NODES+node];
            px += cx[q]*df;
            py += cy[q]*df;
            pz += cz[q]*df;
        }
        wallExchange[i] = -2*px;
        wallExchange[NUM_SOLID_NODES+i] = -2*py;
        wallExchange[2*NUM_SOLID_NODES+i] = -2*pz;
    }
}
#endif

#ifdef HAS_BOUNDARY_NODES
/**
 * Local regularized boundary condition (Latt 2007, section 5.2; docs/theory.md, section 1), after the streaming and
 * the bounce-back, one thread per boundary node, with the arithmetic of applyBoundaries() of the Reference platform.
 * At a boundary node x the populations of the directions q whose source x - c_q is solid (bits solid) or lies
 * beyond an open face are unknown (bits unknown).  The 19 populations are rebuilt as feq(rho, v) + fneq(Pi), with
 * v = u - g/2 so that the velocity of the fluid (j + F/2)/rho is the velocity u imposed on the node:
 *  - on a wall (kind 0) u = 0, on a Velocity face (kind 1) u is the velocity of the face, and rho follows from the
 *    populations that arrive: the known ones, those that x sent into the solid nodes in this streaming (in the
 *    solid node x - c_q, direction opposite to q), and for an unknown direction q beyond a face f_qbar + 6 w_q rho
 *    c_q.v, or feq_q(rho, v) if qbar is unknown as well: the generalization of eq. 5.3 of Latt;
 *  - on a Density face (kind 2) rho is that of the face, the velocity along the face is zero and the velocity
 *    across it follows from the same balance, averaged with that of the node at the start of the step, which damps
 *    the staggered mode;
 *  - on the nodes shared by several Density faces (kind 3) rho is that of the first face and u = 0;
 *  - Pi = sum_q H2(c_q) fneq_q, with fneq_q = f_q - feq_q(rho, v) for the known populations and the bounce-back of
 *    the non-equilibrium part, fneq_q = fneq_qbar, for the unknown ones (zero if qbar is unknown as well).
 * kindAndFace = kind + 4*(face + 1); faceParameters holds the velocity and rho - 1 of each face (4 per face).  A
 * thread reads its own populations and the solid slots that its node wrote in the streaming, and writes its own
 * populations, so no two threads touch the same value.  On walls it writes the momentum given to the wall by the
 * deviations f - w (the populations that streamed into the solid nodes, minus the momentum that the rebuild adds)
 * to boundaryExchange[k*NUM_BOUNDARY_NODES + b]; the host adds the part of the weights w, computed once.
 */
KERNEL void applyBoundaries(GLOBAL mixed* RESTRICT f, GLOBAL const int* RESTRICT boundaryNodes,
        GLOBAL const int* RESTRICT boundaryUnknown, GLOBAL const int* RESTRICT boundarySolid,
        GLOBAL const int* RESTRICT kindAndFace, GLOBAL const mixed* RESTRICT densityDeviation,
        GLOBAL const mixed* RESTRICT momentum, GLOBAL const mixed* RESTRICT faceParameters,
        GLOBAL mixed* RESTRICT boundaryExchange, mixed gx, mixed gy, mixed gz) {
    DECLARE_D3Q19_VELOCITIES
    for (int b = GLOBAL_ID; b < NUM_BOUNDARY_NODES; b += GLOBAL_SIZE) {
        int node = boundaryNodes[b], unknown = boundaryUnknown[b], solid = boundarySolid[b];
        int kind = kindAndFace[b]%4, face = kindAndFace[b]/4 - 1;
        int x = node%NX, y = (node/NX)%NY, z = node/(NX*NY);

        // The populations that are available: the known ones and, for the solid directions, the reflected ones.

        mixed a[19];
        int available = 0;
        mixed sum = 0, kx = 0, ky = 0, kz = 0, ix = 0, iy = 0, iz = 0;
        for (int q = 0; q < 19; q++) {
            a[q] = 0;
            if (!(unknown & (1<<q))) {
                a[q] = f[q*NUM_NODES+node];
                kx += cx[q]*a[q];
                ky += cy[q]*a[q];
                kz += cz[q]*a[q];
                available |= 1<<q;
                sum += a[q];
            }
            else if (solid & (1<<q)) {
                int source = (x-cx[q]+NX)%NX + NX*((y-cy[q]+NY)%NY + NY*((z-cz[q]+NZ)%NZ));
                int opposite = (q%2 == 1 ? q+1 : q-1);
                a[q] = f[opposite*NUM_NODES+source];
                ix -= cx[q]*a[q];
                iy -= cy[q]*a[q];
                iz -= cz[q]*a[q];
                available |= 1<<q;
                sum += a[q];
            }
        }

        // Density and velocity of the populations.  The populations are deviations f - w, and w_q = w_qbar.

        mixed v[3] = {-0.5f*gx, -0.5f*gy, -0.5f*gz};
        mixed dr;
        if (kind == 2) {
            int axis = face/2;
            mixed inward = (face%2 == 0 ? 1 : -1);
            for (int q = 1; q < 19; q++)
                if (!(available & (1<<q)))
                    sum += a[q%2 == 1 ? q+1 : q-1];
            dr = faceParameters[4*face+3];
            v[axis] = 0.5f*(inward*(dr-sum)/(1+dr) + momentum[axis*NUM_NODES+node]/(1+densityDeviation[node]));
        }
        else {
            if (kind == 1)
                for (int k = 0; k < 3; k++)
                    v[k] += faceParameters[4*face+k];
            if (kind == 3)
                dr = faceParameters[4*face+3];
            else {
                mixed numerator = sum, denominator = 1, vv = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
                for (int q = 1; q < 19; q++) {
                    if (available & (1<<q))
                        continue;
                    int opposite = (q%2 == 1 ? q+1 : q-1);
                    mixed w = latticeWeight(q);
                    mixed cv = cx[q]*v[0] + cy[q]*v[1] + cz[q]*v[2];
                    if (available & (1<<opposite)) {
                        numerator += a[opposite] + 6*w*cv;
                        denominator -= 6*w*cv;
                    }
                    else {
                        mixed e = w*(3*cv + 4.5f*cv*cv - 1.5f*vv);
                        numerator += e;
                        denominator -= w + e;
                    }
                }
                dr = numerator/denominator;
            }
        }

        // Stress from the known non-equilibrium parts and their bounce-back, then the regularized populations.

        mixed rho = 1 + dr;
        mixed vv = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        mixed dfeq[19], fneq[19];
        for (int q = 0; q < 19; q++) {
            mixed cv = cx[q]*v[0] + cy[q]*v[1] + cz[q]*v[2];
            dfeq[q] = latticeWeight(q)*(dr + rho*(3.0f*cv + 4.5f*cv*cv - 1.5f*vv));
            fneq[q] = (unknown & (1<<q) ? 0 : a[q] - dfeq[q]);
        }
        for (int q = 1; q < 19; q++) {
            int opposite = (q%2 == 1 ? q+1 : q-1);
            if ((unknown & (1<<q)) && !(unknown & (1<<opposite)))
                fneq[q] = fneq[opposite];
        }
        mixed pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
        for (int q = 0; q < 19; q++) {
            pxx += ((mixed) (cx[q]*cx[q]) - CS2)*fneq[q];
            pyy += ((mixed) (cy[q]*cy[q]) - CS2)*fneq[q];
            pzz += ((mixed) (cz[q]*cz[q]) - CS2)*fneq[q];
            pxy += (cx[q]*cy[q])*fneq[q];
            pxz += (cx[q]*cz[q])*fneq[q];
            pyz += (cy[q]*cz[q])*fneq[q];
        }
        mixed rx = 0, ry = 0, rz = 0;
        for (int q = 0; q < 19; q++) {
            mixed w = latticeWeight(q);
            mixed hxx = (mixed) (cx[q]*cx[q]) - CS2, hyy = (mixed) (cy[q]*cy[q]) - CS2, hzz = (mixed) (cz[q]*cz[q]) - CS2;
            mixed value = dfeq[q] + 4.5f*w*(hxx*pxx + hyy*pyy + hzz*pzz + 2.0f*((cx[q]*cy[q])*pxy + (cx[q]*cz[q])*pxz + (cy[q]*cz[q])*pyz));
            f[q*NUM_NODES+node] = value;
            rx += cx[q]*value;
            ry += cy[q]*value;
            rz += cz[q]*value;
        }
        boundaryExchange[b] = (kind == 0 ? ix - (rx - kx) : 0);
        boundaryExchange[NUM_BOUNDARY_NODES+b] = (kind == 0 ? iy - (ry - ky) : 0);
        boundaryExchange[2*NUM_BOUNDARY_NODES+b] = (kind == 0 ? iz - (rz - kz) : 0);
    }
}
#endif
