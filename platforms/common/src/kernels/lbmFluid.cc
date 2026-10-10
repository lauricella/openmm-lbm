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
 * NUM_BOUNDARY_NODES are defined; the boundary nodes are fluid nodes like the others, except that applyBoundaries
 * rebuilds their unknown populations after the streaming.  OPEN_X, OPEN_Y and OPEN_Z are defined for the axes with
 * open faces.
 *
 * With the domain decomposition (DOMAIN_DECOMPOSITION, docs/theory.md, section 8) NX, NY and NZ are the extents of
 * the block of the rank and NUM_NODES its number of nodes, and node indices are local, i + NX*(j + NY*k) within the
 * block.  Along the divided axes (PAD_X, PAD_Y, PAD_Z nonzero) the arrays hold PAD_X (PAD_Y, PAD_Z) more layers of
 * nodes on each side, the halo: one, or two with the Keys interpolation stencil, whose coupling reads the moments of
 * nodes two layers away (CommonLBMKernels.cpp).  The streaming pushes the populations that leave the block into the
 * first layer; the host sends them to the ranks that own those nodes (packPopulations, unpackPopulations).  Along the
 * other axes the block is the whole axis and the streaming wraps around within it, as without the decomposition.  The node n of the block is stored at
 * STORAGE_INDEX(n) of arrays of NUM_STORED entries per component (populations, moments, isFluid).
 * Without the decomposition the storage index is the node index and NUM_STORED is NUM_NODES.
 */

#define DECLARE_D3Q19_VELOCITIES \
    const int cx[19] = {0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 0,  0,  0,  0, 1, -1, -1,  1}; \
    const int cy[19] = {0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 1, -1,  1, -1, 0,  0,  0,  0}; \
    const int cz[19] = {0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1,  1, -1};

#ifdef DOMAIN_DECOMPOSITION
#define STORAGE_INDEX(n) ((n)%NX + PAD_X + SX*(((n)/NX)%NY + PAD_Y + SY*((n)/(NX*NY) + PAD_Z)))
#else
#define NUM_STORED NUM_NODES
#define PAD_X 0
#define PAD_Y 0
#define PAD_Z 0
#define SX NX
#define SY NY
#define STORAGE_INDEX(n) (n)
#endif

/**
 * The storage index of the node (i + dx, j + dy, k + dz), with (i, j, k) a node of the block and |dx|, |dy|, |dz| <= 1:
 * in the halo along the divided axes, wrapped around along the others.
 */
DEVICE int neighborIndex(int i, int j, int k, int dx, int dy, int dz) {
#if PAD_X
    int a = i+dx+PAD_X;
#else
    int a = (i+dx+NX)%NX;
#endif
#if PAD_Y
    int b = j+dy+PAD_Y;
#else
    int b = (j+dy+NY)%NY;
#endif
#if PAD_Z
    int c = k+dz+PAD_Z;
#else
    int c = (k+dz+NZ)%NZ;
#endif
    return a + SX*(b + SY*c);
}

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
        int s = STORAGE_INDEX(node);
#ifdef HAS_SOLID_NODES
        if (!isFluid[s]) {
            densityDeviation[s] = -1;
            for (int k = 0; k < 3; k++)
                momentum[k*NUM_STORED+s] = 0;
            for (int k = 0; k < 6; k++)
                piNeq[k*NUM_STORED+s] = 0;
            continue;
        }
#endif
        mixed df[19];
        mixed dr = 0, jx = 0, jy = 0, jz = 0;
        for (int q = 0; q < 19; q++) {
            df[q] = f[q*NUM_STORED+s];
            dr += df[q];
            jx += cx[q]*df[q];
            jy += cy[q]*df[q];
            jz += cz[q]*df[q];
        }
        densityDeviation[s] = dr;
        momentum[s] = jx;
        momentum[NUM_STORED+s] = jy;
        momentum[2*NUM_STORED+s] = jz;
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
        piNeq[s] = pxx;
        piNeq[NUM_STORED+s] = pyy;
        piNeq[2*NUM_STORED+s] = pzz;
        piNeq[3*NUM_STORED+s] = pxy;
        piNeq[4*NUM_STORED+s] = pxz;
        piNeq[5*NUM_STORED+s] = pyz;
    }
}

/**
 * First stage of the removal of the fluid momentum: each work group sums rho - 1 and j over its nodes and writes
 * the four sums to partialSums[4*group + k].
 */
KERNEL void sumFluidMomentum(GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL const mixed* RESTRICT momentum,
        GLOBAL mixed* RESTRICT partialSums, GLOBAL const int* RESTRICT isFluid) {
    LOCAL mixed sums[4*LBM_BLOCK_SIZE];
    mixed dr = 0, px = 0, py = 0, pz = 0;
    for (int node = GLOBAL_ID; node < NUM_NODES; node += GLOBAL_SIZE) {
        int s = STORAGE_INDEX(node);
        dr += densityDeviation[s];
        px += momentum[s];
        py += momentum[NUM_STORED+s];
        pz += momentum[2*NUM_STORED+s];
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
 * centre of mass of the fluid, u_cm = sum(j)/sum(rho), with sum(rho) = NUM_NODES + sum(rho - 1).  With the domain
 * decomposition it writes the four sums of the block instead, which the host adds over the ranks.
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
#ifdef DOMAIN_DECOMPOSITION
        for (int k = 0; k < 4; k++)
            centerVelocity[k] = sums[k*LBM_BLOCK_SIZE];
#else
        mixed mass = NUM_NODES + sums[0];
        centerVelocity[0] = sums[LBM_BLOCK_SIZE]/mass;
        centerVelocity[1] = sums[2*LBM_BLOCK_SIZE]/mass;
        centerVelocity[2] = sums[3*LBM_BLOCK_SIZE]/mass;
#endif
    }
}

/**
 * Third stage: subtract the velocity of the centre of mass from every node, j <- j - rho*u_cm.  The
 * non-equilibrium moments are left as they are.
 */
KERNEL void removeFluidMomentum(GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL mixed* RESTRICT momentum,
        GLOBAL const mixed* RESTRICT centerVelocity, GLOBAL const int* RESTRICT isFluid) {
    mixed ux = centerVelocity[0], uy = centerVelocity[1], uz = centerVelocity[2];
    for (int node = GLOBAL_ID; node < NUM_NODES; node += GLOBAL_SIZE) {
        int s = STORAGE_INDEX(node);
        mixed rho = 1 + densityDeviation[s];
        momentum[s] -= rho*ux;
        momentum[NUM_STORED+s] -= rho*uy;
        momentum[2*NUM_STORED+s] -= rho*uz;
    }
}

/**
 * Regularized collision with Guo forcing and push streaming,
 *   f_q(x + c_q) = feq_q(rho, u) + (1 - omega) fneq_q(Pi_neq) + S_q(u, F)/2,  u = (j + F/2)/rho,
 * with F = rho*g plus, with coupled particles (HAS_COUPLED_PARTICLES), the reaction of the particles of the node,
 * stored as the deviation f_q - w_q.  A fluctuating fluid (FLUID_FLUCTUATIONS) adds a random part that conserves
 * the mass and momentum of the node.  Each population is computed from the moments of its own node only, so the
 * populations can be overwritten in place: every (q, target node) is written by exactly one thread.  With the domain
 * decomposition the kernel advances the numListed nodes of nodeList, so that the host can advance the frame of the
 * block (the nodes that push into other ranks) first and the interior while their populations travel.
 */
KERNEL void collideAndStream(GLOBAL mixed* RESTRICT f, GLOBAL const int* RESTRICT isFluid, GLOBAL const mixed* RESTRICT densityDeviation,
        GLOBAL const mixed* RESTRICT momentum, GLOBAL const mixed* RESTRICT piNeq, GLOBAL const mixed* RESTRICT cellReaction,
        mixed omega, mixed gx, mixed gy, mixed gz
#ifdef FLUID_FLUCTUATIONS
        , GLOBAL const float4* RESTRICT random, GLOBAL const mixed* RESTRICT fluctuationBasis, mixed mu, int randomIndex
#endif
#ifdef DOMAIN_DECOMPOSITION
        , GLOBAL const int* RESTRICT nodeList, int numListed
#endif
        ) {
    DECLARE_D3Q19_VELOCITIES
#ifdef DOMAIN_DECOMPOSITION
    for (int listed = GLOBAL_ID; listed < numListed; listed += GLOBAL_SIZE) {
        int node = nodeList[listed];
#else
    for (int node = GLOBAL_ID; node < NUM_NODES; node += GLOBAL_SIZE) {
#endif
        int s = STORAGE_INDEX(node);
#ifdef HAS_SOLID_NODES
        if (!isFluid[s])
            continue;
#endif
        int i = node%NX, j = (node/NX)%NY, k = node/(NX*NY);
        mixed dr = densityDeviation[s];
        mixed rho = 1 + dr;
        mixed fx = rho*gx, fy = rho*gy, fz = rho*gz;
#ifdef HAS_COUPLED_PARTICLES
        fx += cellReaction[s];
        fy += cellReaction[NUM_STORED+s];
        fz += cellReaction[2*NUM_STORED+s];
#endif
        mixed ux = (momentum[s] + 0.5f*fx)/rho;
        mixed uy = (momentum[NUM_STORED+s] + 0.5f*fy)/rho;
        mixed uz = (momentum[2*NUM_STORED+s] + 0.5f*fz)/rho;
        mixed uu = ux*ux + uy*uy + uz*uz;
        mixed uf = ux*fx + uy*fy + uz*fz;
        mixed pxx = piNeq[s], pyy = piNeq[NUM_STORED+s], pzz = piNeq[2*NUM_STORED+s];
        mixed pxy = piNeq[3*NUM_STORED+s], pxz = piNeq[4*NUM_STORED+s], pyz = piNeq[5*NUM_STORED+s];
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
            mixed source = w*(3.0f*(cf - uf) + 9.0f*cu*cf);
            int target = neighborIndex(i, j, k, cx[q], cy[q], cz[q]);
            mixed value = dfeq + (1-omega)*fneq + 0.5f*source;
#ifdef FLUID_FLUCTUATIONS
            if (fluctuate)
                value += xi[q];
#endif
            f[q*NUM_STORED+target] = value;
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
        int s = STORAGE_INDEX(node);
#ifdef HAS_SOLID_NODES
        if (!isFluid[s])
            continue;
#endif
        mixed dr = 0, jx = 0, jy = 0, jz = 0;
        for (int q = 0; q < 19; q++) {
            mixed df = f[q*NUM_STORED+s];
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
 * touches.  The deviations f - w are copied as they are, since opposite directions have the same weight.  With the
 * domain decomposition a solid node in the halo holds what the node pushed into it, since the exchange writes only
 * populations of fluid nodes of the block.
 */
KERNEL void bounceBack(GLOBAL mixed* RESTRICT f, GLOBAL const int* RESTRICT wallNodes, GLOBAL const int* RESTRICT wallLinks) {
    DECLARE_D3Q19_VELOCITIES
    for (int b = GLOBAL_ID; b < NUM_WALL_NODES; b += GLOBAL_SIZE) {
        int node = wallNodes[b], links = wallLinks[b];
        int x = node%NX, y = (node/NX)%NY, z = node/(NX*NY), s = STORAGE_INDEX(node);
        for (int q = 1; q < 19; q++) {
            if (!(links & (1<<q)))
                continue;
            int solid = neighborIndex(x, y, z, cx[q], cy[q], cz[q]);
            f[(q%2 == 1 ? q+1 : q-1)*NUM_STORED+s] = f[q*NUM_STORED+solid];
        }
    }
}

/**
 * Momentum exchange of the halfway bounce-back (Ladd 1994), one thread per solid node: the population that streamed
 * from the fluid node s + c_q into the solid node s, moving along -c_q, went back to s + c_q moving along c_q, so the
 * wall at rest receives -2 f c_q on each link, with the full population f = (f - w) + w.  Only links to fluid nodes
 * that do not cross an open face count, the bits 1 << q of solidLinks[i] (found by the host; with the domain
 * decomposition only the links to fluid nodes of the block, so that every link counts on one rank, and the solid
 * nodes may lie in the halo), and a thread reads only populations of its own solid node, stored at solidNodes[i].
 * Each thread writes the part of f - w, -2 sum c_q (f - w), to wallExchange[k*NUM_SOLID_NODES + i]; the host sums it
 * over the solid nodes in the order of the list and adds the part of w, the static pressure, which depends only on
 * the geometry and is computed once in double precision.  Kept apart, the static pressure does not hide the
 * hydrodynamic part in single precision.
 */
KERNEL void computeWallExchange(GLOBAL const mixed* RESTRICT f, GLOBAL const int* RESTRICT solidLinks, GLOBAL const int* RESTRICT solidNodes,
        GLOBAL mixed* RESTRICT wallExchange) {
    DECLARE_D3Q19_VELOCITIES
    for (int i = GLOBAL_ID; i < NUM_SOLID_NODES; i += GLOBAL_SIZE) {
        int node = solidNodes[i], links = solidLinks[i];
        mixed px = 0, py = 0, pz = 0;
        for (int q = 1; q < 19; q++) {
            if (!(links & (1<<q)))
                continue;
            int opposite = (q%2 == 1 ? q+1 : q-1);
            mixed df = f[opposite*NUM_STORED+node];
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
 * Regularized boundaries (docs/theory.md, section 1), after the streaming and the bounce-back, one thread per
 * boundary node, with the arithmetic of applyBoundaries() of the Reference platform.  At a boundary node x the
 * populations of the directions q whose source x - c_q is solid (bits solid) or lies beyond an open face are unknown
 * (bits unknown).  Each of them is rebuilt as the population that a node at x - c_q would send, a node with the
 * moments of x except the imposed one:
 *   feq_q(rho_b, u_b) + (1 - omega) fneq_q(Pi_neq of x) + S_q(u_b, rho_b g)/2,
 * plus, with a fluctuating fluid, a random part of its own, from the four float4 of OpenMM's random numbers of the
 * node after those of all the nodes.  The known populations of x are not changed.  kindAndFace = kind + 4*(face + 1);
 * faceParameters holds the velocity and rho - 1 of each face (4 per face).
 *  - Walls (kind 0, next to solid nodes) and Velocity faces (kind 1): u_b = 0 or the velocity of the face; rho_b
 *    from the mass balance of the rebuilt links: the mass that x sent into the solid nodes in this streaming (in the
 *    solid node x - c_q, direction opposite to q) and, across a face, the population that arrived at x moving out of
 *    the face plus the inflow 6 w_q rho_b c_q.u_b; links whose opposite is unknown too do not count.
 *  - Density faces (kind 2): rho_b of the face; the velocity along the face is that of x, (j + rho g/2)/rho, without
 *    the reaction of the coupled particles of x (the rebuilt populations are those of a node beyond the face), and
 *    the velocity across it the mean of the velocity that gives x the density of the face, from the populations that
 *    have arrived (the unknown ones replaced by the bounce-back of their opposites), and that velocity of x at the
 *    start of the step.
 *  - Nodes shared by several Density faces (kind 3): rho_b of the first face and u_b = 0.
 * On walls the node writes the momentum given to the solid nodes by the deviations f - w (the populations that it
 * sent into them minus those that come back) to boundaryExchange[k*NUM_BOUNDARY_NODES + b]; the host adds the part of
 * the weights w, computed once.  A thread reads its own populations and moments and the solid slots that its node
 * wrote in the streaming, and writes its own populations, so no two threads touch the same value.
 */
KERNEL void applyBoundaries(GLOBAL mixed* RESTRICT f, GLOBAL const int* RESTRICT boundaryNodes,
        GLOBAL const int* RESTRICT boundaryUnknown, GLOBAL const int* RESTRICT boundarySolid,
        GLOBAL const int* RESTRICT kindAndFace, GLOBAL const mixed* RESTRICT densityDeviation,
        GLOBAL const mixed* RESTRICT momentum, GLOBAL const mixed* RESTRICT faceParameters,
        GLOBAL mixed* RESTRICT boundaryExchange, mixed gx, mixed gy, mixed gz, GLOBAL const mixed* RESTRICT piNeq,
        mixed omega
#ifdef FLUID_FLUCTUATIONS
        , GLOBAL const float4* RESTRICT random, GLOBAL const mixed* RESTRICT fluctuationBasis, mixed mu, int randomIndex
#endif
        ) {
    DECLARE_D3Q19_VELOCITIES
    for (int b = GLOBAL_ID; b < NUM_BOUNDARY_NODES; b += GLOBAL_SIZE) {
        int node = boundaryNodes[b], unknown = boundaryUnknown[b], solid = boundarySolid[b];
        int kind = kindAndFace[b]%4, face = kindAndFace[b]/4 - 1;
        int x = node%NX, y = (node/NX)%NY, z = node/(NX*NY), s = STORAGE_INDEX(node);
        mixed dr = densityDeviation[s];
        mixed rho = 1 + dr;
        mixed u[3] = {(momentum[s] + 0.5f*(rho*gx))/rho, (momentum[NUM_STORED+s] + 0.5f*(rho*gy))/rho,
                      (momentum[2*NUM_STORED+s] + 0.5f*(rho*gz))/rho};
        if (kind == 0 || kind == 3)
            u[0] = u[1] = u[2] = 0;
        if (kind == 1)
            for (int a = 0; a < 3; a++)
                u[a] = faceParameters[4*face+a];
        if (kind == 2 || kind == 3) {
            dr = faceParameters[4*face+3];
            rho = 1 + dr;
        }
        if (kind == 2) {
            int axis = face/2;
            mixed inward = (face%2 == 0 ? 1 : -1), sum = 0;
            for (int q = 0; q < 19; q++)
                sum += f[(unknown & (1<<q) ? (q == 0 ? 0 : (q%2 == 1 ? q+1 : q-1)) : q)*NUM_STORED+s];
            u[axis] = 0.5f*(inward*(dr-sum)/rho + u[axis]);
        }
        mixed pxx = piNeq[s], pyy = piNeq[NUM_STORED+s], pzz = piNeq[2*NUM_STORED+s];
        mixed pxy = piNeq[3*NUM_STORED+s], pxz = piNeq[4*NUM_STORED+s], pyz = piNeq[5*NUM_STORED+s];
        mixed rest[19];
        for (int q = 0; q < 19; q++) {
            mixed w = latticeWeight(q);
            mixed hxx = (mixed) (cx[q]*cx[q]) - CS2, hyy = (mixed) (cy[q]*cy[q]) - CS2, hzz = (mixed) (cz[q]*cz[q]) - CS2;
            rest[q] = (1-omega)*(4.5f*w*(hxx*pxx + hyy*pyy + hzz*pzz + 2.0f*((cx[q]*cy[q])*pxy + (cx[q]*cz[q])*pxz + (cy[q]*cz[q])*pyz)));
        }
#ifdef FLUID_FLUCTUATIONS
        if (mu > 0) {
            mixed stressAmplitude = sqrt(mu*rho*omega*(2-omega)), ghostAmplitude = sqrt(mu*rho);
            for (int b4 = 0; b4 < 4; b4++) {
                float4 r4 = random[randomIndex + 4*NUM_NODES + 4*b + b4];
                float r[4] = {r4.x, r4.y, r4.z, r4.w};
                for (int c = 0; c < 4; c++) {
                    int m = 4*b4 + c;
                    if (m < 15) {
                        mixed a = (m < 6 ? stressAmplitude : ghostAmplitude)*r[c];
                        for (int q = 0; q < 19; q++)
                            rest[q] += fluctuationBasis[q*15+m]*a;
                    }
                }
            }
        }
#endif
        mixed uu = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
        mixed ug = u[0]*gx + u[1]*gy + u[2]*gz;
        if (kind == 0 || kind == 1) {
            // Mass balance of the rebuilt links, linear in rho_b = 1 + dr and written for dr.
            mixed numerator = 0, denominator = 0;
            for (int q = 1; q < 19; q++) {
                if (!(unknown & (1<<q)))
                    continue;
                int opposite = (q%2 == 1 ? q+1 : q-1);
                mixed arriving, flux = 0;
                if (solid & (1<<q)) {
                    int source = neighborIndex(x, y, z, -cx[q], -cy[q], -cz[q]);
                    arriving = f[opposite*NUM_STORED+source];
                }
                else if (!(unknown & (1<<opposite))) {
                    arriving = f[opposite*NUM_STORED+s];
                    flux = 6*latticeWeight(q)*(cx[q]*u[0] + cy[q]*u[1] + cz[q]*u[2]);
                }
                else
                    continue;
                mixed w = latticeWeight(q);
                mixed cu = cx[q]*u[0] + cy[q]*u[1] + cz[q]*u[2];
                mixed e0 = w*(3.0f*cu + 4.5f*cu*cu - 1.5f*uu);
                mixed cg = cx[q]*gx + cy[q]*gy + cz[q]*gz;
                mixed sg = w*(3.0f*(cg - ug) + 9.0f*cu*cg);
                numerator += arriving - rest[q] - e0 - 0.5f*sg + flux;
                denominator += w + e0 + 0.5f*sg - flux;
            }
            if (denominator != 0) {
                dr = numerator/denominator;
                rho = 1 + dr;
            }
        }
        mixed fx = rho*gx, fy = rho*gy, fz = rho*gz;
        mixed uf = u[0]*fx + u[1]*fy + u[2]*fz;
        mixed ex = 0, ey = 0, ez = 0;
        for (int q = 1; q < 19; q++) {
            if (!(unknown & (1<<q)))
                continue;
            mixed w = latticeWeight(q);
            mixed cu = cx[q]*u[0] + cy[q]*u[1] + cz[q]*u[2];
            mixed dfeq = w*(dr + rho*(3.0f*cu + 4.5f*cu*cu - 1.5f*uu));
            mixed cf = cx[q]*fx + cy[q]*fy + cz[q]*fz;
            mixed sq = w*(3.0f*(cf - uf) + 9.0f*cu*cf);
            mixed value = dfeq + rest[q] + 0.5f*sq;
            if (solid & (1<<q)) {
                int source = neighborIndex(x, y, z, -cx[q], -cy[q], -cz[q]);
                mixed sent = f[(q%2 == 1 ? q+1 : q-1)*NUM_STORED+source];
                ex -= cx[q]*(sent + value);
                ey -= cy[q]*(sent + value);
                ez -= cz[q]*(sent + value);
            }
            f[q*NUM_STORED+s] = value;
        }
        boundaryExchange[b] = ex;
        boundaryExchange[NUM_BOUNDARY_NODES+b] = ey;
        boundaryExchange[2*NUM_BOUNDARY_NODES+b] = ez;
    }
}
#endif

#ifdef DOMAIN_DECOMPOSITION
/**
 * The exchange of the populations between the ranks: packPopulations copies the populations at the slots
 * q*NUM_STORED + s of the halo that the streaming filled into a buffer, which the host sends to the ranks that own
 * those nodes; unpackPopulations writes the populations received from them at the slots of the fluid nodes of the
 * block.  Every slot appears once.
 */
KERNEL void packPopulations(GLOBAL const mixed* RESTRICT f, GLOBAL const int* RESTRICT slots, GLOBAL mixed* RESTRICT buffer, int numSlots) {
    for (int i = GLOBAL_ID; i < numSlots; i += GLOBAL_SIZE)
        buffer[i] = f[slots[i]];
}

KERNEL void unpackPopulations(GLOBAL mixed* RESTRICT f, GLOBAL const int* RESTRICT slots, GLOBAL const mixed* RESTRICT buffer, int numSlots) {
    for (int i = GLOBAL_ID; i < numSlots; i += GLOBAL_SIZE)
        f[slots[i]] = buffer[i];
}

/**
 * The exchange of the halo of the density and the velocity (setDensityHaloExchange(), setVelocityHaloExchange()):
 * copy rho - 1 and j of the listed nodes (storage indices), four values per node, into a buffer for the host, which
 * computes the fields as getFluidFields() does and sends them to the ranks whose halo holds the nodes.
 */
KERNEL void packFields(GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL const mixed* RESTRICT momentum,
        GLOBAL const int* RESTRICT nodes, GLOBAL mixed* RESTRICT buffer, int numListed) {
    for (int i = GLOBAL_ID; i < numListed; i += GLOBAL_SIZE) {
        int s = nodes[i];
        buffer[4*i] = densityDeviation[s];
        buffer[4*i+1] = momentum[s];
        buffer[4*i+2] = momentum[NUM_STORED+s];
        buffer[4*i+3] = momentum[2*NUM_STORED+s];
    }
}
#endif
