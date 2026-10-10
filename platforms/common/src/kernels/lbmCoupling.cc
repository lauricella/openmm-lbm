/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

/**
 * The coupling of the particles to the fluid, with the arithmetic of the Reference platform
 * (ReferenceLBMKernels.cpp, docs/theory.md section 2).
 *
 * The atoms are stored in the order of the platform, which changes when OpenMM reorders them: atomIndex[j] is
 * the index in the System of the atom stored at j, and couplingIndex[atom] is the index of a coupled atom in the
 * list of the force (from 0 to NUM_COUPLED - 1), or -1.  Per-particle arrays are indexed by the coupling index,
 * with the components at [k*NUM_COUPLED + i].
 *
 * Defines, besides those of lbmFluid.cc: NUM_ATOMS, PADDED_NUM_ATOMS, NUM_COUPLED, DX (lattice spacing, nm),
 * VELOCITY_SCALE (dx/dt) and FORCE_SCALE (m_c dx/dt^2, the force unit of the lattice in kJ/mol/nm); with the
 * Centered drag, HAS_FLOAT_FORCE_BUFFERS if the platform also accumulates forces in floating point buffers.
 * NUM_KEYS is the number of sort keys and KEY_STRIDE the factor of the node in a key: NUM_COUPLED for the nearest node;
 * with an interpolation stencil (docs/theory.md, section 9) STENCIL_WIDTH is its number of nodes along an axis,
 * STENCIL_SIZE = STENCIL_WIDTH^3, and every particle has a key per node of its stencil, so that both are
 * NUM_COUPLED*STENCIL_SIZE.
 *
 * With the domain decomposition (DOMAIN_DECOMPOSITION, kernels/lbmFluid.cc) GNX, GNY and GNZ are the sizes of the
 * lattice and OX, OY, OZ the first node of the block of the rank; the moments and reactions are stored at the storage
 * index of the node in the block.  Every rank holds all the particles: the nearest node and the wall normal come from
 * the whole lattice (isFluid is then the mask of the whole lattice), the rank that owns the nearest node computes the
 * coupling force, the others set it to zero and give the particle a key that sorts after all the nodes of the block and
 * is skipped (otherRankKey()); the host sums the forces over the ranks.  SKIP_REFLECTION_MOMENTUM is defined on the
 * ranks other than 0, which reflect their copies of the particles but do not count the momentum of the wall.  With an
 * interpolation stencil the owner of the nearest node reads the moments of the nodes of the stencil in the block and in
 * its halo, where the host writes those of the other ranks (the coupling halo); every rank writes the weights and the
 * keys of all the stencils (stencilKey()) and adds the reactions on the nodes of its block.
 */

#ifndef DOMAIN_DECOMPOSITION
#define NUM_STORED NUM_NODES
#define GNX NX
#define GNY NY
#define GNZ NZ
#define SX NX
#define SY NY
#endif

DEVICE mixed4 loadPosition(GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT posqCorrection, int index) {
#ifdef USE_MIXED_PRECISION
    real4 pos1 = posq[index];
    real4 pos2 = posqCorrection[index];
    return make_mixed4(pos1.x+(mixed) pos2.x, pos1.y+(mixed) pos2.y, pos1.z+(mixed) pos2.z, pos1.w);
#else
    return posq[index];
#endif
}

/**
 * Lattice coordinate of a position along one axis, wrapped into [0, size).
 */
DEVICE mixed wrapCoordinate(mixed x, int size) {
    mixed s = x/DX;
    return s - floor(s/size)*size;
}

/**
 * The nearest node: after wrapping the position into the box, node i owns the interval [(i - 1/2) dx, (i + 1/2) dx).
 */
DEVICE int nearestNode(mixed4 pos) {
    int i = ((int) floor(wrapCoordinate(pos.x, GNX) + 0.5f))%GNX;
    int j = ((int) floor(wrapCoordinate(pos.y, GNY) + 0.5f))%GNY;
    int k = ((int) floor(wrapCoordinate(pos.z, GNZ) + 0.5f))%GNZ;
    return i + GNX*(j + GNY*k);
}

/**
 * The storage index of a node of the lattice, or NUM_STORED if it is not in the block of the rank (only with the
 * domain decomposition).
 */
DEVICE int storedNode(int node) {
#ifdef DOMAIN_DECOMPOSITION
    int i = node%GNX - OX, j = (node/GNX)%GNY - OY, k = node/(GNX*GNY) - OZ;
    if (i < 0 || i >= NX || j < 0 || j >= NY || k < 0 || k >= NZ)
        return NUM_STORED;
    return (i+PAD_X) + SX*((j+PAD_Y) + SY*(k+PAD_Z));
#else
    return node;
#endif
}

/**
 * The node of a sort key node*KEY_STRIDE + e, or NUM_STORED for the keys of the particles of other ranks.
 */
DEVICE int keyNode(mm_long key) {
    mm_long node = key/KEY_STRIDE;
    return (node < NUM_STORED ? (int) node : NUM_STORED);
}

#ifdef DOMAIN_DECOMPOSITION
/**
 * The sort key of particle i whose nearest node, of index node in the lattice, belongs to another rank: it sorts after
 * the keys of the nodes of the block, and keyNode() gives NUM_STORED.  The keys follow the node rather than sharing a
 * few values, because OpenMM's sort puts the keys into buckets by value and sorts a bucket larger than a work group
 * with a single work group: with one key value for the particles of the other ranks, most of them in a large System,
 * the sort made the step several times slower.
 */
DEVICE mm_long otherRankKey(int node, int i) {
    return (NUM_STORED + (mm_long) node)*NUM_COUPLED + i;
}
#endif

#ifdef HAS_SOLID_NODES
/**
 * Gradient of the solid indicator (1 at solid nodes, 0 at fluid nodes), interpolated trilinearly between the
 * eight nodes of the lattice cell that contains the position: it points from the fluid into the wall.
 */
DEVICE mixed3 wallNormal(mixed4 pos, GLOBAL const int* RESTRICT isFluid) {
    mixed s[3] = {wrapCoordinate(pos.x, GNX), wrapCoordinate(pos.y, GNY), wrapCoordinate(pos.z, GNZ)};
    int size[3] = {GNX, GNY, GNZ};
    int index[3][2];
    mixed weight[3][2];
    for (int k = 0; k < 3; k++) {
        mixed lower = floor(s[k]);
        index[k][0] = ((int) lower)%size[k];
        index[k][1] = (index[k][0]+1)%size[k];
        weight[k][1] = s[k]-lower;
        weight[k][0] = 1-weight[k][1];
    }
    mixed solid[2][2][2];
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++)
            for (int c = 0; c < 2; c++)
                solid[a][b][c] = (isFluid[index[0][a] + GNX*(index[1][b] + GNY*index[2][c])] ? 0 : 1);
    mixed3 gradient = make_mixed3(0, 0, 0);
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++) {
            gradient.x += weight[1][a]*weight[2][b]*(solid[1][a][b]-solid[0][a][b]);
            gradient.y += weight[0][a]*weight[2][b]*(solid[a][1][b]-solid[a][0][b]);
            gradient.z += weight[0][a]*weight[1][b]*(solid[a][b][1]-solid[a][b][0]);
        }
    return gradient;
}

/**
 * At the start of a step, a coupled particle whose nearest node is solid and that moves into the wall,
 * v.n > 0, has every component of its velocity reversed.  The wall receives 2 m v (lattice units), stored in
 * wallMomentum[k*NUM_COUPLED + i]; the array is reset for every coupled particle.
 */
KERNEL void reflectParticles(GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT posqCorrection,
        GLOBAL mixed4* RESTRICT velm, GLOBAL const int* RESTRICT atomIndex, GLOBAL const int* RESTRICT couplingIndex,
        GLOBAL const int* RESTRICT isFluid, GLOBAL const mixed* RESTRICT particleMass, GLOBAL mixed* RESTRICT wallMomentum) {
    for (int j = GLOBAL_ID; j < NUM_ATOMS; j += GLOBAL_SIZE) {
        int i = couplingIndex[atomIndex[j]];
        if (i < 0)
            continue;
        mixed4 pos = loadPosition(posq, posqCorrection, j);
        mixed4 v = velm[j];
        mixed3 p = make_mixed3(0, 0, 0);
        mixed3 n = wallNormal(pos, isFluid);
        if (!isFluid[nearestNode(pos)] && v.x*n.x + v.y*n.y + v.z*n.z > 0) {
            mixed scale = 2*particleMass[i]/VELOCITY_SCALE;
            p = make_mixed3(v.x*scale, v.y*scale, v.z*scale);
            velm[j] = make_mixed4(-v.x, -v.y, -v.z, v.w);
        }
#ifdef SKIP_REFLECTION_MOMENTUM
        p = make_mixed3(0, 0, 0);
#endif
        wallMomentum[i] = p.x;
        wallMomentum[NUM_COUPLED+i] = p.y;
        wallMomentum[2*NUM_COUPLED+i] = p.z;
    }
}
#endif

/**
 * Explicit Euler-Maruyama coupling at the nearest node, in lattice units:
 *   F = -gamma m (v - j/rho) + sqrt(2 gamma m kT) xi,
 * with v the velocity of the leapfrog and j the momentum of the fluid after the momentum removal.  A solid node
 * has rho = 0 and is at rest.  The force is stored in particleForce and the key node*NUM_COUPLED + i in sortKeys
 * (sorted next, to sum the reactions of each node in particle order).  In a lattice step (isStep) the reaction
 * on a solid node is added to wallMomentum; the boundary nodes of regularized walls and open faces are fluid nodes
 * and keep it.  xi is noise[i]; with drawNoise it is first copied from OpenMM's
 * random numbers, random[randomIndex + i].
 */
KERNEL void coupleParticles(GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT posqCorrection,
        GLOBAL const mixed4* RESTRICT velm, GLOBAL const int* RESTRICT atomIndex, GLOBAL const int* RESTRICT couplingIndex,
        GLOBAL const mixed* RESTRICT particleMass, GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL const mixed* RESTRICT momentum,
        GLOBAL mixed* RESTRICT particleForce, GLOBAL mm_long* RESTRICT sortKeys, GLOBAL mixed* RESTRICT wallMomentum,
        GLOBAL const float4* RESTRICT random, GLOBAL float4* RESTRICT noise, int randomIndex, int drawNoise, int isStep,
        mixed gamma, mixed kT, GLOBAL const int* RESTRICT isFluid) {
    for (int j = GLOBAL_ID; j < NUM_ATOMS; j += GLOBAL_SIZE) {
        int i = couplingIndex[atomIndex[j]];
        if (i < 0)
            continue;
        mixed4 pos = loadPosition(posq, posqCorrection, j);
        int latticeNode = nearestNode(pos);
        int node = storedNode(latticeNode);
#ifdef DOMAIN_DECOMPOSITION
        if (node == NUM_STORED) {
            // Another rank owns the node: it computes the force.  The random numbers are copied all the same.
            if (kT > 0 && gamma > 0 && drawNoise)
                noise[i] = random[randomIndex+i];
            particleForce[i] = 0;
            particleForce[NUM_COUPLED+i] = 0;
            particleForce[2*NUM_COUPLED+i] = 0;
            sortKeys[i] = otherRankKey(latticeNode, i);
            continue;
        }
#endif
        mixed rho = 1 + densityDeviation[node];
        mixed ux = 0, uy = 0, uz = 0;
        if (rho > 0) {
            ux = momentum[node]*(1/rho);
            uy = momentum[NUM_STORED+node]*(1/rho);
            uz = momentum[2*NUM_STORED+node]*(1/rho);
        }
        mixed4 v = velm[j];
        mixed vx = v.x*(1/(mixed) VELOCITY_SCALE), vy = v.y*(1/(mixed) VELOCITY_SCALE), vz = v.z*(1/(mixed) VELOCITY_SCALE);
        mixed m = particleMass[i];
        mixed fx = (vx-ux)*(-gamma*m), fy = (vy-uy)*(-gamma*m), fz = (vz-uz)*(-gamma*m);
        if (kT > 0 && gamma > 0) {
            mixed sigma = sqrt(2*gamma*m*kT);
            float4 xi;
            if (drawNoise) {
                xi = random[randomIndex+i];
                noise[i] = xi;
            }
            else
                xi = noise[i];
            fx += xi.x*sigma;
            fy += xi.y*sigma;
            fz += xi.z*sigma;
        }
        particleForce[i] = fx;
        particleForce[NUM_COUPLED+i] = fy;
        particleForce[2*NUM_COUPLED+i] = fz;
        sortKeys[i] = ((mm_long) node)*NUM_COUPLED + i;
#ifdef HAS_SOLID_NODES
        if (isStep && rho == 0) {
            wallMomentum[i] -= fx;
            wallMomentum[NUM_COUPLED+i] -= fy;
            wallMomentum[2*NUM_COUPLED+i] -= fz;
        }
#endif
    }
}

/**
 * After sorting the keys, the first entry of each node sums the reactions -F of the particles of that node in
 * particle order and writes the sum to cellReaction: one writer per node, no atomic operations.
 */
KERNEL void sumCellReactions(GLOBAL const mm_long* RESTRICT sortKeys, GLOBAL const mixed* RESTRICT particleForce,
        GLOBAL mixed* RESTRICT cellReaction) {
    for (int k = GLOBAL_ID; k < NUM_COUPLED; k += GLOBAL_SIZE) {
        int node = keyNode(sortKeys[k]);
        if (k > 0 && keyNode(sortKeys[k-1]) == node)
            continue;
#ifdef DOMAIN_DECOMPOSITION
        if (node == NUM_STORED)
            continue;
#endif
        mixed fx = 0, fy = 0, fz = 0;
        for (int m = k; m < NUM_COUPLED && keyNode(sortKeys[m]) == node; m++) {
            int i = (int) (sortKeys[m] - ((mm_long) node)*NUM_COUPLED);
            fx -= particleForce[i];
            fy -= particleForce[NUM_COUPLED+i];
            fz -= particleForce[2*NUM_COUPLED+i];
        }
        cellReaction[node] = fx;
        cellReaction[NUM_STORED+node] = fy;
        cellReaction[2*NUM_STORED+node] = fz;
    }
}

#ifdef STENCIL_WIDTH
/**
 * The kernel phi(r) of the interpolation stencil, r being the distance from the node in lattice spacings, as in
 * internal/LBMStencils.h: trilinear (STENCIL_WIDTH 2), three-point of Roma, Peskin and Berger (3), Keys (4).
 */
DEVICE mixed stencilKernel(mixed r) {
    mixed s = fabs(r);
#if STENCIL_WIDTH == 2
    return (s < 1 ? 1-s : 0);
#elif STENCIL_WIDTH == 3
    if (s <= (mixed) 0.5)
        return (1 + sqrt(1-3*s*s))/3;
    if (s < (mixed) 1.5)
        return (5 - 3*s - sqrt(max((mixed) 0, 1-3*(1-s)*(1-s))))/6;
    return 0;
#else
    if (s <= 1)
        return 1 - (mixed) 2.5*s*s + (mixed) 1.5*s*s*s;
    if (s < 2)
        return 2 - 4*s + (mixed) 2.5*s*s - (mixed) 0.5*s*s*s;
    return 0;
#endif
}

/**
 * The nodes of the stencil along one axis and their weights, for a coordinate s in lattice spacings wrapped into
 * [0, size): STENCIL_WIDTH nodes in increasing order from floor(s) (trilinear), round(s) - 1 (three-point) or
 * floor(s) - 1 (Keys), wrapped periodically, or along an axis with open faces (open, OPEN_X/Y/Z) moved onto the
 * nearest node of the lattice, 0 or size - 1; a node in several slots has the sum of their weights in its first slot
 * and 0 in the others, as LBMStencils::axisWeights().
 */
DEVICE int stencilFirst(mixed s) {
#if STENCIL_WIDTH == 3
    return (int) floor(s + (mixed) 0.5) - 1;
#elif STENCIL_WIDTH == 4
    return (int) floor(s) - 1;
#else
    return (int) floor(s);
#endif
}

DEVICE void stencilAxis(mixed s, int size, int open, int* index, mixed* weight) {
    int first = stencilFirst(s);
    for (int a = 0; a < STENCIL_WIDTH; a++) {
        weight[a] = stencilKernel(s - (first + a));
        index[a] = (open ? min(max(first + a, 0), size - 1) : ((first + a)%size + size)%size);
    }
    for (int a = 1; a < STENCIL_WIDTH; a++)
        for (int b = 0; b < a; b++)
            if (index[b] == index[a]) {
                weight[b] += weight[a];
                weight[a] = 0;
                break;
            }
}

/**
 * The node whose rank couples a particle with a stencil: the nearest node, or along an axis with open faces the nearest
 * node of the lattice, so that the particles between node size - 1 and the end of the box belong to node size - 1, like
 * their stencils (LBMStencils::ownerIndex()).
 */
DEVICE int stencilOwnerNode(mixed4 pos) {
    int i = (int) floor(wrapCoordinate(pos.x, GNX) + 0.5f);
    int j = (int) floor(wrapCoordinate(pos.y, GNY) + 0.5f);
    int k = (int) floor(wrapCoordinate(pos.z, GNZ) + 0.5f);
    i = (OPEN_X ? min(i, GNX - 1) : i%GNX);
    j = (OPEN_Y ? min(j, GNY - 1) : j%GNY);
    k = (OPEN_Z ? min(k, GNZ - 1) : k%GNZ);
    return i + GNX*(j + GNY*k);
}

#ifdef DOMAIN_DECOMPOSITION
/**
 * With the domain decomposition, for a particle whose nearest node belongs to the block: the positions along one axis,
 * in the arrays of the block, of the nodes of its stencil.  They are at most pad nodes from the nearest node (pad = 1, or
 * 2 with Keys), so they lie in the block or in its halo, at the same offsets from the nearest node as in the lattice; a
 * node of the lattice may appear at two positions of the halo of a small block, which both hold its moments
 * (unpackCouplingHalo).  s is the coordinate wrapped into [0, size), origin the first node of the block along the axis,
 * and pad its layers of halo, 0 along an axis that is not divided, where the block is the whole axis.  Along an axis
 * with open faces the nodes are those of stencilAxis(), which do not wrap.
 */
DEVICE void stencilAxisStored(mixed s, int size, int origin, int pad, int open, int* stored) {
    int first = stencilFirst(s);
    int nearest = (int) floor(s + 0.5f);
    for (int a = 0; a < STENCIL_WIDTH; a++) {
        if (open)
            stored[a] = min(max(first + a, 0), size - 1) - origin + pad;
        else
            stored[a] = (pad > 0 ? nearest%size - origin + (first + a - nearest) + pad : ((first + a)%size + size)%size);
    }
}
#endif

/**
 * The sort key of slot e of a stencil on a node of the lattice, pi(node)*KEY_STRIDE + e.  The node is its index in the
 * lattice, which without the decomposition is its storage index; with the decomposition the keys, and so the segments
 * of the sorted keys, are the same on every rank.  pi(n) = n*STENCIL_SCRAMBLE mod GNX*GNY*GNZ, a permutation that spreads
 * the keys uniformly over their range wherever the particles are, so that the buckets of equal width of OpenMM's sort
 * stay small (CouplingSortTrait in CommonLBMKernels.cpp, where the keys of the nearest node are permuted only for the
 * sort, so that their kernels stay those of version 0.4.0); stencilKeyNode() inverts it with STENCIL_UNSCRAMBLE.
 */
DEVICE mm_long stencilKey(int ix, int iy, int iz, int e) {
    const mm_long numNodes = (mm_long) GNX*GNY*GNZ;
    mm_long node = ((ix + GNX*(iy + (mm_long) GNY*iz))*STENCIL_SCRAMBLE)%numNodes;
    return node*KEY_STRIDE + e;
}

/**
 * The node of the arrays of the block that receives the reaction of a stencil key, or NUM_STORED for the nodes outside
 * the block (only with the decomposition).
 */
DEVICE int stencilReactionNode(mm_long key) {
    const mm_long numNodes = (mm_long) GNX*GNY*GNZ;
    mm_long scrambled = key/KEY_STRIDE;
    int node = (int) ((scrambled*STENCIL_UNSCRAMBLE)%numNodes);
#ifdef DOMAIN_DECOMPOSITION
    return storedNode(node);
#else
    return node;
#endif
}

/**
 * The segments of the sorted stencil keys, one per node, listed after every sort so that the sums over a segment run one
 * thread per segment, in sorted order, instead of one thread per key of which only the first of each segment works.  In
 * three passes over chunks of LBM_BLOCK_SIZE sorted keys: countKeySegments() counts the segments that start in each chunk,
 * scanKeySegments() (one work group) turns the counts into offsets, with the number of segments in
 * chunkOffset[NUM_KEY_CHUNKS], and listKeySegments() writes the first key of each segment s to segmentStart[s]
 * (segmentStart[number of segments] = NUM_KEYS), and the weight and particle of each sorted key m to sortedWeight[m] and
 * sortedParticle[m], which the sums then read contiguously; with the centred drag also the segment of each slot e,
 * keySegment[e], where the conjugate gradients keep the vector spread on its node.
 */
#define NUM_KEY_CHUNKS ((NUM_KEYS+LBM_BLOCK_SIZE-1)/LBM_BLOCK_SIZE)

DEVICE bool startsKeySegment(GLOBAL const mm_long* RESTRICT sortKeys, int k) {
    return (k == 0 || sortKeys[k-1]/KEY_STRIDE != sortKeys[k]/KEY_STRIDE);
}

KERNEL void countKeySegments(GLOBAL const mm_long* RESTRICT sortKeys, GLOBAL int* RESTRICT chunkCount) {
    LOCAL int counts[LBM_BLOCK_SIZE];
    for (int c = GROUP_ID; c < NUM_KEY_CHUNKS; c += NUM_GROUPS) {
        int k = c*LBM_BLOCK_SIZE + LOCAL_ID;
        counts[LOCAL_ID] = (k < NUM_KEYS && startsKeySegment(sortKeys, k) ? 1 : 0);
        for (int step = LBM_BLOCK_SIZE/2; step > 0; step /= 2) {
            SYNC_THREADS;
            if (LOCAL_ID < step)
                counts[LOCAL_ID] += counts[LOCAL_ID+step];
        }
        if (LOCAL_ID == 0)
            chunkCount[c] = counts[0];
        SYNC_THREADS;
    }
}

KERNEL void scanKeySegments(GLOBAL const int* RESTRICT chunkCount, GLOBAL int* RESTRICT chunkOffset,
        GLOBAL int* RESTRICT segmentStart) {
    LOCAL int sums[LBM_BLOCK_SIZE];
    const int chunksPerThread = (NUM_KEY_CHUNKS+LBM_BLOCK_SIZE-1)/LBM_BLOCK_SIZE;
    int first = LOCAL_ID*chunksPerThread, last = min(first+chunksPerThread, NUM_KEY_CHUNKS);
    int total = 0;
    for (int c = first; c < last; c++)
        total += chunkCount[c];
    sums[LOCAL_ID] = total;
    for (int step = 1; step < LBM_BLOCK_SIZE; step *= 2) {
        SYNC_THREADS;
        int add = (LOCAL_ID >= step ? sums[LOCAL_ID-step] : 0);
        SYNC_THREADS;
        sums[LOCAL_ID] += add;
    }
    int offset = sums[LOCAL_ID]-total;
    for (int c = first; c < last; c++) {
        chunkOffset[c] = offset;
        offset += chunkCount[c];
    }
    if (LOCAL_ID == LBM_BLOCK_SIZE-1) {
        chunkOffset[NUM_KEY_CHUNKS] = sums[LOCAL_ID];
        segmentStart[sums[LOCAL_ID]] = NUM_KEYS;
    }
}

KERNEL void listKeySegments(GLOBAL const mm_long* RESTRICT sortKeys, GLOBAL const int* RESTRICT chunkOffset,
        GLOBAL const mixed* RESTRICT stencilWeight, GLOBAL int* RESTRICT segmentStart, GLOBAL mixed* RESTRICT sortedWeight,
        GLOBAL int* RESTRICT sortedParticle
#ifdef CENTERED_DRAG
        , GLOBAL int* RESTRICT keySegment
#endif
        ) {
    LOCAL int sums[LBM_BLOCK_SIZE];
    for (int c = GROUP_ID; c < NUM_KEY_CHUNKS; c += NUM_GROUPS) {
        int k = c*LBM_BLOCK_SIZE + LOCAL_ID;
        bool start = (k < NUM_KEYS && startsKeySegment(sortKeys, k));
        sums[LOCAL_ID] = (start ? 1 : 0);
        for (int step = 1; step < LBM_BLOCK_SIZE; step *= 2) {
            SYNC_THREADS;
            int add = (LOCAL_ID >= step ? sums[LOCAL_ID-step] : 0);
            SYNC_THREADS;
            sums[LOCAL_ID] += add;
        }
        if (k < NUM_KEYS) {
            int s = chunkOffset[c] + sums[LOCAL_ID] - 1;
            if (start)
                segmentStart[s] = k;
            mm_long key = sortKeys[k];
            int e = (int) (key - (key/KEY_STRIDE)*KEY_STRIDE);
            sortedWeight[k] = stencilWeight[e];
            sortedParticle[k] = e/STENCIL_SIZE;
#ifdef CENTERED_DRAG
            keySegment[e] = s;
#endif
        }
        SYNC_THREADS;
    }
}

/**
 * Explicit drag with an interpolation stencil (docs/theory.md, section 9), as on the Reference platform: the
 * particle sees u = sum_j xi_j j_j/rho_j over the STENCIL_SIZE nodes of its stencil, x fastest, a solid node being a wall
 * at rest (rho = 0), and F = -gamma m (v - u) + sqrt(2 gamma m kT) xi.  The weight of slot n of particle i is stored in
 * stencilWeight[i*STENCIL_SIZE + n] and its key node*KEY_STRIDE + i*STENCIL_SIZE + n in sortKeys, sorted next to sum the
 * reactions -xi_j F of each node in particle order and, for a particle, in slot order.  In a lattice step the reaction
 * on the solid nodes of the stencil is added to wallMomentum.  The arguments are those of coupleParticles(), and then
 * stencilWeight.  With the domain decomposition the rank that owns the nearest node computes the force, from the
 * moments of the block and of its coupling halo; the other ranks set it to zero, and every rank writes the weights and
 * keys of all the stencils (stencilKey()), for the reactions on its own nodes once the forces are summed.
 */
KERNEL void coupleParticlesStencil(GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT posqCorrection,
        GLOBAL const mixed4* RESTRICT velm, GLOBAL const int* RESTRICT atomIndex, GLOBAL const int* RESTRICT couplingIndex,
        GLOBAL const mixed* RESTRICT particleMass, GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL const mixed* RESTRICT momentum,
        GLOBAL mixed* RESTRICT particleForce, GLOBAL mm_long* RESTRICT sortKeys, GLOBAL mixed* RESTRICT wallMomentum,
        GLOBAL const float4* RESTRICT random, GLOBAL float4* RESTRICT noise, int randomIndex, int drawNoise, int isStep,
        mixed gamma, mixed kT, GLOBAL const int* RESTRICT isFluid, GLOBAL mixed* RESTRICT stencilWeight) {
    for (int j = GLOBAL_ID; j < NUM_ATOMS; j += GLOBAL_SIZE) {
        int i = couplingIndex[atomIndex[j]];
        if (i < 0)
            continue;
        mixed4 pos = loadPosition(posq, posqCorrection, j);
        int ix[STENCIL_WIDTH], iy[STENCIL_WIDTH], iz[STENCIL_WIDTH];
        mixed wx[STENCIL_WIDTH], wy[STENCIL_WIDTH], wz[STENCIL_WIDTH];
        mixed sx = wrapCoordinate(pos.x, GNX), sy = wrapCoordinate(pos.y, GNY), sz = wrapCoordinate(pos.z, GNZ);
        stencilAxis(sx, GNX, OPEN_X, ix, wx);
        stencilAxis(sy, GNY, OPEN_Y, iy, wy);
        stencilAxis(sz, GNZ, OPEN_Z, iz, wz);
#ifdef DOMAIN_DECOMPOSITION
        // Positions of the nodes of the stencil in the arrays of the block, if the rank owns the nearest node.
        bool owned = (storedNode(stencilOwnerNode(pos)) != NUM_STORED);
        int jx[STENCIL_WIDTH], jy[STENCIL_WIDTH], jz[STENCIL_WIDTH];
        if (owned) {
            stencilAxisStored(sx, GNX, OX, PAD_X, OPEN_X, jx);
            stencilAxisStored(sy, GNY, OY, PAD_Y, OPEN_Y, jy);
            stencilAxisStored(sz, GNZ, OZ, PAD_Z, OPEN_Z, jz);
        }
#else
        const bool owned = true;
        int* jx = ix;
        int* jy = iy;
        int* jz = iz;
#endif
        mixed ux = 0, uy = 0, uz = 0;
        for (int c = 0; c < STENCIL_WIDTH; c++)
            for (int b = 0; b < STENCIL_WIDTH; b++)
                for (int a = 0; a < STENCIL_WIDTH; a++) {
                    int slot = a + STENCIL_WIDTH*(b + STENCIL_WIDTH*c);
                    mixed w = wx[a]*wy[b]*wz[c];
                    stencilWeight[i*STENCIL_SIZE+slot] = w;
                    sortKeys[i*STENCIL_SIZE+slot] = stencilKey(ix[a], iy[b], iz[c], i*STENCIL_SIZE + slot);
                    if (!owned)
                        continue;
                    int node = jx[a] + SX*(jy[b] + SY*jz[c]);
                    mixed rho = 1 + densityDeviation[node];
                    if (rho > 0) {
                        ux += momentum[node]*(w/rho);
                        uy += momentum[NUM_STORED+node]*(w/rho);
                        uz += momentum[2*NUM_STORED+node]*(w/rho);
                    }
                }
        if (!owned) {
            // Another rank owns the nearest node: it computes the force.  The random numbers are copied all the same.
            if (kT > 0 && gamma > 0 && drawNoise)
                noise[i] = random[randomIndex+i];
            particleForce[i] = 0;
            particleForce[NUM_COUPLED+i] = 0;
            particleForce[2*NUM_COUPLED+i] = 0;
            continue;
        }
        mixed4 v = velm[j];
        mixed vx = v.x*(1/(mixed) VELOCITY_SCALE), vy = v.y*(1/(mixed) VELOCITY_SCALE), vz = v.z*(1/(mixed) VELOCITY_SCALE);
        mixed m = particleMass[i];
        mixed fx = (vx-ux)*(-gamma*m), fy = (vy-uy)*(-gamma*m), fz = (vz-uz)*(-gamma*m);
        if (kT > 0 && gamma > 0) {
            mixed sigma = sqrt(2*gamma*m*kT);
            float4 xi;
            if (drawNoise) {
                xi = random[randomIndex+i];
                noise[i] = xi;
            }
            else
                xi = noise[i];
            fx += xi.x*sigma;
            fy += xi.y*sigma;
            fz += xi.z*sigma;
        }
        particleForce[i] = fx;
        particleForce[NUM_COUPLED+i] = fy;
        particleForce[2*NUM_COUPLED+i] = fz;
#ifdef HAS_SOLID_NODES
        if (isStep)
            for (int c = 0; c < STENCIL_WIDTH; c++)
                for (int b = 0; b < STENCIL_WIDTH; b++)
                    for (int a = 0; a < STENCIL_WIDTH; a++) {
                        int node = jx[a] + SX*(jy[b] + SY*jz[c]);
                        if (!(1 + densityDeviation[node] > 0)) {
                            mixed w = stencilWeight[i*STENCIL_SIZE + a + STENCIL_WIDTH*(b + STENCIL_WIDTH*c)];
                            wallMomentum[i] -= w*fx;
                            wallMomentum[NUM_COUPLED+i] -= w*fy;
                            wallMomentum[2*NUM_COUPLED+i] -= w*fz;
                        }
                    }
#endif
    }
}

/**
 * After sorting the keys of the stencils and listing their segments, each segment sums the reactions -xi_j F of the
 * particles whose stencils contain its node, in particle order, and writes the sum to cellReaction: one writer per node.
 * With the domain decomposition only the nodes of the block receive the reactions.
 */
KERNEL void sumStencilReactions(GLOBAL const mm_long* RESTRICT sortKeys, GLOBAL const mixed* RESTRICT particleForce,
        GLOBAL mixed* RESTRICT cellReaction, GLOBAL const int* RESTRICT segmentStart, GLOBAL const int* RESTRICT chunkOffset,
        GLOBAL const mixed* RESTRICT sortedWeight, GLOBAL const int* RESTRICT sortedParticle) {
    const int numSegments = chunkOffset[NUM_KEY_CHUNKS];
    for (int s = GLOBAL_ID; s < numSegments; s += GLOBAL_SIZE) {
        int first = segmentStart[s], end = segmentStart[s+1];
        int node = stencilReactionNode(sortKeys[first]);
        if (node == NUM_STORED)
            continue;
        mixed fx = 0, fy = 0, fz = 0;
        for (int m = first; m < end; m++) {
            int i = sortedParticle[m];
            mixed w = sortedWeight[m];
            fx -= w*particleForce[i];
            fy -= w*particleForce[NUM_COUPLED+i];
            fz -= w*particleForce[2*NUM_COUPLED+i];
        }
        cellReaction[node] = fx;
        cellReaction[NUM_STORED+node] = fy;
        cellReaction[2*NUM_STORED+node] = fz;
    }
}

/**
 * Centred drag with an interpolation stencil (docs/theory.md, section 9), first part, one thread per atom, as on the
 * Reference platform (lattice units, h = 1/2, a = gamma h): the linear system
 *   (1 + a) F_k + a m_k sum_l K_kl F_l = b_k,   K_kl = sum_j xi_jk xi_jl/rho_j,
 *   b_k = -gamma m_k (v~_k - sum_j xi_jk u~_j) + sqrt(2 gamma m_k kT) xi_k,
 * divided by m_k, with v~_k = v_k(t - h) + h Fc_k/m_k and u~_j = (j_j + h rho_j g)/rho_j; a solid node is a wall at rest
 * (u = 0, no 1/rho term).  Writes the right-hand side b_k/m_k (rhs), the diagonal (1 + a)/m_k + a K_kk, and for each slot
 * e = i*STENCIL_SIZE + n of the stencil its weight, xi/rho (0 at a solid node), its node and its sort key.  The other
 * forces Fc and the random numbers are handled as in prepareCenteredDrag().  With the domain decomposition the rank that
 * owns the nearest node writes the right-hand side, the diagonal and the nodes of a particle, which the host sums over
 * the ranks, and the others write zero and NUM_STORED as the nodes; every rank writes the weights and keys of all the
 * stencils (stencilKey()), and xi/rho of all of them comes later, from the densities of the nodes summed over the
 * ranks (gatherSegmentDensity(), stencilInterpolationWeights()).
 */
#if STENCIL_SIZE >= 27
/*
 * With the three-point kernel and Keys a work group of LBM_BLOCK_SIZE threads takes LBM_BLOCK_SIZE/STENCIL_SIZE atoms at
 * a time: one thread per slot computes its weight, key and node and the terms of the interpolated velocity and of the
 * self weight, written to local memory; four threads per particle add them in slot order (the three components of u
 * and the self weight); then one thread per particle writes the right-hand side and the diagonal.  The arithmetic is
 * that of the loop over the slots below, so the results are the same.
 */
#define STENCIL_ATOMS_PER_GROUP (LBM_BLOCK_SIZE/STENCIL_SIZE)

KERNEL void prepareCenteredStencil(GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT posqCorrection,
        GLOBAL const mixed4* RESTRICT velm, GLOBAL const int* RESTRICT atomIndex, GLOBAL const int* RESTRICT couplingIndex,
        GLOBAL const mixed* RESTRICT particleMass, GLOBAL const mm_long* RESTRICT longForces,
        GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL const mixed* RESTRICT momentum, GLOBAL mm_long* RESTRICT sortKeys,
        GLOBAL const float4* RESTRICT random, GLOBAL float4* RESTRICT noise, int randomIndex, int drawNoise, mixed gamma,
        mixed kT, mixed gx, mixed gy, mixed gz, GLOBAL mixed* RESTRICT stencilWeight, GLOBAL mixed* RESTRICT interpWeight,
        GLOBAL int* RESTRICT stencilNode, GLOBAL mixed* RESTRICT rhs, GLOBAL mixed* RESTRICT diagonal
#ifdef HAS_FLOAT_FORCE_BUFFERS
        , GLOBAL const real4* RESTRICT floatForces, int numFloatForces
#endif
        ) {
    LOCAL mixed term[5*LBM_BLOCK_SIZE];
    LOCAL int termUsed[LBM_BLOCK_SIZE];
    LOCAL mixed sums[4*STENCIL_ATOMS_PER_GROUP];
    const mixed h = 0.5f;
    const mixed a = gamma*h;
    const mixed fixedPointScale = 1/(mixed) 0x100000000;
    for (int firstAtom = GROUP_ID*STENCIL_ATOMS_PER_GROUP; firstAtom < NUM_ATOMS; firstAtom += NUM_GROUPS*STENCIL_ATOMS_PER_GROUP) {
        // One thread per slot.
        int slotAtom = firstAtom + LOCAL_ID/STENCIL_SIZE;
        termUsed[LOCAL_ID] = 0;
        int i = (LOCAL_ID < STENCIL_ATOMS_PER_GROUP*STENCIL_SIZE && slotAtom < NUM_ATOMS ? couplingIndex[atomIndex[slotAtom]] : -1);
        if (i >= 0) {
            int slot = LOCAL_ID%STENCIL_SIZE;
            int a2 = slot%STENCIL_WIDTH, b = (slot/STENCIL_WIDTH)%STENCIL_WIDTH, c = slot/(STENCIL_WIDTH*STENCIL_WIDTH);
            mixed4 pos = loadPosition(posq, posqCorrection, slotAtom);
            int ix[STENCIL_WIDTH], iy[STENCIL_WIDTH], iz[STENCIL_WIDTH];
            mixed wx[STENCIL_WIDTH], wy[STENCIL_WIDTH], wz[STENCIL_WIDTH];
            mixed sx = wrapCoordinate(pos.x, GNX), sy = wrapCoordinate(pos.y, GNY), sz = wrapCoordinate(pos.z, GNZ);
            stencilAxis(sx, GNX, OPEN_X, ix, wx);
            stencilAxis(sy, GNY, OPEN_Y, iy, wy);
            stencilAxis(sz, GNZ, OPEN_Z, iz, wz);
#ifdef DOMAIN_DECOMPOSITION
            bool owned = (storedNode(stencilOwnerNode(pos)) != NUM_STORED);
            int jx[STENCIL_WIDTH], jy[STENCIL_WIDTH], jz[STENCIL_WIDTH];
            if (owned) {
                stencilAxisStored(sx, GNX, OX, PAD_X, OPEN_X, jx);
                stencilAxisStored(sy, GNY, OY, PAD_Y, OPEN_Y, jy);
                stencilAxisStored(sz, GNZ, OZ, PAD_Z, OPEN_Z, jz);
            }
#else
            const bool owned = true;
            int* jx = ix;
            int* jy = iy;
            int* jz = iz;
#endif
            int e = i*STENCIL_SIZE + slot;
            mixed w = wx[a2]*wy[b]*wz[c];
            stencilWeight[e] = w;
            sortKeys[e] = stencilKey(ix[a2], iy[b], iz[c], e);
            interpWeight[e] = 0;
            stencilNode[e] = NUM_STORED;
            if (owned) {
                int node = jx[a2] + SX*(jy[b] + SY*jz[c]);
                mixed rho = 1 + densityDeviation[node];
                stencilNode[e] = node;
                if (rho > 0) {
                    term[5*LOCAL_ID] = momentum[node] + h*(rho*gx);
                    term[5*LOCAL_ID+1] = momentum[NUM_STORED+node] + h*(rho*gy);
                    term[5*LOCAL_ID+2] = momentum[2*NUM_STORED+node] + h*(rho*gz);
                    term[5*LOCAL_ID+3] = w/rho;
                    term[5*LOCAL_ID+4] = w*w/rho;
                    termUsed[LOCAL_ID] = 1;
                    interpWeight[e] = w/rho;
                }
            }
        }
        SYNC_THREADS;
        // Four threads per particle: the components of the interpolated velocity and the self weight, in slot order.
        int sumAtom = LOCAL_ID/4, component = LOCAL_ID%4;
        if (sumAtom < STENCIL_ATOMS_PER_GROUP) {
            mixed sum = 0;
            for (int n = 0; n < STENCIL_SIZE; n++) {
                int t = sumAtom*STENCIL_SIZE + n;
                if (termUsed[t]) {
                    if (component < 3)
                        sum += term[5*t+component]*term[5*t+3];
                    else
                        sum += term[5*t+4];
                }
            }
            sums[LOCAL_ID] = sum;
        }
        SYNC_THREADS;
        // One thread per particle.
        int j = firstAtom + LOCAL_ID;
        i = (LOCAL_ID < STENCIL_ATOMS_PER_GROUP && j < NUM_ATOMS ? couplingIndex[atomIndex[j]] : -1);
        if (i >= 0) {
            mixed fx = longForces[j]*fixedPointScale;
            mixed fy = longForces[j+PADDED_NUM_ATOMS]*fixedPointScale;
            mixed fz = longForces[j+2*PADDED_NUM_ATOMS]*fixedPointScale;
#ifdef HAS_FLOAT_FORCE_BUFFERS
            for (int b = 0; b < numFloatForces; b++) {
                real4 f = floatForces[j+b*PADDED_NUM_ATOMS];
                fx += f.x;
                fy += f.y;
                fz += f.z;
            }
#endif
            mixed4 v = velm[j];
            mixed m = particleMass[i];
            mixed scale = h/(m*FORCE_SCALE);
            mixed kx = v.x*(1/(mixed) VELOCITY_SCALE) + fx*scale;
            mixed ky = v.y*(1/(mixed) VELOCITY_SCALE) + fy*scale;
            mixed kz = v.z*(1/(mixed) VELOCITY_SCALE) + fz*scale;
#ifdef DOMAIN_DECOMPOSITION
            bool owned = (storedNode(stencilOwnerNode(loadPosition(posq, posqCorrection, j))) != NUM_STORED);
#else
            const bool owned = true;
#endif
            mixed ux = sums[4*LOCAL_ID], uy = sums[4*LOCAL_ID+1], uz = sums[4*LOCAL_ID+2], self = sums[4*LOCAL_ID+3];
            if (!owned) {
                if (kT > 0 && gamma > 0 && drawNoise)
                    noise[i] = random[randomIndex+i];
                rhs[i] = 0;
                rhs[NUM_COUPLED+i] = 0;
                rhs[2*NUM_COUPLED+i] = 0;
                diagonal[i] = 0;
            }
            else {
                mixed bx = (kx-ux)*(-gamma*m), by = (ky-uy)*(-gamma*m), bz = (kz-uz)*(-gamma*m);
                if (kT > 0 && gamma > 0) {
                    mixed sigma = sqrt(2*gamma*m*kT);
                    float4 xi;
                    if (drawNoise) {
                        xi = random[randomIndex+i];
                        noise[i] = xi;
                    }
                    else
                        xi = noise[i];
                    bx += xi.x*sigma;
                    by += xi.y*sigma;
                    bz += xi.z*sigma;
                }
                rhs[i] = bx*(1/m);
                rhs[NUM_COUPLED+i] = by*(1/m);
                rhs[2*NUM_COUPLED+i] = bz*(1/m);
                diagonal[i] = (1+a)/m + a*self;
            }
        }
        SYNC_THREADS;
    }
}
#else
KERNEL void prepareCenteredStencil(GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT posqCorrection,
        GLOBAL const mixed4* RESTRICT velm, GLOBAL const int* RESTRICT atomIndex, GLOBAL const int* RESTRICT couplingIndex,
        GLOBAL const mixed* RESTRICT particleMass, GLOBAL const mm_long* RESTRICT longForces,
        GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL const mixed* RESTRICT momentum, GLOBAL mm_long* RESTRICT sortKeys,
        GLOBAL const float4* RESTRICT random, GLOBAL float4* RESTRICT noise, int randomIndex, int drawNoise, mixed gamma,
        mixed kT, mixed gx, mixed gy, mixed gz, GLOBAL mixed* RESTRICT stencilWeight, GLOBAL mixed* RESTRICT interpWeight,
        GLOBAL int* RESTRICT stencilNode, GLOBAL mixed* RESTRICT rhs, GLOBAL mixed* RESTRICT diagonal
#ifdef HAS_FLOAT_FORCE_BUFFERS
        , GLOBAL const real4* RESTRICT floatForces, int numFloatForces
#endif
        ) {
    const mixed h = 0.5f;
    const mixed a = gamma*h;
    const mixed fixedPointScale = 1/(mixed) 0x100000000;
    for (int j = GLOBAL_ID; j < NUM_ATOMS; j += GLOBAL_SIZE) {
        int i = couplingIndex[atomIndex[j]];
        if (i < 0)
            continue;
        mixed fx = longForces[j]*fixedPointScale;
        mixed fy = longForces[j+PADDED_NUM_ATOMS]*fixedPointScale;
        mixed fz = longForces[j+2*PADDED_NUM_ATOMS]*fixedPointScale;
#ifdef HAS_FLOAT_FORCE_BUFFERS
        for (int b = 0; b < numFloatForces; b++) {
            real4 f = floatForces[j+b*PADDED_NUM_ATOMS];
            fx += f.x;
            fy += f.y;
            fz += f.z;
        }
#endif
        mixed4 v = velm[j];
        mixed m = particleMass[i];
        mixed scale = h/(m*FORCE_SCALE);
        mixed kx = v.x*(1/(mixed) VELOCITY_SCALE) + fx*scale;
        mixed ky = v.y*(1/(mixed) VELOCITY_SCALE) + fy*scale;
        mixed kz = v.z*(1/(mixed) VELOCITY_SCALE) + fz*scale;
        mixed4 pos = loadPosition(posq, posqCorrection, j);
        int ix[STENCIL_WIDTH], iy[STENCIL_WIDTH], iz[STENCIL_WIDTH];
        mixed wx[STENCIL_WIDTH], wy[STENCIL_WIDTH], wz[STENCIL_WIDTH];
        mixed sx = wrapCoordinate(pos.x, GNX), sy = wrapCoordinate(pos.y, GNY), sz = wrapCoordinate(pos.z, GNZ);
        stencilAxis(sx, GNX, OPEN_X, ix, wx);
        stencilAxis(sy, GNY, OPEN_Y, iy, wy);
        stencilAxis(sz, GNZ, OPEN_Z, iz, wz);
#ifdef DOMAIN_DECOMPOSITION
        bool owned = (storedNode(stencilOwnerNode(pos)) != NUM_STORED);
        int jx[STENCIL_WIDTH], jy[STENCIL_WIDTH], jz[STENCIL_WIDTH];
        if (owned) {
            stencilAxisStored(sx, GNX, OX, PAD_X, OPEN_X, jx);
            stencilAxisStored(sy, GNY, OY, PAD_Y, OPEN_Y, jy);
            stencilAxisStored(sz, GNZ, OZ, PAD_Z, OPEN_Z, jz);
        }
#else
        const bool owned = true;
        int* jx = ix;
        int* jy = iy;
        int* jz = iz;
#endif
        mixed ux = 0, uy = 0, uz = 0, self = 0;
        for (int c = 0; c < STENCIL_WIDTH; c++)
            for (int b = 0; b < STENCIL_WIDTH; b++)
                for (int a2 = 0; a2 < STENCIL_WIDTH; a2++) {
                    int e = i*STENCIL_SIZE + a2 + STENCIL_WIDTH*(b + STENCIL_WIDTH*c);
                    mixed w = wx[a2]*wy[b]*wz[c];
                    stencilWeight[e] = w;
                    sortKeys[e] = stencilKey(ix[a2], iy[b], iz[c], e);
                    interpWeight[e] = 0;
                    stencilNode[e] = NUM_STORED;
                    if (!owned)
                        continue;
                    int node = jx[a2] + SX*(jy[b] + SY*jz[c]);
                    mixed rho = 1 + densityDeviation[node];
                    stencilNode[e] = node;
                    if (rho > 0) {
                        ux += (momentum[node] + h*(rho*gx))*(w/rho);
                        uy += (momentum[NUM_STORED+node] + h*(rho*gy))*(w/rho);
                        uz += (momentum[2*NUM_STORED+node] + h*(rho*gz))*(w/rho);
                        self += w*w/rho;
                        interpWeight[e] = w/rho;
                    }
                }
        if (!owned) {
            if (kT > 0 && gamma > 0 && drawNoise)
                noise[i] = random[randomIndex+i];
            rhs[i] = 0;
            rhs[NUM_COUPLED+i] = 0;
            rhs[2*NUM_COUPLED+i] = 0;
            diagonal[i] = 0;
            continue;
        }
        mixed bx = (kx-ux)*(-gamma*m), by = (ky-uy)*(-gamma*m), bz = (kz-uz)*(-gamma*m);
        if (kT > 0 && gamma > 0) {
            mixed sigma = sqrt(2*gamma*m*kT);
            float4 xi;
            if (drawNoise) {
                xi = random[randomIndex+i];
                noise[i] = xi;
            }
            else
                xi = noise[i];
            bx += xi.x*sigma;
            by += xi.y*sigma;
            bz += xi.z*sigma;
        }
        rhs[i] = bx*(1/m);
        rhs[NUM_COUPLED+i] = by*(1/m);
        rhs[2*NUM_COUPLED+i] = bz*(1/m);
        diagonal[i] = (1+a)/m + a*self;
    }
}
#endif

/**
 * Conjugate gradients of the centred drag: the vector p of the particles spread on the nodes, sum_l xi_jl p_l in
 * particle order, written for the node of segment s at spreadValue[3*s ... 3*s+2], the three components together for
 * the scattered reads of multiplyStencilMatrix().  With the domain decomposition every rank runs the conjugate
 * gradients for all the particles, and spreads p on all the nodes of the stencils.
 */
KERNEL void spreadStencilVector(GLOBAL const int* RESTRICT segmentStart, GLOBAL const int* RESTRICT chunkOffset,
        GLOBAL const mixed* RESTRICT sortedWeight, GLOBAL const int* RESTRICT sortedParticle, GLOBAL const mixed* RESTRICT p,
        GLOBAL mixed* RESTRICT spreadValue) {
    const int numSegments = chunkOffset[NUM_KEY_CHUNKS];
    for (int s = GLOBAL_ID; s < numSegments; s += GLOBAL_SIZE) {
        int first = segmentStart[s], end = segmentStart[s+1];
        mixed sx = 0, sy = 0, sz = 0;
        for (int m = first; m < end; m++) {
            int i = sortedParticle[m];
            mixed w = sortedWeight[m];
            sx += w*p[i];
            sy += w*p[NUM_COUPLED+i];
            sz += w*p[2*NUM_COUPLED+i];
        }
        spreadValue[3*s] = sx;
        spreadValue[3*s+1] = sy;
        spreadValue[3*s+2] = sz;
    }
}

/**
 * The product of the matrix of the centred drag with p: (1 + a) p_k/m_k + a sum_j xi_jk/rho_j (spread p)_j over the
 * stencil of particle k, in slot order.  The values of the slots are scattered: with the three-point kernel and Keys a
 * work group of LBM_BLOCK_SIZE threads takes LBM_BLOCK_SIZE/STENCIL_SIZE particles at a time, its threads read the
 * values of the slots together, one per slot, into local memory, and then three threads per particle, one per
 * component, add them in slot order; with the trilinear kernel, whose 8 slots this does not speed up, one thread per
 * particle reads and adds them.
 */
#if STENCIL_SIZE >= 27
#define STENCIL_PARTICLES_PER_GROUP (LBM_BLOCK_SIZE/STENCIL_SIZE)

KERNEL void multiplyStencilMatrix(GLOBAL const mixed* RESTRICT particleMass, GLOBAL const mixed* RESTRICT interpWeight,
        GLOBAL const int* RESTRICT keySegment, GLOBAL const mixed* RESTRICT spreadValue, GLOBAL const mixed* RESTRICT p,
        GLOBAL mixed* RESTRICT result, mixed gamma) {
    LOCAL mixed slotValue[4*LBM_BLOCK_SIZE];
    const mixed a = gamma*0.5f;
    for (int firstParticle = GROUP_ID*STENCIL_PARTICLES_PER_GROUP; firstParticle < NUM_COUPLED;
            firstParticle += NUM_GROUPS*STENCIL_PARTICLES_PER_GROUP) {
        int slotParticle = firstParticle + LOCAL_ID/STENCIL_SIZE;
        if (LOCAL_ID < STENCIL_PARTICLES_PER_GROUP*STENCIL_SIZE && slotParticle < NUM_COUPLED) {
            int e = slotParticle*STENCIL_SIZE + LOCAL_ID%STENCIL_SIZE;
            int k = keySegment[e];
            slotValue[4*LOCAL_ID] = spreadValue[3*k];
            slotValue[4*LOCAL_ID+1] = spreadValue[3*k+1];
            slotValue[4*LOCAL_ID+2] = spreadValue[3*k+2];
            slotValue[4*LOCAL_ID+3] = interpWeight[e];
        }
        SYNC_THREADS;
        int localParticle = LOCAL_ID/3, component = LOCAL_ID%3;
        int i = firstParticle + localParticle;
        if (localParticle < STENCIL_PARTICLES_PER_GROUP && i < NUM_COUPLED) {
            mixed sum = 0;
            for (int n = 0; n < STENCIL_SIZE; n++) {
                int slot = 4*(localParticle*STENCIL_SIZE + n);
                sum += slotValue[slot+component]*slotValue[slot+3];
            }
            mixed d = (1+a)/particleMass[i];
            result[component*NUM_COUPLED+i] = p[component*NUM_COUPLED+i]*d + sum*a;
        }
        SYNC_THREADS;
    }
}
#else
KERNEL void multiplyStencilMatrix(GLOBAL const mixed* RESTRICT particleMass, GLOBAL const mixed* RESTRICT interpWeight,
        GLOBAL const int* RESTRICT keySegment, GLOBAL const mixed* RESTRICT spreadValue, GLOBAL const mixed* RESTRICT p,
        GLOBAL mixed* RESTRICT result, mixed gamma) {
    const mixed a = gamma*0.5f;
    for (int i = GLOBAL_ID; i < NUM_COUPLED; i += GLOBAL_SIZE) {
        mixed sx = 0, sy = 0, sz = 0;
        for (int n = 0; n < STENCIL_SIZE; n++) {
            int e = i*STENCIL_SIZE + n;
            int k = keySegment[e];
            mixed w = interpWeight[e];
            sx += spreadValue[3*k]*w;
            sy += spreadValue[3*k+1]*w;
            sz += spreadValue[3*k+2]*w;
        }
        mixed d = (1+a)/particleMass[i];
        result[i] = p[i]*d + sx*a;
        result[NUM_COUPLED+i] = p[NUM_COUPLED+i]*d + sy*a;
        result[2*NUM_COUPLED+i] = p[2*NUM_COUPLED+i]*d + sz*a;
    }
}
#endif

/**
 * The three components of the dot product of two vectors of the particles, in two stages so that the result does not
 * depend on the timing: DOT_GROUPS work groups of LBM_BLOCK_SIZE threads add the products of a fixed subset of the
 * particles in a fixed order and write their partial sums (dotStencilPartials), then a single work group adds the partial
 * sums in a fixed order and writes the result to scalars[offset + k] (dotStencilFinish).  DOT_GROUPS depends only on
 * the number of particles, so that ranks with different devices compute the same sums.
 */
KERNEL void dotStencilPartials(GLOBAL const mixed* RESTRICT x, GLOBAL const mixed* RESTRICT y, GLOBAL mixed* RESTRICT partials) {
    LOCAL mixed sums[3*LBM_BLOCK_SIZE];
    mixed s[3] = {0, 0, 0};
    for (int i = GROUP_ID*LBM_BLOCK_SIZE + LOCAL_ID; i < NUM_COUPLED; i += DOT_GROUPS*LBM_BLOCK_SIZE)
        for (int k = 0; k < 3; k++)
            s[k] += x[k*NUM_COUPLED+i]*y[k*NUM_COUPLED+i];
    for (int k = 0; k < 3; k++)
        sums[k*LBM_BLOCK_SIZE+LOCAL_ID] = s[k];
    for (int step = LBM_BLOCK_SIZE/2; step > 0; step /= 2) {
        SYNC_THREADS;
        if (LOCAL_ID < step)
            for (int k = 0; k < 3; k++)
                sums[k*LBM_BLOCK_SIZE+LOCAL_ID] += sums[k*LBM_BLOCK_SIZE+LOCAL_ID+step];
    }
    if (LOCAL_ID == 0)
        for (int k = 0; k < 3; k++)
            partials[3*GROUP_ID+k] = sums[k*LBM_BLOCK_SIZE];
}

KERNEL void dotStencilFinish(GLOBAL const mixed* RESTRICT partials, GLOBAL mixed* RESTRICT scalars, int offset) {
    LOCAL mixed sums[3*LBM_BLOCK_SIZE];
    mixed s[3] = {0, 0, 0};
    for (int g = LOCAL_ID; g < DOT_GROUPS; g += LBM_BLOCK_SIZE)
        for (int k = 0; k < 3; k++)
            s[k] += partials[3*g+k];
    for (int k = 0; k < 3; k++)
        sums[k*LBM_BLOCK_SIZE+LOCAL_ID] = s[k];
    for (int step = LBM_BLOCK_SIZE/2; step > 0; step /= 2) {
        SYNC_THREADS;
        if (LOCAL_ID < step)
            for (int k = 0; k < 3; k++)
                sums[k*LBM_BLOCK_SIZE+LOCAL_ID] += sums[k*LBM_BLOCK_SIZE+LOCAL_ID+step];
    }
    if (LOCAL_ID == 0)
        for (int k = 0; k < 3; k++)
            scalars[offset+k] = sums[k*LBM_BLOCK_SIZE];
}

/**
 * Start of the conjugate gradients, from the solution of the diagonal: F = b/d, then (after F has been multiplied, in
 * Ap) r = b - Ap, z = r/d, p = z.  stage 0 does the first part, stage 1 the second.
 */
KERNEL void startStencilGradients(GLOBAL const mixed* RESTRICT rhs, GLOBAL const mixed* RESTRICT diagonal,
        GLOBAL mixed* RESTRICT F, GLOBAL const mixed* RESTRICT Ap, GLOBAL mixed* RESTRICT r, GLOBAL mixed* RESTRICT z,
        GLOBAL mixed* RESTRICT p, int stage) {
    for (int i = GLOBAL_ID; i < NUM_COUPLED; i += GLOBAL_SIZE) {
        mixed inverse = 1/diagonal[i];
        for (int k = 0; k < 3; k++) {
            int index = k*NUM_COUPLED+i;
            if (stage == 0)
                F[index] = rhs[index]*inverse;
            else {
                r[index] = rhs[index]-Ap[index];
                z[index] = r[index]*inverse;
                p[index] = z[index];
            }
        }
    }
}

/**
 * One step of the conjugate gradients, for each component: alpha = (r.z)/(p.Ap) from scalars[rz + k] and
 * scalars[pAp + k] (0 if p.Ap is not positive), F += alpha p, r -= alpha Ap, z = r/d.
 */
KERNEL void advanceStencilGradients(GLOBAL const mixed* RESTRICT diagonal, GLOBAL mixed* RESTRICT F,
        GLOBAL mixed* RESTRICT r, GLOBAL mixed* RESTRICT z, GLOBAL const mixed* RESTRICT p, GLOBAL const mixed* RESTRICT Ap,
        GLOBAL const mixed* RESTRICT scalars, int rz, int pAp) {
    for (int i = GLOBAL_ID; i < NUM_COUPLED; i += GLOBAL_SIZE) {
        mixed inverse = 1/diagonal[i];
        for (int k = 0; k < 3; k++) {
            mixed alpha = (scalars[pAp+k] > 0 ? scalars[rz+k]/scalars[pAp+k] : 0);
            int index = k*NUM_COUPLED+i;
            F[index] += alpha*p[index];
            r[index] -= alpha*Ap[index];
            z[index] = r[index]*inverse;
        }
    }
}

/**
 * The new direction p = z + beta p, beta = (r.z)_new/(r.z)_old from scalars[rzNew + k] and scalars[rzOld + k].
 */
KERNEL void updateStencilDirection(GLOBAL const mixed* RESTRICT z, GLOBAL mixed* RESTRICT p,
        GLOBAL const mixed* RESTRICT scalars, int rzOld, int rzNew) {
    for (int i = GLOBAL_ID; i < NUM_COUPLED; i += GLOBAL_SIZE)
        for (int k = 0; k < 3; k++) {
            mixed beta = (scalars[rzOld+k] > 0 ? scalars[rzNew+k]/scalars[rzOld+k] : 0);
            int index = k*NUM_COUPLED+i;
            p[index] = z[index] + beta*p[index];
        }
}

/**
 * In a lattice step with the centred drag: the reaction -xi_j F on the solid nodes of the stencil of each particle goes
 * to the walls, in wallMomentum.  With the domain decomposition the rank that owns the nearest node counts it (the
 * nodes of the other particles are NUM_STORED).
 */
KERNEL void addStencilWallMomentum(GLOBAL const int* RESTRICT stencilNode, GLOBAL const mixed* RESTRICT stencilWeight,
        GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL const mixed* RESTRICT particleForce,
        GLOBAL mixed* RESTRICT wallMomentum) {
    for (int i = GLOBAL_ID; i < NUM_COUPLED; i += GLOBAL_SIZE)
        for (int n = 0; n < STENCIL_SIZE; n++) {
            int e = i*STENCIL_SIZE + n;
            int node = stencilNode[e];
            if (node < NUM_STORED && !(1 + densityDeviation[node] > 0)) {
                mixed w = stencilWeight[e];
                wallMomentum[i] -= w*particleForce[i];
                wallMomentum[NUM_COUPLED+i] -= w*particleForce[NUM_COUPLED+i];
                wallMomentum[2*NUM_COUPLED+i] -= w*particleForce[2*NUM_COUPLED+i];
            }
        }
}

#ifdef DOMAIN_DECOMPOSITION
/**
 * The coupling halo (CommonCalcLBMForceKernel::exchangeCouplingHalo()): write rho - 1 and j received from the owners,
 * four values per node, into the positions of the halo of the moments; slots lists the storage index of each position
 * and sources the node received for it, which may fill several positions.
 */
KERNEL void unpackCouplingHalo(GLOBAL mixed* RESTRICT densityDeviation, GLOBAL mixed* RESTRICT momentum,
        GLOBAL const int* RESTRICT slots, GLOBAL const int* RESTRICT sources, GLOBAL const mixed* RESTRICT received,
        int numSlots) {
    for (int i = GLOBAL_ID; i < numSlots; i += GLOBAL_SIZE) {
        int s = slots[i], h = sources[i];
        densityDeviation[s] = received[4*h];
        momentum[s] = received[4*h+1];
        momentum[NUM_STORED+s] = received[4*h+2];
        momentum[2*NUM_STORED+s] = received[4*h+3];
    }
}

#ifdef CENTERED_DRAG
/**
 * With the domain decomposition and the centred drag: the density of the node of each segment of the sorted keys (the
 * same on every rank), from the rank whose block holds the node and 0 on the others, which the host sums over the
 * ranks; then xi/rho of every slot of every stencil, as prepareCenteredStencil() computes it for the particles of the
 * rank (0 at a solid node).  So every rank runs the conjugate gradients for all the particles, with the same numbers.
 */
KERNEL void gatherSegmentDensity(GLOBAL const mm_long* RESTRICT sortKeys, GLOBAL const int* RESTRICT segmentStart,
        GLOBAL const int* RESTRICT chunkOffset, GLOBAL const mixed* RESTRICT densityDeviation,
        GLOBAL mixed* RESTRICT segmentDensity) {
    const int numSegments = chunkOffset[NUM_KEY_CHUNKS];
    for (int s = GLOBAL_ID; s < numSegments; s += GLOBAL_SIZE) {
        int node = stencilReactionNode(sortKeys[segmentStart[s]]);
        segmentDensity[s] = (node == NUM_STORED ? 0 : 1 + densityDeviation[node]);
    }
}

KERNEL void stencilInterpolationWeights(GLOBAL const mixed* RESTRICT stencilWeight, GLOBAL const int* RESTRICT keySegment,
        GLOBAL const mixed* RESTRICT segmentDensity, GLOBAL mixed* RESTRICT interpWeight) {
    for (int e = GLOBAL_ID; e < NUM_KEYS; e += GLOBAL_SIZE) {
        mixed rho = segmentDensity[keySegment[e]];
        interpWeight[e] = (rho > 0 ? stencilWeight[e]/rho : 0);
    }
}
#endif
#endif
#endif

/**
 * After the collision, set the reactions of the nodes that received any back to zero, for the next step.
 */
KERNEL void clearCellReactions(GLOBAL const mm_long* RESTRICT sortKeys, GLOBAL mixed* RESTRICT cellReaction) {
    for (int k = GLOBAL_ID; k < NUM_KEYS; k += GLOBAL_SIZE) {
#ifdef STENCIL_WIDTH
        int node = stencilReactionNode(sortKeys[k]);
#else
        int node = keyNode(sortKeys[k]);
#endif
#ifdef DOMAIN_DECOMPOSITION
        if (node == NUM_STORED)
            continue;
#endif
        cellReaction[node] = 0;
        cellReaction[NUM_STORED+node] = 0;
        cellReaction[2*NUM_STORED+node] = 0;
    }
}

/**
 * Centred drag, first part, one thread per atom (docs/theory.md, section 2; lattice units, h = 1/2): the velocity
 * with half the other forces, v~ = v(t - h) + h Fc/m, and the random force sqrt(2 gamma m kT) xi of every coupled
 * particle, and its sort key node*NUM_COUPLED + i.  Fc is read from OpenMM's fixed point force buffer and, with
 * HAS_FLOAT_FORCE_BUFFERS, from its floating point buffers, which hold the other forces at the end of the force
 * evaluation.  The random numbers are handled as in coupleParticles().
 */
KERNEL void prepareCenteredDrag(GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT posqCorrection,
        GLOBAL const mixed4* RESTRICT velm, GLOBAL const int* RESTRICT atomIndex, GLOBAL const int* RESTRICT couplingIndex,
        GLOBAL const mixed* RESTRICT particleMass, GLOBAL const mm_long* RESTRICT longForces, GLOBAL mixed* RESTRICT knownVelocity,
        GLOBAL mixed* RESTRICT randomForce, GLOBAL mm_long* RESTRICT sortKeys, GLOBAL const float4* RESTRICT random,
        GLOBAL float4* RESTRICT noise, int randomIndex, int drawNoise, mixed gamma, mixed kT
#ifdef HAS_FLOAT_FORCE_BUFFERS
        , GLOBAL const real4* RESTRICT floatForces, int numFloatForces
#endif
#ifdef DOMAIN_DECOMPOSITION
        , GLOBAL mixed* RESTRICT particleForce
#endif
        ) {
    const mixed h = 0.5f;
    const mixed fixedPointScale = 1/(mixed) 0x100000000;
    for (int j = GLOBAL_ID; j < NUM_ATOMS; j += GLOBAL_SIZE) {
        int i = couplingIndex[atomIndex[j]];
        if (i < 0)
            continue;
        mixed4 pos = loadPosition(posq, posqCorrection, j);
        int latticeNode = nearestNode(pos);
        int node = storedNode(latticeNode);
#ifdef DOMAIN_DECOMPOSITION
        if (node == NUM_STORED) {
            // Another rank owns the node and solves the drag: the force is zero here.
            particleForce[i] = 0;
            particleForce[NUM_COUPLED+i] = 0;
            particleForce[2*NUM_COUPLED+i] = 0;
        }
#endif
        mixed fx = longForces[j]*fixedPointScale;
        mixed fy = longForces[j+PADDED_NUM_ATOMS]*fixedPointScale;
        mixed fz = longForces[j+2*PADDED_NUM_ATOMS]*fixedPointScale;
#ifdef HAS_FLOAT_FORCE_BUFFERS
        for (int b = 0; b < numFloatForces; b++) {
            real4 f = floatForces[j+b*PADDED_NUM_ATOMS];
            fx += f.x;
            fy += f.y;
            fz += f.z;
        }
#endif
        mixed4 v = velm[j];
        mixed m = particleMass[i];
        mixed scale = h/(m*FORCE_SCALE);
        knownVelocity[i] = v.x*(1/(mixed) VELOCITY_SCALE) + fx*scale;
        knownVelocity[NUM_COUPLED+i] = v.y*(1/(mixed) VELOCITY_SCALE) + fy*scale;
        knownVelocity[2*NUM_COUPLED+i] = v.z*(1/(mixed) VELOCITY_SCALE) + fz*scale;
        mixed rx = 0, ry = 0, rz = 0;
        if (kT > 0 && gamma > 0) {
            mixed sigma = sqrt(2*gamma*m*kT);
            float4 xi;
            if (drawNoise) {
                xi = random[randomIndex+i];
                noise[i] = xi;
            }
            else
                xi = noise[i];
            rx = xi.x*sigma;
            ry = xi.y*sigma;
            rz = xi.z*sigma;
        }
        randomForce[i] = rx;
        randomForce[NUM_COUPLED+i] = ry;
        randomForce[2*NUM_COUPLED+i] = rz;
#ifdef DOMAIN_DECOMPOSITION
        if (node == NUM_STORED)
            sortKeys[i] = otherRankKey(latticeNode, i);
        else
#endif
        sortKeys[i] = ((mm_long) node)*NUM_COUPLED + i;
    }
}

/**
 * Centred drag, second part, after sorting the keys: the first entry of each node solves the particles of that
 * node together, in particle order, with a = gamma h, M, P~ and R the sums of m_k, m_k v~_k and of the random
 * forces, and u~ = (j + h rho g)/rho, m_c = rho:
 *   S = [-gamma (P~ - M u~) + R]/(1 + a + a M/m_c),   u_c(t) = u~ - h S/m_c,
 *   F_k = [-gamma m_k (v~_k - u_c(t)) + R_k]/(1 + a).
 * A solid node is a wall at rest of infinite mass: u_c(t) = 0; the boundary nodes of regularized walls and open
 * faces are fluid nodes like the others.  The first entry writes the force of every particle of the segment (one writer per particle) and, in a lattice step
 * (isStep), the reaction -S of the node (one writer per node); the reaction on a solid node goes to the walls, in wallMomentum of the first particle of the segment.
 */
KERNEL void solveCenteredDrag(GLOBAL const mm_long* RESTRICT sortKeys, GLOBAL const mixed* RESTRICT particleMass,
        GLOBAL const mixed* RESTRICT knownVelocity, GLOBAL const mixed* RESTRICT randomForce,
        GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL const mixed* RESTRICT momentum, GLOBAL mixed* RESTRICT particleForce,
        GLOBAL mixed* RESTRICT cellReaction, GLOBAL mixed* RESTRICT wallMomentum, int isStep, mixed gamma, mixed gx, mixed gy, mixed gz,
        GLOBAL const int* RESTRICT isFluid) {
    const mixed h = 0.5f;
    const mixed a = gamma*h;
    for (int k = GLOBAL_ID; k < NUM_COUPLED; k += GLOBAL_SIZE) {
        int node = keyNode(sortKeys[k]);
        if (k > 0 && keyNode(sortKeys[k-1]) == node)
            continue;
#ifdef DOMAIN_DECOMPOSITION
        if (node == NUM_STORED)
            continue;
#endif
        int end = k;
        mixed mass = 0, px = 0, py = 0, pz = 0, rx = 0, ry = 0, rz = 0;
        for (; end < NUM_COUPLED && keyNode(sortKeys[end]) == node; end++) {
            int i = (int) (sortKeys[end] - ((mm_long) node)*NUM_COUPLED);
            mixed m = particleMass[i];
            mass += m;
            px += knownVelocity[i]*m;
            py += knownVelocity[NUM_COUPLED+i]*m;
            pz += knownVelocity[2*NUM_COUPLED+i]*m;
            rx += randomForce[i];
            ry += randomForce[NUM_COUPLED+i];
            rz += randomForce[2*NUM_COUPLED+i];
        }
        mixed rho = 1 + densityDeviation[node];
        mixed ux = 0, uy = 0, uz = 0, sx, sy, sz;
        if (rho > 0) {
            mixed kx = (momentum[node] + h*(rho*gx))*(1/rho);
            mixed ky = (momentum[NUM_STORED+node] + h*(rho*gy))*(1/rho);
            mixed kz = (momentum[2*NUM_STORED+node] + h*(rho*gz))*(1/rho);
            mixed scale = 1/(1 + a + a*mass/rho);
            sx = ((px - kx*mass)*(-gamma) + rx)*scale;
            sy = ((py - ky*mass)*(-gamma) + ry)*scale;
            sz = ((pz - kz*mass)*(-gamma) + rz)*scale;
            ux = kx - sx*(h/rho);
            uy = ky - sy*(h/rho);
            uz = kz - sz*(h/rho);
        }
        else {
            mixed scale = 1/(1 + a);
            sx = ((px - ux*mass)*(-gamma) + rx)*scale;
            sy = ((py - uy*mass)*(-gamma) + ry)*scale;
            sz = ((pz - uz*mass)*(-gamma) + rz)*scale;
        }
        mixed scale = 1/(1 + a);
        for (int m = k; m < end; m++) {
            int i = (int) (sortKeys[m] - ((mm_long) node)*NUM_COUPLED);
            mixed mi = particleMass[i];
            particleForce[i] = ((knownVelocity[i] - ux)*(-gamma*mi) + randomForce[i])*scale;
            particleForce[NUM_COUPLED+i] = ((knownVelocity[NUM_COUPLED+i] - uy)*(-gamma*mi) + randomForce[NUM_COUPLED+i])*scale;
            particleForce[2*NUM_COUPLED+i] = ((knownVelocity[2*NUM_COUPLED+i] - uz)*(-gamma*mi) + randomForce[2*NUM_COUPLED+i])*scale;
        }
        if (isStep) {
            cellReaction[node] = -sx;
            cellReaction[NUM_STORED+node] = -sy;
            cellReaction[2*NUM_STORED+node] = -sz;
#ifdef HAS_SOLID_NODES
            if (rho == 0) {
                int i = (int) (sortKeys[k] - ((mm_long) node)*NUM_COUPLED);
                wallMomentum[i] -= sx;
                wallMomentum[NUM_COUPLED+i] -= sy;
                wallMomentum[2*NUM_COUPLED+i] -= sz;
            }
#endif
        }
    }
}

/**
 * Add the coupling forces of the last lattice step to OpenMM's force buffer (fixed point), one thread per atom.
 */
KERNEL void applyCouplingForces(GLOBAL const int* RESTRICT atomIndex, GLOBAL const int* RESTRICT couplingIndex,
        GLOBAL const mixed* RESTRICT particleForce, GLOBAL mm_long* RESTRICT forceBuffers) {
    for (int j = GLOBAL_ID; j < NUM_ATOMS; j += GLOBAL_SIZE) {
        int i = couplingIndex[atomIndex[j]];
        if (i < 0)
            continue;
        forceBuffers[j] += (mm_long) (particleForce[i]*FORCE_SCALE*0x100000000);
        forceBuffers[j+PADDED_NUM_ATOMS] += (mm_long) (particleForce[NUM_COUPLED+i]*FORCE_SCALE*0x100000000);
        forceBuffers[j+2*PADDED_NUM_ATOMS] += (mm_long) (particleForce[2*NUM_COUPLED+i]*FORCE_SCALE*0x100000000);
    }
}
