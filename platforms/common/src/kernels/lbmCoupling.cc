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
 * NUM_KEYS is the number of sort keys and KEY_STRIDE the factor of the node in a key, both NUM_COUPLED for the nearest
 * node; with an interpolation stencil (docs/theory.md, section 9) STENCIL_WIDTH is its number of nodes along an axis,
 * STENCIL_SIZE = STENCIL_WIDTH^3, NUM_KEYS = NUM_COUPLED*STENCIL_SIZE the number of slots of the stencils, and every
 * particle has a sort key of its own, from its brick (BRICK_SIZE, NBX, NBY, NBZ, NUM_BRICKS, BRICK_SCRAMBLE; see the
 * segments of the stencils below).
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
 * keys of all the stencils (particleKey()) and adds the reactions on the nodes of its block.
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
 * The segments of the stencils, one per node that a stencil reaches: the reactions of a node, and the vector of the
 * conjugate gradients spread on it, are sums over the slots e = i*STENCIL_SIZE + n of the stencils on that node, in
 * particle order and, for a particle, in slot order.  segmentStart[s] is the first entry of segment s, segmentNode[s] its
 * node (index in the lattice), and sortedWeight[k] and sortedParticle[k] the weight and particle of entry k, so that the
 * sums read their entries contiguously; with the centred drag keySegment[e] is the segment of slot e, where the
 * conjugate gradients keep the vector spread on its node.  chunkOffset[c] is the first segment that starts in chunk c of
 * LBM_BLOCK_SIZE entries, and chunkOffset[NUM_KEY_CHUNKS] the number of segments.
 *
 * The segments are built from one sort key per particle.  Along each axis the cell of a particle is the first node of
 * its stencil, wrapped into [0, size) or, along an axis with open faces, moved into [0, size - 1]; the cells are grouped
 * into bricks of BRICK_SIZE^3 nodes, the last brick along an axis taking the rest of the axis (NBX, NBY, NBZ bricks), so
 * that every brick has at least BRICK_SIZE >= STENCIL_WIDTH nodes, or the whole axis.  The particles whose stencils reach
 * a node then have their cells in at most two bricks along each axis.  A brick per node would need an array as large
 * as the lattice on every rank.  The key of particle i is pi(b)*NUM_COUPLED + i, b its brick and pi(b) = b*BRICK_SCRAMBLE
 * mod NUM_BRICKS a permutation that spreads the keys uniformly over their range wherever the particles are, so that the
 * buckets of equal width of OpenMM's sort stay small (CouplingSortTrait in CommonLBMKernels.cpp).  After the sort:
 * findBrickStarts() finds where each brick starts in the sorted keys, and listCandidates() writes the particle and the
 * first nodes of the stencil of each sorted key; countBrickSegments() counts, for every brick of nodes (one work group
 * each), the nodes that the stencils reach and their slots, testing the particles of the at most 8 bricks of cells that
 * can reach it; scanBrickChunks() and scanBrickTotals() turn the counts into offsets; listBrickSegments() writes the
 * segments of every brick, merging the particles of its bricks of cells in particle order; findChunkSegments() writes
 * chunkOffset.  Every write has one writer, and the results are the same as those of sorting a key per slot.
 */
#define NUM_KEY_CHUNKS ((NUM_KEYS+LBM_BLOCK_SIZE-1)/LBM_BLOCK_SIZE)
#define NUM_BRICK_CHUNKS ((NUM_BRICKS+LBM_BLOCK_SIZE-1)/LBM_BLOCK_SIZE)
#define MAX_SOURCE_BRICKS 8

/**
 * The node of the lattice of slot a of a stencil whose first node along an axis is first, as stencilAxis().
 */
DEVICE int stencilIndex(int first, int a, int size, int open) {
    return (open ? min(max(first + a, 0), size - 1) : ((first + a)%size + size)%size);
}

/**
 * The brick along an axis of a cell c.
 */
DEVICE int brickOfCell(int c, int numBricks) {
    return min(c/BRICK_SIZE, numBricks - 1);
}

DEVICE int cellOfStencil(int first, int size, int open) {
    return (open ? min(max(first, 0), size - 1) : (first%size + size)%size);
}

/**
 * The position of brick b in the sorted keys, pi(b).
 */
DEVICE int sortedBrick(int b) {
    return (int) ((((mm_long) b)*BRICK_SCRAMBLE)%NUM_BRICKS);
}

/**
 * The sort key of particle i whose stencil starts at the nodes first.x, first.y, first.z of the lattice.
 */
DEVICE mm_long particleKey(int4 first, int i) {
    int b = brickOfCell(cellOfStencil(first.x, GNX, OPEN_X), NBX) + NBX*(brickOfCell(cellOfStencil(first.y, GNY, OPEN_Y), NBY) +
            NBY*brickOfCell(cellOfStencil(first.z, GNZ, OPEN_Z), NBZ));
    return ((mm_long) sortedBrick(b))*NUM_COUPLED + i;
}

/**
 * The node of the arrays of the block that receives the reaction of the node of a segment, or NUM_STORED for the nodes
 * outside the block (only with the decomposition).
 */
DEVICE int segmentReactionNode(int node) {
#ifdef DOMAIN_DECOMPOSITION
    return storedNode(node);
#else
    return node;
#endif
}

/**
 * The nodes [lo, hi) of brick b along an axis, and the bricks of the cells of the stencils that can reach them: brick b
 * and the brick of cell lo - (STENCIL_WIDTH - 1), wrapped or, along an axis with open faces, moved to 0.  Returns their
 * number, 1 or 2.
 */
DEVICE int sourceBricks(int b, int size, int numBricks, int open, int* lo, int* hi, int* source) {
    *lo = b*BRICK_SIZE;
    *hi = (b == numBricks - 1 ? size : *lo + BRICK_SIZE);
    int c = *lo - (STENCIL_WIDTH - 1);
    int previous = brickOfCell(open ? max(c, 0) : (c%size + size)%size, numBricks);
    source[0] = b;
    if (previous == b)
        return 1;
    source[1] = previous;
    return 2;
}

/**
 * The ranges in the sorted keys of the bricks of cells that can reach brick tb of nodes, and the nodes of tb.  Returns
 * the number of ranges.
 */
DEVICE int brickSources(int tb, GLOBAL const int* RESTRICT brickStart, int* lo, int* hi, int* first, int* end) {
    int bx = tb%NBX, by = (tb/NBX)%NBY, bz = tb/(NBX*NBY);
    int sx[2], sy[2], sz[2];
    int nx = sourceBricks(bx, GNX, NBX, OPEN_X, &lo[0], &hi[0], sx);
    int ny = sourceBricks(by, GNY, NBY, OPEN_Y, &lo[1], &hi[1], sy);
    int nz = sourceBricks(bz, GNZ, NBZ, OPEN_Z, &lo[2], &hi[2], sz);
    int n = 0;
    for (int c = 0; c < nz; c++)
        for (int b = 0; b < ny; b++)
            for (int a = 0; a < nx; a++) {
                int s = sortedBrick(sx[a] + NBX*(sy[b] + NBY*sz[c]));
                first[n] = brickStart[s];
                end[n] = brickStart[s+1];
                n++;
            }
    return n;
}

/**
 * brickStart[s], for s from 0 to NUM_BRICKS, is the first sorted key of the brick at position s of the sorted keys.
 */
KERNEL void findBrickStarts(GLOBAL const mm_long* RESTRICT sortKeys, GLOBAL int* RESTRICT brickStart) {
    for (int s = GLOBAL_ID; s <= NUM_BRICKS; s += GLOBAL_SIZE) {
        mm_long key = ((mm_long) s)*NUM_COUPLED;
        int lo = 0, hi = NUM_COUPLED;
        while (lo < hi) {
            int mid = (lo + hi)/2;
            if (sortKeys[mid] < key)
                lo = mid + 1;
            else
                hi = mid;
        }
        brickStart[s] = lo;
    }
}

/**
 * The particle and the first nodes of the stencil of each sorted key, read contiguously by the bricks.
 */
KERNEL void listCandidates(GLOBAL const mm_long* RESTRICT sortKeys, GLOBAL const int4* RESTRICT stencilOrigin,
        GLOBAL int* RESTRICT candidateParticle, GLOBAL int4* RESTRICT candidateFirst) {
    for (int k = GLOBAL_ID; k < NUM_COUPLED; k += GLOBAL_SIZE) {
        mm_long key = sortKeys[k];
        int i = (int) (key - (key/NUM_COUPLED)*NUM_COUPLED);
        candidateParticle[k] = i;
        candidateFirst[k] = stencilOrigin[i];
    }
}

/**
 * The slots of a stencil whose first node along an axis is first that fall on each node lo + j of a brick of width nodes
 * along that axis, packed 3 bits per node: (packed >> 3j) & 7 (width < 2*BRICK_SIZE <= 8 nodes, at most STENCIL_WIDTH
 * slots per node).  The product over the three axes of these counts at the coordinates of a node in the brick is the
 * number of slots that fall on it, so that testing a candidate against a node costs a few shifts.
 */
DEVICE int axisCoverage(int first, int lo, int width, int size, int open) {
    int packed = 0;
    for (int a = 0; a < STENCIL_WIDTH; a++) {
        int j = stencilIndex(first, a, size, open) - lo;
        if (j >= 0 && j < width)
            packed += 1 << (3*j);
    }
    return packed;
}

DEVICE int brickSlotsAt(int px, int py, int pz, int lx, int ly, int lz) {
    return ((px >> (3*lx)) & 7)*((py >> (3*ly)) & 7)*((pz >> (3*lz)) & 7);
}

/**
 * For every brick of nodes, one thread: its number of candidates (the particles of its bricks of cells) and whether it
 * has any, in brickCandidates[tb] and brickActive[tb]; its segments and slots are set to zero, and countBrickSegments()
 * counts them for the bricks with candidates only, from the list of those bricks (writeActiveBricks()), so that the
 * kernels of the bricks do not visit the empty bricks of a large lattice.
 */
KERNEL void flagBricks(GLOBAL const int* RESTRICT brickStart, GLOBAL int* RESTRICT brickActive,
        GLOBAL int* RESTRICT brickCandidates, GLOBAL int* RESTRICT brickSegments, GLOBAL int* RESTRICT brickSlots) {
    for (int tb = GLOBAL_ID; tb < NUM_BRICKS; tb += GLOBAL_SIZE) {
        int lo[3], hi[3], first[MAX_SOURCE_BRICKS], end[MAX_SOURCE_BRICKS];
        int n = brickSources(tb, brickStart, lo, hi, first, end);
        int candidates = 0;
        for (int r = 0; r < n; r++)
            candidates += end[r] - first[r];
        brickCandidates[tb] = candidates;
        brickActive[tb] = (candidates > 0 ? 1 : 0);
        brickSegments[tb] = 0;
        brickSlots[tb] = 0;
    }
}

/**
 * After the scan of brickActive (its offsets, in place, and those of the chunks): the list of the bricks of nodes with
 * candidates, in increasing order.
 */
KERNEL void writeActiveBricks(GLOBAL const int* RESTRICT brickActive, GLOBAL const int* RESTRICT chunkActive,
        GLOBAL int* RESTRICT activeBrick) {
    for (int tb = GLOBAL_ID; tb < NUM_BRICKS; tb += GLOBAL_SIZE) {
        int next = (tb+1 < NUM_BRICKS ? chunkActive[(tb+1)/LBM_BLOCK_SIZE] + brickActive[tb+1] : chunkActive[NUM_BRICK_CHUNKS]);
        int offset = chunkActive[tb/LBM_BLOCK_SIZE] + brickActive[tb];
        if (next > offset)
            activeBrick[offset] = tb;
    }
}

/**
 * The kernels of the bricks of nodes run work groups of BRICKS_PER_GROUP*LBM_BLOCK_SIZE threads, BRICKS_PER_GROUP
 * bricks at a time, LBM_BLOCK_SIZE threads per brick: OpenMM launches a limited number of work groups, and larger groups
 * keep more threads on the device to hide the latency of the reads.  For its brick, the first thread of each part of
 * the group finds the ranges of the candidates in the sorted keys (brickSources()) and writes them, with the nodes of
 * the brick, to local memory; the part then reads the candidates CANDIDATE_CHUNK at a time into local memory, and every
 * thread tests them against its node.  The loops run up to the largest brick and number of candidates of the group, so
 * that all its threads meet the barriers.
 */
#define CANDIDATE_CHUNK (4*LBM_BLOCK_SIZE)

/**
 * For every brick of nodes with candidates (activeBrick): the number of its nodes that the stencils reach (segments) and
 * of the slots that fall on them, in brickSegments[tb] and brickSlots[tb].
 */
KERNEL void countBrickSegments(GLOBAL const int* RESTRICT brickStart, GLOBAL const int4* RESTRICT candidateFirst,
        GLOBAL int* RESTRICT brickSegments, GLOBAL int* RESTRICT brickSlots, GLOBAL const int* RESTRICT activeBrick,
        GLOBAL const int* RESTRICT chunkActive) {
    LOCAL int sums[2*BRICKS_PER_GROUP*LBM_BLOCK_SIZE];
    LOCAL int rangeFirst[BRICKS_PER_GROUP*MAX_SOURCE_BRICKS], rangeOffset[BRICKS_PER_GROUP*(MAX_SOURCE_BRICKS+1)];
    LOCAL int brickNodes[8*BRICKS_PER_GROUP];
    LOCAL int chunkX[BRICKS_PER_GROUP*CANDIDATE_CHUNK], chunkY[BRICKS_PER_GROUP*CANDIDATE_CHUNK];
    LOCAL int chunkZ[BRICKS_PER_GROUP*CANDIDATE_CHUNK];
    const int q = LOCAL_ID/LBM_BLOCK_SIZE, lane = LOCAL_ID%LBM_BLOCK_SIZE;
    const int R = q*MAX_SOURCE_BRICKS, O = q*(MAX_SOURCE_BRICKS+1), B = 8*q, C = q*CANDIDATE_CHUNK, S = 2*q*LBM_BLOCK_SIZE;
    const int numActive = chunkActive[NUM_BRICK_CHUNKS];
    for (int firstBrick = GROUP_ID*BRICKS_PER_GROUP; firstBrick < numActive; firstBrick += NUM_GROUPS*BRICKS_PER_GROUP) {
        const int tb = (firstBrick + q < numActive ? activeBrick[firstBrick + q] : NUM_BRICKS);
        if (lane == 0) {
            // The ranges of the candidates of brick tb (none beyond the last active brick) and its nodes.
            int lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1}, first[MAX_SOURCE_BRICKS], end[MAX_SOURCE_BRICKS];
            int n = (tb < NUM_BRICKS ? brickSources(tb, brickStart, lo, hi, first, end) : 0);
            rangeOffset[O] = 0;
            for (int r = 0; r < MAX_SOURCE_BRICKS; r++) {
                rangeFirst[R+r] = (r < n ? first[r] : 0);
                rangeOffset[O+r+1] = rangeOffset[O+r] + (r < n ? end[r] - first[r] : 0);
            }
            for (int a = 0; a < 3; a++) {
                brickNodes[B+a] = lo[a];
                brickNodes[B+3+a] = hi[a];
            }
            brickNodes[B+6] = n;
            brickNodes[B+7] = (rangeOffset[O+n] > 0 ? (hi[0]-lo[0])*(hi[1]-lo[1])*(hi[2]-lo[2]) : 0);
        }
        SYNC_THREADS;
        const int numRanges = brickNodes[B+6], numCandidates = rangeOffset[O+numRanges], numNodes = brickNodes[B+7];
        const int wx = brickNodes[B+3]-brickNodes[B], wy = brickNodes[B+4]-brickNodes[B+1];
        int maxNodes = 0, maxCandidates = 0;
        for (int b = 0; b < BRICKS_PER_GROUP; b++) {
            maxNodes = max(maxNodes, brickNodes[8*b+7]);
            maxCandidates = max(maxCandidates, rangeOffset[(MAX_SOURCE_BRICKS+1)*b + brickNodes[8*b+6]]);
        }
        const int wz = brickNodes[B+5]-brickNodes[B+2];
        int segments = 0, slots = 0, loaded = -1;
        for (int base = 0; base < maxNodes; base += LBM_BLOCK_SIZE) {
            int l = base + lane;
            int lx = l%max(wx, 1), ly = (l/max(wx, 1))%max(wy, 1), lz = l/max(wx*wy, 1);
            int count = 0;
            for (int cb = 0; cb < maxCandidates; cb += CANDIDATE_CHUNK) {
                int n = min(numCandidates-cb, CANDIDATE_CHUNK);
                if (loaded != cb) {
                    SYNC_THREADS;
                    for (int c = lane; c < n; c += LBM_BLOCK_SIZE) {
                        int r = 0;
                        while (cb+c >= rangeOffset[O+r+1])
                            r++;
                        int4 f = candidateFirst[rangeFirst[R+r] + cb + c - rangeOffset[O+r]];
                        chunkX[C+c] = axisCoverage(f.x, brickNodes[B], wx, GNX, OPEN_X);
                        chunkY[C+c] = axisCoverage(f.y, brickNodes[B+1], wy, GNY, OPEN_Y);
                        chunkZ[C+c] = axisCoverage(f.z, brickNodes[B+2], wz, GNZ, OPEN_Z);
                    }
                    SYNC_THREADS;
                    loaded = cb;
                }
                if (l < numNodes)
                    for (int c = 0; c < n; c++)
                        count += brickSlotsAt(chunkX[C+c], chunkY[C+c], chunkZ[C+c], lx, ly, lz);
            }
            segments += (count > 0 ? 1 : 0);
            slots += count;
        }
        sums[S+lane] = segments;
        sums[S+LBM_BLOCK_SIZE+lane] = slots;
        for (int step = LBM_BLOCK_SIZE/2; step > 0; step /= 2) {
            SYNC_THREADS;
            if (lane < step) {
                sums[S+lane] += sums[S+lane+step];
                sums[S+LBM_BLOCK_SIZE+lane] += sums[S+LBM_BLOCK_SIZE+lane+step];
            }
        }
        SYNC_THREADS;
        if (lane == 0 && tb < NUM_BRICKS) {
            brickSegments[tb] = sums[S];
            brickSlots[tb] = sums[S+LBM_BLOCK_SIZE];
        }
        SYNC_THREADS;
    }
}

/**
 * For each chunk of LBM_BLOCK_SIZE bricks, one work group: two counts per brick become their offsets within the chunk,
 * in place, and the totals of the chunk go to chunkFirst[c] and chunkSecond[c].  Run on the active flags and the
 * candidates of the bricks, and then on their segments and slots.
 */
KERNEL void scanBrickChunks(GLOBAL int* RESTRICT brickFirst, GLOBAL int* RESTRICT brickSecond,
        GLOBAL int* RESTRICT chunkFirst, GLOBAL int* RESTRICT chunkSecond) {
    LOCAL int sums[2*LBM_BLOCK_SIZE];
    for (int c = GROUP_ID; c < NUM_BRICK_CHUNKS; c += NUM_GROUPS) {
        int tb = c*LBM_BLOCK_SIZE + LOCAL_ID;
        int first = (tb < NUM_BRICKS ? brickFirst[tb] : 0), second = (tb < NUM_BRICKS ? brickSecond[tb] : 0);
        sums[LOCAL_ID] = first;
        sums[LBM_BLOCK_SIZE+LOCAL_ID] = second;
        for (int step = 1; step < LBM_BLOCK_SIZE; step *= 2) {
            SYNC_THREADS;
            int addFirst = (LOCAL_ID >= step ? sums[LOCAL_ID-step] : 0);
            int addSecond = (LOCAL_ID >= step ? sums[LBM_BLOCK_SIZE+LOCAL_ID-step] : 0);
            SYNC_THREADS;
            sums[LOCAL_ID] += addFirst;
            sums[LBM_BLOCK_SIZE+LOCAL_ID] += addSecond;
        }
        if (tb < NUM_BRICKS) {
            brickFirst[tb] = sums[LOCAL_ID] - first;
            brickSecond[tb] = sums[LBM_BLOCK_SIZE+LOCAL_ID] - second;
        }
        if (LOCAL_ID == LBM_BLOCK_SIZE-1) {
            chunkFirst[c] = sums[LOCAL_ID];
            chunkSecond[c] = sums[LBM_BLOCK_SIZE+LOCAL_ID];
        }
        SYNC_THREADS;
    }
}

/**
 * One work group: the totals of the chunks of bricks become their offsets, in place, followed by the totals in
 * chunkFirst[NUM_BRICK_CHUNKS] and chunkSecond[NUM_BRICK_CHUNKS]; with segments nonzero (the scan of the segments and
 * slots), segmentStart of the end of the last segment.
 */
KERNEL void scanBrickTotals(GLOBAL int* RESTRICT chunkFirst, GLOBAL int* RESTRICT chunkSecond,
        GLOBAL int* RESTRICT segmentStart, int segments) {
    LOCAL int sums[2*LBM_BLOCK_SIZE];
    const int chunksPerThread = (NUM_BRICK_CHUNKS+LBM_BLOCK_SIZE-1)/LBM_BLOCK_SIZE;
    int firstChunk = LOCAL_ID*chunksPerThread, lastChunk = min(firstChunk+chunksPerThread, NUM_BRICK_CHUNKS);
    int totalFirst = 0, totalSecond = 0;
    for (int c = firstChunk; c < lastChunk; c++) {
        totalFirst += chunkFirst[c];
        totalSecond += chunkSecond[c];
    }
    sums[LOCAL_ID] = totalFirst;
    sums[LBM_BLOCK_SIZE+LOCAL_ID] = totalSecond;
    for (int step = 1; step < LBM_BLOCK_SIZE; step *= 2) {
        SYNC_THREADS;
        int addFirst = (LOCAL_ID >= step ? sums[LOCAL_ID-step] : 0);
        int addSecond = (LOCAL_ID >= step ? sums[LBM_BLOCK_SIZE+LOCAL_ID-step] : 0);
        SYNC_THREADS;
        sums[LOCAL_ID] += addFirst;
        sums[LBM_BLOCK_SIZE+LOCAL_ID] += addSecond;
    }
    int offsetFirst = sums[LOCAL_ID] - totalFirst, offsetSecond = sums[LBM_BLOCK_SIZE+LOCAL_ID] - totalSecond;
    for (int c = firstChunk; c < lastChunk; c++) {
        int countFirst = chunkFirst[c], countSecond = chunkSecond[c];
        chunkFirst[c] = offsetFirst;
        chunkSecond[c] = offsetSecond;
        offsetFirst += countFirst;
        offsetSecond += countSecond;
    }
    if (LOCAL_ID == LBM_BLOCK_SIZE-1) {
        chunkFirst[NUM_BRICK_CHUNKS] = sums[LOCAL_ID];
        chunkSecond[NUM_BRICK_CHUNKS] = sums[LBM_BLOCK_SIZE+LOCAL_ID];
        if (segments)
            segmentStart[sums[LOCAL_ID]] = NUM_KEYS;
    }
}

#define LIST_CHUNK (4*CANDIDATE_CHUNK)

/**
 * The candidates of a brick with merged positions [first, first + n) into local memory, by all the threads of the work
 * group: particle, position in the sorted keys and coverage of the brick along the three axes.  lo and width are the
 * first node and the width of the brick along each axis.
 */
DEVICE void loadMergedCandidates(int first, int n, int numCandidates, int numRanges, LOCAL_ARG const int* rangeFirst,
        LOCAL_ARG const int* rangeOffset, const int* lo, const int* width, GLOBAL const int* RESTRICT candidateParticle,
        GLOBAL const int4* RESTRICT candidateFirst, LOCAL_ARG int* chunkParticle, LOCAL_ARG int* chunkSource,
        LOCAL_ARG int* chunkX, LOCAL_ARG int* chunkY, LOCAL_ARG int* chunkZ) {
    for (int c = LOCAL_ID; c < numCandidates; c += LOCAL_SIZE) {
        int r = 0;
        while (c >= rangeOffset[r+1])
            r++;
        int p = rangeFirst[r] + c - rangeOffset[r];
        int i = candidateParticle[p];
        int position = c - rangeOffset[r];
        for (int t = 0; t < numRanges; t++) {
            if (t == r)
                continue;
            int low = rangeFirst[t], high = rangeFirst[t] + rangeOffset[t+1] - rangeOffset[t];
            while (low < high) {
                int mid = (low + high)/2;
                if (candidateParticle[mid] < i)
                    low = mid + 1;
                else
                    high = mid;
            }
            position += low - rangeFirst[t];
        }
        if (position >= first && position < first+n) {
            int4 f = candidateFirst[p];
            chunkParticle[position-first] = i;
            chunkSource[position-first] = p;
            chunkX[position-first] = axisCoverage(f.x, lo[0], width[0], GNX, OPEN_X);
            chunkY[position-first] = axisCoverage(f.y, lo[1], width[1], GNY, OPEN_Y);
            chunkZ[position-first] = axisCoverage(f.z, lo[2], width[2], GNZ, OPEN_Z);
        }
    }
}

/**
 * For every brick of nodes with candidates (activeBrick), one work group of BRICKS_PER_GROUP*LBM_BLOCK_SIZE threads: the
 * segments of the nodes that the stencils reach, in the order of the nodes in the brick (x fastest), and the merged list
 * of its candidates.  The
 * candidates are read into local memory, LIST_CHUNK at a time, merged in particle order: the position of a candidate is
 * its position in its range plus, for each other range, the number of candidates with a smaller particle index (a binary
 * search; the ranges are sorted by particle).  The group copies them, with their coverage of the brick, to
 * mergedCoverage (the coverage along the three axes, then the particle) and mergedSource (the position in the sorted
 * keys), from the offset of the candidates of the brick.  Each node has BRICKS_PER_GROUP threads, which count the slots
 * of contiguous blocks of the candidates; their sum and a scan over the nodes give the segment and first entry of the
 * node.  writeSegmentEntries() then writes the entries, one thread per segment, so that the work goes to the nodes that
 * the stencils reach.
 */
KERNEL void listBrickSegments(GLOBAL const int* RESTRICT brickStart, GLOBAL const int* RESTRICT candidateParticle,
        GLOBAL const int4* RESTRICT candidateFirst, GLOBAL const int* RESTRICT brickSegments,
        GLOBAL const int* RESTRICT brickSlots, GLOBAL const int* RESTRICT brickCandidates,
        GLOBAL const int* RESTRICT chunkSegments, GLOBAL const int* RESTRICT chunkSlots,
        GLOBAL const int* RESTRICT chunkCandidates, GLOBAL int* RESTRICT segmentStart, GLOBAL int* RESTRICT segmentNode,
        GLOBAL int4* RESTRICT mergedCoverage, GLOBAL int* RESTRICT mergedSource, GLOBAL const int* RESTRICT activeBrick,
        GLOBAL const int* RESTRICT chunkActive) {
    LOCAL int sums[2*LBM_BLOCK_SIZE];
    LOCAL int blockCount[BRICKS_PER_GROUP*LBM_BLOCK_SIZE];
    LOCAL int rangeFirst[MAX_SOURCE_BRICKS];
    LOCAL int rangeOffset[MAX_SOURCE_BRICKS+1];
    LOCAL int brickNodes[8];
    LOCAL int chunkParticle[LIST_CHUNK];
    LOCAL int chunkSource[LIST_CHUNK];
    LOCAL int chunkX[LIST_CHUNK];
    LOCAL int chunkY[LIST_CHUNK];
    LOCAL int chunkZ[LIST_CHUNK];
    const int part = LOCAL_ID/LBM_BLOCK_SIZE, lane = LOCAL_ID%LBM_BLOCK_SIZE;
    const int numActive = chunkActive[NUM_BRICK_CHUNKS];
    for (int a = GROUP_ID; a < numActive; a += NUM_GROUPS) {
        const int tb = activeBrick[a];
        if (LOCAL_ID == 0) {
            int lo[3], hi[3], first[MAX_SOURCE_BRICKS], end[MAX_SOURCE_BRICKS];
            int n = brickSources(tb, brickStart, lo, hi, first, end);
            rangeOffset[0] = 0;
            for (int r = 0; r < MAX_SOURCE_BRICKS; r++) {
                rangeFirst[r] = (r < n ? first[r] : 0);
                rangeOffset[r+1] = rangeOffset[r] + (r < n ? end[r] - first[r] : 0);
            }
            for (int a = 0; a < 3; a++) {
                brickNodes[a] = lo[a];
                brickNodes[3+a] = hi[a];
            }
            brickNodes[6] = n;
            brickNodes[7] = (rangeOffset[n] > 0 ? (hi[0]-lo[0])*(hi[1]-lo[1])*(hi[2]-lo[2]) : 0);
        }
        SYNC_THREADS;
        const int numRanges = brickNodes[6], numCandidates = rangeOffset[numRanges], numNodes = brickNodes[7];
        int lo[3], width[3];
        for (int a = 0; a < 3; a++) {
            lo[a] = brickNodes[a];
            width[a] = brickNodes[3+a]-brickNodes[a];
        }
        const int c0 = tb/LBM_BLOCK_SIZE;
        int segmentBase = chunkSegments[c0] + brickSegments[tb];
        int slotBase = chunkSlots[c0] + brickSlots[tb];
        const int candidateBase = chunkCandidates[c0] + brickCandidates[tb];

        // The merged candidates, with their coverage of the brick, to global memory.
        int loaded = -1;
        for (int cb = 0; cb < numCandidates; cb += LIST_CHUNK) {
            int n = min(numCandidates-cb, LIST_CHUNK);
            SYNC_THREADS;
            loadMergedCandidates(cb, n, numCandidates, numRanges, rangeFirst, rangeOffset, lo, width, candidateParticle,
                    candidateFirst, chunkParticle, chunkSource, chunkX, chunkY, chunkZ);
            SYNC_THREADS;
            loaded = cb;
            for (int c = LOCAL_ID; c < n; c += LOCAL_SIZE) {
                int4 coverage;
                coverage.x = chunkX[c];
                coverage.y = chunkY[c];
                coverage.z = chunkZ[c];
                coverage.w = chunkParticle[c];
                mergedCoverage[candidateBase+cb+c] = coverage;
                mergedSource[candidateBase+cb+c] = chunkSource[c];
            }
        }

        // The segments: the slots that fall on each node, counted by blocks of candidates, and a scan over the nodes.
        for (int base = 0; base < numNodes; base += LBM_BLOCK_SIZE) {
            int l = base + lane;
            int lx = l%width[0], ly = (l/width[0])%width[1], lz = l/(width[0]*width[1]);
            bool isNode = (l < numNodes);
            int count = 0;
            for (int cb = 0; cb < numCandidates; cb += LIST_CHUNK) {
                int n = min(numCandidates-cb, LIST_CHUNK);
                if (loaded != cb) {
                    SYNC_THREADS;
                    loadMergedCandidates(cb, n, numCandidates, numRanges, rangeFirst, rangeOffset, lo, width,
                            candidateParticle, candidateFirst, chunkParticle, chunkSource, chunkX, chunkY, chunkZ);
                    SYNC_THREADS;
                    loaded = cb;
                }
                int blockFirst = (part*n)/BRICKS_PER_GROUP, blockEnd = ((part+1)*n)/BRICKS_PER_GROUP;
                if (isNode)
                    for (int c = blockFirst; c < blockEnd; c++)
                        count += brickSlotsAt(chunkX[c], chunkY[c], chunkZ[c], lx, ly, lz);
            }
            blockCount[LOCAL_ID] = count;
            SYNC_THREADS;
            count = 0;
            for (int b = 0; b < BRICKS_PER_GROUP; b++)
                count += blockCount[b*LBM_BLOCK_SIZE+lane];
            if (part == 0) {
                sums[lane] = (count > 0 ? 1 : 0);
                sums[LBM_BLOCK_SIZE+lane] = count;
            }
            for (int step = 1; step < LBM_BLOCK_SIZE; step *= 2) {
                SYNC_THREADS;
                int addSegments = 0, addSlots = 0;
                if (part == 0 && lane >= step) {
                    addSegments = sums[lane-step];
                    addSlots = sums[LBM_BLOCK_SIZE+lane-step];
                }
                SYNC_THREADS;
                if (part == 0) {
                    sums[lane] += addSegments;
                    sums[LBM_BLOCK_SIZE+lane] += addSlots;
                }
            }
            SYNC_THREADS;
            if (part == 0 && count > 0) {
                int s = segmentBase + sums[lane] - 1;
                segmentStart[s] = slotBase + sums[LBM_BLOCK_SIZE+lane] - count;
                segmentNode[s] = (lo[0] + lx) + GNX*((lo[1] + ly) + GNY*(lo[2] + lz));
            }
            segmentBase += sums[LBM_BLOCK_SIZE-1];
            slotBase += sums[2*LBM_BLOCK_SIZE-1];
            SYNC_THREADS;
        }
        SYNC_THREADS;
    }
}

/**
 * One thread per segment: the entries of its node, from the merged candidates of the brick of the node (written by
 * listBrickSegments()), for each candidate in particle order and each of its slots that falls on the node in slot order:
 * the weight and particle of the entry and, with the centred drag, the segment of the slot.
 */
KERNEL void writeSegmentEntries(GLOBAL const int* RESTRICT segmentStart, GLOBAL const int* RESTRICT segmentNode,
        GLOBAL const int* RESTRICT chunkSegments, GLOBAL const int* RESTRICT brickCandidates,
        GLOBAL const int* RESTRICT chunkCandidates, GLOBAL const int4* RESTRICT mergedCoverage,
        GLOBAL const int* RESTRICT mergedSource, GLOBAL const int4* RESTRICT candidateFirst,
        GLOBAL const mixed* RESTRICT stencilWeight, GLOBAL mixed* RESTRICT sortedWeight, GLOBAL int* RESTRICT sortedParticle
#ifdef CENTERED_DRAG
        , GLOBAL int* RESTRICT keySegment
#endif
        ) {
    const int numSegments = chunkSegments[NUM_BRICK_CHUNKS];
    for (int s = GLOBAL_ID; s < numSegments; s += GLOBAL_SIZE) {
        int node = segmentNode[s];
        int jx = node%GNX, jy = (node/GNX)%GNY, jz = node/(GNX*GNY);
        int bx = brickOfCell(jx, NBX), by = brickOfCell(jy, NBY), bz = brickOfCell(jz, NBZ);
        int tb = bx + NBX*(by + NBY*bz);
        int lx = jx - bx*BRICK_SIZE, ly = jy - by*BRICK_SIZE, lz = jz - bz*BRICK_SIZE;
        int first = chunkCandidates[tb/LBM_BLOCK_SIZE] + brickCandidates[tb];
        int end = (tb+1 < NUM_BRICKS ? chunkCandidates[(tb+1)/LBM_BLOCK_SIZE] + brickCandidates[tb+1] :
                chunkCandidates[NUM_BRICK_CHUNKS]);
        int k = segmentStart[s];
        for (int m = first; m < end; m++) {
            int4 coverage = mergedCoverage[m];
            if (brickSlotsAt(coverage.x, coverage.y, coverage.z, lx, ly, lz) == 0)
                continue;
            int i = coverage.w;
            int4 f = candidateFirst[mergedSource[m]];
            for (int cz = 0; cz < STENCIL_WIDTH; cz++) {
                if (stencilIndex(f.z, cz, GNZ, OPEN_Z) != jz)
                    continue;
                for (int cy = 0; cy < STENCIL_WIDTH; cy++) {
                    if (stencilIndex(f.y, cy, GNY, OPEN_Y) != jy)
                        continue;
                    for (int cx = 0; cx < STENCIL_WIDTH; cx++) {
                        if (stencilIndex(f.x, cx, GNX, OPEN_X) != jx)
                            continue;
                        int e = i*STENCIL_SIZE + cx + STENCIL_WIDTH*(cy + STENCIL_WIDTH*cz);
                        sortedWeight[k] = stencilWeight[e];
                        sortedParticle[k] = i;
#ifdef CENTERED_DRAG
                        keySegment[e] = s;
#endif
                        k++;
                    }
                }
            }
        }
    }
}

/**
 * chunkOffset[c], for c from 0 to NUM_KEY_CHUNKS: the first segment that starts at entry c*LBM_BLOCK_SIZE or later.
 */
KERNEL void findChunkSegments(GLOBAL const int* RESTRICT segmentStart, GLOBAL const int* RESTRICT chunkSegments,
        GLOBAL int* RESTRICT chunkOffset) {
    const int numSegments = chunkSegments[NUM_BRICK_CHUNKS];
    for (int c = GLOBAL_ID; c <= NUM_KEY_CHUNKS; c += GLOBAL_SIZE) {
        int entry = c*LBM_BLOCK_SIZE;
        int lo = 0, hi = numSegments;
        while (lo < hi) {
            int mid = (lo + hi)/2;
            if (segmentStart[mid] < entry)
                lo = mid + 1;
            else
                hi = mid;
        }
        chunkOffset[c] = lo;
    }
}

/**
 * Explicit drag with an interpolation stencil (docs/theory.md, section 9), as on the Reference platform: the
 * particle sees u = sum_j xi_j j_j/rho_j over the STENCIL_SIZE nodes of its stencil, x fastest, a solid node being a wall
 * at rest (rho = 0), and F = -gamma m (v - u) + sqrt(2 gamma m kT) xi.  The weight of slot n of particle i is stored in
 * stencilWeight[i*STENCIL_SIZE + n], the first nodes of the stencil along the three axes in stencilOrigin[i] and the
 * sort key of the particle in sortKeys[i] (particleKey()), from which the segments of the nodes are built to sum the
 * reactions -xi_j F of each node in particle order and, for a particle, in slot order.  In a lattice step the reaction
 * on the solid nodes of the stencil is added to wallMomentum.  The arguments are those of coupleParticles(), and then
 * stencilWeight and stencilOrigin.  With the domain decomposition the rank that owns the nearest node computes the
 * force, from the moments of the block and of its coupling halo; the other ranks set it to zero, and every rank writes
 * the weights, first nodes and keys of all the stencils, for the reactions on its own nodes once the forces are summed.
 */
KERNEL void coupleParticlesStencil(GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT posqCorrection,
        GLOBAL const mixed4* RESTRICT velm, GLOBAL const int* RESTRICT atomIndex, GLOBAL const int* RESTRICT couplingIndex,
        GLOBAL const mixed* RESTRICT particleMass, GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL const mixed* RESTRICT momentum,
        GLOBAL mixed* RESTRICT particleForce, GLOBAL mm_long* RESTRICT sortKeys, GLOBAL mixed* RESTRICT wallMomentum,
        GLOBAL const float4* RESTRICT random, GLOBAL float4* RESTRICT noise, int randomIndex, int drawNoise, int isStep,
        mixed gamma, mixed kT, GLOBAL const int* RESTRICT isFluid, GLOBAL mixed* RESTRICT stencilWeight,
        GLOBAL int4* RESTRICT stencilOrigin) {
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
        int4 firstNodes;
        firstNodes.x = stencilFirst(sx);
        firstNodes.y = stencilFirst(sy);
        firstNodes.z = stencilFirst(sz);
        firstNodes.w = 0;
        stencilOrigin[i] = firstNodes;
        sortKeys[i] = particleKey(firstNodes, i);
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
 * After listing the segments of the stencils, each segment sums the reactions -xi_j F of the particles whose stencils
 * contain its node, in particle order, and writes the sum to cellReaction: one writer per node.  With the domain
 * decomposition only the nodes of the block receive the reactions.
 */
KERNEL void sumStencilReactions(GLOBAL const int* RESTRICT segmentNode, GLOBAL const mixed* RESTRICT particleForce,
        GLOBAL mixed* RESTRICT cellReaction, GLOBAL const int* RESTRICT segmentStart, GLOBAL const int* RESTRICT chunkOffset,
        GLOBAL const mixed* RESTRICT sortedWeight, GLOBAL const int* RESTRICT sortedParticle) {
    const int numSegments = chunkOffset[NUM_KEY_CHUNKS];
    for (int s = GLOBAL_ID; s < numSegments; s += GLOBAL_SIZE) {
        int first = segmentStart[s], end = segmentStart[s+1];
        int node = segmentReactionNode(segmentNode[s]);
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
 * (u = 0, no 1/rho term).  Writes the right-hand side b_k/m_k (rhs), the diagonal (1 + a)/m_k + a K_kk, for each slot
 * e = i*STENCIL_SIZE + n of the stencil its weight, xi/rho (0 at a solid node) and its node, and for each particle the
 * first nodes of its stencil (stencilOrigin) and its sort key (particleKey()).  The other forces Fc and the random
 * numbers are handled as in prepareCenteredDrag().  With the domain decomposition the rank that owns the nearest node
 * writes the right-hand side, the diagonal and the nodes of a particle, which the host sums over the ranks, and the
 * others write zero and NUM_STORED as the nodes; every rank writes the weights, first nodes and keys of all the
 * stencils, and xi/rho of all of them comes later, from the densities of the nodes summed over the ranks
 * (gatherSegmentDensity(), stencilInterpolationWeights()).
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
        GLOBAL int* RESTRICT stencilNode, GLOBAL mixed* RESTRICT rhs, GLOBAL mixed* RESTRICT diagonal,
        GLOBAL int4* RESTRICT stencilOrigin
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
            if (slot == 0) {
                int4 firstNodes;
                firstNodes.x = stencilFirst(sx);
                firstNodes.y = stencilFirst(sy);
                firstNodes.z = stencilFirst(sz);
                firstNodes.w = 0;
                stencilOrigin[i] = firstNodes;
                sortKeys[i] = particleKey(firstNodes, i);
            }
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
        GLOBAL int* RESTRICT stencilNode, GLOBAL mixed* RESTRICT rhs, GLOBAL mixed* RESTRICT diagonal,
        GLOBAL int4* RESTRICT stencilOrigin
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
        int4 firstNodes;
        firstNodes.x = stencilFirst(sx);
        firstNodes.y = stencilFirst(sy);
        firstNodes.z = stencilFirst(sz);
        firstNodes.w = 0;
        stencilOrigin[i] = firstNodes;
        sortKeys[i] = particleKey(firstNodes, i);
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
 * gradients for all the particles, and spreads p on all the nodes of the stencils.  A work group of LBM_BLOCK_SIZE
 * threads takes the segments that start in a chunk of LBM_BLOCK_SIZE entries, one thread per segment: the threads read
 * the weights and the values of p of the entries of these segments together, LBM_BLOCK_SIZE entries at a time, into
 * local memory, and the thread of each segment adds those of its segment in order.
 */
KERNEL void spreadStencilVector(GLOBAL const int* RESTRICT segmentStart, GLOBAL const int* RESTRICT chunkOffset,
        GLOBAL const mixed* RESTRICT sortedWeight, GLOBAL const int* RESTRICT sortedParticle, GLOBAL const mixed* RESTRICT p,
        GLOBAL mixed* RESTRICT spreadValue) {
    LOCAL mixed keyValue[4*LBM_BLOCK_SIZE];
    for (int c = GROUP_ID; c < NUM_KEY_CHUNKS; c += NUM_GROUPS) {
        int firstSegment = chunkOffset[c], endSegment = chunkOffset[c+1];
        int s = firstSegment + LOCAL_ID;
        bool owner = (s < endSegment);
        int first = (owner ? segmentStart[s] : 0), end = (owner ? segmentStart[s+1] : 0);
        int chunkEnd = segmentStart[endSegment];
        mixed sx = 0, sy = 0, sz = 0;
        for (int base = segmentStart[firstSegment]; base < chunkEnd; base += LBM_BLOCK_SIZE) {
            int m = base + LOCAL_ID;
            if (m < chunkEnd) {
                int i = sortedParticle[m];
                keyValue[4*LOCAL_ID] = sortedWeight[m];
                keyValue[4*LOCAL_ID+1] = p[i];
                keyValue[4*LOCAL_ID+2] = p[NUM_COUPLED+i];
                keyValue[4*LOCAL_ID+3] = p[2*NUM_COUPLED+i];
            }
            SYNC_THREADS;
            int to = min(end, base+LBM_BLOCK_SIZE);
            for (int k = max(first, base); k < to; k++) {
                int slot = 4*(k-base);
                mixed w = keyValue[slot];
                sx += w*keyValue[slot+1];
                sy += w*keyValue[slot+2];
                sz += w*keyValue[slot+3];
            }
            SYNC_THREADS;
        }
        if (owner) {
            spreadValue[3*s] = sx;
            spreadValue[3*s+1] = sy;
            spreadValue[3*s+2] = sz;
        }
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
 * With the domain decomposition and the centred drag: the density of the node of each segment (the segments are the
 * same on every rank), from the rank whose block holds the node and 0 on the others, which the host sums over the
 * ranks; then xi/rho of every slot of every stencil, as prepareCenteredStencil() computes it for the particles of the
 * rank (0 at a solid node).  So every rank runs the conjugate gradients for all the particles, with the same numbers.
 */
KERNEL void gatherSegmentDensity(GLOBAL const int* RESTRICT segmentNode, GLOBAL const int* RESTRICT segmentStart,
        GLOBAL const int* RESTRICT chunkOffset, GLOBAL const mixed* RESTRICT densityDeviation,
        GLOBAL mixed* RESTRICT segmentDensity) {
    const int numSegments = chunkOffset[NUM_KEY_CHUNKS];
    for (int s = GLOBAL_ID; s < numSegments; s += GLOBAL_SIZE) {
        int node = segmentReactionNode(segmentNode[s]);
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
 * After the collision, set the reactions of the nodes that received any back to zero, for the next step: the nodes of
 * the keys, or with an interpolation stencil those of the segments.
 */
#ifdef STENCIL_WIDTH
KERNEL void clearCellReactions(GLOBAL const int* RESTRICT segmentNode, GLOBAL const int* RESTRICT chunkOffset,
        GLOBAL mixed* RESTRICT cellReaction) {
    const int numSegments = chunkOffset[NUM_KEY_CHUNKS];
    for (int s = GLOBAL_ID; s < numSegments; s += GLOBAL_SIZE) {
        int node = segmentReactionNode(segmentNode[s]);
        if (node == NUM_STORED)
            continue;
        cellReaction[node] = 0;
        cellReaction[NUM_STORED+node] = 0;
        cellReaction[2*NUM_STORED+node] = 0;
    }
}
#else
KERNEL void clearCellReactions(GLOBAL const mm_long* RESTRICT sortKeys, GLOBAL mixed* RESTRICT cellReaction) {
    for (int k = GLOBAL_ID; k < NUM_KEYS; k += GLOBAL_SIZE) {
        int node = keyNode(sortKeys[k]);
#ifdef DOMAIN_DECOMPOSITION
        if (node == NUM_STORED)
            continue;
#endif
        cellReaction[node] = 0;
        cellReaction[NUM_STORED+node] = 0;
        cellReaction[2*NUM_STORED+node] = 0;
    }
}
#endif

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
