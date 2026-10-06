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
 * VELOCITY_SCALE (dx/dt) and FORCE_SCALE (m_c dx/dt^2, the force unit of the lattice in kJ/mol/nm).
 */

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
    int i = ((int) floor(wrapCoordinate(pos.x, NX) + 0.5f))%NX;
    int j = ((int) floor(wrapCoordinate(pos.y, NY) + 0.5f))%NY;
    int k = ((int) floor(wrapCoordinate(pos.z, NZ) + 0.5f))%NZ;
    return i + NX*(j + NY*k);
}

#ifdef HAS_SOLID_NODES
/**
 * Gradient of the solid indicator (1 at solid nodes, 0 at fluid nodes), interpolated trilinearly between the
 * eight nodes of the lattice cell that contains the position: it points from the fluid into the wall.
 */
DEVICE mixed3 wallNormal(mixed4 pos, GLOBAL const int* RESTRICT isFluid) {
    mixed s[3] = {wrapCoordinate(pos.x, NX), wrapCoordinate(pos.y, NY), wrapCoordinate(pos.z, NZ)};
    int size[3] = {NX, NY, NZ};
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
                solid[a][b][c] = (isFluid[index[0][a] + NX*(index[1][b] + NY*index[2][c])] ? 0 : 1);
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
 * has rho = 0 and is at rest.  The force is stored in particleForce, the key node*NUM_COUPLED + i in sortKeys
 * (sorted next, to sum the reactions of each node in particle order), and the reaction on a solid node is
 * added to wallMomentum.  xi comes from OpenMM's random numbers, random[randomIndex + i].
 */
KERNEL void coupleParticles(GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT posqCorrection,
        GLOBAL const mixed4* RESTRICT velm, GLOBAL const int* RESTRICT atomIndex, GLOBAL const int* RESTRICT couplingIndex,
        GLOBAL const mixed* RESTRICT particleMass, GLOBAL const mixed* RESTRICT densityDeviation, GLOBAL const mixed* RESTRICT momentum,
        GLOBAL mixed* RESTRICT particleForce, GLOBAL mm_long* RESTRICT sortKeys, GLOBAL mixed* RESTRICT wallMomentum,
        GLOBAL const float4* RESTRICT random, int randomIndex, mixed gamma, mixed kT) {
    for (int j = GLOBAL_ID; j < NUM_ATOMS; j += GLOBAL_SIZE) {
        int i = couplingIndex[atomIndex[j]];
        if (i < 0)
            continue;
        mixed4 pos = loadPosition(posq, posqCorrection, j);
        int node = nearestNode(pos);
        mixed rho = 1 + densityDeviation[node];
        mixed ux = 0, uy = 0, uz = 0;
        if (rho > 0) {
            ux = momentum[node]*(1/rho);
            uy = momentum[NUM_NODES+node]*(1/rho);
            uz = momentum[2*NUM_NODES+node]*(1/rho);
        }
        mixed4 v = velm[j];
        mixed vx = v.x*(1/(mixed) VELOCITY_SCALE), vy = v.y*(1/(mixed) VELOCITY_SCALE), vz = v.z*(1/(mixed) VELOCITY_SCALE);
        mixed m = particleMass[i];
        mixed fx = (vx-ux)*(-gamma*m), fy = (vy-uy)*(-gamma*m), fz = (vz-uz)*(-gamma*m);
        if (kT > 0 && gamma > 0) {
            mixed sigma = sqrt(2*gamma*m*kT);
            float4 xi = random[randomIndex+i];
            fx += xi.x*sigma;
            fy += xi.y*sigma;
            fz += xi.z*sigma;
        }
        particleForce[i] = fx;
        particleForce[NUM_COUPLED+i] = fy;
        particleForce[2*NUM_COUPLED+i] = fz;
        sortKeys[i] = ((mm_long) node)*NUM_COUPLED + i;
#ifdef HAS_SOLID_NODES
        if (rho == 0) {
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
        int node = (int) (sortKeys[k]/NUM_COUPLED);
        if (k > 0 && (int) (sortKeys[k-1]/NUM_COUPLED) == node)
            continue;
        mixed fx = 0, fy = 0, fz = 0;
        for (int m = k; m < NUM_COUPLED && (int) (sortKeys[m]/NUM_COUPLED) == node; m++) {
            int i = (int) (sortKeys[m] - ((mm_long) node)*NUM_COUPLED);
            fx -= particleForce[i];
            fy -= particleForce[NUM_COUPLED+i];
            fz -= particleForce[2*NUM_COUPLED+i];
        }
        cellReaction[node] = fx;
        cellReaction[NUM_NODES+node] = fy;
        cellReaction[2*NUM_NODES+node] = fz;
    }
}

/**
 * After the collision, set the reactions of the nodes that received any back to zero, for the next step.
 */
KERNEL void clearCellReactions(GLOBAL const mm_long* RESTRICT sortKeys, GLOBAL mixed* RESTRICT cellReaction) {
    for (int k = GLOBAL_ID; k < NUM_COUPLED; k += GLOBAL_SIZE) {
        int node = (int) (sortKeys[k]/NUM_COUPLED);
        cellReaction[node] = 0;
        cellReaction[NUM_NODES+node] = 0;
        cellReaction[2*NUM_NODES+node] = 0;
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
