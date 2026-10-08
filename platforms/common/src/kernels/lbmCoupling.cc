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
 *
 * With the domain decomposition (DOMAIN_DECOMPOSITION, kernels/lbmFluid.cc) GNX, GNY and GNZ are the sizes of the
 * lattice and OX, OY, OZ the first node of the block of the rank; the moments and reactions are stored at the storage
 * index of the node in the block.  Every rank holds all the particles: the nearest node and the wall normal come from
 * the whole lattice (isFluid is then the mask of the whole lattice), the rank that owns the nearest node computes the
 * coupling force, the others set it to zero and give the particle the key NUM_STORED*NUM_COUPLED + i, which sorts after
 * all the nodes and is skipped; the host sums the forces over the ranks.  SKIP_REFLECTION_MOMENTUM is defined on the
 * ranks other than 0, which reflect their copies of the particles but do not count the momentum of the wall.
 */

#ifndef DOMAIN_DECOMPOSITION
#define NUM_STORED NUM_NODES
#define GNX NX
#define GNY NY
#define GNZ NZ
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
        int node = storedNode(nearestNode(pos));
#ifdef DOMAIN_DECOMPOSITION
        if (node == NUM_STORED) {
            // Another rank owns the node: it computes the force.  The random numbers are copied all the same.
            if (kT > 0 && gamma > 0 && drawNoise)
                noise[i] = random[randomIndex+i];
            particleForce[i] = 0;
            particleForce[NUM_COUPLED+i] = 0;
            particleForce[2*NUM_COUPLED+i] = 0;
            sortKeys[i] = ((mm_long) NUM_STORED)*NUM_COUPLED + i;
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
        int node = (int) (sortKeys[k]/NUM_COUPLED);
        if (k > 0 && (int) (sortKeys[k-1]/NUM_COUPLED) == node)
            continue;
#ifdef DOMAIN_DECOMPOSITION
        if (node == NUM_STORED)
            continue;
#endif
        mixed fx = 0, fy = 0, fz = 0;
        for (int m = k; m < NUM_COUPLED && (int) (sortKeys[m]/NUM_COUPLED) == node; m++) {
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

/**
 * After the collision, set the reactions of the nodes that received any back to zero, for the next step.
 */
KERNEL void clearCellReactions(GLOBAL const mm_long* RESTRICT sortKeys, GLOBAL mixed* RESTRICT cellReaction) {
    for (int k = GLOBAL_ID; k < NUM_COUPLED; k += GLOBAL_SIZE) {
        int node = (int) (sortKeys[k]/NUM_COUPLED);
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
        int node = storedNode(nearestNode(pos));
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
        int node = (int) (sortKeys[k]/NUM_COUPLED);
        if (k > 0 && (int) (sortKeys[k-1]/NUM_COUPLED) == node)
            continue;
#ifdef DOMAIN_DECOMPOSITION
        if (node == NUM_STORED)
            continue;
#endif
        int end = k;
        mixed mass = 0, px = 0, py = 0, pz = 0, rx = 0, ry = 0, rz = 0;
        for (; end < NUM_COUPLED && (int) (sortKeys[end]/NUM_COUPLED) == node; end++) {
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
