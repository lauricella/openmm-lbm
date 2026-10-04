# Model, units and conventions

This document describes the physics implemented by `LBMForce`, the conversion between OpenMM and
lattice units, and the conventions every platform must follow. Each section says whether it is
already implemented (version 0.1.0) or still to be implemented; the target scheme is the one of the
CUDA lattice Boltzmann library of the DragOpenMM project, which this plugin reproduces.

## 1. Fluid model (to be implemented)

The fluid is a D3Q19 lattice Boltzmann model, weakly compressible:

- moment 0 is the density, rho = sum_i f_i;
- moment 1 is the momentum, j = rho u = sum_i c_i f_i (the arrays of the plugin store j, not u);
- the equilibrium is the second-order Hermite expansion

  f_i^eq(rho, u) = w_i rho [1 + c_i.u/cs^2 + (c_i.u)^2/(2 cs^4) - u.u/(2 cs^2)],  cs^2 = 1/3.

**Collision: regularized, with Guo forcing [1, 2].** With the force density F acting on a node
during the step, the post-collision populations are

  f_i* = f_i^eq(rho, u*) + (1 - omega) f_i^neq,reg + (1/2) S_i,

where:

- u* = (j + F/2)/rho is the velocity shifted by half a force;
- f_i^neq,reg = w_i/(2 cs^4) (c_i c_i - cs^2 I) : Pi^neq is rebuilt from the non-equilibrium stress
  Pi^neq = sum_i c_i c_i (f_i - f_i^eq(rho, j/rho));
- S_i = w_i [(c_i - u*)/cs^2 + (c_i.u*) c_i/cs^4] . F is the Guo source.

The prefactor of S_i is 1/2, not the (1 - omega/2) of the BGK form. The equilibrium is already
shifted by F/2 and the regularized f^neq has no first-order part, so with 1/2 the momentum increases
by exactly F per step, at every relaxation time [3, 4].

**Relaxation and streaming.** The relaxation frequency is omega = 1/tau, with
tau = 3 nu dt/dx^2 + 1/2.

The update is thread-safe and uses a single copy of the populations:

1. A first kernel computes rho, j and Pi^neq at every node from its populations.
2. A second kernel reads only these moments at a node and writes the 19 post-collision populations
   to the neighbouring nodes (push streaming).

Each population of the destination is written by exactly one thread.

## 2. Particle-fluid coupling (to be implemented)

**Euler-Maruyama scheme.** Each coupled particle k of mass m_k feels

  F_k = -gamma m_k (v_k - u(x_k)) + R_k,   <R_k R_k> = 2 gamma m_k kT/dt (per component).

- u(x_k) is the fluid velocity j/rho at the nearest lattice node.
- The fluid at that node receives -F_k.
- Drag and noise are part of the force, so the System is integrated with `VerletIntegrator`.

**Time levels.** OpenMM's Verlet integrator is a leapfrog: during the force evaluation of step t
the velocities are v(t - dt/2). The fluid momentum before the step is j(t - dt/2), because
j(t + dt/2) = j(t - dt/2) + F(t). The drag therefore compares particle and fluid velocities at the
same half step, explicitly.

**Kinetic temperature.** Measured from half-step velocities it is kT/(1 - gamma dt/2) for a free
particle. Measured from full-step velocities v(t) = (v(t - dt/2) + v(t + dt/2))/2 it is kT.

**Per-cell reaction.** The reaction forces of the particles in the same cell are summed without
atomic operations:

1. the (cell, particle) pairs are sorted with OpenMM's `ComputeSort`, using unique keys;
2. one thread per cell adds the forces of its particles, in particle order.

The result is bitwise reproducible.

**Fluid update.** The fluid is advanced once per time step. Further force evaluations within the
same step (for example `getState(getForces=True)`) reuse the forces already computed.

## 3. Units and conversions (implemented)

The public API uses OpenMM units: nm, ps, Da (g/mol), K and kJ/mol. Internally the plugin works in
lattice units. The conversion is computed in one place, `LBMForceImpl::computeLatticeParameters()`:

| Quantity | Lattice unit | OpenMM value |
|---|---|---|
| length | dx | box length / number of nodes (cells must be cubic) |
| time | dt | integrator step size |
| mass | m_c = rho0 dx^3 | rho0 = `setFluidDensity()`, in Da/nm^3 |
| density | rho0 | the fluid at rest has lattice density 1 |
| velocity | dx/dt | u_lattice = u dt/dx |
| acceleration | dx/dt^2 | g_lattice = g dt^2/dx |
| force on a cell | m_c dx/dt^2 | F_lattice = F dt^2/(m_c dx) |
| kinematic viscosity | dx^2/dt | tau = 3 nu dt/dx^2 + 1/2 > 1/2 |

The thermal energy is kT = k_B T with k_B = `BOLTZ` of OpenMM, in kJ/mol. The body force on the
fluid is set as an acceleration (`setBodyAcceleration()`); the force density on a node is
rho g_lattice.

Changing the integrator step size after creating the Context would change the lattice time step,
so it is rejected.

## 4. Storage and ordering (implemented)

- **Node index.** Node (i, j, k) has index i + nx (j + ny k).
- **Populations.** Stored as f[q numNodes + node], with the velocity set of
  `openmmapi/include/internal/D3Q19.h`:

| q | c_q | weight |
|---|---|---|
| 0 | (0, 0, 0) | 1/3 |
| 1, 2 | (1, 0, 0), (-1, 0, 0) | 1/18 |
| 3, 4 | (0, 1, 0), (0, -1, 0) | 1/18 |
| 5, 6 | (0, 0, 1), (0, 0, -1) | 1/18 |
| 7, 8 | (1, 1, 0), (-1, -1, 0) | 1/36 |
| 9, 10 | (1, -1, 0), (-1, 1, 0) | 1/36 |
| 11, 12 | (0, 1, 1), (0, -1, -1) | 1/36 |
| 13, 14 | (0, 1, -1), (0, -1, 1) | 1/36 |
| 15, 16 | (1, 0, 1), (-1, 0, -1) | 1/36 |
| 17, 18 | (-1, 0, 1), (1, 0, -1) | 1/36 |

- **Fluid state.** `getFluidState()` returns the populations in this layout, in lattice units.
- **Initial state.** A new Context starts from the equilibrium at lattice density 1 and the initial
  velocity.

## 5. Precision (implemented)

The fluid (populations and moments) uses the "mixed" type of the platform:

- float when the platform `Precision` is `single`;
- double when it is `mixed` or `double`.

The Reference platform always uses double precision.

Weak uniform forces and the conservation of momentum are limited by single precision (relative
resolution about 1e-7 on populations of order 0.05), so `mixed` is recommended for production.

## References

1. J. Latt and B. Chopard, Math. Comput. Simul. 72, 165 (2006): regularized collision.
2. Z. Guo, C. Zheng and B. Shi, Phys. Rev. E 65, 046308 (2002): forcing scheme.
3. accLB, arXiv:2505.01126 (2025), eq. 6: f = f^eq + (1 - omega) f^neq + S/2.
4. LBFAST, arXiv:2609.09160 (2026), eq. 3.
5. P. Ahlrichs and B. Dünweg, J. Chem. Phys. 111, 8225 (1999): frictional particle-fluid coupling.
6. B. Dünweg and A. J. C. Ladd, Adv. Polym. Sci. 221, 89 (2009): review of lattice Boltzmann for soft matter.
