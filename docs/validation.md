# Validation

Tests of openmm-lbm, what each one checks, its tolerance and the values measured. The C++ tests run with
`ctest` (one executable per platform, in `platforms/*/tests`); the Python tests run with `pytest` in
`python/tests`.

## Fluid on its own (`tests/TestLBMFluid.h`, Reference platform)

The fluid tests use no coupled particles. The lattice spacing is 0.5 nm, the time step 0.01 ps and the
density 602.2 Da/nm^3; the viscosity is chosen to give the relaxation time tau of each test. The momentum
removal is off unless stated.

| Test | Setup | Check | Tolerance |
|---|---|---|---|
| `testUniformFlowIsSteady` | 6x5x4 nodes, tau = 0.8, uniform velocity (0.03, -0.02, 0.01) in lattice units, 50 steps | populations unchanged | 1e-13 |
| `testFluidConservation` | 6x5x4 nodes, tau = 0.7, populations perturbed by up to 1%, 100 steps | total mass and momentum unchanged | 1e-13 (relative to the mass) |
| `testBodyForce` | 4x3x5 nodes, tau = 0.9, uniform lattice density rho0 = 0.98, 1 or 1.02 at rest, body acceleration g, 20 steps | momentum n rho0 g per node (lattice units), density rho0, velocity (n + 1/2) g dt | 1e-13 (momentum), 1e-12 nm/ps (velocity) |
| `testFluidMomentumRemoval` | 4x4x4 nodes, tau = 1, initial uniform velocity, body force F per node, removal frequency 3 | total momentum ((n-1)%3 + 1) F per node after n = 1...7 steps: removal on step indices 0, 3, 6, before the collision | 1e-13 |
| `testShearWaveViscosity` | 2x64x2 nodes, u_x = 1e-3 sin(2 pi y/64) in lattice units, tau = 0.6, 1, 1.5; amplitude at steps 200 and 1200 | decay rate nu k^2 with nu = (tau - 1/2)/3 | 2e-3 (relative) |
| `testMachNumberCheck` | 4x4x4 nodes, uniform flow at Ma = 0.35, check every 10 steps | `getFluidMachNumber()` = 0.35; exception at step 10 with limit 0.3, none with the check off or with limit 0.5 | 1e-12 |
| `testLatticeParameters` | 4x6x8 nodes, tau = 0.9 | `getLatticeParametersInContext()` returns dx, dt and tau | 1e-12 |
| `testRelaxationTimeWarning` | tau = 0.503, 1, 2.2 | warning on stderr at Context creation only outside [0.505, 2] | exact |
| `testSolidNodeChecks` | solid node index out of range, repeated, or all nodes solid | exception at Context creation | exact |
| `testPoiseuille` | 2x12x2 nodes, solid plane j = 0, body force along x, tau = 0.7, 0.875, 1.2, 4 H^2/nu steps | steady profile equal to the exact solution of the scheme (`docs/theory.md`, solid nodes), zero density and velocity at the solid nodes | 1e-9 (relative to the maximum velocity) |
| `testWallConservation` | 6x5x4 nodes with a solid block of 8 nodes, initial uniform flow, without and with momentum removal | mass of the fluid conserved; after a removal step the momentum of the fluid is zero | 1e-13 |
| `testQueriesDoNotAdvanceFluid` | 4x4x4 nodes, body force, 10 steps with and without `getState(Forces)`, `getState(Forces, Energy)` and `setVelocitiesToTemperature()` after each step | identical populations | exact |

**Body force at lattice densities different from 1.** The test checks the convention of the weakly
compressible model: the half force shifts the momentum, u* = (j + F/2)/rho with F = rho g, so that each
step adds exactly rho g to the momentum. Shifting the velocity by F/2 instead would add (1/(2 rho) + 1/2) F.

**Viscosity.** Measured relative difference nu_measured/nu - 1:

| tau | 64 nodes per wavelength | 32 nodes per wavelength |
|---|---|---|
| 0.6 | +5.1e-4 | +2.1e-3 |
| 1.0 | -1.7e-7 | -2.8e-6 |
| 1.5 | +8.0e-4 | +3.2e-3 |

The difference decreases as k^2 (by a factor of 4 when the wavelength doubles), as expected for a
second-order scheme. It vanishes at tau = 1, where the collision relaxes the populations to equilibrium
in one step.

**Walls.** With halfway bounce-back the steady Poiseuille profile matches the exact solution of the
scheme to 1e-12 for every tau tested (0.7 to 1.5, channel widths 11 and 19). Its wall position differs
from the halfway position by (3 - 16 Lambda)/(12 H), Lambda = (tau - 1/2)/2; measured in a channel of
width H = 23:

| tau | 0.55 | 0.6 | 0.7 | 0.8 | 0.875 | 0.9 | 1.0 | 1.2 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| wall shift (lattice units) | +0.00942 | +0.00797 | +0.00507 | +0.00217 | 0.00000 | -0.00072 | -0.00362 | -0.00942 | -0.01810 | -0.03256 |

## Coupling of particles and fluid (`tests/TestLBMCoupling.h`, Reference platform)

Particles of 100 Da in a fluid of 8x8x8 nodes (dx = 0.5 nm, dt = 0.01 ps, tau = 0.8), without removal of
the fluid momentum.

| Test | Checks | Tolerance |
|---|---|---|
| First step | a particle in a fluid at rest: v1 = v0 (1 - gamma dt); `getState()` then returns the force of the step, m (v1 - v0)/dt | 1e-14, 1e-12 relative |
| Momentum conservation | particles and fluid with drag and random force, 50 steps, two particles at the same node, one crossing the periodic boundary, lattice densities 1, 0.98 and 1.02 | 1e-11 of the particle momentum (measured 7e-13: rounding of the sum over 19x512 populations) |
| Moving with the fluid | particle and fluid at the same velocity along x, y, z and a diagonal, two cells crossed | 1e-13 |
| Partial coupling | uncoupled particles keep their velocity and feel no force | rounding of OpenMM's Verlet |
| Force evaluations and seeds | `getState()` before and after every step does not change the trajectory; two runs with seed 0 differ | bitwise |
| Walls | a coupled particle reaching a solid node: v2 = -v0 (1 - gamma dt)^2; an uncoupled one passes | 1e-14 |
| Restart | checkpoint plus `setFluidState()` at step 13, removal every 5 steps, T = 0 | bitwise |
| Warning | friction*dt > 1 is reported at Context creation | |

**Temperature** (Reference, 200 free beads of 100 Da, 16^3 nodes, tau = 1.10, gamma dt = 0.1, T = 300 K,
20000 steps): 295.8 +- 0.4 K from full-step velocities, 311.7 +- 0.4 K from half-step velocities
(T/(1 - gamma dt/2) = 315.8 K). The kinetic energy reported by OpenMM gives 359 K and is not valid for
coupled particles (`docs/theory.md`, section 2).

## Equivalence with the reference implementation: fluid only

The Reference platform was compared in double precision with the CUDA lattice Boltzmann library from
which openmm-lbm is derived (the library of the DragOpenMM plugin, version tagged `ref-explicit-2026-10`),
built in double precision.
- **Setup.** 16x12x10 nodes, 500 steps, no particles acting on the fluid.
- **Initial state, the same in both codes.** Density modulated by up to 1.5% around rho0, a shear wave
  and smaller velocity modes, and a non-zero non-equilibrium stress, given as moments; the populations
  are rebuilt from them as feq(rho, j/rho) + fneq,reg(Pi_neq).
- **Cases.** tau = 1.102 with a body force, without and with removal of the fluid momentum at every
  step; tau = 0.62 with rho0 = 0.98 and a body force.
- **Result.** Density and momentum agree within the float32 rounding of the output of the reference
  library: relative differences of at most 5e-8. The non-equilibrium stress agrees within 5e-16 in
  absolute value, the double-precision rounding of f - feq for populations of order 0.05.
- **Sensitivity.** A difference in the algorithm would appear at 1e-3 or above. For example, a Guo
  prefactor of 1 - omega/2 instead of 1/2 changes the momentum input by 4.6% at tau = 1.102, and
  shifting the velocity instead of the momentum by half a force changes it by 1% at rho0 = 0.98.

## Planned

- The same fluid tests on the CUDA, OpenCL and HIP platforms, in single, mixed and double precision.
- Equivalence with the reference library for the coupling, at T = 0 in double precision: the
  deterministic tests of its validation campaign (first step and reaction, co-moving particle,
  nearest-node artefacts, kick, pair mobility, composite sphere) and the mobility of a dragged particle
  as a function of tau and box size.
