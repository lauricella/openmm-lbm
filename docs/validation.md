# Validation

Tests of openmm-lbm, what each one checks, its tolerance and the values measured. The C++ tests run with
`ctest` (one executable per platform, in `platforms/*/tests`); the Python tests run with `pytest` in
`python/tests`.

## Fluid on its own (`tests/TestLBMFluid.h`, all platforms)

The fluid tests use no coupled particles. The lattice spacing is 0.5 nm, the time step 0.01 ps and the
density 602.2 Da/nm^3; the viscosity is chosen to give the relaxation time tau of each test. The momentum
removal is off unless stated.
- `runFluidTests()` and `runWallTests()` (`testSolidNodeChecks`, `testPoiseuille`, `testWallConservation`)
  run on every platform, in the `single`, `mixed` and `double` precision modes.
- The tolerances below hold on the Reference platform and in `mixed` and `double` precision. In `single`
  precision (`getFluidTolerance()`) the tolerances of 1e-12 and below become 2e-6, the resolution of the
  stored type, and those of `testPoiseuille` become 5e-5: its steady state is the result of thousands of
  steps rounded in single precision, and on an A100 the profile and the force on the walls differ from
  the exact values by up to 1.1e-5 and 8e-6.

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
scheme to 1e-12 for every tau tested (0.7 to 1.5, channel widths 11 and 19). In the steady state the
force on the walls from the momentum exchange (`getWallForce()`) equals the body force on the fluid, g
times its mass, to 1e-8 (`testPoiseuille`). Its wall position differs
from the halfway position by (3 - 16 Lambda)/(12 H), Lambda = (tau - 1/2)/2; measured in a channel of
width H = 23:

| tau | 0.55 | 0.6 | 0.7 | 0.8 | 0.875 | 0.9 | 1.0 | 1.2 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| wall shift (lattice units) | +0.00942 | +0.00797 | +0.00507 | +0.00217 | 0.00000 | -0.00072 | -0.00362 | -0.00942 | -0.01810 | -0.03256 |

## Coupling of particles and fluid (`tests/TestLBMCoupling.h`, all platforms)

Particles of 100 Da in a fluid of 8x8x8 nodes (dx = 0.5 nm, dt = 0.01 ps, tau = 0.8), without removal of
the fluid momentum. The tests run on every platform and precision mode. The tolerances below hold on the
Reference platform and in `double` precision; otherwise the tolerances below 2e-6 become 2e-6
(`getCouplingTolerance()`), because OpenMM handles the forces on the particles in single precision unless
the platform runs in double precision (`docs/theory.md`, section 2).

| Test | Checks | Tolerance |
|---|---|---|
| First step | a particle in a fluid at rest: v1 = v0 (1 - gamma dt); `getState()` then returns the force of the step, m (v1 - v0)/dt | 1e-14, 1e-12 relative |
| Momentum conservation | particles and fluid with drag and random force, 50 steps, two particles at the same node, one crossing the periodic boundary, lattice densities 1, 0.98 and 1.02 | 1e-11 of the particle momentum (measured 7e-13: rounding of the sum over 19x512 populations) |
| Moving with the fluid | particle and fluid at the same velocity along x, y, z and a diagonal, two cells crossed | 1e-13 |
| Partial coupling | uncoupled particles keep their velocity and feel no force | rounding of OpenMM's Verlet |
| Force evaluations and seeds | `getState()` before and after every step does not change the trajectory; two runs with seed 0 differ | bitwise |
| Walls | a coupled particle reaching a solid node: v2 = -v0 (1 - gamma dt)^2; an uncoupled one passes | 1e-14 |
| Walls, direction | at a wall one node thick, a particle is reversed only if it moves into the wall, from either side | 1e-14 |
| Momentum with walls | fluid flowing against two walls, a particle reflected, drag and random force, 100 steps: particles + fluid + the sum of `getWallForce()` dt is constant | 1e-12 relative |
| Restart | checkpoint plus `setFluidState()` at step 13, removal every 5 steps, T = 0 | bitwise |
| Checkpoint with random force | OpenMM checkpoint plus `createCheckpoint()` at step 9, 300 K, a wall, removal every 4 steps, a force evaluation just before (random numbers already drawn), 14 more steps (`testCheckpointWithRandomForce`) | bitwise, also the wall force |
| Checkpoint refused | different number of coupled particles, or data that are not a checkpoint (`testCheckpointMismatch`) | exception |
| Warning | friction*dt > 1 is reported at Context creation | |
| Equipartition | 100 free particles, gamma dt = 0.1, T = 300 K: full-step temperature close to T, half-step temperature close to T/(1 - gamma dt/2) | 10% (measured 4% below, from the missing fluid fluctuations) |
| Warning on tau | tau > 1.7 with coupled particles and the explicit drag is reported at Context creation | |
| Repeated force evaluation | 8000 particles with a short-range `CustomNonbondedForce`, compressed into a cube of 2 nm so that the neighbor list overflows and the GPU platforms repeat the force evaluation of the step: the total momentum is conserved in that step (`testRepeatedForceEvaluation`; GPU platforms only) | 1e-11 relative (1.3e-2 before the fix, on CUDA and OpenCL) |

The tests that do not depend on the drag (full-step kinetic energy, momentum conservation, moving with the
fluid, partial coupling, force evaluations and seeds, momentum with walls, restart, checkpoint with random
force, equipartition) run again with the centred drag (next section).

## Centred drag (`tests/TestLBMCentered.h`, all platforms)

| Test | Checks | Tolerance |
|---|---|---|
| First step | a particle in a fluid at rest: v1 = v0 (1 - a + a m/m_c)/(1 + a + a m/m_c), a = gamma dt/2; then the force of the next step | 1e-14 (5e-14 on the GPU platforms, see below), 1e-12 |
| Shared node | three particles of 80, 120 and 150 Da at one node and a fourth alone, external forces, uniform flow and body acceleration: velocities after the first step against the direct solution of the linear system of the drag, and the momentum received by the fluid (j + G at every node, G = body force - S) | 1e-12, 1e-11 |
| Wall | a particle at a solid node moving out of the wall: v1 = v0 (1 - a)/(1 + a), and the wall receives the opposite of its coupling force | 1e-14 (5e-14 on the GPU platforms), 1e-12 |
| Large friction | gamma dt = 3, four particles at one node: velocities decay, total momentum conserved | 1e-11 |
| Repeated evaluation | 8000 particles with a short-range pair force compressed into a cube of 2 nm: a step whose force evaluation overflows the neighbor list gives the same velocities as the same step after the list has grown in an evaluation between steps, so the step uses the complete other forces (`testCenteredRepeatedEvaluation`; GPU platforms only) | 1e-12 relative to the largest velocity |
| Requirements | `LBMForce` not last, virtual sites, change of the drag in `updateParametersInContext()`, checkpoint loaded with the other drag | exception |
| Equipartition | as above, with a fluid 100 times denser: half-step temperature close to T, full-step close to T/(1 + gamma dt/2) | 10% |

The tests of the previous section that do not depend on the drag run again with the centred drag, among them
the momentum conservation in a step whose force evaluation is repeated (`testRepeatedForceEvaluation`).
In single and mixed precision the tolerances are at least 2e-6, as for the explicit drag. In double
precision the GPU platforms add the forces in fixed point, with a resolution of 2^-32 kJ/mol/nm, that is
2.3e-14 nm/ps of velocity in one step of 0.01 ps for a particle of 100 Da: measured 1.2e-14 to 1.9e-14 for
the centred first step on CUDA (the explicit one happens to be exact there), hence 5e-14 for the one-step
tests on these platforms.

**Python tests.** `test_centered_drag_sees_all_forces` checks, on every platform, that the drag of a step
uses all the other forces: four charged particles with a constant field and a `NonbondedForce` with PME
(whose reciprocal part the CUDA platform computes on a separate stream), each alone at its node, in a fluid
at rest at T = 0; the velocities after the first step agree within 1e-11 with the closed form computed with
the forces of the other force groups, read from the State. Without the PME forces in the drag the error would
be about 1e-3. `test_coupling_agrees_with_reference` compares both drags with the Reference platform (next
sections).

**Verifications done once, outside the test suite** (NVIDIA A100, OpenMM 8.6.1, `develop` before the commit
of the centred drag on the GPU platforms).
- *Floating point force buffers of OpenCL.* OpenMM's own forces do not write them during a force evaluation
  on one device, so the tests cannot reach the code that reads them. In a modified build a kernel moved all
  the other forces from the fixed point buffer into the floating point buffer just before the centred drag
  read them: all the OpenCL tests passed in single, mixed and double precision. With the same build and the
  floating point buffers not read, `testCenteredSharedNode` failed (velocities wrong by 3e-3).
- *Repeated evaluation.* In a build without the check of the validity of the evaluation, the centred drag
  of a step whose evaluation overflows the neighbor list used the incomplete forces, and
  `testCenteredRepeatedEvaluation` failed with velocities wrong by 1e-3.
- *Explicit drag unchanged.* The final state of a run with random force, momentum removal, walls and force
  evaluations between steps is identical bit for bit to that of version 0.1.0 on the Reference platform and
  with CUDA in double and mixed precision (SHA-256 of positions, velocities and fluid state).

**Fluctuation-dissipation balance without the response of the fluid** (Reference, fluid 10^4 times denser,
100 particles of 100 Da, 300 K, 200000 steps, three seeds, error from 20 blocks). Ratio of the measured
temperature to the exact value of the discretization (explicit: T at full steps, T/(1 - gamma dt/2) at half
steps; centred: T at half steps, T/(1 + gamma dt/2) at full steps), mean of the three seeds:

| gamma dt | explicit, full step | explicit, half step | centred, half step | centred, full step |
|---|---|---|---|---|
| 0.1 | 1.0002 | 1.0001 | 1.0002 | 1.0002 |
| 0.5 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| 1.5 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |

The statistical error of each mean is about 4e-4 at gamma dt = 0.1 and 1.5e-4 at 1.5. The kinetic energy that
OpenMM reports gives the full-step values.

**Temperature with the fluid** and **self-mobility y(tau)** of the two drags: `docs/theory.md`, section 2.

**Temperature** (Reference, 200 free beads of 100 Da, 16^3 nodes, tau = 1.10, gamma dt = 0.1, T = 300 K,
20000 steps): 295.8 +- 0.4 K from full-step velocities, 311.7 +- 0.4 K from half-step velocities
(T/(1 - gamma dt/2) = 315.8 K). The kinetic energy that OpenMM reports is that of the full step: on 200
samples of the same system (seed 7) it gives 293.98 K, the same as the full-step velocities within 4e-12 K
(`testFullStepKineticEnergy` checks it on every platform; `docs/theory.md`, section 2). Before the change of
the coupling forces between steps to those of the next step it read 359 K.

## Equivalence with the reference implementation: coupled particles (E0)

The Reference platform was compared with the validation campaign of the reference CUDA library (version
tagged `ref-explicit-2026-10`, built in double precision), run with the same scripts, at T = 0, with all
particles coupled and no removal of the fluid momentum: 80 cases.
- **Cases.**
  - Nearest-node artefacts of a dragged particle (8 cases);
  - kick of a particle in a fluid at rest (6 cases, lattices up to 64^3, 20000 steps);
  - pair mobility (15 cases);
  - composite sphere of 300 beads in translation and rotation (7 cases);
  - mobility of a dragged particle (44 cases) as a function of mass, friction, force, direction, box size,
    tau and time step.
- **Result.**
  - Trajectories agree within 5e-13 relative over 30000 steps.
  - Derived quantities (mobilities, self-mobility y, hydrodynamic radii, velocity decay) agree within
    2e-6, and within 1.1e-5 for the velocity of a kicked particle at its last plateau (5e-6 of its initial
    velocity).
  - The criterion was 1e-4.
- **Rounding of small momenta.** With 10 Da nm/ps spread over a fluid of 2e7 Da (64^3 nodes, 20000 steps),
  the total momentum of particles and fluid drifts by at most 1e-8 of the momentum; the reference library
  drifts by 4.5e-6 (read from its single-precision output). Such a momentum is a difference between
  populations of order 0.05 at the ninth digit. Before the populations were stored as deviations from the
  rest equilibrium (`docs/theory.md`, section 4), the drift of openmm-lbm was 3e-5.

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

## GPU platforms against the Reference platform: fluid and walls

The Python test `test_fluid_agrees_with_reference` runs the same fluid on the Reference platform and on
each available GPU platform: 6x5x4 nodes, tau = 0.8, populations perturbed by up to 1e-3, a body force and
the removal of the fluid momentum every third step, 40 steps, without solid nodes and with the solid plane
j = 0 and a block of 8 solid nodes. The largest difference of the fluid states, relative to the largest
deviation |f - w|, must be below 1e-12 in `mixed` and `double` precision, and the forces on the walls must
agree to 1e-10. Measured on an NVIDIA A100 with OpenMM 8.6.1, without solid nodes:

| Platform | Precision | 40 steps | 1000 steps |
|---|---|---|---|
| CUDA | mixed, double | 1.8e-15 | 2.8e-15 |
| CUDA | single | 9.1e-7 | 1.3e-5 |
| OpenCL | mixed, double | 1.8e-15 | 7.6e-15 |
| OpenCL | single | 1.1e-6 | 2.7e-5 |

With walls (6x7x5 nodes, the solid plane j = 0 and a block of 8 solid nodes, 300 steps), the fluid states
differ by 6e-16 in `mixed` and `double` precision and by 1.9e-5 in `single` precision, and the forces on
the walls by 1e-14 and 4e-8 (CUDA and OpenCL alike).

The two platforms are not identical bit for bit: the GPU compilers contract multiplications and additions
into fused multiply-adds. Each GPU platform is deterministic: the removal of the momentum and the Mach
number use reductions in a fixed order, without atomic operations, and each population returned by the
bounce-back is written by one thread, so two runs give identical results.

## GPU platforms against the Reference platform: coupled particles

The Python test `test_coupling_agrees_with_reference` runs four coupled particles at T = 0 (friction
10/ps, 60 steps; two particles share a node, one crosses the periodic boundary), without and with the solid
plane j = 0, which reflects two of them, on the Reference platform and in double precision on each GPU
platform, with each drag scheme. A constant field and a soft pair force act on the particles, so that the
centred drag sees other forces. Positions, velocities, fluid state and force on the walls must agree to
1e-10. Largest differences relative to max(1, largest value), with the walls, measured on an NVIDIA A100
with OpenMM 8.6.1 (`develop` at the commit of the centred drag on the GPU platforms):

| Platform and precision | drag | positions | velocities | fluid | force on the walls |
|---|---|---|---|---|---|
| CUDA and OpenCL double | explicit | 2e-14 | 2e-13 | 2e-16 | 7e-13 |
| | centred | 2e-14 | 2e-13 | 1e-16 | 8e-13 |
| CUDA mixed | explicit | 2e-9 | 1e-8 | 2e-16 | 7e-13 |
| | centred | 3e-10 | 4e-9 | 1e-16 | 8e-13 |
| OpenCL mixed | explicit | 2e-9 | 1e-8 | 9e-14 | 3e-9 |
| | centred | 3e-10 | 4e-9 | 2e-13 | 7e-9 |
| CUDA and OpenCL single | explicit | 3e-7 | 3e-7 | 1e-8 | 1.4e-6 |
| | centred | 3e-7 | 3e-7 | 1e-8 | 1.6e-6 |

In double precision the velocities differ at 1e-13 because OpenMM adds the forces in fixed point, with a
resolution of 2^-32 kJ/mol/nm. In mixed precision OpenMM computes the other forces (field and pair force) in
single precision; the OpenCL platform also rounds the total force on a particle to single precision.

**Momentum conservation** (the system of `testMomentumConservation`: two particles at one node, 50 steps,
300 K): relative drift of the total momentum, measured with `getFluidFields()`:

| Platform and precision | explicit | centred |
|---|---|---|
| Reference | 4e-14 | 7e-14 |
| CUDA double and mixed | 7e-14 | 9e-14 |
| OpenCL double | 5e-14 | 1.4e-13 |
| OpenCL mixed | 7e-9 | 1.1e-8 |
| CUDA and OpenCL single | 1.5e-7 to 2.1e-7 | 1.4e-7 to 1.7e-7 |

**Speed.** With 110 coupled particles on a 30^3 lattice at 300 K and the removal of the fluid momentum at
every step, a step takes 61 us with CUDA in mixed precision (51 us in single, 55 us in double) and 64 us
with OpenCL in mixed precision, on an NVIDIA A100 with OpenMM 8.6.1. The CUDA library of the DragOpenMM
project takes 517 us for the same system. The centred drag costs 0 to 6% more (A100, OpenMM 8.6.1, 300 K,
removal at every step, 5000 steps after 200), in us per step:

| System | precision | CUDA explicit | CUDA centred | OpenCL explicit | OpenCL centred |
|---|---|---|---|---|---|
| 110 particles, 30^3 | mixed | 61.4 | 58.7 | 67.3 | 70.5 |
| | single | 53.7 | 53.8 | 62.9 | 62.8 |
| | double | 58.1 | 57.8 | 66.9 | 68.7 |
| 1000 particles, 30^3 | mixed | 78.6 | 81.1 | 86.8 | 89.6 |
| | single | 73.8 | 77.1 | 81.4 | 83.5 |
| | double | 78.2 | 80.5 | 87.1 | 91.3 |
| 8000 particles, 40^3 | mixed | 103.6 | 104.9 | 116.6 | 119.2 |
| | single | 88.0 | 93.1 | 101.7 | 104.7 |
| | double | 100.8 | 104.6 | 114.1 | 118.6 |

## Examples compared with the DragOpenMM plugin

The scripts of `examples/` are ports of the examples of the DragOpenMM plugin, with the Euler-Maruyama
coupling and `VerletIntegrator`. Run with the same parameters, the DragOpenMM plugin (reference library
in double precision, CUDA platform in double precision, OpenMM 8.2) and openmm-lbm (OpenMM 8.6.1, NVIDIA
A100) give the same trajectories:

- **Kick of a bead** (`particle/kick.py`, 100^3 nodes, 2000 steps, T = 0): velocities within 1e-13 of
  v0 and positions within 1e-11 of the distance travelled, with CUDA in double and mixed precision;
  5e-8 in single precision. With the parameters of the alanine bead (30^3 nodes, 100 steps) the
  Reference platform agrees within 7e-14.
- **Thermalization of a bead** (`particle/thermal.py`, 300 K, 20000 steps, same seed): velocities within
  3e-11 of their largest value with CUDA in double and mixed precision, random force included, because
  both plugins draw the same numbers from OpenMM's generator ([theory.md](theory.md), section 2);
  2e-6 in single precision and 5e-6 with OpenCL, whose generator rounds differently.
- **Bead in a uniform flow** (`particle/uniform_flow.py`, NVE, 400 steps): equal to the 6 printed digits.
- **Kick of the peptide GRGDSPYS** (`cocomo/kick.py --preset peptide`, COCOMO2 forces, 2000 steps,
  T = 0, the density of the original script): velocity of the centre of mass within 1e-13 of v0, with
  CUDA in double precision.

The reference outputs of the old examples (`*.LBLatticeOn.dat`) came from the Langevin scheme of the old
plugin, in which OpenMM's `LangevinIntegrator` supplies friction and noise (first step v1/v0 =
exp(-gamma dt)); they were regenerated with the Euler-Maruyama scheme for this comparison
(v1/v0 = 1 - gamma dt).
- **Disordered protein rlp** (`cocomo/diffusion.py --preset rlp`, 166 beads, friction 100/ps, dt 2 fs,
  298 K): full-step temperature 259.1 K and half-step 293.6 K over 10 ns; the DragOpenMM plugin with the
  same parameters gives 259.9 K and 294.7 K. The 13% deficit is that of the model without thermal
  fluctuations of the fluid, large at this friction (T2 and T6 below).
- **COCOMO2 model** (`cocomo/cocomo2.py`), written from the article: on SOD1 (`cocomo/data/sod1.pdb`,
  box 15 nm, Reference platform) the energy of every term (bonds, angles, electrostatics, short range,
  cation-pi, pi-pi, elastic network with 478 bonds) is identical to that of the COCOMO2 script of the
  DragOpenMM runs, and the forces agree within 1e-16 of the largest force, at the reference structure
  and with the beads displaced at random by 0.03 nm.
- **Diffusion of SOD1** (`cocomo/diffusion.py --preset sod1`, three runs of 200 ns with seeds 1, 2, 3,
  CUDA in mixed precision): the apparent diffusion coefficient of the centre of mass, MSD(t)/(6t), is
  2.34, 2.41, 2.42, 2.49, 2.64, 2.83 and 2.86 A^2/ns (mean of the three runs; standard error 0.01 to
  0.03 up to 2 ns, 0.06 to 0.19 at longer lags) at lag times of 0.1, 1, 2, 3, 5, 10 and 18 ns. Three runs of the DragOpenMM plugin with the same parameters and the Euler-Maruyama coupling
  give 2.3, 2.4, 2.3, 2.3, 2.4, 2.4 and 2.6 A^2/ns, with a spread between runs of 1.9 to 3.4 A^2/ns at
  18 ns. Both stay close to kT/(M gamma) = 2.26 A^2/ns, as expected without thermal fluctuations of
  the fluid. At the longest lags the two sets differ by about 1.5 standard errors. The full-step
  temperature is 293-294 K at 298 K.

## Kinetic energy budget (`python/tests/TestEnergyBudget.py`, Reference)

The kinetic energy plus the viscous and drag dissipation of [theory.md](theory.md), section 2, is
constant only to O(Ma^2, Kn^2). The test checks the size and scaling of the residual (E + dissipated)/E0 - 1:

| Case | Residual |
|---|---|
| shear wave, 16 nodes per wavelength, tau = 1.1 | -3.06% |
| shear wave, 32 nodes per wavelength, tau = 1.1 | -0.77% (ratio 3.97: O(k^2)) |
| bead kicked at Mach 0.035, 12^3 nodes, NVE | -4.448% |
| bead kicked at Mach 0.10, same system | -4.445% (independent of the Mach number) |

Tolerances: below 1% for the finer shear wave and a ratio between 3 and 5; below 6% for the kick and a
difference below 1e-3 between the two Mach numbers.

## Stochastic tests T2, T6, T7 (CUDA, mixed precision, NVIDIA A100)

The stochastic tests of the reference campaign of the DragOpenMM plugin, run with openmm-lbm with the same
cases, seeds, protocol and analysis (zero initial total momentum in T6). Both plugins draw the random
numbers from OpenMM's generator in the same order, so the agreement is closer than the statistical error.
The fluid has no thermal fluctuations of its own: the diffusion coefficient is kT/(m gamma) and the
Einstein relation with the hydrodynamic mobility is not satisfied, as in the reference.

**T2, equipartition** (100 free particles of 100 Da, 300 K, L = 8 nm, viscosity fixed, 200 ps). Mean
temperatures in K, openmm-lbm / reference; statistical error about 0.5 K.

| dt (ps) | tau | gamma (1/ps) | half step | full step |
|---|---|---|---|---|
| 0.005 | 0.80 | 1 | 299.7 / 299.7 | 298.9 / 298.9 |
| 0.005 | 0.80 | 5 | 299.0 / 299.2 | 295.4 / 295.3 |
| 0.005 | 0.80 | 10 | 300.5 / 300.1 | 292.9 / 292.6 |
| 0.01 | 1.10 | 1 | 303.6 / 303.6 | 302.0 / 302.1 |
| 0.01 | 1.10 | 5 | 306.5 / 306.6 | 298.9 / 298.8 |
| 0.01 | 1.10 | 10 | 310.6 / 311.0 | 294.8 / 295.0 |
| 0.02 | 1.70 | 1 | 302.8 / 302.5 | 299.8 / 299.6 |
| 0.02 | 1.70 | 5 | 315.6 / 316.1 | 300.1 / 300.0 |
| 0.02 | 1.70 | 10 | 333.1 / 333.1 | 299.6 / 299.8 |

**T6, diffusion** (64 particles of 1000 Da, 300 K, 5 ns), openmm-lbm / reference:

| gamma (1/ps) | L (nm) | D/(kT/m gamma), MSD | D/(kT/m gamma), Green-Kubo | full-step T (K) |
|---|---|---|---|---|
| 1 | 16 | 1.038 / 1.038 | 1.010 / 1.010 | 292.4 / 292.4 |
| 5 | 16 | 0.984 / 0.984 | 0.984 / 0.985 | 270.4 / 270.4 |
| 10 | 16 | 1.010 / 1.011 | 1.007 / 1.004 | 254.0 / 254.0 |
| 5 | 8 | 0.979 / 0.983 | 0.979 / 0.980 | 271.4 / 271.4 |

**T7, velocity autocorrelation against the kick response** (gamma = 10, L = 16 nm): the ratio of the
normalized VACF at half steps to the kick response minus its plateau is 0.93, 0.90, 0.85, 0.81, 0.69 and
0.50 at 0.05, 0.1, 0.2, 0.3, 0.5 and 1 ps (reference: 0.93, 0.90, 0.85, 0.81, 0.70, 0.47). With thermal
fluctuations of the fluid the two curves would coincide (fluctuation-dissipation theorem).

