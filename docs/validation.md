# Validation

Tests of openmm-lbm, what each one checks, its tolerance and the values measured. A value measured with an earlier
version, which is then named, holds for the current one as long as the code that it tests has not changed: the tests
of the suite check it at every change.
- **C++ tests**, run with `ctest`: `TestSerializeLBMForce` (the XML of the force, written in version 8, and the
  reading of versions 1 to 7); one executable per platform, `Test<Platform>LBMForce` in `platforms/*/tests`, which
  runs the tests shared by all the platforms in `tests/*.h` in each precision; and, with the CMake option
  `OPENMM_LBM_MPI`, `TestMPIReferenceLBMForce`, which `ctest` runs with two MPI ranks. The shared tests of
  `tests/TestLBMForce.h` (`runPlatformTests()`) check the parameters and their accessors (`testParameters`), that the
  coupling force is zero before the first step and the energy always zero (`testZeroForce`), the initial fluid
  (`testInitialFluidFields`), reading, changing and writing back the state of the fluid (`testFluidStateRoundTrip`)
  and the rejection of invalid setups (`testInvalidSetup`); the others are described below.
- **Python tests**, run with `pytest` in `python/tests`: `TestLBMForce.py` (the Python API, units and errors, and the
  GPU platforms against the Reference platform), `TestCheckpoint.py`, `TestVTKReporter.py`, `TestEnergyBudget.py`
  and `TestExamples.py`, which runs every script of `examples/` for a few steps.
- **MPI scripts**: `python/tests/mpi_decomposition.py` and `python/tests/mpi_abort.py` run under MPI, so pytest does
  not collect them ([domain decomposition](#domain-decomposition-pythontestsmpi_decompositionpy-mpi)).

## Fluid on its own (`tests/TestLBMFluid.h`, all platforms)

The fluid tests use no coupled particles. The lattice spacing is 0.5 nm, the time step 0.01 ps and the
density 602.2 Da/nm³; the viscosity is chosen to give the relaxation time $`\tau`$ of each test. The momentum
removal is off unless stated.
- `runFluidTests()` and `runWallTests()` (`testSolidNodeChecks`, `testPoiseuille`, `testWallConservation`,
  `testWallBalance` with bounce-back, and the tests of the regularized walls and of the open faces,
  [walls and open faces](#walls-and-open-faces-teststestlbmfluidh-all-platforms)) run on every platform, in the
  `single`, `mixed` and `double` precision modes.
- The tolerances below hold on the Reference platform and in `mixed` and `double` precision. In `single`
  precision (`getFluidTolerance()`) the tolerances of 1e-12 and below become 2e-6, the resolution of the
  stored type, and those of `testPoiseuille` become 5e-5: its steady state is the result of thousands of
  steps rounded in single precision, and on an A100 the profile and the force on the walls differ from
  the exact values by up to 1.1e-5 and 8e-6.

| Test | Setup | Check | Tolerance |
|---|---|---|---|
| `testUniformFlowIsSteady` | 6x5x4 nodes, $`\tau = 0.8`$, uniform velocity (0.03, -0.02, 0.01) in lattice units, 50 steps | populations unchanged | 1e-13 |
| `testFluidConservation` | 6x5x4 nodes, $`\tau = 0.7`$, populations perturbed by up to $`10^{-3}`$ (lattice units), 100 steps | total mass and momentum unchanged | 1e-13 (relative to the mass) |
| `testBodyForce` | 4x3x5 nodes, $`\tau = 0.9`$, uniform lattice density $`\rho_0`$ = 0.98, 1 or 1.02 at rest, body acceleration $`\mathbf g`$, 20 steps | momentum $`n\rho_0\mathbf g`$ per node (lattice units), density $`\rho_0`$, velocity $`(n + 1/2)\,\mathbf g\,\Delta t`$ | 1e-13 (momentum), 1e-12 nm/ps (velocity) |
| `testFluidMomentumRemoval` | 4x4x4 nodes, $`\tau = 1`$, initial uniform velocity, body force $`\mathbf F`$ per node, removal frequency 3 | total momentum $`((n-1) \bmod 3 + 1)\,\mathbf F`$ per node after $`n = 1,\dots,7`$ steps: removal on step indices 0, 3, 6, before the collision | 1e-13 |
| `testShearWaveViscosity` | 2x64x2 nodes, $`u_x = 10^{-3}\sin(2\pi y/64)`$ in lattice units, $`\tau`$ = 0.6, 1, 1.5; amplitude at steps 200 and 1200 | decay rate $`\nu k^2`$ with $`\nu = (\tau - 1/2)/3`$ | 2e-3 (relative) |
| `testMachNumberCheck` | 4x4x4 nodes, uniform flow at $`\mathrm{Ma} = 0.35`$, check every 10 steps | `getFluidMachNumber()` = 0.35; exception at step 10 with limit 0.3, none with the check off or with limit 0.5 | 1e-12 |
| `testLatticeParameters` | 4x6x8 nodes, $`\tau = 0.9`$ | `getLatticeParametersInContext()` returns $`\Delta x`$, $`\Delta t`$ and $`\tau`$ | 1e-12 |
| `testRelaxationTimeWarning` | $`\tau`$ = 0.503, 1, 2.2 | warning on stderr at Context creation only outside [0.505, 2] | exact |
| `testSolidNodeChecks` | solid node index out of range, repeated, or all nodes solid | exception at Context creation | exact |
| `testPoiseuille` | 2x12x2 nodes, solid plane $`j = 0`$, body force along x, $`\tau`$ = 0.7, 0.875, 1.2, $`4H^2/\nu`$ steps | steady profile equal to the exact solution of the scheme (`docs/theory.md`, solid nodes), zero density and velocity at the solid nodes | 1e-9 (relative to the maximum velocity) |
| `testWallConservation` | 6x5x4 nodes with a solid block of 8 nodes, initial uniform flow, without and with momentum removal | mass of the fluid conserved; after a removal step the momentum of the fluid is zero | 1e-13 |
| `testRestartFromCheckpoint` | removal of the fluid momentum every few steps, a checkpoint of OpenMM at a step that is not a multiple of the period, the fluid restored with `setFluidState()` | the restarted run against the uninterrupted one | identical bit for bit |
| `testUpdateParameters` | an existing Context | `updateParametersInContext()` changes the body acceleration, the removal of the fluid momentum and the Mach number check, and rejects a change of the viscosity | |
| `testQueriesDoNotAdvanceFluid` | 4x4x4 nodes, body force, 10 steps with and without `getState(Forces)`, `getState(Forces, Energy)` and `setVelocitiesToTemperature()` after each step | identical populations | exact |

**Body force at lattice densities different from 1.** The test checks the convention of the weakly compressible model:
the half force shifts the momentum, $`\mathbf u^* = (\mathbf j + \mathbf F/2)/\rho`$ with
$`\mathbf F = \rho\mathbf g`$, so that each step adds exactly $`\rho\mathbf g`$ to the momentum. Shifting the velocity
by $`\mathbf F/2`$ instead would add $`(1/(2\rho) + 1/2)\,\mathbf F`$.

**Viscosity.** Measured relative difference $`\nu_{\mathrm{measured}}/\nu - 1`$:

| $`\tau`$ | 64 nodes per wavelength | 32 nodes per wavelength |
|---|---|---|
| 0.6 | +5.1e-4 | +2.1e-3 |
| 1.0 | -1.7e-7 | -2.8e-6 |
| 1.5 | +8.0e-4 | +3.2e-3 |

The difference decreases as $`k^2`$ (by a factor of 4 when the wavelength doubles), as expected for a
second-order scheme. It vanishes at $`\tau = 1`$, where the collision relaxes the populations to equilibrium
in one step.

**Walls.** With halfway bounce-back the steady Poiseuille profile matches the exact solution of the scheme to 1e-12
for every $`\tau`$ tested (0.7 to 1.5, channel widths 11 and 19; at most 1.2e-13 relative to the centre-line velocity,
measured again for version 0.3.0). `testPoiseuille` checks $`\tau`$ = 0.7, 7/8 and 1.2 at width 11 to 1e-9. In the
steady state the force on the walls from the momentum exchange (`getWallForce()`) equals the body force on the fluid,
$`\mathbf g`$ times its mass, to 1e-8 (`testPoiseuille`). Its wall position differs from the halfway position by
$`(3 - 16\Lambda)/(12H)`$, $`\Lambda = (\tau - 1/2)/2`$; measured in a channel of width $`H = 23`$:

| $`\tau`$ | 0.55 | 0.6 | 0.7 | 0.8 | 0.875 | 0.9 | 1.0 | 1.2 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| wall shift (lattice units) | +0.00942 | +0.00797 | +0.00507 | +0.00217 | 0.00000 | -0.00072 | -0.00362 | -0.00942 | -0.01810 | -0.03256 |

**Bounce-back at the fluid nodes.** Since version 0.3.0 the halfway bounce-back is done by the fluid nodes next
to the walls, which take back the populations that they built for the directions towards solid nodes
(`docs/theory.md`, section 1), instead of a pass over the solid nodes that wrote into their fluid neighbours.
The two forms were compared on seven cases, each run for 300 steps from a perturbed fluid on a 6x5x4 lattice
with a body force: a channel, a slab one node thick with a 2x2x2 block (edges and corners), the same with six
coupled particles, with a fluctuating fluid and the explicit or the centred drag, with a `Velocity` and a
`Density` face, and with faces, fluctuations and particles together. Populations, density and velocity of the
fluid, positions and velocities of the particles and wall force are identical bit for bit in all 98 runs: the
Reference platform, CUDA and OpenCL in single, mixed and double precision, OpenMM 8.6.1 and 8.3.1, NVIDIA A100.
The cost per step does not change (CUDA, mixed precision: 88.6 against 86.3 us in a 64x34x64 channel, 170.7
against 171.0 us in a $`64^3`$ lattice with 13 % solid nodes in spheres).

Writing the returned population inside the collision kernel, without a separate pass, is the same arithmetic,
and it was 3 % faster in a channel and 8 % faster in a porous medium, but the compiler then rounded the collision
differently in some variants of the kernel: with CUDA in single precision the results
differed in the last bit of a float, and with CUDA in double and mixed precision a fluid with fluctuations at
zero temperature and coupled particles was no longer identical bit for bit to a fluid without fluctuations
(`testFluctuationsAtZeroTemperature`). The separate pass copies stored values and does not depend on the
rounding of the compiler, so it is used.

## Coupling of particles and fluid (`tests/TestLBMCoupling.h`, all platforms)

Particles of 100 Da in a fluid of 8x8x8 nodes ($`\Delta x`$ = 0.5 nm, $`\Delta t`$ = 0.01 ps, $`\tau = 0.8`$), without
removal of the fluid momentum. The tests run on every platform and precision mode. The tolerances below hold on the
Reference platform and in `double` precision; otherwise the tolerances below 2e-6 become 2e-6
(`getCouplingTolerance()`), because OpenMM handles the forces on the particles in single precision unless the platform
runs in double precision (`docs/theory.md`, section 2).

| Test | Checks | Tolerance |
|---|---|---|
| First step | a particle in a fluid at rest: $`v_1 = v_0(1 - \gamma\Delta t)`$; `getState()` then returns the force of the step, $`m(v_1 - v_0)/\Delta t`$ | 1e-14, 1e-12 relative |
| Full-step kinetic energy | with drag, random force and momentum removal, the kinetic energy of the State is that of the full-step velocities, because the force between steps is that of the next step (`testFullStepKineticEnergy`) | 1e-10 relative |
| Momentum conservation | particles and fluid with drag and random force, 50 steps, two particles at the same node, one crossing the periodic boundary, lattice densities 1, 0.98 and 1.02 | 1e-11 of the particle momentum (measured 7e-13: rounding of the sum over 19x512 populations) |
| Moving with the fluid | particle and fluid at the same velocity along x, y, z and a diagonal, two cells crossed | 1e-13 |
| Partial coupling | uncoupled particles keep their velocity and feel no force | rounding of OpenMM's Verlet |
| NVE scheme | at 300 K the first step is the deterministic drag, and two runs with different seeds are identical (`testNVEScheme`) | 1e-14, bitwise |
| Force evaluations and seeds | `getState()` before and after every step does not change the trajectory; two runs with seed 0 differ | bitwise |
| Walls | a coupled particle reaching a solid node: $`v_2 = -v_0(1 - \gamma\Delta t)^2`$; an uncoupled one passes | 1e-14 |
| Walls, direction | at a wall one node thick, a particle is reversed only if it moves into the wall, from either side | 1e-14 |
| Momentum with walls | fluid flowing against two walls, a particle reflected, drag and random force, 100 steps: particles + fluid + the sum of `getWallForce()` $`\Delta t`$ is constant | 1e-12 relative |
| Restart | checkpoint plus `setFluidState()` at step 13, removal every 5 steps, $`T = 0`$ | bitwise |
| Checkpoint with random force | OpenMM checkpoint plus `createCheckpoint()` at step 9, 300 K, a wall, removal every 4 steps, a force evaluation just before (random numbers already drawn), 14 more steps (`testCheckpointWithRandomForce`) | bitwise, also the wall force |
| Checkpoint refused | different number of coupled particles, or data that are not a checkpoint (`testCheckpointMismatch`) | exception |
| Warning | $`\text{friction}\cdot\Delta t > 1`$ with coupled particles and the explicit drag is reported at Context creation, not with the centred drag (`testFrictionWarning`) | |
| Equipartition | 100 free particles, $`\gamma\Delta t = 0.1`$, $`T`$ = 300 K: full-step temperature close to $`T`$, half-step temperature close to $`T/(1 - \gamma\Delta t/2)`$ | 10% (measured 4% below, from the missing fluid fluctuations) |
| Warning on $`\tau`$ | $`\tau > 1.7`$ with coupled particles and the explicit drag is reported at Context creation, not without coupled particles or with the centred drag (`testSelfMobilityWarning`) | |
| Repeated force evaluation | 8000 particles with a short-range `CustomNonbondedForce`, compressed into a cube of 2 nm so that the neighbor list overflows and the GPU platforms repeat the force evaluation of the step: the total momentum is conserved in that step (`testRepeatedForceEvaluation`; GPU platforms only) | 1e-11 relative (1.3e-2 on CUDA and OpenCL with the defect this test guards against, corrected in version 0.2.0) |

The tests that do not depend on the drag (full-step kinetic energy, momentum conservation, moving with the
fluid, partial coupling, force evaluations and seeds, momentum with walls, restart, checkpoint with random
force, equipartition) run again with the centred drag (next section).

## Centred drag (`tests/TestLBMCentered.h`, all platforms)

| Test | Checks | Tolerance |
|---|---|---|
| First step | a particle in a fluid at rest: $`v_1 = v_0(1 - a + a m/m_c)/(1 + a + a m/m_c)`$, $`a = \gamma\Delta t/2`$; then the force of the next step | 1e-14 (5e-14 on the GPU platforms, see below), 1e-12 |
| Shared node | three particles of 80, 120 and 150 Da at one node and a fourth alone, external forces, uniform flow and body acceleration: velocities after the first step against the direct solution of the linear system of the drag, and the momentum received by the fluid ($`\mathbf j + \mathbf G`$ at every node, $`\mathbf G = \mathbf F_{\mathrm{body}} - \mathbf S`$) | 1e-12, 1e-11 |
| Wall | a particle at a solid node moving out of the wall: $`v_1 = v_0(1 - a)/(1 + a)`$, and the wall receives the opposite of its coupling force | 1e-14 (5e-14 on the GPU platforms), 1e-12 |
| Large friction | $`\gamma\Delta t = 3`$, four particles at one node: velocities decay, total momentum conserved | 1e-11 |
| Repeated evaluation | 8000 particles with a short-range pair force compressed into a cube of 2 nm: a step whose force evaluation overflows the neighbor list gives the same velocities as the same step after the list has grown in an evaluation between steps, so the step uses the complete other forces (`testCenteredRepeatedEvaluation`; GPU platforms only) | 1e-12 relative to the largest velocity |
| Requirements | `LBMForce` not last, virtual sites, change of the drag in `updateParametersInContext()`, checkpoint loaded with the other drag | exception |
| Equipartition | as above, with a fluid 100 times denser: half-step temperature close to $`T`$, full-step close to $`T/(1 + \gamma\Delta t/2)`$ | 10% |

The tests of the previous section that do not depend on the drag run again with the centred drag, among them
the momentum conservation in a step whose force evaluation is repeated (`testRepeatedForceEvaluation`).
In single and mixed precision the tolerances are at least 2e-6, as for the explicit drag. In double
precision the GPU platforms add the forces in fixed point, with a resolution of $`2^{-32}`$ kJ/mol/nm, that is
2.3e-14 nm/ps of velocity in one step of 0.01 ps for a particle of 100 Da: measured 1.2e-14 to 1.9e-14 for
the centred first step on CUDA (the explicit one happens to be exact there), hence 5e-14 for the one-step
tests on these platforms.

**Python tests.** `test_centered_drag_sees_all_forces` checks, on every platform, that the drag of a step
uses all the other forces: four charged particles with a constant field and a `NonbondedForce` with PME
(whose reciprocal part the CUDA platform computes on a separate stream), each alone at its node, in a fluid
at rest at $`T = 0`$; the velocities after the first step agree within 1e-11 with the closed form computed with
the forces of the other force groups, read from the State. Without the PME forces in the drag the error would
be about 1e-3. `test_coupling_agrees_with_reference` compares both drags with the Reference platform (next
sections).

**Verifications done once, outside the test suite** (NVIDIA A100, OpenMM 8.6.1, during the development of
version 0.2.0).
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

**Fluctuation-dissipation balance without the response of the fluid** (Reference, fluid $`10^4`$ times denser, 100
particles of 100 Da, 300 K, 200000 steps, three seeds, error from 20 blocks). Ratio of the measured temperature to the
exact value of the discretization (explicit: $`T`$ at full steps, $`T/(1 - \gamma\Delta t/2)`$ at half steps; centred:
$`T`$ at half steps, $`T/(1 + \gamma\Delta t/2)`$ at full steps), mean of the three seeds:

| $`\gamma\Delta t`$ | explicit, full step | explicit, half step | centred, half step | centred, full step |
|---|---|---|---|---|
| 0.1 | 1.0002 | 1.0001 | 1.0002 | 1.0002 |
| 0.5 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| 1.5 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |

The statistical error of each mean is about 4e-4 at $`\gamma\Delta t = 0.1`$ and 1.5e-4 at 1.5. The kinetic energy
that OpenMM reports gives the full-step values. The same runs on CUDA in double precision (one seed, NVIDIA A100) give
ratios between 1.0000 and 1.0003 for both drags, all three values of $`\gamma\Delta t`$ and the four temperatures,
within their statistical error of 2e-4 to 7e-4.

**Temperature with the fluid** and **self-mobility $`y(\tau)`$** of the two drags: `docs/theory.md`, section 2.

**Temperature** (Reference, 200 free beads of 100 Da, $`16^3`$ nodes, $`\tau = 1.10`$, $`\gamma\Delta t = 0.1`$, $`T`$
= 300 K, 20000 steps): 295.8 ± 0.4 K from full-step velocities, 311.7 ± 0.4 K from half-step velocities
($`T/(1 - \gamma\Delta t/2)`$ = 315.8 K). The kinetic energy that OpenMM reports is that of the full step: on 200
samples of the same system (seed 7) it gives 293.98 K, the same as the full-step velocities within 4e-12 K
(`testFullStepKineticEnergy` checks it on every platform; `docs/theory.md`, section 2). It needs the coupling
forces between steps to be those of the next step, as for every force of OpenMM: with those of the step just done it
read 359 K.

## Interpolation stencils (`tests/TestLBMStencils.h`, Reference platform)

In development for version 0.5.0 (`docs/theory.md`, section 9). The tests run on every platform and precision, with
the tolerances of the coupling tests in mixed and single precision, except `testStencilWeights` and
`testStencilCanonicalTemperature` (Reference platform); in single precision the conjugate gradients of the centred drag
stop at 1e-5 instead of 1e-13. `test_coupling_agrees_with_reference` (Python, section GPU platforms against the Reference
platform: coupled particles) also runs the three stencils with both drags, without and with walls.
- `testStencilWeights`: along an axis, at 1001 positions in a cell, the weights sum to one and have a zero first
  moment (1e-14); the second moment is 0 to 1/4 (trilinear), 1/4 to 1/3 (three-point) and 0 (Keys); the squares of the
  three-point weights sum to 1/2.
- `testStencilLinearField`: fluid at equilibrium with a velocity and a density that vary linearly in space: the
  first step gives a particle at rest $`\gamma\Delta t\,\mathbf u(\mathbf X)`$ with the linear field at the particle,
  to 1e-12, with every stencil.
- `testStencilAtNode`: the trilinear kernel and Keys at a node give the first step of the nearest node, bit for bit,
  for the particle and the fluid.
- `testStencilMomentumConservation`: total momentum conserved to 1e-11 over 50 steps with the random force,
  overlapping stencils and a particle crossing the periodic boundary, and with the solid plane $`j = 0`$ covered by
  the stencils, including the momentum given to the wall; with both drags.
- `testCenteredStencilSolve`: the forces of the centred drag (conjugate gradients) against a solution by Gaussian
  elimination of the same system, built from the fluid state, positions, velocities and the other forces: three
  particles with overlapping stencils near a wall, one isolated, masses 100 to 1000 Da, to 1e-10.
- `testStencilTranslation`: particles and fluid moved by one node along each axis move the run by one node, to 1e-12.
- `testStencilErrors`: a different stencil in `updateParametersInContext()` or in a checkpoint, an unknown stencil
  and open faces stop with an error.
- `testStencilWarnings`: with the explicit drag at $`\tau = 1.75`$ the warning on the self-mobility is printed with the
  trilinear and Keys stencils, not with the three-point one; with the fluctuating fluid the warning on the heating of
  the explicit drag gives 1.968%, 0.830% and 3.586% for a particle of 100 Da with $`\gamma\Delta t = 0.1`$ (the
  6.64% of the nearest node times the self weight averaged over a cell, 8/27, 1/8 and $`(57/70)^3`$).
- `testStencilCanonicalTemperature`: centred drag, fluctuating fluid, $`4^3`$ nodes, $`\tau = 1.1`$,
  $`\gamma\Delta t = 0.1`$, two particles of 100 and 1000 Da with overlapping stencils held at their positions (set
  again before every step) for 40000 steps, with the nearest node and every stencil: the half-step temperature is that
  of the canonical ensemble with fixed total momentum, $`m\langle v^2\rangle/3 = k_BT\,(1 - m/M)`$ with $`M`$ the
  mass of particles and fluid, within 5% (statistical error 1.3%; the explicit drag or the fluid without
  fluctuations give 9% to 48% off).
- The nearest node is unchanged: 64 arrays (fluid state, positions, velocities, forces) of 16 runs with both drags,
  both coupling schemes, with and without fluid fluctuations, with bounce-back and regularized walls, identical bit
  for bit to version 0.4.0.

**Self-mobility.** Reference platform, $`16^3`$ nodes, $`\Delta x`$ = 0.5 nm, $`\Delta t`$ = 0.01 ps, $`T = 0`$, one
particle of 1000 Da with $`\gamma`$ = 5/ps held almost at its place by a constant force along x (velocity
$`10^{-5}`$ nm/ps: it moves less than 0.3% of a cell), fluid momentum removed at every step (a uniform compensation,
as in the Fourier calculation of `docs/theory.md`, section 2), $`y = V/F - 1/(m\gamma)`$ averaged over the last 2000
of 4000 to 6500 steps. Positions in units of $`\Delta x`$ from a node; $`y\,\eta\,\Delta x`$ at $`\tau`$ = 0.62 / 1.1
/ 3.51:

| Stencil | Position | centred | explicit |
|---|---|---|---|
| nearest node | any | 0.0815 / 0.1443 / 0.3592 | 0.0615 / 0.0443 / -0.1425 |
| `Trilinear` | node | 0.0813 / 0.1442 / 0.3588 | 0.0614 / 0.0444 / -0.1422 |
| `Trilinear` | centre | 0.0387 / 0.0471 / 0.0787 | 0.0362 / 0.0346 / 0.0160 |
| `Trilinear` | (0.25, 0.1, 0.4) | 0.0480 / 0.0654 / 0.1280 | 0.0427 / 0.0388 / -0.0056 |
| `ThreePoint` | node | 0.0362 / 0.0445 / 0.0755 | 0.0337 / 0.0320 / 0.0128 |
| `ThreePoint` | centre | 0.0387 / 0.0471 / 0.0787 | 0.0362 / 0.0346 / 0.0160 |
| `ThreePoint` | (0.25, 0.1, 0.4) | 0.0372 / 0.0455 / 0.0767 | 0.0347 / 0.0330 / 0.0140 |
| `Keys` | node | 0.0815 / 0.1443 / 0.3592 | 0.0615 / 0.0443 / -0.1425 |
| `Keys` | centre | 0.0543 / 0.0715 / 0.1347 | 0.0490 / 0.0452 / 0.0028 |
| `Keys` | (0.25, 0.1, 0.4) | 0.0647 / 0.0975 / 0.2126 | 0.0543 / 0.0454 / -0.0488 |

All 60 values agree with the linearized calculation of `docs/theory.md`, section 9, within 0.0004, which checks the
interpolation, the spreading and both drags independently of the tests above. The nearest node agrees with section 2
within 0.002: there the particle moved across the cells at 0.05 nm/ps.

**Temperature with the particles held in place: exact comparison.** The linearized lattice of `docs/theory.md`,
section 9 (the fluid of section 1 linearized about rest, Guo forcing, the fluctuations of section 7, particles at fixed
positions with their random force) is a linear map with Gaussian noise, whose stationary covariance is computed
exactly, by summing $`A^nQA^{n\mathsf T}`$ with repeated squaring after removing the conserved modes (eigenvalue 1:
mass and total momentum) and the staggered modes that the particles do not reach (eigenvalue -1, for example the
momentum $`(-1)^{x+t}j_x`$ at the wavevector $`\pi`$, an exact invariant of the fluid on a periodic lattice with an even
number of nodes). The plugin runs on the same lattice of $`4^3`$ nodes ($`\Delta x`$ = 0.5 nm, $`\Delta t`$ = 0.01 ps,
$`\tau = 1.1`$, $`\gamma\Delta t = 0.1`$, 300 K) with the particles held at their positions (set again before every
step), $`4\times10^6`$ steps per case: 56 cases, every stencil at a node, at the centre of a cell and at a generic
point, $`m/m_c`$ = 1.33 and 13.3, both drags, with and without fluctuations of the fluid, and two particles with
overlapping stencils (three-point and Keys, $`\tau = 0.62`$). The 128 temperatures, at the half and at the full step,
relative to the canonical value $`k_BT\,(1 - m/M)`$:

| Drag, fluid | exact values | largest deviation |
|---|---|---|
| centred, fluctuating | 1 | 0.9 standard errors |
| centred, without fluctuations | 0.517 to 0.988 | 1.2 |
| explicit, fluctuating | 1.006 to 1.593 | 0.8 |
| explicit, without fluctuations | 0.668 to 0.994 | 1.2 |

All agree within 1.2 standard errors (about 0.13%, from block averages), and within 0.18%: the random force, its
spreading, the fluid noise and the solution of the centred drag are right for every stencil, also with overlapping
stencils. The same 8 cases with two particles, with the domain decomposition (two ranks, the stencils across the
blocks along x, y or z), agree within 1.7 standard errors (32 values). The centred drag in the fluctuating fluid gives
exactly the canonical temperature; the others do not, and on this small lattice their exact values differ from the
approximate formulas of linear response of `docs/theory.md` (sections 2 and 7) by up to 18% at $`\tau = 1.1`$ and more
at $`\tau = 0.62`$; with free particles on $`16^3`$ nodes the formulas are much closer (below).

**Stochastic tests with free particles (T2, T6, T7).** The tests of the section Stochastic tests T2, T6, T7 below,
with the same cases, protocol and analysis, on the Reference platform with every stencil, both drags, and the fluid
with and without fluctuations; T6 and T7 at $`L`$ = 8 nm ($`16^3`$ nodes, the lattice of the self-mobility above),
with $`\gamma`$ = 5 and 10/ps and runs of 2 ns. A particle that moves across the lattice visits every position of a
cell alike, so it sees the self weight and the self-mobility of the stencil averaged over a cell: the mean self weight
is $`(2/3)^3 = 8/27`$ for the trilinear kernel, 1/8 for the three-point kernel and $`(57/70)^3 = 0.540`$ for Keys, and
on this lattice at $`\tau = 1.1`$ the mean $`y_{\mathrm{centred}}`$ is 9.57e-5 (nearest node), 4.47e-5 (trilinear),
3.02e-5 (three-point) and 6.58e-5 ps/Da (Keys), and $`y_{\mathrm{explicit}}`$ 2.93e-5, 2.53e-5, 2.18e-5 and 2.99e-5
ps/Da (averages over $`8^3`$ positions of the linearized calculation). The predictions in parentheses are the formulas
of `docs/theory.md` (sections 2, 7 and 9) averaged in this way: with the fluctuating fluid
$`T\,\langle 1 + \gamma\Delta t\,mK/(2m_c(1 + \zeta y_{\mathrm{explicit}}))\rangle`$ for the explicit drag, and
without fluctuations $`T\,\langle 1/(1 + \zeta y)\rangle`$ with the $`y`$ of the drag.

T2 (100 particles of 100 Da, $`L`$ = 8 nm, 200 ps; four runs with different random numbers), mean temperatures in K.
The statistical error, from the scatter of the four runs, is about 1 K at $`\gamma`$ = 1/ps and 0.3 to 0.4 K at 5 and
10/ps. Fluctuating fluid:

| $`\Delta t`$ (ps) | $`\tau`$ | $`\gamma`$ (1/ps) | centred, half step: `Trilinear` / `ThreePoint` / `Keys` | explicit, full step: `Trilinear` / `ThreePoint` / `Keys` |
|---|---|---|---|---|
| 0.005 | 0.80 | 1 | 300.8 / 300.8 / 300.8 | 301.0 (300.3) / 300.9 (300.1) / 301.3 (300.5) |
| 0.005 | 0.80 | 5 | 299.9 / 299.9 / 299.9 | 301.2 (301.4) / 300.4 (300.6) / 302.4 (302.6) |
| 0.005 | 0.80 | 10 | 299.9 / 299.9 / 299.9 | 302.6 (302.8) / 301.0 (301.2) / 304.9 (305.2) |
| 0.01 | 1.10 | 1 | 301.4 / 301.4 / 301.4 | 302.0 (300.6) / 301.7 (300.2) / 302.5 (301.1) |
| 0.01 | 1.10 | 5 | 299.7 / 299.7 / 299.7 | 302.5 (302.9) / 300.9 (301.2) / 304.9 (305.3) |
| 0.01 | 1.10 | 10 | 299.9 / 299.8 / 299.9 | 305.4 (305.7) / 302.1 (302.4) / 310.1 (310.4) |
| 0.02 | 1.70 | 1 | 298.7 / 298.7 / 298.6 | 299.8 (301.2) / 299.2 (300.5) / 300.8 (302.1) |
| 0.02 | 1.70 | 5 | 299.8 / 299.8 / 299.8 | 305.4 (305.8) / 302.1 (302.5) / 310.3 (310.7) |
| 0.02 | 1.70 | 10 | 299.8 / 299.8 / 299.8 | 310.9 (311.5) / 304.2 (304.9) / 320.5 (321.2) |

With the centred drag the three stencils give almost the same half-step temperature, because they share the random
numbers, and it is the set one within the statistical error (at most 1.4 errors). With the explicit drag the particles
are much less hot than with the nearest node (310.9, 304.2 and 320.5 K at $`\Delta t`$ = 0.02 ps and
$`\gamma`$ = 10/ps, against 340.7 K), and the prediction holds within 1.5 K. Fluid without fluctuations:

| $`\Delta t`$ (ps) | $`\tau`$ | $`\gamma`$ (1/ps) | centred, half step: `Trilinear` / `ThreePoint` / `Keys` | explicit, full step: `Trilinear` / `ThreePoint` / `Keys` |
|---|---|---|---|---|
| 0.005 | 0.80 | 1 | 298.0 (298.9) / 298.3 (299.2) / 297.6 (298.4) | 298.3 (299.2) / 298.5 (299.3) / 298.1 (299.0) |
| 0.005 | 0.80 | 5 | 295.4 (294.5) / 296.9 (296.0) / 293.4 (292.3) | 296.8 (295.9) / 297.5 (296.6) / 295.9 (294.9) |
| 0.005 | 0.80 | 10 | 291.9 (289.2) / 294.7 (292.1) / 288.2 (285.0) | 294.5 (291.9) / 295.7 (293.3) / 292.9 (289.9) |
| 0.01 | 1.10 | 1 | 298.8 (298.7) / 299.2 (299.1) / 298.2 (298.0) | 299.4 (299.2) / 299.5 (299.3) / 299.3 (299.1) |
| 0.01 | 1.10 | 5 | 294.4 (293.4) / 296.4 (295.5) / 291.5 (290.5) | 297.1 (296.3) / 297.6 (296.8) / 296.6 (295.6) |
| 0.01 | 1.10 | 10 | 289.6 (287.2) / 293.5 (291.2) / 284.3 (281.5) | 295.0 (292.6) / 295.7 (293.6) / 294.0 (291.3) |
| 0.02 | 1.70 | 1 | 298.2 (298.3) / 298.9 (298.9) / 297.3 (297.4) | 299.4 (299.5) / 299.4 (299.4) / 299.5 (299.5) |
| 0.02 | 1.70 | 5 | 292.3 (291.7) / 295.4 (294.7) / 287.9 (287.3) | 297.7 (297.3) / 297.6 (297.1) / 297.9 (297.5) |
| 0.02 | 1.70 | 10 | 286.3 (283.8) / 292.4 (289.6) / 278.0 (275.7) | 296.8 (294.6) / 296.6 (294.3) / 297.2 (295.1) |

The predictions hold within 1.1%.

T6 (64 particles of 1000 Da, $`m/m_c`$ = 13.3, $`L`$ = 8 nm, $`\tau`$ = 1.1, 2 ns). The total momentum is zero, so the
exact half-step temperature of the centred drag is $`T\,(1 - m/M)`$ = 299.2 K, $`M`$ being the mass of particles and
fluid, and the predictions include the same factor. Fluctuating fluid:

| Stencil | $`\gamma`$ | $`D/(k_BT/m\gamma)`$, centred / explicit | $`D/(k_BT(1/\zeta + y_{\mathrm{centred}}))`$, centred / explicit | half-step $`T`$, centred (exact 299.2) | full-step $`T`$, explicit (prediction) |
|---|---|---|---|---|---|
| nearest node | 5 | 1.486 / 1.488 | 1.005 / 1.007 | 298.6 | 388.8 (385.9) |
| nearest node | 10 | 1.976 / 1.954 | 1.010 / 0.998 | 299.3 | 467.0 (453.0) |
| `Trilinear` | 5 | 1.222 / 1.220 | 0.999 / 0.997 | 299.3 | 325.2 (325.0) |
| `Trilinear` | 10 | 1.455 / 1.451 | 1.005 / 1.003 | 299.2 | 346.6 (345.5) |
| `ThreePoint` | 5 | 1.155 / 1.156 | 1.004 / 1.004 | 299.3 | 310.2 (310.4) |
| `ThreePoint` | 10 | 1.334 / 1.334 | 1.025 / 1.025 | 299.3 | 319.5 (319.6) |
| `Keys` | 5 | 1.353 / 1.322 | 1.018 / 0.994 | 299.2 | 346.5 (345.9) |
| `Keys` | 10 | 1.654 / 1.702 | 0.998 / 1.026 | 299.2 | 385.5 (381.8) |

The centred drag has the exact temperature with every stencil. The diffusion coefficient is the same for both drags
within 3%, and the Einstein relation holds with the self-mobility of the centred drag of the stencil, averaged over the
cell, within 2.6%, inside the statistical error of $`D`$ in a run of 2 ns (about 4%, from the number of independent
intervals of the fit; the stencils share the random numbers, so their errors are correlated): the stencils reduce the
hydrodynamic part of $`D`$ as they reduce the self-mobility. The explicit drag is too hot by the predicted amount, within 1% with the
stencils (the nearest node is 3% above it at $`\gamma`$ = 10/ps, as at $`L`$ = 16 nm below). Fluid without
fluctuations:

| Stencil | $`\gamma`$ | $`D/(k_BT/m\gamma)`$, centred / explicit | half-step $`T`$, centred (prediction) | full-step $`T`$, explicit (prediction) |
|---|---|---|---|---|
| nearest node | 5 | 0.940 / 0.941 | 208.3 (202.4) | 271.2 (261.0) |
| nearest node | 10 | 0.933 / 0.934 | 161.6 (152.9) | 255.6 (231.5) |
| `Trilinear` | 5 | 0.940 / 0.941 | 252.6 (244.8) | 274.6 (265.7) |
| `Trilinear` | 10 | 0.933 / 0.933 | 222.9 (207.5) | 259.6 (238.9) |
| `ThreePoint` | 5 | 0.941 / 0.941 | 268.5 (260.0) | 278.1 (269.7) |
| `ThreePoint` | 10 | 0.933 / 0.933 | 248.8 (229.9) | 265.1 (245.5) |
| `Keys` | 5 | 0.940 / 0.941 | 232.9 (225.4) | 269.9 (260.3) |
| `Keys` | 10 | 0.933 / 0.933 | 193.9 (181.1) | 252.4 (230.4) |

Without fluctuations of the fluid $`D = k_BT/(m\gamma)`$ within the statistical error, the same for every stencil and
drag, which share the random numbers. The half-step temperature of the centred drag is much closer to the set one with
the stencils than with the nearest node (268.5 K with the three-point kernel at $`\gamma`$ = 5/ps, against 208.3 K),
and the predictions of linear response are 3% to 10% below the measured temperatures, as for the nearest node.

T7, ratio of the normalized velocity autocorrelation at half steps (T6 at $`\gamma`$ = 10/ps, fluctuating fluid) to
the response to a kick minus its plateau. The response is computed at $`T = 0`$ in the starting configuration of T6,
with the same stencil and drag, the kicked particle being placed at the 27 points
$`((i + 1/2)/3, (j + 1/2)/3, (k + 1/2)/3)\,\Delta x`$ of a cell, and averaged:

| Stencil | drag | 0.05 ps | 0.1 ps | 0.2 ps | 0.3 ps | 0.5 ps |
|---|---|---|---|---|---|---|
| nearest node | centred | 1.002 | 1.001 | 0.998 | 0.995 | 0.995 |
| `Trilinear` | centred | 1.007 | 1.010 | 1.015 | 1.019 | 1.024 |
| `ThreePoint` | centred | 1.002 | 1.001 | 0.999 | 0.997 | 0.978 |
| `Keys` | centred | 1.003 | 1.002 | 1.000 | 0.999 | 0.998 |
| nearest node | explicit | 0.974 | 0.960 | 0.940 | 0.925 | 0.878 |
| `Trilinear` | explicit | 0.993 | 0.987 | 0.978 | 0.968 | 0.935 |
| `ThreePoint` | explicit | 0.998 | 0.994 | 0.985 | 0.978 | 0.944 |
| `Keys` | explicit | 0.985 | 0.976 | 0.959 | 0.946 | 0.938 |

With the centred drag and the fluctuating fluid the fluctuation-dissipation theorem holds for the dynamics with every
stencil, within 2.5%; at 1 ps the autocorrelation is 0.2% to 1% of its initial value and the ratio is dominated by the
statistical error. With the trilinear kernel and Keys the response depends on where the particle is in the cell:
kicked within $`0.2\,\Delta x`$ of a node, where these kernels approach the nearest node, the particle keeps its
velocity longer, and the ratio of the autocorrelation to that response falls to 0.69 (trilinear) and 0.75 (Keys) at 0.5 ps.

## Fluctuating fluid (`tests/TestLBMFluctuations.h`, all platforms)

The platforms draw different random numbers (the generator of the force on the Reference platform, OpenMM's on the
others), so they are compared through the statistics; the tolerances of mass and momentum follow the precision (1e-15
and 1e-13 in double and mixed precision, 2e-6 and 1e-5 in single). $`\mu = 3k_BT`$ in lattice units (`docs/theory.md`,
section 7); $`T`$ = 300 K, $`\Delta x`$ = 0.5 nm, $`\Delta t`$ = 0.01 ps and the density of water give $`k_BT`$ =
1.3e-5.

| Test | Checks | Tolerance |
|---|---|---|
| Basis | the polynomials $`e_k`$ of the test, written independently of the plugin, are orthogonal with the norms $`b_k`$ | 1e-15 |
| Single node | a 1x1x1 lattice streams every population back to its node, so the state is the post-collision state: density and momentum unchanged at every step; over 20000 steps the moments $`k = 4,\dots,18`$ have the variance $`\mu b_k`$ and are uncorrelated, for $`\tau`$ = 1 and 0.8 | 1e-15; 0.05 (statistical error 0.01) |
| Equilibrium | fluid at rest, 8x8x8, 300 steps of transient, 300 samples every 5 steps: variances of density, momentum and moments $`k = 4,\dots,18`$ of a node equal to $`\mu\rho`$, $`\rho k_BT`$ and $`\mu\rho b_k`$, for $`\tau`$ = 0.8 and 2.5; total mass and momentum conserved | 0.05; 1e-13 |
| Velocity spectrum (`testVelocitySpectrum`) | fluid on 8x8x8 nodes, $`\tau = 1`$, 300 steps of transient, 400 samples every 5 steps: the velocity of `getFluidFields()` minus the mean velocity of the sample, Fourier transformed; the longitudinal and transverse parts of the spectrum equal $`k_BT/\rho`$ and $`2k_BT/\rho`$ in three bands of $`\lvert\mathbf k\rvert`$ ($`\lvert\mathbf m\rvert^2 \le 3`$, $`4 \dots 12`$, $`> 12`$, with $`\mathbf k = 2\pi\mathbf m/8`$), and the variance per node is $`(N - 1)/N\,k_BT/\rho`$ per component; at rest, in a uniform flow (0.05, 0.02, -0.03) and with a body force $`g = 10^{-5}`$ along x, whose mean velocity $`u_0 + gt`$ is checked to 2e-4 | 5% (spectrum), 3% (per node) |
| Zero temperature | with walls, body force and particles with the NVE scheme, the run with the fluctuations switched on at $`T = 0`$ equals the run without them | bitwise |
| Reproducibility | same seed: identical runs, also with force evaluations between steps; another seed: a different run | bitwise |
| Restart | with coupled particles at 300 K and seed 0, a run restarted from the checkpoints of OpenMM and of the force equals the uninterrupted run (fluid, velocities and force on the walls); also with a wall (regularized with the explicit and the centred drag, bounce-back with the centred drag), open faces along x (a `Velocity` inlet and a `Density` outlet) and a body force | bitwise |
| NVE | with the NVE scheme, particles at rest are set in motion by a fluctuating fluid, and stay at rest without fluctuations | exact |
| Parameters | `updateParametersInContext()` changes the temperature and refuses to switch the fluctuations; a checkpoint is refused by a Context with the fluctuations switched differently | exceptions |
| Warning | with fluid fluctuations a warning is printed only for the explicit drag with the EM scheme at $`T > 0`$, with the estimate $`\text{friction}\cdot\Delta t\,m/(2m_c)`$ of the heaviest coupled particle | exact |

**Measured** (OpenMM 8.6.1 and 8.3.1; Reference, and CUDA and OpenCL on an NVIDIA A100 in the three precisions): all
pass. OpenMM's buffer of random numbers is enlarged when the Context is created (`docs/theory.md`, section 7): enlarged
in the first step instead, with OpenMM 8.3.1 the restart test failed on CUDA and OpenCL in all precisions (the restarted
fluid was a different realization, with populations such as 5.4e-3 against 5.2e-4), because OpenMM's checkpoint reads
the buffer back with the size it has in the new Context; with 8.6.1 it passed. Cost of the fluctuations on the A100
(CUDA, mixed precision, version 0.3.0): 59.8 → 76.6, 168.9 → 240.4 and 1115.8 → 1590.4 us per step on $`32^3`$, $`64^3`$
and $`128^3`$ nodes. Without fluctuations, and with fluctuations at zero temperature, the fluid, positions and
velocities after 40 steps with walls, body force and six coupled particles (explicit and centred drag, EM and NVE) are
identical, bit for bit, to those of version 0.2.1.

### Equilibrium spectra (CUDA, NVIDIA A100)

The protocol of the article of the model (reference 17 of `docs/theory.md`), on a periodic fluid of $`64^3`$ nodes
with one particle that is not coupled, without removal of the fluid momentum, from rest. $`k_BT = 1/3000`$ in lattice
units, the value of the article ($`\Delta x`$ = 1 nm, the density of water, 300 K and $`\Delta t`$ = 0.284 ps give
it), and $`k_BT`$ = 1.3e-5 (water at $`\Delta x`$ = 0.5 nm and $`\Delta t`$ = 0.01 ps). After a transient of 2e4
steps, or $`5/(2\nu k_1^2)`$ steps if longer ($`k_1 = 2\pi/64`$, the slowest mode), 400 snapshots every 2500 steps.
Each snapshot gives, at every node, the density, the momentum, the non-equilibrium stress
$`a^{(2)}_{ab} = \sum_q (c_{qa} c_{qb} - \delta_{ab}/3)\,(f_q - f_q^{\mathrm{eq}}(\rho, \mathbf u))`$ (the moments
$`k = 4,\dots,9`$) and the nine non-equilibrium ghost moments of the basis of `docs/theory.md`, section 7, from
`getFluidState()`, with the velocities of the populations read from `getFluidFields()`. For each field: the variance
per node and the spherically averaged spectrum
$`S_m(\lvert\mathbf k\rvert) = \langle\lvert m(\mathbf k)\rvert^2\rangle`$, with the normalized FFT
$`m(\mathbf k) = \sum_{\mathbf x} m(\mathbf x)\exp(-i\mathbf k\cdot\mathbf x)/\sqrt N`$ and shells of width 1 in the
integer wave vector. In the linear regime the populations of all nodes are independent Gaussian variables of variance
$`\mu\rho w_q`$: this state is stationary, because collision acts mode by mode with the variances of section 7 and
streaming is a permutation. So the theory is $`S_m(\lvert\mathbf k\rvert) = \mu\rho b_k`$ for every
$`\lvert\mathbf k\rvert`$, and the equilibration ratio is $`\mathrm{ER} = \text{measured}/\text{theory}`$. The four
observables of the article are $`\rho`$, $`\sum_a j_a`$, $`\sum_a a^{(2)}_{aa}`$ and $`\sum_{a<b} a^{(2)}_{ab}`$
(theory $`\mu`$, $`\mu`$, $`2\mu/3`$ and $`\mu/3`$). Shells 1 to 16 of $`64^3`$ cover the window
$`\lvert\mathbf k\rvert \in [4, 64]`$ of the $`256^3`$ lattice of the article. Statistical errors: about 1e-4 for the
ER per node (10 blocks), 0.02 to 0.04 for $`\mathrm{ER}(\lvert\mathbf k\rvert)`$ in the first shells and 1e-3 above
shell 10.

ER per node of the four observables, range of $`\mathrm{ER}(\lvert\mathbf k\rvert)`$ over shells 1 to 16, and range of
the ER per node of all 19 moments ($`k_BT = 1/3000`$):

| $`\tau`$ | $`\rho`$ | $`\sum_a j_a`$ | $`\sum_a a^{(2)}_{aa}`$ | $`\sum_{a<b} a^{(2)}_{ab}`$ | $`\mathrm{ER}(\lvert\mathbf k\rvert)\;\rho`$ | $`\mathrm{ER}(\lvert\mathbf k\rvert)\;\sum_a j_a`$ | $`\mathrm{ER}(\lvert\mathbf k\rvert)\;\sum_a a^{(2)}_{aa}`$ | $`\mathrm{ER}(\lvert\mathbf k\rvert)\;\sum_{a<b} a^{(2)}_{ab}`$ | $`\mathrm{ER}(\lvert\mathbf k\rvert)`$ ghosts | 19 moments, per node |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.505 | 1.0060 | 1.0215 | 1.0526 | 1.0408 | 1.02-1.12 | 0.96-1.04 | 1.08-1.15 | 1.03-1.26 | 0.988-1.031 | 1.0018-1.0407 |
| 0.51 | 1.0046 | 1.0101 | 1.0234 | 1.0193 | 1.01-1.06 | 1.00-1.02 | 1.03-1.10 | 1.02-1.11 | 0.986-1.029 | 1.0016-1.0193 |
| 0.55 | 1.0023 | 1.0027 | 1.0052 | 1.0045 | 0.99-1.02 | 1.00-1.01 | 0.99-1.02 | 1.00-1.02 | 0.966-1.015 | 1.0011-1.0047 |
| 0.7 | 1.0012 | 1.0009 | 1.0022 | 1.0017 | 1.00-1.03 | 1.00-1.02 | 0.98-1.01 | 0.99-1.01 | 0.966-1.016 | 1.0004-1.0019 |
| 1 | 1.0007 | 1.0004 | 1.0015 | 1.0012 | 1.00-1.04 | 1.00-1.02 | 0.99-1.01 | 0.98-1.01 | 0.965-1.016 | 1.0001-1.0013 |
| 1.5 | 1.0005 | 1.0001 | 1.0013 | 1.0011 | 0.99-1.05 | 1.00-1.02 | 0.99-1.01 | 0.98-1.01 | 0.964-1.016 | 0.9999-1.0011 |
| 2 | 1.0003 | 1.0000 | 1.0012 | 1.0011 | 0.99-1.06 | 1.00-1.01 | 1.00-1.01 | 0.98-1.01 | 0.964-1.016 | 0.9999-1.0011 |
| 5 | 1.0001 | 0.9999 | 1.0010 | 1.0010 | 0.99-1.05 | 0.99-1.01 | 1.00-1.01 | 0.99-1.00 | 0.966-1.017 | 0.9999-1.0011 |
| 10 | 1.0000 | 0.9999 | 1.0009 | 1.0009 | 0.99-1.04 | 0.99-1.01 | 1.00-1.01 | 0.99-1.00 | 0.967-1.018 | 0.9998-1.0011 |
| 50 | 0.9999 | 0.9999 | 1.0008 | 1.0008 | 1.00-1.02 | 0.99-1.01 | 1.00-1.01 | 0.99-1.01 | 0.968-1.019 | 0.9998-1.0010 |
| 100 | 0.9999 | 0.9999 | 1.0008 | 1.0008 | 1.00-1.02 | 0.99-1.00 | 1.00-1.01 | 0.99-1.02 | 0.968-1.018 | 0.9998-1.0010 |

The same with $`k_BT`$ = 1.3e-5:

| $`\tau`$ | $`\rho`$ | $`\sum_a j_a`$ | $`\sum_a a^{(2)}_{aa}`$ | $`\sum_{a<b} a^{(2)}_{ab}`$ | $`\mathrm{ER}(\lvert\mathbf k\rvert)\;\rho`$ | $`\mathrm{ER}(\lvert\mathbf k\rvert)\;\sum_a j_a`$ | $`\mathrm{ER}(\lvert\mathbf k\rvert)\;\sum_a a^{(2)}_{aa}`$ | $`\mathrm{ER}(\lvert\mathbf k\rvert)\;\sum_{a<b} a^{(2)}_{ab}`$ | $`\mathrm{ER}(\lvert\mathbf k\rvert)`$ ghosts | 19 moments, per node |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.501 | 1.0004 | 1.0028 | 1.0179 | 1.0114 | 0.95-1.04 | 0.94-1.01 | 1.03-1.04 | 1.00-1.06 | 0.966-1.019 | 0.9999-1.0117 |
| 0.505 | 1.0000 | 1.0007 | 1.0021 | 1.0017 | 0.99-1.00 | 0.96-1.00 | 1.00-1.01 | 1.00-1.03 | 0.988-1.030 | 0.9998-1.0017 |
| 0.62 | 1.0000 | 1.0002 | 1.0001 | 1.0000 | 0.99-1.01 | 1.00-1.01 | 0.97-1.01 | 0.99-1.01 | 0.966-1.015 | 0.9998-1.0003 |
| 1 | 1.0000 | 1.0000 | 0.9999 | 1.0002 | 1.00-1.04 | 1.00-1.02 | 0.99-1.01 | 0.98-1.01 | 0.964-1.016 | 0.9998-1.0002 |
| 3.5 | 0.9999 | 0.9999 | 1.0000 | 1.0001 | 0.99-1.05 | 0.99-1.01 | 0.99-1.01 | 0.99-1.00 | 0.965-1.017 | 0.9999-1.0002 |

- **$`\tau \ge 0.55`$**: the ER per node of every moment is within 0.5% of 1, within 0.12% for $`\tau \ge 2`$;
  $`\mathrm{ER}(\lvert\mathbf k\rvert)`$ is within 1.6% of 1 from shell 4 on (1.2% for $`\tau \ge 0.7`$), and in the
  first three shells within 6%, about two statistical errors (0.02 to 0.04: they have few wave vectors; the runs share
  the seed, so their deviations there are correlated from one $`\tau`$ to the next). The small excess (0.1% to 0.2% on
  the stress) is nonlinear: it grows with $`k_BT`$ and is absent (within 3e-4) at $`k_BT`$ = 1.3e-5. The article, on
  D3Q27, reports ER about 1.002 to 1.004 at the same $`\tau`$.
- **$`\tau \to 1/2`$**: the fluctuations of the stress and of the density are too large at long wavelengths (at
  $`\tau = 0.505`$ and $`k_BT = 1/3000`$, $`\mathrm{ER}(\lvert\mathbf k\rvert)`$ of the stress 1.15 to 1.26 in the
  first shells, decreasing to 1.03 (off-diagonal) and 1.08 (diagonal) at shell 16), and the ER per node grows (stress
  1.03 to 1.04 at $`\tau = 0.505`$). This excess is nonlinear as well: at $`k_BT`$ = 1.3e-5 and $`\tau = 0.505`$ the
  ER per node is within 0.25% of 1, and up to 1.8% above it at $`\tau = 0.501`$. The article, on D3Q27, finds 1.005 to
  1.016 at $`\tau = 0.505`$ and 1.04 to 1.07 at $`\tau = 0.5001`$ with $`k_BT = 1/3000`$. At $`\tau \le 0.501`$ with
  $`k_BT = 1/3000`$ the fluid of openmm-lbm is unstable (below).
- **Ghost moments**: ER per node within 0.25% of 1 at every $`\tau`$, within 0.1% for $`\tau \ge 0.7`$; their
  $`\mathrm{ER}(\lvert\mathbf k\rvert)`$ is within the statistical error (0.03 in the first shells) at every
  wavelength.
- **Platforms and precisions**: with the same seed CUDA in mixed, double and single precision and OpenCL draw the
  same random numbers, and their ER agree to four digits; the Reference platform (another generator, $`16^3`$ nodes,
  300 snapshots every 500 steps, statistical error about 1e-3) agrees within 0.3%:

  | $`\tau`$ | run | $`\rho`$ | $`\sum_a j_a`$ | $`\sum_a a^{(2)}_{aa}`$ | $`\sum_{a<b} a^{(2)}_{ab}`$ | 19 moments, per node |
  |---|---|---|---|---|---|---|
  | 0.505 | CUDA, mixed | 1.0060 | 1.0215 | 1.0526 | 1.0408 | 1.0018-1.0407 |
  | 0.505 | CUDA, mixed, from the Gaussian state | 1.0061 | 1.0216 | 1.0526 | 1.0406 | 1.0018-1.0407 |
  | 0.505 | CUDA, double | 1.0060 | 1.0215 | 1.0526 | 1.0408 | 1.0018-1.0407 |
  | 0.505 | OpenCL, mixed | 1.0060 | 1.0215 | 1.0526 | 1.0408 | 1.0018-1.0407 |
  | 0.505 | Reference, $`16^3`$ | 1.0063 | 1.0233 | 1.0530 | 1.0391 | 1.0003-1.0419 |
  | 0.7 | CUDA, mixed | 1.0012 | 1.0009 | 1.0022 | 1.0017 | 1.0004-1.0019 |
  | 0.7 | CUDA, double | 1.0012 | 1.0009 | 1.0022 | 1.0017 | 1.0004-1.0019 |
  | 0.7 | CUDA, single | 1.0012 | 1.0009 | 1.0022 | 1.0017 | 1.0004-1.0019 |
  | 0.7 | OpenCL, mixed | 1.0012 | 1.0009 | 1.0022 | 1.0017 | 1.0004-1.0019 |
  | 0.7 | Reference, $`16^3`$ | 0.9990 | 1.0000 | 1.0029 | 1.0003 | 0.9982-1.0023 |
  | 0.501 | CUDA, mixed, $`k_BT`$ = 1.3e-5 | 1.0004 | 1.0028 | 1.0179 | 1.0114 | 0.9999-1.0117 |
  | 0.501 | CUDA, mixed, $`k_BT`$ = 1.3e-5, from the Gaussian state | 1.0003 | 1.0032 | 1.0180 | 1.0113 | 1.0000-1.0114 |

  The state reached does not depend on the initial one: from rest, and from the stationary Gaussian state
  (rows "from the Gaussian state"), the ER agree within 1e-3.
- **Uniform flow** $`u_0 = 0.02`$ along x (Mach number 0.035, as in the article): the ER in the frame of the flow
  minus those at rest, with the same seed, per node and over shells 1 to 16:

  | $`\tau`$ | $`\rho`$ | $`\sum_a j_a`$ | $`\sum_a a^{(2)}_{aa}`$ | $`\sum_{a<b} a^{(2)}_{ab}`$ | $`\Delta\mathrm{ER}(\lvert\mathbf k\rvert)\;\rho`$ | $`\Delta\mathrm{ER}(\lvert\mathbf k\rvert)\;\sum_a j_a`$ | $`\Delta\mathrm{ER}(\lvert\mathbf k\rvert)\;\sum_a a^{(2)}_{aa}`$ | $`\Delta\mathrm{ER}(\lvert\mathbf k\rvert)\;\sum_{a<b} a^{(2)}_{ab}`$ | error of $`\Delta\mathrm{ER}(\lvert\mathbf k\rvert)`$ |
  |---|---|---|---|---|---|---|---|---|---|
  | 0.505 | +0.0017 | +0.0033 | +0.0101 | +0.0092 | -0.002..+0.027 | -0.011..+0.058 | -0.016..+0.013 | -0.016..+0.010 | 0.003 |
  | 1 | +0.0004 | +0.0005 | +0.0008 | +0.0007 | -0.009..+0.002 | -0.001..+0.001 | -0.000..+0.001 | -0.001..+0.001 | 0.003 |

  At $`\tau = 1`$ the flow changes the ER per node by less than 1e-3, and $`\mathrm{ER}(\lvert\mathbf k\rvert)`$
  within the statistical error; at $`\tau = 0.505`$ the ER per node by up to 1%.

### Velocity spectra (CUDA, OpenCL and Reference)

Release requirement: the spectrum of the velocity fluctuations, with the mean velocity of every sample subtracted even
when it should be zero. The velocity and the density are read with `getFluidFields()` (the velocity a user sees,
$`(\mathbf j + \mathbf F/2)/\rho`$), converted to lattice units, and the mean velocity of the sample is subtracted;
then the normalized 3D FFT,
$`\mathbf F(\mathbf k) = \sum_{\mathbf x}\mathbf u(\mathbf x)\exp(-i\mathbf k\cdot\mathbf x)/\sqrt N`$, shells of
width 1 in the integer wave vector, the longitudinal part $`\hat{\mathbf k}\cdot\mathbf F`$ and the transverse part.
Theory (equipartition): $`\langle\lvert F_a(\mathbf k)\rvert^2\rangle = k_BT/\rho`$ per component, so
$`\mathrm{ER}_L = S_L/k_BT`$, $`\mathrm{ER}_T = S_T/(2k_BT)`$, $`\mathrm{ER}_\rho = S_\rho/(3k_BT)`$, white in
$`\mathbf k`$. $`64^3`$ nodes, 400 samples every 2500 steps after 20000 (40000 at $`\tau = 0.55`$), seed 2026. The ER
per node of the velocity is not exactly 1: $`\mathbf u = \mathbf j/\rho`$ with independent density and momentum
fluctuations gives $`\langle u^2\rangle = k_BT\,(1 + 3\langle\delta\rho^2\rangle) = k_BT\,(1 + 9k_BT)`$, 1.003 at
$`k_BT = 1/3000`$ and 1.0001 at the $`k_BT`$ of water, which is what is measured.

| Run | ER per node $`u_x`$, $`u_y`$, $`u_z`$, $`\rho`$ | $`\mathrm{ER}_L(\lvert\mathbf k\rvert)`$, shells 1 - 16 (4 - 16) | $`\mathrm{ER}_T(\lvert\mathbf k\rvert)`$, shells 1 - 16 (4 - 16) | $`\mathrm{ER}_\rho(\lvert\mathbf k\rvert)`$, shells 1 - 16 (4 - 16) |
|---|---|---|---|---|
| $`\tau = 0.55`$ | 1.005, 1.005, 1.005, 1.002 | 1.004 - 1.046 (1.005 - 1.013) | 0.993 - 1.009 (1.004 - 1.009) | 0.993 - 1.016 (1.005 - 1.013) |
| $`\tau = 0.7`$ | 1.003, 1.003, 1.004, 1.001 | 1.000 - 1.008 (1.000 - 1.007) | 0.993 - 1.007 (0.999 - 1.007) | 0.999 - 1.029 (0.999 - 1.006) |
| $`\tau = 1`$ | 1.003, 1.003, 1.003, 1.001 | 0.999 - 1.010 (0.999 - 1.007) | 0.997 - 1.007 (0.997 - 1.007) | 0.997 - 1.039 (0.998 - 1.003) |
| $`\tau = 2`$ | 1.003, 1.003, 1.003, 1.000 | 1.000 - 1.013 (1.000 - 1.006) | 0.997 - 1.014 (0.997 - 1.007) | 0.988 - 1.057 (0.998 - 1.005) |
| $`\tau = 5`$ | 1.003, 1.003, 1.003, 1.000 | 1.000 - 1.012 (1.000 - 1.006) | 0.996 - 1.017 (0.996 - 1.007) | 0.990 - 1.047 (0.998 - 1.005) |
| $`\tau = 1`$, uniform flow $`u_0 = 0.05`$ along x | 1.009, 1.004, 1.005, 1.003 | 1.002 - 1.016 (1.002 - 1.011) | 1.000 - 1.010 (1.000 - 1.010) | 1.000 - 1.022 (1.001 - 1.007) |
| $`\tau = 1`$, body force $`g = 5\cdot 10^{-8}`$ along x (mean velocity 0 → 0.05) | 1.005, 1.004, 1.004, 1.002 | 1.000 - 1.014 (1.000 - 1.008) | 0.999 - 1.008 (0.999 - 1.008) | 0.999 - 1.024 (0.999 - 1.004) |
| $`\tau = 0.6`$, $`k_BT`$ of water (1.3e-5) | 1.000, 1.000, 1.000, 1.000 | 0.996 - 1.016 (0.996 - 1.002) | 0.990 - 1.002 (0.997 - 1.002) | 0.992 - 1.009 (0.996 - 1.003) |
| $`\tau = 1`$, $`k_BT`$ of water | 1.000, 1.000, 1.000, 1.000 | 0.995 - 1.006 (0.995 - 1.003) | 0.994 - 1.004 (0.994 - 1.004) | 0.996 - 1.038 (0.996 - 1.002) |

CUDA in mixed precision unless said otherwise, $`k_BT = 1/3000`$ in lattice units. The statistical error of the first
shell (6 wave vectors) is 1.5 - 2 %, of the shells from 4 on below 0.6 %. CUDA in single and double precision and OpenCL
in mixed precision draw the same random numbers and give the same values as CUDA mixed to the digits shown; the
Reference platform ($`16^3`$ nodes, its own generator) gives ER per node 1.002, 1.001, 1.003 and spectra within 2.5 %,
within its larger statistical error. The mean velocity is subtracted at every sample: in the run with the body force it
grows from 0 to 0.05 and the spectra are those of the fluid at rest. In the uniform flow and in the accelerated fluid
the ER per node of the velocity along the flow is 0.6 % and 0.2 % above that at rest, an effect of the second-order
equilibrium of D3Q19 at Mach 0.09, which grows as $`u^2`$. The test `testVelocitySpectrum` checks the same quantities on
$`8^3`$ nodes on every platform (Fluctuating fluid, above).

### Time correlations (CUDA, NVIDIA A100)

The decay of the thermal fluctuations at long wavelengths is that of hydrodynamics (Onsager's regression). On $`32^3`$
nodes at $`k_BT = 1/3000`$, from the stationary Gaussian state, the amplitudes of the density and of the momentum at
the wave vectors $`k = 2\pi m/32`$ along the three axes ($`m`$ = 1, 2) were recorded every 1 to 5 steps over 129 to
1542 decay times of the shear mode. The transverse momentum decays as $`\exp(-\nu k^2 t)`$ (linear fit of its
logarithm after 20 steps, or a fifth of the decay time if shorter, while the correlation is above 0.3; the first steps
contain a fast decay of 0.4% ($`m = 1`$) to 1.6% ($`m = 2`$) at $`\tau = 0.55`$ from the non-hydrodynamic moments);
the density as a damped sound wave, $`\exp(-\Gamma k^2 t)\,[\cos(Wt) + (\Gamma k^2/W)\sin(Wt)]`$ with
$`W^2 = c_s^2 k^2 - \Gamma^2 k^4`$, where $`\Gamma = (\tfrac43\nu + \nu_{\mathrm{bulk}})/2 = \nu`$ for the bulk
viscosity $`2\nu/3`$ of a single relaxation time of the stress. Errors from 10 blocks of the time series:

| $`\tau`$ | $`m`$ | decay times sampled | $`\nu_{\mathrm{fit}}/\nu`$: fluctuations / deterministic | $`c/c_s`$: fluctuations / deterministic | $`\Gamma/\nu`$: fluctuations / deterministic |
|---|---|---|---|---|---|
| 0.55 | 1 | 129 | 1.090 ± 0.055 / 1.003 | 0.9985 ± 0.0003 / 0.9990 | 1.025 ± 0.073 / 1.000 |
| 0.55 | 2 | 514 | 1.038 ± 0.030 / 1.010 | 0.9957 ± 0.0004 / 0.9958 | 1.006 ± 0.049 / 1.000 |
| 0.7 | 1 | 257 | 0.975 ± 0.046 / 1.001 | 1.0008 ± 0.0013 / 0.9994 | 0.989 ± 0.029 / 1.000 |
| 0.7 | 2 | 1028 | 1.033 ± 0.038 / 1.005 | 0.9977 ± 0.0012 / 0.9978 | 0.995 ± 0.028 / 1.002 |
| 1 | 1 | 257 | 0.938 ± 0.051 / 1.000 | 1.0078 ± 0.0024 / 1.0021 | 0.915 ± 0.031 / 1.003 |
| 1 | 2 | 1028 | 1.024 ± 0.014 / 1.000 | 1.0111 ± 0.0026 / 1.0086 | 1.010 ± 0.032 / 1.013 |
| 2 | 1 | 386 | 1.076 ± 0.027 / 1.013 | 1.0410 ± 0.0053 / 1.0282 | 0.998 ± 0.030 / 1.031 |
| 2 | 2 | 1542 | 1.050 ± 0.018 / 1.046 | 1.1183 ± 0.0077 / 1.1148 | 1.147 ± 0.019 / 1.144 |

The deterministic values come from a shear wave and a sound wave of amplitude 1e-4 started on the same lattice without
fluctuations (CUDA, double precision) and fitted in the same way: they contain the dispersion of the lattice at these
wave numbers (for example a sound speed 2.8% above $`c_s`$ at $`\tau = 2`$ and $`m = 1`$). The thermal fluctuations
relax as the deterministic hydrodynamics of the model, within 2.8 standard errors (most within 1.5): the shear
viscosity within 9% (errors 1.4% to 5.5%), the sound speed within 1.3% and the attenuation within 9%.

### Stability near tau = 1/2

On $`64^3`$ nodes over $`10^5`$ steps (CUDA, mixed precision; the same in double precision and on the Reference
platform at $`16^3`$): the fluctuating fluid exceeds the Mach number limit within a few thousand steps at

| $`k_BT`$ (lattice units) | unstable at $`\tau`$ | stable at $`\tau`$ |
|---|---|---|
| 1/3000 | 0.5001, 0.5002, 0.5005, 0.501 | 0.502, 0.503, 0.505 |
| 1e-4 | 0.5001, 0.5002, 0.5005 | 0.501, 0.502, 0.503, 0.505 |
| 1.3e-5 | 0.5001 | 0.5002, 0.5005, 0.501, 0.502, 0.503, 0.505 |

Without noise, from the same thermal initial state, only $`\tau`$ = 0.5001 and 0.5002 at $`k_BT = 1/3000`$ and 0.5001
at $`k_BT = 10^{-4}`$ are unstable, because the velocities decay. The explanation is in `docs/theory.md`, section 7
(stability near $`\tau = 1/2`$). The article of the model, on D3Q27, is stable down to $`\tau = 0.5001`$ at
$`k_BT = 1/3000`$.

## Walls and open faces (`tests/TestLBMFluid.h`, all platforms)

The tests of the regularized walls (`setWallScheme(Regularized)`) and of the open faces (`setFaceBoundary()`)
run on every platform, in the three precision modes. The tolerances below hold on the Reference platform and
in `mixed` and `double` precision (`testWallMomentumBalance`: in `double`; in `mixed` it has 2e-6, as the other
coupling tests). In `single` precision the tolerances of 1e-12 and below become 2e-6, those of
`testRegularizedPoiseuille` and `testCouette` 5e-5, as for `testPoiseuille`, the steadiness of
`testPressureDrivenDuct` 1e-4 and `testWallMomentumBalance` with regularized walls 1e-5. Theory:
`docs/theory.md`, section 1.

| Test | What it checks | Tolerance |
|---|---|---|
| `testRegularizedPoiseuille` ($`\tau`$ = 0.7, 1, 1.5) | channel between regularized walls on the solid nodes $`j = 0`$ and $`j = n_y`$, driven by $`g`$: $`u(y) = g/(2\nu)\,y\,(n_y - y) + 3g(\tau - 1)/(\tau - 1/2)`$ at every fluid node, uniform density, wall force $`= gM`$ | 1e-9 (wall force 1e-8) |
| `testWallBalance` (both wall schemes) | a plate one node thick and a block, a flow against them and a body force: mass conserved and $`\mathbf P(t + \Delta t) - \mathbf P(t) = (M\mathbf g - \mathbf F_{\mathrm{wall}})\,\Delta t`$ at every step | 1e-13, 1e-12 |
| `testWallMomentumBalance` (`tests/TestLBMCoupling.h`; both drags, regularized walls) | particles, fluid and walls: total momentum conserved | 1e-11 relative (1e-12 with bounce-back): rebuilding the boundary nodes adds rounding (up to 4e-6 in `single` precision) |
| `testCouette` ($`\tau`$ = 0.6, 1, 1.5) | ZMin `Velocity` at rest, ZMax `Velocity` $`U`$ along x: $`u_x = U(z + 1)/(n_z + 1)`$, the faces holding on the nodes beyond them, uniform density | 1e-9 |
| `testUniformFlowThroughFaces` | a uniform flow from a `Velocity` inlet to a `Density` outlet stays unchanged | 1e-13 |
| `testPressureDrivenDuct` | square duct between bounce-back walls, `Density` faces at 1.01 and 1: density at the node $`n_y/2`$ on the straight line between those of the faces, held at $`y = -1`$ and $`y = n_y`$, gradient in the middle between those of the lengths $`n_y + 1`$ and $`n_y - 3`$, velocity in the middle within 2 % of the incompressible duct solution for that gradient, steady to 1e-9 between two steps | see text |
| `testCheckpointBoundaries` | a checkpoint with regularized walls and `Density` faces loads in the same configuration and is refused with another wall scheme or other face types | exceptions |
| `testFluidStateRestartWithBoundaries` (bounce-back and regularized walls) | a wall, a `Velocity` inlet, a `Density` outlet and a body force: a run restarted with `getFluidState()`/`setFluidState()` at step 11 equals the uninterrupted run at step 30 | bitwise |
| `testFaceChecks` | one open face on an axis, open faces with momentum removal, fewer than 3 nodes on an open axis: errors | |
| Python `test_wall_scheme_and_faces` | API with units, serialization, Couette flow | 1e-10 |

**Measured** (NVIDIA A100, OpenMM 8.6.1 and 8.3.1): the C++ tests pass on the Reference platform and on CUDA and OpenCL
in the three precisions, and the Python tests pass. The GPU platforms against the Reference platform with regularized
walls and open faces: [below](#gpu-platforms-against-the-reference-platform-fluid-and-walls).

**Where the walls and the faces are** (Reference, Poiseuille flow driven by $`g`$, 12 fluid nodes). The wall of the
bounce-back scheme lies halfway between the fluid and the solid nodes, the walls of the regularized scheme and the
open faces on the solid nodes or on the nodes beyond the faces. Zero of the fitted parabola (the profile is a parabola
to 1e-15 in every case):

| $`\tau`$ | 0.6 | 0.8 | 0.875 | 1 | 1.5 | 2.5 |
|---|---|---|---|---|---|---|
| bounce-back, solid nodes at 0 and 13 | 0.515 | | 0.500 | 0.493 | 0.465 | |
| regularized walls, solid nodes at 0 and 13 | 0.062 | | 0.019 | 0.000 | -0.077 | |
| two `Velocity` faces at rest, nodes beyond the faces at -1 and 12 | -0.938 | -0.969 | | -1.000 | -1.077 | -1.227 |

Started from either of the two parabolas (zero on the solid nodes or halfway), the velocity of the first fluid
node goes to the profile of its scheme in about a thousand steps. A Couette flow between two `Velocity` faces is
linear to 1e-14 between the nodes beyond the faces.

**Accuracy of the two walls** (body-force driven channel, largest error over the profile relative to the centre-line
velocity; $`H`$ the distance between the walls: the width of the fluid for bounce-back, the distance between the solid
nodes for regularized walls). From the exact profiles, $`\lvert 8\tau - 7\rvert/(3H^2)`$ with bounce-back and
$`8\lvert\tau - 1\rvert/H^2`$ with regularized walls, relative to the parabola that vanishes on the walls; measured
with $`H = 13`$, relative to the computed centre-line velocity: bounce-back 3e-14 at $`\tau = 7/8`$, regularized walls
1.9e-2, 9.6e-3, 1e-14, 2.3e-2 and 6.7e-2 at $`\tau`$ = 0.6, 0.8, 1, 1.5 and 2.5. Both are second order; at the
$`\tau`$ of water (about 0.6) the bounce-back wall is about four times more accurate. The local regularized condition of
Latt (`docs/theory.md`, section 1), tried for the regularized walls and left out because of its fluctuations
(below), has $`8\lvert\tau - 1\rvert/(3H^2)`$, with the wall on the boundary nodes.

**Mass with regularized walls and open faces** (Reference, 8 x 12 x 8 nodes, a solid plane and a block,
$`\tau = 0.8`$, a decaying flow; relative change of the total mass):

| | deterministic, 10000 steps | fluctuating ($`k_BT = 1/3000`$), 1000 steps | 10000 steps |
|---|---|---|---|
| bounce-back | 3e-16 | 2e-16 | 2e-16 |
| regularized, $`\rho_b`$ of the boundary node | -1.8e-5 | -5.4e-2 | -4.9e-1 |
| regularized, $`\rho_b`$ from the mass balance (the plugin) | 2e-16 | 2e-16 | 2e-16 |

Between two `Velocity` faces at rest with a fluctuating fluid ($`8^3`$ nodes), the mean density fell to 0.82 in 3000
steps and 0.40 in 10000 with the density of the face node, and stays within 2e-4 of 1 with the mass balance. With
a `Velocity` inlet and a `Density` outlet it stays within 0.5 %, held by the outlet.

**Density faces as inlets** (Reference, channel of 10 x 32 x 2 nodes between bounce-back walls, the faces along y):

| inlet - outlet | $`\tau = 0.52`$ | 0.55 | 0.6 | 0.7 to 2 |
|---|---|---|---|---|
| `Density` 1.01 - `Density` 1.0, with the time filter | unstable | unstable | steady | steady |
| the same, without the time filter | | unstable | unstable | steady |
| `Density` 1.003 - `Density` 1.0, with the time filter | unstable | unstable | | |
| `Velocity` 0.03 or 0.05 - `Density` 1.0 | steady | steady | steady | |

"Unstable": the Mach number check stops the run within a few thousand steps. Without the filter, three other variants
(velocity along the face zero, no stress copied, velocity taken from the next node) made $`\tau = 0.6`$ steady but not
$`\tau = 0.55`$.

**Fluctuations next to the walls** (Reference, 12 x 14 x 12 nodes with the solid planes $`j = 0`$ and $`j = 13`$,
fluid fluctuations at $`k_BT = 1/3000`$ in lattice units, $`\tau = 0.8`$, 600 samples; ER = variance of the mode over
the plane / $`\mu\rho b_k`$, as in the equilibrium spectra above):

| plane | bounce-back: $`\rho`$, $`\mathbf j`$, stress, ghosts | regularized: $`\rho`$ | $`\mathbf j`$ along, $`\mathbf j`$ normal | stress | ghosts |
|---|---|---|---|---|---|
| $`y = 1`$ (next to the wall) | 0.990 - 1.011 | 0.998 | 0.961 - 0.970, 1.002 | 0.912 - 1.002 | 0.941 - 1.003 |
| $`y = 2`$ | 0.993 - 1.008 | 1.001 | 1.001 - 1.006, 0.995 | 0.996 - 1.012 | 0.989 - 1.005 |
| $`y = 3`$ | 0.989 - 1.005 | 0.996 | 0.997 - 1.008, 1.001 | 0.995 - 1.009 | 0.988 - 1.005 |
| $`y`$ = 4 to 9 | 0.987 - 1.016 | 0.996 - 1.010 | 0.994 - 1.010, 0.995 - 1.001 | 0.990 - 1.013 | 0.990 - 1.008 |

Bounce-back walls are in thermal equilibrium with the fluctuating fluid at every distance, within the
statistical error (about 1 %): bounce-back permutes the populations and keeps the Gaussian equilibrium state.
Next to regularized walls the fluctuations are at equilibrium from the second node on; on the first node the
momentum along the wall is 3 to 4 % low and some stress and ghost modes up to 9 % low. Latt's condition, tried and
left out, was much further from equilibrium: on the boundary node the momentum was zero (imposed)
and the density 0.777, and on the next node the density was 0.954 and the momentum normal to the wall 0.910.

**Velocity spectra next to the walls and the faces** (release requirement, script and protocol of the velocity spectra
above; ER of the velocity of `getFluidFields()` minus the mean velocity of each sample, per plane and as a function of
the wave number $`\lvert\mathbf k\rvert`$ along the plane, 2D FFT, $`k_BT = 1/3000`$, $`\tau = 0.8`$, 400 samples):
- Bounce-back walls, 64 x 34 x 64 nodes on CUDA (mixed precision) and 16 x 18 x 16 on the Reference platform:
  every plane, the first fluid nodes included, within 1.003 ± 0.005 per node on CUDA ($`1 + 9k_BT`$, as in the bulk)
  and 1 ± 0.01 on the Reference platform, and white in $`\lvert\mathbf k\rvert`$ within the statistical error.
- Regularized walls (16 x 18 x 16): on the first fluid node next to the wall the velocity normal to the wall has
  ER 1.004 per node and 0.985 - 1.019 at every wavelength along the wall, the velocity along the wall 0.97
  (0.95 - 1.04) and the density 1.002 (0.95 - 1.04); from the second node on every plane is within the statistical
  error. With Latt's condition, tried and left out, the deficit of the first fluid node depended on
  the wavelength: the normal velocity had ER 0.72 at the longest wavelength (16 nodes) and 0.97 at the shortest,
  the density 0.83 - 0.86 at the longest; imposing the velocity on the boundary node damped the long-wavelength
  fluctuations next to it.
- `Density` faces (16 x 16 x 16, faces YMin and YMax at the fluid density, a solid plane $`x = 0`$): on the nodes of the
  faces the velocity along the face fluctuates at equilibrium (0.96 - 0.99), the velocity across the face 51 % more than
  in equilibrium with bounce-back walls and 38 % more with regularized walls, and the density 0.43 and 0.57 of the
  equilibrium variance; the next plane is within 2 % and the following ones within the statistical error. With Latt's
  condition on the faces, the density and the velocity along the face were imposed on the face nodes (ER 0) and the
  velocity across it fluctuated 21 to 29 % more. Without the solid plane, with x and z periodic, the mean flow across
  the two faces has no restoring force (equal pressures, no friction): the fluctuations make it wander like a free
  Brownian particle, and the run stopped with a Mach number of 0.3 after 87000 steps (measured with Latt's condition on
  the faces). A duct with walls is stable (mean velocity below 1e-3), because the viscous friction at the walls damps
  the mean flow (`docs/theory.md`, section 1, Open faces).

**Duct driven by a difference of density** (8 x 16 x 8 nodes, bounce-back walls, `Density` faces at 1.01 and 1): the
flow is steady (1e-18 between two steps) with a uniform mass flux, the staggered mode at 1e-16. The density falls
linearly in the middle, but on the face nodes it overshoots the densities of the faces (1.0104 and 0.9996 at
$`\tau = 1`$): next to the faces the flow enters and leaves the duct, and the gradient in the middle is that of a
length of 14.6 nodes instead of the 17 between the nodes beyond the faces. With that gradient the largest error in the
middle cross-section relative to the incompressible solution is 0.9 % at $`\tau = 1`$ (15 % with the gradient of 17
nodes), and 1.3 % at $`\tau = 1.1`$ in the example of the user guide (gradient of a length of 7.3 nm for 8.5 nm
between the nodes beyond the faces). With Latt's condition on the faces, which imposed the densities on the
face nodes, the gradient was that of the faces, and the errors were 1.3 % at $`\tau = 0.6`$ and 0.5 % at $`\tau = 1`$
(10 x 32 x 10).

**Staggered mode.** With the faces of the plugin the staggered momentum is at 1e-16 in the duct above with the time
filter; without it, it decayed by itself at $`\tau = 1`$ (to 1e-12 in 3000 steps) and slowly at $`\tau = 0.6`$, and
the `Density` inlets were unstable at $`\tau = 0.6`$ (above). With Latt's condition on the faces and without the
time filter of the `Density` faces (`docs/theory.md`, section 1, Time filter of the Density faces), the duct between
bounce-back walls kept an oscillation of the velocity from one node to the next and from one step to the next, of 0.6
to 2.9 % of the velocity at $`\tau = 0.8`$ (at $`\tau`$ = 0.6 and 1 the mass flux through neighbouring cross sections
differed by 9 to 18 %): the staggered momentum $`\sum_y (-1)^{y+t} j_y`$, an exact invariant of the bulk (eigenvalue
-1 of the linearized step at $`k = \pi`$, for every $`\tau`$), was excited by the start and then kept constant (-0.047
in lattice units from step 2000 to 16000). With regularized walls, or with `Velocity` faces, it was at the level of
rounding. With that condition the time filter ($`\beta = 1/2`$) damped it to rounding (1e-16) in every combination of
walls and faces, with the same steady state; the alternative of taking the velocity of the `Density` face from the
next node (zero gradient, as in the outflow of Malaspinas) also damped it, but raised the error with regularized walls
to 8 % at $`\tau = 0.6`$.

## Checkpoints from Python (`python/tests/TestCheckpoint.py`, Reference)

| Test | Checks | Tolerance |
|---|---|---|
| `test_checkpoint_bytes` | `createCheckpoint()` returns bytes starting with the tag of the format, which `loadCheckpoint()` accepts; other data raise an exception | exact |
| `test_restart_is_exact` | eight beads at 300 K; a `Simulation` restarted from the file of `LBMCheckpointReporter` at step 20 equals the uninterrupted run at step 35 (positions, fluid, force on the walls), also when a `StateDataReporter` at the same step has drawn the random numbers of the next step; with a bounce-back wall and the momentum removal, and with a fluctuating fluid, a regularized wall and open faces along x (a `Velocity` inlet and a `Density` outlet) | bitwise |
| `test_save_and_load_functions` | `openmmlbm.saveCheckpoint()` and `loadCheckpoint()` restore time and force on the walls; a damaged file raises `ValueError` | exact |

## VTK output (`python/tests/TestVTKReporter.py`, Reference)

`openmmlbm.LBMVTKReporter` on 6x5x4 nodes of 0.5 nm with two solid nodes, a body force and four particles, three
of them coupled, one outside the box:

| Test | Checks | Tolerance |
|---|---|---|
| Fluid | each of the two `.vti` files, of the density and of the velocity, has the extent of the lattice, spacing $`\Delta x`$ and origin 0; its field equals that of `getFluidFields()`; the solid nodes are flagged; the field data array `units` lists the unit of each array | 1e-6 relative in single precision, exact in double |
| Particles | the `.vtp` file has the positions of the State wrapped into the box, its velocities, the masses, the indices, the coupled flags and one vertex per particle, and the array `units`; with `wrap=False` the position outside the box stays outside | as above |
| Series | the `.pvd` file lists the fluid and particle files of each report with the time in ps; with `append=True` a new reporter keeps the files already listed | exact |
| Parts | `fluid=False` and `particles=False` write only the other part, `density=False` and `velocity=False` only the other field of the fluid | exact |
| No effect on the run | a run with the reporter equals, bit for bit, the run without it | bitwise |

**Measured** (OpenMM 8.6.1): all pass. The files written by the test, with the density and the velocity in one `.vti`
file as version 0.3.0 wrote them, were also read with the VTK readers of ParaView 5.13 (`vtkXMLImageDataReader`,
`vtkXMLPolyDataReader`, the `.pvd` reader): dimensions, spacing, the coordinates of the nodes, the fields and the times
agree, within 5e-8 relative in single precision and exactly in double precision, and the readers return the strings of
the `units` array of the field data.

## Domain decomposition (`python/tests/mpi_decomposition.py`, MPI)

With `-DOPENMM_LBM_MPI=ON`, `ctest` also runs `TestMPIReferenceLBMForce` (`platforms/reference/tests/mpi/`) with two
ranks: on the Reference platform, with two domains along x, the fluid nodes of each domain, the state gathered on rank
0, the fields of each domain with the exchanged halo and four coupled particles (one reflected at the wall) must be
identical bit for bit to those of one domain. The script below runs the complete comparisons, on every platform.

The script, `python/tests/mpi_decomposition.py`, needs the plugin built with MPI and runs under MPI, so pytest does not
collect it:

```bash
mpirun -n 4 python python/tests/mpi_decomposition.py 2 2 1                                   # Reference platform
mpirun -n 4 python python/tests/mpi_decomposition.py 2 2 1 --platform CUDA --precision double --devices 4
mpirun -n 4 python python/tests/mpi_decomposition.py 2 2 1 OpenCL     # also rank 0 on Reference, the others on OpenCL
```

It prints one line per rank and case, and a line with `FAILED` for any difference. It runs each case on every rank
twice, with one domain and with the decomposition given on the command line, and compares the populations of the fluid
nodes that the rank owns (`docs/theory.md`, section 8). Lattice $`8 \times 6 \times 6`$, $`\tau = 0.8`$, a body force,
an initial velocity and populations perturbed by up to $`10^{-3}`$, 60 steps; cases: periodic; the solid plane $`j = 0`$
and a block of 8 solid nodes with bounce-back walls, with regularized walls, and with regularized walls and open faces
along x (a `Velocity` inlet and a `Density` outlet); periodic with the removal of the fluid momentum every third step.
Run with OpenMPI 4.1.6 on one node, plugin built with `-DOPENMM_LBM_MPI=ON`, OpenMM 8.6.1, decompositions
$`2 \times 1 \times 1`$, $`1 \times 2 \times 1`$, $`1 \times 1 \times 2`$, $`2 \times 2 \times 1`$,
$`4 \times 1 \times 1`$, $`1 \times 2 \times 2`$ and $`2 \times 2 \times 2`$:

| Case | Fluid nodes of each rank | Force on the walls (relative) |
|---|---|---|
| periodic, bounce-back, regularized walls, open faces | identical bit for bit on every rank and decomposition (104 of 104) | at most $`3.3 \cdot 10^{-13}`$ |
| removal of the fluid momentum | at most $`1.1 \cdot 10^{-19}`$ | |

The force on the walls and the removal differ by rounding because the ranks add their sums in another order. The
slots of the solid nodes are not compared: they hold what the fluid nodes pushed into them, on the rank of each
fluid node, and are not part of the state of the fluid.

**Fields and state of each domain.** The initial state of these cases is set from rank 0 (`getFluidState()` with
`gather=True`, perturbed as a function of the global index, then `setFluidState()` with `scatter=True`). After the
run every rank compares with one domain: the state of its domain (`getFluidState()`, `getLocalDomain()`) with the
same block of the lattice; on rank 0 the state gathered from all the ranks (`gather=True`) with the whole lattice;
the density and velocity of its domain with the halo (`getFluidFields()` with `halo=True` and the exchange of both
fields on) with the fields of one domain extended periodically, NaN beyond the open faces; and on rank 0 the fields
gathered from all the ranks. A sixth case, periodic, exchanges only the velocity, and the density of the halo must be
NaN; a seventh exchanges only the density, and the velocity of the halo must be NaN. Over the decompositions
$`2 \times 1 \times 1`$, $`1 \times 2 \times 1`$, $`1 \times 1 \times 2`$, $`2 \times 2 \times 1`$,
$`1 \times 2 \times 2`$ and $`2 \times 2 \times 2`$ (22 ranks): identical bit for bit in the four cases without the
removal of the fluid momentum and in the sixth (110 of 110); with the removal the state agrees to
$`1.1 \cdot 10^{-19}`$ and the fields to $`3.1 \cdot 10^{-17}`$ (density relative to $`\rho_0`$, velocity in nm/ps).
The seventh case is identical bit for bit on the Reference platform ($`2 \times 1 \times 1`$,
$`1 \times 2 \times 2`$), on OpenCL on the CPU and on CUDA and OpenCL with A100 GPUs (in double and single
precision). With one domain `test_local_fields_halo_and_gather` checks the same rules (pytest, with and without open
faces).

**Interpolation stencils** (in development for version 0.5.0; `docs/theory.md`, section 9). The particle case with
the walls, with the explicit and the centred drag and each of the three stencils (`Trilinear`, `ThreePoint`, `Keys`):
fluid and particles identical bit for bit to one domain on every rank, on the Reference platform with
$`2 \times 1 \times 1`$, $`1 \times 2 \times 2`$, $`2 \times 2 \times 1`$ and $`2 \times 2 \times 2`$ domains, and on
OpenCL on the CPU in double precision with $`2 \times 1 \times 1`$, $`2 \times 2 \times 1`$, $`1 \times 1 \times 4`$ and
$`2 \times 2 \times 2`$ and in single precision with $`2 \times 2 \times 1`$, and on four A100 GPUs with CUDA in double
($`2 \times 2 \times 1`$), mixed ($`1 \times 2 \times 2`$) and single precision ($`1 \times 1 \times 4`$) and OpenCL in
double precision ($`2 \times 1 \times 1`$) (the particles share nodes, cross the borders of the blocks and reach the
solid nodes with their stencils). With the fluctuating fluid and `Keys` the copies
of the particles stay identical on every rank.

**CUDA and OpenCL platforms.** All the cases of this section, the fluid ones with the exchange of the halo and the
coupled particles, on one node with four A100 GPUs, one GPU per rank, against one domain on the same platform and
precision, plugin built with `-DOPENMM_LBM_MPI=ON`, OpenMM 8.6.1 (`--platform`, `--precision` and `--devices` of the
script; on CUDA with `DeterministicForces`). CUDA in double precision with $`2 \times 1 \times 1`$,
$`1 \times 2 \times 1`$, $`1 \times 1 \times 2`$, $`2 \times 2 \times 1`$, $`1 \times 2 \times 2`$ and
$`4 \times 1 \times 1`$, in mixed and single precision with $`2 \times 1 \times 1`$ and $`2 \times 2 \times 1`$; OpenCL
in double precision with $`2 \times 1 \times 1`$ and $`1 \times 2 \times 2`$, in single precision with
$`2 \times 2 \times 1`$ (40 ranks in all):

| Precision | Fluid: state, gathered state, fields with the halo, gathered fields | Particles: fluid and particles | Removal of the fluid momentum | Force on the walls |
|---|---|---|---|---|
| double, mixed | identical bit for bit on every rank (150 of 150) | identical bit for bit (90 of 90) | at most $`1.0 \cdot 10^{-17}`$ | fluid cases at most $`1.3 \cdot 10^{-13}`$ relative; particles $`2.8 \cdot 10^{-16}`$ of the pressure force |
| single | identical bit for bit on every rank (50 of 50) | identical bit for bit (30 of 30) | at most $`4.4 \cdot 10^{-9}`$ | fluid cases at most $`1.8 \cdot 10^{-8}`$ relative; particles $`4.9 \cdot 10^{-13}`$ of the pressure force |

On two nodes with four A100 GPUs each (8 ranks, the populations between the nodes over InfiniBand): CUDA in double
precision with $`2 \times 2 \times 2`$ and $`8 \times 1 \times 1`$ (blocks one node thick), in single precision with
$`4 \times 2 \times 1`$, OpenCL in double precision with $`2 \times 2 \times 2`$ and in single precision with
$`4 \times 2 \times 1`$ (40 ranks): fluid identical bit for bit in 200 of 200, particles in 120 of 120, removal of the
momentum at most $`1.0 \cdot 10^{-17}`$ ($`2.9 \cdot 10^{-9}`$ in single precision), checks 144 of 144. They were
launched with `mpirun` of OpenMPI 4.1.6, which passes to the processes of the other node only the environment variables
named with `-x` (here `PATH`, `LD_LIBRARY_PATH` and, for OpenCL, `OCL_ICD_VENDORS`), and with UCX on one port of the
network (`UCX_NET_DEVICES`): with the four ports of the nodes the setup of UCX between the nodes stopped in
`MPI_Comm_split_type` ("endpoint reconfiguration not supported yet"), before any call of the plugin.

In single precision each rank sums in float the links of a solid node to its own fluid nodes, and one domain sums all
of them, so the force on the walls agrees to float rounding. With the fluctuating fluid the copies of the particles
stay identical on every rank (13 of 13 runs), and a fluctuating fluid without particles conserves the mass as one
domain does. The ranks draw independent random numbers: a fluctuating fluid without walls and without the initial
perturbation is invariant under translations apart from the noise, so the fluctuations of corresponding nodes of two
equal blocks along x would be identical if two ranks drew the same numbers; their correlation coefficient is between
-0.09 and 0.01 on every platform (the statistics of the fluctuations with the decomposition are in the next
subsection). The checks that must stop every rank together all do (150 of 150): copies of the particles that differ
($`10^{-12}`$ in the velocities, $`10^{-6}`$ in single precision, where the velocities are stored as float), an
`AndersenThermostat`, `DeterministicForces` off on CUDA, and `LBMForce.createCheckpoint()`, which holds the state of
one domain (the checkpoint files are below). Without the decomposition nothing changes on
these platforms: the regression cases of the walls and the open faces (7 cases, with and without coupled particles and
fluctuations) are identical bit for bit to those of the code of version 0.3.0 on CUDA and OpenCL in the three
precisions.

**Coupled particles.** Seven particles of 50 Da, with a constant field and a soft pair force, at $`T = 0`$: near the
borders of the blocks, two on the same node, and one crossing the periodic boundaries into the solid plane
$`j = 0`$, where it is reflected; friction 10 ps⁻¹, 60 steps. Cases: explicit drag with bounce-back walls,
centred drag with regularized walls, centred drag with regularized walls and open faces along x. On every rank the
script compares the fluid nodes of the rank and the positions and velocities of all the particles. Decompositions
$`2 \times 1 \times 1`$, $`1 \times 2 \times 1`$, $`1 \times 1 \times 2`$, $`2 \times 2 \times 1`$,
$`1 \times 2 \times 2`$ and $`2 \times 2 \times 2`$ (22 ranks in all):

| Case | Fluid nodes and particles | Force on the walls |
|---|---|---|
| explicit drag, centred drag, open faces | identical bit for bit on every rank and decomposition (66 of 66) | absolute difference at most $`2.9 \cdot 10^{-9}`$ kJ/mol/nm |

Without a body force the force on the walls, 58 to $`2.5 \cdot 10^{4}`$ kJ/mol/nm here, is a small difference between
the pressure forces on the two sides of the solid plane, each about $`\rho c_s^2`$ times its area, $`6.0 \cdot 10^{6}`$
kJ/mol/nm: the difference is $`5 \cdot 10^{-16}`$ of that, the rounding of the sums added in another order. Relative to
the net force it reaches $`3 \cdot 10^{-11}`$, which is why the script compares it with the pressure force.

With a fluctuating fluid (centred drag, $`T = 300`$ K, 50 steps) the ranks draw different random numbers, so the run
is not compared with one domain; the copies of the particles stay identical on every rank (the plugin compares them
every 10 steps, and the hash of positions and velocities printed by every rank is the same). The checks that must stop
every rank together all do, without any rank waiting: copies of the particles that differ by $`10^{-12}`$ in the
velocities of one rank (stopped at the first step), an `AndersenThermostat` (refused when the Context is created) and,
with $`2 \times 1 \times 1`$ and $`2 \times 2 \times 1`$, rank 0 on the Reference platform and the others on OpenCL
(50 of 50). With the check of the copies off (`setParticleCopiesCheck(False)`) the same differing copies run without an
error.

An exception in the script on one rank only cannot be made collective: `python/tests/mpi_abort.py` raises one on rank 1
while rank 0 waits in the check made when the Context is created (`mpirun -n 2 python mpi_abort.py`). With the
`excepthook` of `openmmlbm` the job stops after 3 s with exit code 1 (`MPI_Abort`); with the default one of Python
(`mpi_abort.py nohook`) it hangs.

### Checkpoint files and VTK files across the domains (all platforms)

The last part of `python/tests/mpi_decomposition.py` checks the files written by all the ranks with MPI-IO
(`saveCheckpointFile()`, `LBMVTKReporter`), in a folder of the working directory shared by the ranks. For three cases
with the coupled particles above (centred drag with regularized walls and with open faces, at $`T = 0`$; centred drag
with the fluctuating fluid at 300 K) a run of 60 steps is interrupted after 30 by `openmmlbm.saveCheckpoint()` and
continued in new Contexts with the same decomposition, with another one (the ranks along one axis) and with one domain
on every rank (each rank reading the whole file):

| Loaded with | Restored state (rank 0, gathered) | Continued run against the uninterrupted one |
|---|---|---|
| the same decomposition | fluid and particles identical bit for bit | identical bit for bit, also with the fluctuating fluid |
| another decomposition, or one domain | fluid identical bit for bit; particles identical bit for bit in double precision, positions to float rounding in mixed and single precision | without random numbers: identical bit for bit in double precision, to rounding in mixed and single precision (fluid within $`1.4 \cdot 10^{-9}`$, particles within $`9 \cdot 10^{-8}`$ nm and nm/ps); with the fluctuating fluid it continues with the new random numbers of the ranks |

In mixed and single precision OpenMM keeps the positions in float, wrapped into the box, while the State gives them
unwrapped: a particle that has crossed a periodic boundary is set again at a position rounded to float (seen on two
nodes in single precision, in the fluctuating case), and the float representation of the positions is rebuilt, which
changes the rounding of the steps that follow. The same decomposition loads the OpenMM checkpoint of each rank instead,
which restores that representation. A file written by `saveCheckpointFile()` with one domain loads with the
decomposition, and the run continues as the uninterrupted one in the same way; a file written by
`openmmlbm.saveCheckpoint()` with one domain is refused, with an error on every rank. The VTK files of the density, the
velocity and the particles written with the decomposition are identical, byte for byte, to those of one domain, in
single and double precision. Runs: on a CPU node the Reference platform with $`2 \times 1 \times 1`$,
$`2 \times 2 \times 1`$, $`1 \times 2 \times 2`$ and $`2 \times 2 \times 2`$, OpenCL on the CPU (pocl) in double
precision with $`2 \times 1 \times 1`$ and $`1 \times 2 \times 2`$, in mixed precision with $`1 \times 2 \times 1`$ and
in single precision with $`2 \times 2 \times 1`$; on one node with four A100 GPUs CUDA in double precision with
$`2 \times 1 \times 1`$, $`1 \times 2 \times 2`$ and $`4 \times 1 \times 1`$, in mixed and single precision with
$`2 \times 1 \times 1`$ and $`2 \times 2 \times 1`$, OpenCL in the three precisions; on two nodes (8 ranks, one file
on the parallel file system written from both nodes) CUDA in double precision with $`2 \times 2 \times 2`$, in mixed
precision with $`2 \times 2 \times 2`$ and in single precision with $`4 \times 2 \times 1`$, OpenCL in double precision
with $`2 \times 2 \times 2`$ and in single precision with $`4 \times 2 \times 1`$. All passed.

`TestMPIReferenceLBMForce` (CTest, two ranks) continues the run of two domains from a checkpoint file with two domains
and with one domain, bit for bit. Without MPI the C++ tests continue a run with the fluctuating fluid, regularized walls
and open faces from a file of `saveCheckpointFile()` bit for bit on every platform, and `test_checkpoint_file` does the
same in Python; the files of `openmmlbm.saveCheckpoint()` with one domain are those of version 0.3, byte for byte (the
part of the force; the OpenMM checkpoint inside them is not reproducible from one run to the next on OpenCL with pocl,
with any version), and the arrays of the VTK files are those of version 0.3, byte for byte, on the Reference platform
and on CUDA and OpenCL in the three precisions.

### Performance of the domain decomposition (CUDA, NVIDIA A100)

Time per step of the fluid alone (`devtools/benchmark_decomposition.py`: periodic box, body force, one uncoupled
particle, no removal of the fluid momentum; 200 steps timed after 20), CUDA platform, OpenMM 8.6.1, Open MPI 4.1.6
with UCX 1.16, one A100 (64 GB) per rank, four per node, NVLink between the GPUs of a node, InfiniBand between nodes
(one port of UCX, `UCX_NET_DEVICES=mlx5_0:1`, unless stated otherwise). In milliseconds per step; "device" is the
default exchange, from GPU to GPU between the ranks of a node (CUDA-aware MPI) and through the host between nodes,
"host" the exchange through the host also within a node (`OPENMM_LBM_DEVICE_MPI=0`).

| Lattice | GPUs (domains) | Mixed: device | host | Single: device | host |
|---|---|---|---|---|---|
| $`128^3`$ | 1 | 0.87 | | 0.48 | |
| $`256 \times 128 \times 128`$ | 2 ($`2 \times 1 \times 1`$) | 1.19 | 1.46 | 0.73 | 0.89 |
| $`128 \times 128 \times 256`$ | 2 ($`1 \times 1 \times 2`$) | 0.91 | 1.18 | 0.49 | 0.63 |
| $`256 \times 256 \times 128`$ | 4 ($`2 \times 2 \times 1`$) | 1.20 | 1.69 | 0.75 | 1.02 |
| $`128 \times 256 \times 256`$ | 4 ($`1 \times 2 \times 2`$) | 0.91 | 1.40 | 0.49 | 0.76 |
| $`128 \times 128 \times 512`$ | 4 ($`1 \times 1 \times 4`$) | 0.91 | 1.21 | 0.49 | 0.64 |
| $`256^3`$ | 1 | 7.42 | | 4.18 | |
| $`256^3`$ | 4 ($`2 \times 2 \times 1`$) | 2.50 | 3.43 | 1.57 | 2.08 |
| $`256^3`$ | 4 ($`4 \times 1 \times 1`$) | 2.77 | 3.71 | 1.89 | 2.41 |
| $`256^3`$ | 4 ($`1 \times 2 \times 2`$) | 1.84 | 2.75 | 1.01 | 1.49 |
| $`256^3`$ | 4 ($`1 \times 1 \times 4`$) | 1.84 | 2.87 | 1.00 | 1.42 |

With the blocks divided along y and z, four GPUs of a node run $`128^3`$ nodes per GPU at 96% of the speed of one
GPU in mixed precision (98% in single precision), and $`256^3`$ 4.0 times faster than one GPU (4.2 times in single
precision; one GPU runs $`256^3`$ a little slower per node than $`128^3`$). The exchange from GPU to GPU is hidden
behind the collision of the interior; through the host it is not.

Dividing x, the axis along which consecutive nodes are consecutive in memory, costs 30% to 90% more (more in single
precision and with more domains along x). The cause is the granularity of the accesses to the memory of the device,
which reads and writes segments of 32 bytes. Nsight Compute on rank 0 (block of $`128^3`$ nodes, mixed precision,
$`2 \times 1 \times 1`$ against $`1 \times 1 \times 2`$, each kernel with the caches flushed) shows two effects:
- the frame of the block and the slots of the faces normal to x are one node per row of the block, scattered in
  memory, so that every access of a thread is a segment of its own: the collision of the 32768 nodes of the frame
  moved 61 MB instead of about 7 MB, 90 µs against 11 µs, and packing and unpacking took 63 µs against 17 µs;
- with the halo along x a row of the block holds $`n_x + 2`$ entries, and the 32 threads of a warp, which read 32
  consecutive nodes, cross the boundaries of the segments: 8.9 segments per access instead of 8 in mixed precision
  (5 instead of 4 in single precision), and the memory delivered 68% of its peak bandwidth instead of 79%. The
  moments took 489 µs against 378 µs, the collision of the interior 612 µs against 500 µs.

A profile of whole steps (Nsight Systems, the same blocks) gives the same picture: collision of the frame 88 against 10
µs, of the interior 554 against 473 µs, moments 458 against 375 µs, packing and unpacking 62 against 14 µs (one domain:
collision 482 µs, moments 372 µs). Rows of the block aligned to 128 bytes, with the interior collided in the order of
the block instead of from a list, made the moments as fast as with one domain (384 µs), but the frame slower (148 µs,
its nodes further apart) and left the interior at 577 µs: the time per step did not change for blocks of $`128^3`$ nodes
($`2 \times 1 \times 1`$: 1.18 ms in mixed precision, 0.73 ms in single) and was 4% to 5% shorter for blocks of
$`256 \times 256 \times 128`$ nodes. Rows aligned to 32 bytes did not change it either. These layouts were left out. So
the automatic decomposition (a 0 in `setDomainDecomposition()`) puts the most domains along z, then y, and x should be
divided only when y and z are not enough.

A fluctuating fluid adds the same time as with one domain: $`256 \times 256 \times 128`$ on four GPUs
($`2 \times 2 \times 1`$) 1.58 ms in mixed precision and 1.06 ms in single (1.21 and 0.75 ms for $`128^3`$ on one GPU).

**Several nodes.** Between nodes the populations go through the host. With $`128^3`$ nodes per GPU (weak scaling) and
$`512^3`$ nodes in all (strong scaling), in mixed precision (single precision in brackets), with blocks divided along
y and z first (as the automatic decomposition does) and along x as well:

| Nodes (GPUs) | $`128^3`$ per GPU, y and z first | x as well | $`512^3`$, y and z first | x as well |
|---|---|---|---|---|
| 1 (4) | $`1 \times 2 \times 2`$: 0.90 (0.50) | $`2 \times 2 \times 1`$: 1.21 (0.75) | $`1 \times 2 \times 2`$: 15.5 (9.05) | $`2 \times 2 \times 1`$: 21.6 (14.7) |
| 2 (8) | $`1 \times 2 \times 4`$: 1.31 (0.78) | $`2 \times 2 \times 2`$: 2.34 (1.27) | $`1 \times 2 \times 4`$: 9.75 (5.38) | $`2 \times 2 \times 2`$: 12.2 (7.76) |
| 4 (16) | $`1 \times 4 \times 4`$: 1.55 (0.88) | $`4 \times 2 \times 2`$: 2.45 (1.42) | $`1 \times 4 \times 4`$: 5.60 (3.05) | $`4 \times 2 \times 2`$: 7.0 (4.33) |
| 8 (32) | $`2 \times 4 \times 4`$: 2.38 (1.43) | $`4 \times 4 \times 2`$: 2.64 (1.52) | $`2 \times 4 \times 4`$: 4.04 (2.39) | $`4 \times 4 \times 2`$: 4.25 (2.48) |

The transfers between nodes take longer than the collision of a block of $`128^3`$ nodes, so the time per step grows
with the nodes until every block has its faces on other nodes. On $`512^3`$ nodes eight nodes are 3.8 times faster
than one in mixed precision (3.8 in single precision), and 5.3 times faster than one node with x divided. With 32 ranks
x had to be divided too. The network is shared with other jobs, and between nodes the times vary from one run to
another by up to about 30%. One run of $`512^3`$ nodes on four nodes with x divided ($`4 \times 2 \times 2`$, mixed
precision) did not end within the 240 s that the script allowed it, and printed nothing; eleven more runs of that
case ended normally, in 6.83 to 7.08 ms (the table gives 7.0 ms). Python buffers its output when it is not a
terminal, so whether that run stopped before or after printing its time is not known; `python -u` prints at once.

**Ports of InfiniBand.** Each node has four ports, one next to each GPU, and UCX, the transport of Open MPI, takes the
ports from `UCX_NET_DEVICES`. With all four ports, which UCX uses when the variable is not set, every run stopped when
the ranks connected (`MPI_Comm_split_type`, `wireup.c: no remote ep address for lane[1]->remote_lane[1]`), also with
`UCX_MAX_RNDV_RAILS=1` or with the transports restricted to `rc` (with `dc` the run hung). With the protocols of
earlier versions of UCX (`UCX_PROTO_ENABLE=n`) all four ports work. On $`512^3`$ nodes in mixed precision, with
$`1 \times 4 \times 4`$ domains on four nodes, one port (`UCX_NET_DEVICES=mlx5_0:1`) took 5.62 to 5.69 ms, one port
per rank, the one next to its GPU, 5.34 to 5.45 ms, two ports (`mlx5_0:1,mlx5_1:1`) 5.14 to 5.25 ms and the four
ports with `UCX_PROTO_ENABLE=n` 5.09 ms; with $`1 \times 2 \times 4`$ on two nodes all of them took 9.3 to 9.6 ms.
Earlier, two ports took 6.55 ms instead of 6.97 ms on four nodes and 3.96 ms instead of 4.25 ms on eight
($`4 \times 2 \times 2`$ and $`4 \times 4 \times 2`$). The tables use one port. Sending from GPU to GPU between the
nodes too (GPUDirect over one port) took 3.8 ms for $`256^3`$ on two nodes against 2.1 ms through the host: between
nodes the default goes through the host.

**End of the run.** UCX registers the host buffers of the exchanges between nodes in the CUDA context of the Context,
which OpenMM destroys with the Context. When MPI was finalized at the exit of the process, after Python had deleted
the Contexts, UCX printed some 200 errors per run on several nodes (`cudaHostUnregister() failed`, `failed to dereg
from md[3]=cuda_cpy`), after the work was done; with `MPI_Finalize` called while the Contexts still existed there
were none, and disabling the registration cache of UCX (`UCX_RCACHE_ENABLE=n`) did not help. The Python module
therefore finalizes MPI as soon as the script ends (`docs/theory.md`, section 8): the runs above, on two and four
nodes, printed no such message.

**Coupled particles.** Every rank computes all the forces of OpenMM on all the particles (the particles are
replicated), so a step takes about $`T_{\mathrm{LB}}/P + T_{\mathrm{MD}} + T_{\mathrm{comm}}`$ with $`P`$ ranks: the
decomposition divides only the time of the fluid, and pays where the fluid takes most of a step. Measured with coupled
particles (`--particles`, beads of 100 Da at random positions with a soft repulsion, cutoff 1 nm, explicit drag,
mixed precision), on one node:

| Coupled particles | $`256^3`$: 1 GPU | 4 GPUs ($`1 \times 2 \times 2`$) | $`128^3`$ per GPU: 1 GPU | 4 GPUs ($`1 \times 2 \times 2`$) |
|---|---|---|---|---|
| 0 | 7.42 | 1.84 | 0.87 | 0.91 |
| $`10^4`$ | 8.08 | 2.21 | 1.07 | 1.26 |
| $`10^5`$ | 8.49 | 3.37 | 1.43 | 2.41 |

$`10^4`$ particles add 0.7 ms per step on $`256^3`$ nodes with one GPU and 0.4 ms with four, $`10^5`$ particles 1.1 and
1.5 ms: the forces of OpenMM and the coupling, which every rank computes for all the particles, plus the sum of the
coupling forces over the ranks ($`3 N_p`$ numbers, copied to the host and back). The centred drag takes the same time.
On $`512^3`$ nodes (a box
of 256 nm) $`10^5`$ particles add 3.7 ms per step on one node ($`2 \times 2 \times 1`$: 25.3 against 21.6 ms) and 4.3 ms
on eight nodes ($`4 \times 4 \times 2`$: 8.51 against 4.25 ms): the difference, 0.6 ms, includes the sum of the forces
over the network. A profile of rank 0 (Nsight Systems, four GPUs of a node, mixed precision, $`10^5`$ particles against
none) separates the cost of the particles into three parts:
- the sum of the coupling forces over the ranks, while the GPU waits: 0.75 to 1.0 ms per step (download of 2.4 MB,
  `MPI_Allreduce` 0.5 to 0.6 ms, upload);
- the kernels of the particles (OpenMM's forces, the coupling, the sort): 0.33 to 0.40 ms;
- the collision, which with coupled particles reads the reaction of every node of the block, three more numbers per
  node: 0.06 ms more with $`256 \times 128 \times 128`$ nodes per GPU, 0.8 to 1.1 ms more with
  $`512 \times 256 \times 256`$ or $`256 \times 256 \times 512`$.

The last part grows with the nodes of the block, which is why the same particles cost more in the larger box: on
$`512^3`$ nodes ($`1 \times 2 \times 2`$) $`10^5`$ particles add 2.5 ms per step (18.0 against 15.5 ms), on $`256^3`$
nodes 1.5 to 1.7 ms. Reading the reaction only at the nodes that have particles would remove it; it is not done. With
$`2 \times 2 \times 1`$ the moments, which do not depend on the particles, also took 1.0 ms more in the run with
particles; that difference is not explained. Starting the sum without blocking (`MPI_Iallreduce`) after the coupling and
waiting for it at the end of the lattice step, so that it would run while the fluid advances, gave 2.99 to 3.09 ms
instead of 3.37 to 3.56 ms on $`256^3`$ nodes with $`10^5`$ particles on one node, but 2.58 instead of 2.41 ms with
$`128^3`$ per GPU, and on four nodes ($`512^3`$, $`1 \times 4 \times 4`$) 10.3 to 11.1 ms instead of 9.1 to 9.3 ms: it
was left out.

**Copies of a protein, and the sort of the coupling keys.** With copies of SOD1 in COCOMO2 (110 beads each, the
`sod1` example of `examples/cocomo` repeated on a grid of $`k^3`$ copies $`d`$ apart; friction 10/ps, time step 10 fs,
removal of the fluid momentum at every step, CUDA, mixed precision, `DeterministicForces`; 500 steps timed after 200),
the forces of OpenMM alone (no `LBMForce`), the fluid alone, and the coupled run with the explicit drag, in
milliseconds per step on one GPU and on four ($`P_x \times P_y \times P_z`$ chosen by MPI):

| Case | MD alone | Fluid: 1 GPU / 4 GPUs | Coupled until version 0.4.0: 1 / 4 | Coupled: 1 / 4 |
|---|---|---|---|---|
| 64 copies, 7040 beads, $`d`$ = 15 nm, $`120^3`$ nodes | 0.125 | 0.920 / 0.269 | 1.158 / 0.640 | 1.148 / 0.624 |
| 512 copies, 56320 beads, $`d`$ = 7.5 nm, $`120^3`$ nodes | 0.254 | 0.918 / 0.271 | 8.660 / 7.046 | 1.336 / 1.392 |
| 512 copies, 56320 beads, $`d`$ = 15 nm, $`240^3`$ nodes | 0.253 | 7.504 / 1.920 | 15.633 / 8.829 | 8.201 / 3.051 |

The forces of OpenMM, which every rank computes for all the beads, take 0.13 to 0.25 ms. Until version 0.4.0 the sort
of the coupling keys took most of the step with 512 copies (7.4 ms on one GPU, 5.8 ms on four, from a profile of rank
0): the copies were in order, and OpenMM's sort, choosing its buckets from 64 keys at fixed intervals of the array,
took all of them from the copies of one layer of the box and put 97% of the keys into one bucket (`docs/theory.md`,
section 2, Per-cell reaction on the GPU platforms). The keys are now sorted by a permutation of their node, with
buckets of equal width; the results are the same bit for bit (CUDA in the three precisions and OpenCL in double
precision, nearest node and the three interpolation stencils, against the version before). On four GPUs the sum of
the coupling forces over the ranks (`MPI_Allreduce` of $`3 N_p`$ numbers) took 0.26 to 0.29 ms of the step.

### Fluctuating fluid and particles across the domains (CUDA, NVIDIA A100; Reference)

With a fluctuating fluid every rank draws its own random numbers, so a run with the decomposition differs from one with
one domain and the agreement is statistical. Three questions: are the fluctuations those of one domain (variances and
spectra), are the planes at the borders of the blocks like the others, and are the ranks independent? The protocol of
the equilibrium spectra above ($`64^3`$ nodes, $`k_BT = 1/3000`$, from rest, $`2 \cdot 10^4`$ steps of transient, then
200 snapshots every 2500 steps, CUDA in mixed precision, one A100 per rank), with the state gathered on rank 0 at every
snapshot (`getFluidState(gather=True)`). Besides the ER per node and $`\mathrm{ER}(\lvert\mathbf k\rvert)`$ of the four
observables, the ER of every plane $`i`$, $`j`$, $`k`$ = const, and the correlation coefficient of the observables
between the corresponding nodes (same index in the block) of the blocks of every pair of ranks, over nodes and
snapshots, whose spread for independent fields is $`1/\sqrt{\text{nodes} \times \text{snapshots}}`$; if two ranks drew
the same numbers it would approach 1. Errors from 10 or 40 blocks of snapshots. ER per node of $`\rho`$,
$`\sum_a j_a`$, $`\sum_a a^{(2)}_{aa}`$, $`\sum_{a<b} a^{(2)}_{ab}`$ (error on the last digit), difference from one
domain in units of the error, $`\chi^2`$ per shell of $`\mathrm{ER}(\lvert\mathbf k\rvert)`$ against one domain (55
shells), and the largest correlation between ranks:

| $`\tau`$, domains (ranks, nodes) | ER per node | difference from one domain | $`\chi^2`$ per shell | largest correlation (spread) |
|---|---|---|---|---|
| 1, one domain | 1.0009(2) 1.0003(1) 1.0017(2) 1.0012(2) | | | |
| 1, $`2 \times 2 \times 1`$ (4, one node) | 1.0002(1) 1.0000(3) 1.0016(1) 1.0010(2) | -2.7 -1.0 -0.5 -0.7 | 1.22 1.12 1.21 1.13 | $`6.4 \cdot 10^{-4}`$ ($`2.8 \cdot 10^{-4}`$) |
| 1, $`4 \times 1 \times 1`$, blocks 16 nodes thick (4, one node) | 1.0006(2) 1.0004(1) 1.0017(2) 1.0012(2) | -0.8 +0.5 -0.3 -0.2 | 0.79 0.88 0.97 0.86 | $`6.8 \cdot 10^{-4}`$ ($`2.8 \cdot 10^{-4}`$) |
| 1, $`2 \times 2 \times 2`$ (8, two nodes) | 1.0010(2) 1.0003(2) 1.0017(2) 1.0018(2) | +0.4 -0.1 -0.1 +2.2 | 1.07 1.11 1.12 1.32 | $`1.1 \cdot 10^{-3}`$ ($`3.9 \cdot 10^{-4}`$) |
| 1, another seed, 600 snapshots: one domain | 1.0007(2) 1.0004(1) 1.0018(1) 1.0012(1) | | | |
| 1, the same, $`2 \times 2 \times 2`$ (8, two nodes) | 1.0006(1) 1.0006(1) 1.0015(1) 1.0011(1) | -0.5 +1.1 -1.8 -1.3 | 1.51 1.17 0.94 1.08 | $`5.6 \cdot 10^{-4}`$ ($`2.3 \cdot 10^{-4}`$) |
| 0.55, $`2 \times 2 \times 1`$ (4), against one domain of the spectra above | 1.0020(1) 1.0023(2) 1.0049(2) 1.0047(2) | -1.2 -1.6 -1.2 +1.0 | 1.01 1.02 1.16 0.85 | $`6.3 \cdot 10^{-4}`$ ($`2.8 \cdot 10^{-4}`$) |
| 0.7, Reference, $`16^3`$, $`2 \times 2 \times 2`$ (8), against one domain of the spectra above | 1.0000(11) 1.0025(9) 1.0010(11) 1.0010(14) | +0.6 +1.6 -1.2 +0.4 | 0.49 0.97 0.72 1.25 | $`7.0 \cdot 10^{-3}`$ ($`2.6 \cdot 10^{-3}`$) |

Two runs with one domain and different seeds differ by -0.7 to +0.3 errors in the ER per node and give $`\chi^2`$ per
shell 0.81 to 1.21, with the largest deviation of a shell at 3.3 errors: the runs with the decomposition differ from one
domain as much as two runs with one domain differ from each other. The correlations between ranks are compatible with
zero: $`\chi^2`$ per value 0.99 to 1.15 over the pairs and observables, the largest at 2.9 errors among 112 values. On
OpenCL ($`2 \times 2 \times 1`$, mixed precision) the results are those of CUDA to all the digits above: the generator
of OpenMM gives the same numbers on both platforms, as with one domain.

**The borders of the blocks.** In the runs above the planes at the borders of the blocks (the first and last plane of
every block, along the divided axes) differ from the others by -1.5 to +2.8 errors, and the same planes of the run
with one domain by -4.0 to +1.5 errors; the errors of plane averages, which are dominated by the slow modes and
correlated between adjacent planes, are somewhat underestimated. A dedicated run settles it: $`32^3`$ nodes,
$`4 \times 1 \times 1`$ (8 border planes out of 32), 15000 snapshots every 300 steps, each rank accumulating the ER of
the planes of its own domain without gathering, errors from 20 blocks of snapshots, and the same with one domain as
control:

| Observable | border planes minus the others, $`4 \times 1 \times 1`$ | the same planes, one domain |
|---|---|---|
| $`\rho`$ | $`-0.7 \cdot 10^{-4} \pm 1.4 \cdot 10^{-4}`$ | $`-1.8 \cdot 10^{-4} \pm 1.8 \cdot 10^{-4}`$ |
| $`\sum_a j_a`$ | $`+1.7 \cdot 10^{-4} \pm 1.8 \cdot 10^{-4}`$ | $`-1.3 \cdot 10^{-4} \pm 0.9 \cdot 10^{-4}`$ |
| $`\sum_a a^{(2)}_{aa}`$ | $`-0.3 \cdot 10^{-4} \pm 1.4 \cdot 10^{-4}`$ | $`-0.0 \cdot 10^{-4} \pm 1.7 \cdot 10^{-4}`$ |
| $`\sum_{a<b} a^{(2)}_{ab}`$ | $`+1.6 \cdot 10^{-4} \pm 1.3 \cdot 10^{-4}`$ | $`+0.9 \cdot 10^{-4} \pm 1.4 \cdot 10^{-4}`$ |

The fluctuations at the borders of the blocks equal the others within $`2 \cdot 10^{-4}`$ of their value, and the mean
ER of the planes (1.00070, 1.00031, 1.00168, 1.00107) is that of one domain (1.00065, 1.00023, 1.00156, 1.00113).

**Walls.** The plane $`j = 0`$ solid with regularized walls (whose boundary nodes draw random numbers of their own),
$`64^3`$, $`\tau = 1`$, the axis normal to the wall divided in four ($`1 \times 4 \times 1`$): the ER of every plane
$`j`$ against the same plane with one domain gives $`\chi^2`$ per plane 0.70 to 1.17, largest deviation 3.4 errors,
including the deficit of the momentum next to the wall (0.979 and 0.981 in the first fluid plane).

**Particles.** 256 free particles of 100 Da coupled with the centred drag ($`\gamma`$ = 5 ps⁻¹) to the fluctuating fluid
at 300 K ($`32^3`$ nodes, $`\Delta x`$ = 0.5 nm, $`\Delta t`$ = 10 fs, the viscosity of water, $`\tau = 0.607`$), the
temperature from the velocities at half step, 20000 samples every 50 steps after $`10^4`$ steps, errors from 20 blocks:
with $`2 \times 2 \times 2`$ domains on two nodes 300.30 ± 0.08 K, with the particles changing domain 91005 times
between samples, and 299.8 to 300.8 K (errors 0.23 to 0.36 K) for the particles of each of the 8 domains; with one
domain 300.11 ± 0.13 K. The decomposition changes the temperature by 0.19 ± 0.15 K.

## Equivalence with the reference implementation: coupled particles (E0)

The Reference platform was compared with the validation campaign of the reference CUDA library (version
tagged `ref-explicit-2026-10`, built in double precision), run with the same scripts, at $`T = 0`$, with all
particles coupled and no removal of the fluid momentum: 80 cases.
- **Cases.**
  - Nearest-node artefacts of a dragged particle (8 cases);
  - kick of a particle in a fluid at rest (6 cases, lattices up to $`64^3`$, 20000 steps);
  - pair mobility (15 cases);
  - composite sphere of 300 beads in translation and rotation (7 cases);
  - mobility of a dragged particle (44 cases) as a function of mass, friction, force, direction, box size,
    $`\tau`$ and time step.
- **Result.**
  - Trajectories agree within 5e-13 relative over 30000 steps.
  - Derived quantities (mobilities, self-mobility $`y`$, hydrodynamic radii, velocity decay) agree within
    2e-6, and within 1.1e-5 for the velocity of a kicked particle at its last plateau (5e-6 of its initial
    velocity).
  - The criterion was 1e-4.
- **Rounding of small momenta.** With 10 Da nm/ps spread over a fluid of 2e7 Da ($`64^3`$ nodes, 20000 steps),
  the total momentum of particles and fluid drifts by at most 1e-8 of the momentum; the reference library
  drifts by 4.5e-6 (read from its single-precision output). Such a momentum is a difference between
  populations of order 0.05 at the ninth digit; the populations are stored as deviations from the rest equilibrium
  for this reason (`docs/theory.md`, section 4): stored whole, the drift was 3e-5.

## Equivalence with the reference implementation: fluid only

The Reference platform was compared in double precision with the CUDA lattice Boltzmann library from
which openmm-lbm is derived (the library of the DragOpenMM plugin, version tagged `ref-explicit-2026-10`),
built in double precision.
- **Setup.** 16x12x10 nodes, 500 steps, no particles acting on the fluid.
- **Initial state, the same in both codes.** Density modulated by up to 1.5% around $`\rho_0`$, a shear wave
  and smaller velocity modes, and a non-zero non-equilibrium stress, given as moments; the populations
  are rebuilt from them as $`f^{\mathrm{eq}}(\rho, \mathbf j/\rho) + f^{\mathrm{neq,reg}}(\Pi^{\mathrm{neq}})`$.
- **Cases.** $`\tau = 1.102`$ with a body force, without and with removal of the fluid momentum at every
  step; $`\tau = 0.62`$ with $`\rho_0 = 0.98`$ and a body force.
- **Result.** Density and momentum agree within the float32 rounding of the output of the reference
  library: relative differences of at most 5e-8. The non-equilibrium stress agrees within 5e-16 in
  absolute value, the double-precision rounding of $`f - f^{\mathrm{eq}}`$ for populations of order 0.05.
- **Sensitivity.** A difference in the algorithm would appear at 1e-3 or above. For example, a Guo
  prefactor of $`1 - \omega/2`$ instead of $`1/2`$ changes the momentum input by 4.6% at $`\tau = 1.102`$, and
  shifting the velocity instead of the momentum by half a force changes it by 1% at $`\rho_0 = 0.98`$.

## GPU platforms against the Reference platform: fluid and walls

The Python test `test_fluid_agrees_with_reference` runs the same fluid on the Reference platform and on each available
GPU platform: 6x5x4 nodes, $`\tau = 0.8`$, populations perturbed by up to 1e-3, a body force and the removal of the
fluid momentum every third step, 40 steps, in four cases: without solid nodes; with the solid plane $`j = 0`$ and a
block of 8 solid nodes, with bounce-back walls and with regularized walls; and with the same regularized walls and
open faces along x, a `Velocity` inlet at `XMin` and a `Density` outlet at `XMax`, without the removal of the fluid
momentum. The largest difference of the fluid states, relative to the largest deviation $`\lvert f - w\rvert`$, must
be below 1e-12 in `mixed` and `double` precision, and the forces on the walls must agree to 1e-10. Measured on an
NVIDIA A100 with OpenMM 8.6.1, without solid nodes:

| Platform | Precision | 40 steps | 1000 steps |
|---|---|---|---|
| CUDA | mixed, double | 1.8e-15 | 2.8e-15 |
| CUDA | single | 9.1e-7 | 1.3e-5 |
| OpenCL | mixed, double | 1.8e-15 | 7.6e-15 |
| OpenCL | single | 1.1e-6 | 2.7e-5 |

With bounce-back walls (6x7x5 nodes, the solid plane $`j = 0`$ and a block of 8 solid nodes, 300 steps), the fluid
states differ by 6e-16 in `mixed` and `double` precision and by 1.9e-5 in `single` precision, and the forces on the
walls by 1e-14 and 4e-8 (CUDA and OpenCL alike). With regularized walls (the same solid nodes, body force, removal
every 3 steps, 300 steps) the populations differ by at most 4e-19 of their largest value in `mixed` and `double`
precision and 7e-8 in `single` precision, the forces on the walls by 3e-13 and 7e-8 relative; with regularized walls
and open faces along x (a `Velocity` inlet and a `Density` outlet, no removal) by 1e-17 and 7e-8, and 1e-14 and 3e-7
relative (CUDA and OpenCL alike, OpenMM 8.6.1).

The two platforms are not identical bit for bit: the GPU compilers contract multiplications and additions
into fused multiply-adds. Each GPU platform is deterministic: the removal of the momentum and the Mach
number use reductions in a fixed order, without atomic operations, and each population returned by the
bounce-back or rebuilt on a boundary node is written by one thread, so two runs give identical results.

## GPU platforms against the Reference platform: coupled particles

The Python test `test_coupling_agrees_with_reference` runs four coupled particles at $`T = 0`$ (friction
10/ps, 60 steps; two particles share a node, one crosses the periodic boundary), without and with the solid
plane $`j = 0`$, which reflects two of them, and with open faces along x (a `Velocity` face `XMin` and a `Density`
face `XMax`, on whose nodes one particle stays), on the Reference platform and in double precision on each GPU
platform, with each drag scheme. A constant field and a soft pair force act on the particles, so that the
centred drag sees other forces. Positions, velocities, fluid state and force on the walls must agree to
1e-10. Largest differences relative to $`\max(1, \text{largest value})`$, with the walls, measured on an NVIDIA A100
with OpenMM 8.6.1 (version 0.2.0):

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

With the open faces (one particle on the nodes of the `Density` face, OpenMM 8.6.1): positions 3e-14, velocities 3e-13
and fluid 2e-16 in double precision (CUDA and OpenCL, both drags); positions 1e-9, velocities 9e-9 and fluid 1e-12 to
2e-12 in mixed precision; positions 4e-7, velocities 1.4e-7 and fluid 2e-9 with CUDA in single precision.

In double precision the velocities differ at 1e-13 because OpenMM adds the forces in fixed point, with a
resolution of $`2^{-32}`$ kJ/mol/nm. In mixed precision OpenMM computes the other forces (field and pair force) in
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

**Speed.** With 110 coupled particles on a $`30^3`$ lattice at 300 K and the removal of the fluid momentum at
every step, a step takes 61 us with CUDA in mixed precision (51 us in single, 55 us in double) and 64 us
with OpenCL in mixed precision, on an NVIDIA A100 with OpenMM 8.6.1. The CUDA library of the DragOpenMM
project takes 517 us for the same system. The centred drag costs 0 to 6% more (A100, OpenMM 8.6.1, 300 K,
removal at every step, 5000 steps after 200), in us per step:

| System | precision | CUDA explicit | CUDA centred | OpenCL explicit | OpenCL centred |
|---|---|---|---|---|---|
| 110 particles, $`30^3`$ | mixed | 61.4 | 58.7 | 67.3 | 70.5 |
| | single | 53.7 | 53.8 | 62.9 | 62.8 |
| | double | 58.1 | 57.8 | 66.9 | 68.7 |
| 1000 particles, $`30^3`$ | mixed | 78.6 | 81.1 | 86.8 | 89.6 |
| | single | 73.8 | 77.1 | 81.4 | 83.5 |
| | double | 78.2 | 80.5 | 87.1 | 91.3 |
| 8000 particles, $`40^3`$ | mixed | 103.6 | 104.9 | 116.6 | 119.2 |
| | single | 88.0 | 93.1 | 101.7 | 104.7 |
| | double | 100.8 | 104.6 | 114.1 | 118.6 |

## Examples compared with the DragOpenMM plugin

The scripts of `examples/` are ports of the examples of the DragOpenMM plugin, with the Euler-Maruyama
coupling and `VerletIntegrator`. Run with the same parameters, the DragOpenMM plugin (reference library
in double precision, CUDA platform in double precision, OpenMM 8.2) and openmm-lbm (OpenMM 8.6.1, NVIDIA
A100) give the same trajectories:

- **Kick of a bead** (`particle/kick.py`, $`100^3`$ nodes, 2000 steps, $`T = 0`$): velocities within 1e-13 of
  $`v_0`$ and positions within 1e-11 of the distance travelled, with CUDA in double and mixed precision;
  5e-8 in single precision. With the parameters of the alanine bead ($`30^3`$ nodes, 100 steps) the
  Reference platform agrees within 7e-14.
- **Thermalization of a bead** (`particle/thermal.py`, 300 K, 20000 steps, same seed): velocities within
  3e-11 of their largest value with CUDA in double and mixed precision, random force included, because
  both plugins draw the same numbers from OpenMM's generator ([theory.md](theory.md), section 2);
  2e-6 in single precision and 5e-6 with OpenCL, whose generator rounds differently.
- **Bead in a uniform flow** (`particle/uniform_flow.py`, NVE, 400 steps): equal to the 6 printed digits.
- **Kick of the peptide GRGDSPYS** (`cocomo/kick.py --preset peptide`, COCOMO2 forces, 2000 steps,
  $`T = 0`$, the density of the original script): velocity of the centre of mass within 1e-13 of $`v_0`$, with
  CUDA in double precision.

The reference outputs of the examples of the old plugin came from its Langevin scheme, in
which OpenMM's `LangevinIntegrator` supplies friction and noise (first step $`v_1/v_0 = \exp(-\gamma\Delta t)`$); they
were regenerated with the Euler-Maruyama scheme for this comparison ($`v_1/v_0 = 1 - \gamma\Delta t`$).
- **Disordered protein rlp** (`cocomo/diffusion.py --preset rlp`, 166 beads, friction 100/ps, $`\Delta t`$ = 2 fs,
  298 K): full-step temperature 259.1 K and half-step 293.6 K over 10 ns; the DragOpenMM plugin with the
  same parameters gives 259.9 K and 294.7 K. The 13% deficit is that of the model without thermal
  fluctuations of the fluid, large at this friction (T2 and T6 below).
- **COCOMO2 model** (`cocomo/cocomo2.py`), written from the article: on SOD1 (`cocomo/data/sod1.pdb`,
  box 15 nm, Reference platform) the energy of every term (bonds, angles, electrostatics, short range,
  cation-pi, pi-pi, elastic network with 478 bonds) is identical to that of the COCOMO2 script of the
  DragOpenMM runs, and the forces agree within 1e-16 of the largest force, at the reference structure
  and with the beads displaced at random by 0.03 nm.
- **Diffusion of SOD1** (`cocomo/diffusion.py --preset sod1`, three runs of 200 ns with seeds 1, 2, 3, CUDA in mixed
  precision): the apparent diffusion coefficient of the centre of mass, $`\mathrm{MSD}(t)/(6t)`$, is 2.34, 2.41, 2.42,
  2.49, 2.64, 2.83 and 2.86 Å²/ns (mean of the three runs; standard error 0.01 to 0.03 up to 2 ns, 0.06 to 0.19 at
  longer lags) at lag times of 0.1, 1, 2, 3, 5, 10 and 18 ns. Three runs of the DragOpenMM plugin with the same
  parameters and the Euler-Maruyama coupling give 2.3, 2.4, 2.3, 2.3, 2.4, 2.4 and 2.6 Å²/ns, with a spread between
  runs of 1.9 to 3.4 Å²/ns at 18 ns. Both stay close to $`k_BT/(M\gamma)`$ = 2.26 Å²/ns, as expected without thermal
  fluctuations of the fluid. At the longest lags the two sets differ by about 1.5 standard errors. The full-step
  temperature is 293-294 K at 298 K.
- **Diffusion of SOD1 with the centred drag** (`--drag Centered`, same three seeds, CUDA in mixed precision,
  NVIDIA A100): 2.34, 2.41, 2.42, 2.49, 2.64, 2.87 and 2.95 Å²/ns at the same lag times (standard error 0.01
  to 0.07 up to 5 ns, 0.04 to 0.17 at longer lags), against 2.34, 2.41, 2.42, 2.49, 2.64, 2.83 and 2.86 with
  the explicit drag: the same random numbers give the same diffusion of the centre of mass to the second
  decimal up to 5 ns, and within the statistical error at longer lags. The temperature of the centred drag,
  at half steps, is 277.5 K (half step 310.3 K and full step 294.0 K with the explicit drag; full step
  264.4 K with the centred drag): 6.9% below $`T`$, against 1.3% for the explicit drag
  ([theory.md](theory.md), section 2, temperature with a fluid without fluctuations).
- **Friction 30/ps** (`--preset sod1-g30`, 50 ns, seed 1): both drags are stable; the apparent diffusion
  coefficient is 0.89, 0.98, 1.00, 1.00, 1.04, 1.24 and 1.25 Å²/ns with the explicit drag and 0.86, 0.93,
  0.96, 0.99, 1.02, 1.28 and 1.35 with the centred drag, within the statistics of one run
  ($`k_BT/(M\gamma)`$ = 0.75 Å²/ns); the temperature is 294.6 K at full steps with the explicit drag and 250.2 K
  at half steps with the centred drag, 16% below $`T`$.

## Many proteins: 64 copies of SOD1 (CUDA, NVIDIA A100)

64 copies of SOD1 (COCOMO2, 7040 beads, well above the size at which the GPU platforms may repeat a force evaluation
to enlarge the neighbor list) on a 4 x 4 x 4 grid in a box of 20 nm ($`40^3`$ nodes), friction 10/ps, 298 K,
$`\Delta t`$ = 0.01 ps, every bead coupled, removal of the fluid momentum switched off so that the total momentum can
be checked; 2 ns in mixed precision and 0.2 ns in double precision, OpenMM 8.6.1. Mean temperatures after the first
quarter of the run, and drift of the total momentum of particles and fluid relative to
$`\sum m\lvert\mathbf v\rvert`$:

| drag | full step (K) | half step (K) | OpenMM, `StateDataReporter` (K) | drift, mixed, 2 ns | drift, double, 0.2 ns |
|---|---|---|---|---|---|
| explicit | 294.4 | 310.7 | 294.4 | 1.4e-8 | 4.0e-10 |
| centred | 264.7 | 277.8 | 264.7 | 1.0e-8 | 4.0e-10 |

Both drags are stable, at about 0.4 ms per step. In double precision the drift grows linearly, 2e-14 per step,
and is the same with the coupling switched off (friction 0, 3.8e-11 after 2000 steps against 4.2e-11): it
comes from the fixed point sum of the COCOMO2 forces in OpenMM, not from the coupling. The temperatures are
those of the right velocity for each drag (half step for the centred one): 1.2% and 6.8% below $`T`$, as for one
protein (previous section).

## Kinetic energy budget (`python/tests/TestEnergyBudget.py`, Reference)

The kinetic energy plus the viscous and drag dissipation of [theory.md](theory.md), section 2, is constant only to
$`O(\mathrm{Ma}^2, \mathrm{Kn}^2)`$. The test checks the size and scaling of the residual
$`(E + \text{dissipated})/E_0 - 1`$:

| Case | Residual |
|---|---|
| shear wave, 16 nodes per wavelength, $`\tau = 1.1`$ | -3.06% |
| shear wave, 32 nodes per wavelength, $`\tau = 1.1`$ | -0.77% (ratio 3.97: $`O(k^2)`$) |
| bead kicked at Mach 0.035, $`12^3`$ nodes, NVE | -4.448% |
| bead kicked at Mach 0.10, same system | -4.445% (independent of the Mach number) |
| bead kicked at Mach 0.035 and 0.10, centred drag | -3.981% and -3.979% |

The dissipation of the drag is computed from the coupling force of each step,
$`\mathbf F = m(\mathbf v' - \mathbf v)`$, and so holds for both drags.

Tolerances: below 1% for the finer shear wave and a ratio between 3 and 5; below 6% for the kick and a
difference below 1e-3 between the two Mach numbers.

## Stochastic tests T2, T6, T7 (CUDA, mixed precision, NVIDIA A100)

The stochastic tests of the reference campaign of the DragOpenMM plugin, run with openmm-lbm with the same
cases, seeds, protocol and analysis (zero initial total momentum in T6). Both plugins draw the random
numbers from OpenMM's generator in the same order, so the agreement is closer than the statistical error.
The fluid has no thermal fluctuations of its own: the diffusion coefficient is $`k_BT/(m\gamma)`$ and the
Einstein relation with the hydrodynamic mobility is not satisfied, as in the reference.

**T2, equipartition** (100 free particles of 100 Da, 300 K, $`L`$ = 8 nm, viscosity fixed, 200 ps). Mean
temperatures in K, openmm-lbm / reference; statistical error about 0.5 K.

| $`\Delta t`$ (ps) | $`\tau`$ | $`\gamma`$ (1/ps) | half step | full step |
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

| $`\gamma`$ (1/ps) | $`L`$ (nm) | $`D/(k_BT/m\gamma)`$, MSD | $`D/(k_BT/m\gamma)`$, Green-Kubo | full-step $`T`$ (K) |
|---|---|---|---|---|
| 1 | 16 | 1.038 / 1.038 | 1.010 / 1.010 | 292.4 / 292.4 |
| 5 | 16 | 0.984 / 0.984 | 0.984 / 0.985 | 270.4 / 270.4 |
| 10 | 16 | 1.010 / 1.011 | 1.007 / 1.004 | 254.0 / 254.0 |
| 5 | 8 | 0.979 / 0.983 | 0.979 / 0.980 | 271.4 / 271.4 |

**T7, velocity autocorrelation against the kick response** ($`\gamma`$ = 10, $`L`$ = 16 nm): the ratio of the
normalized VACF at half steps to the kick response minus its plateau is 0.93, 0.90, 0.85, 0.81, 0.69 and
0.50 at 0.05, 0.1, 0.2, 0.3, 0.5 and 1 ps (reference: 0.93, 0.90, 0.85, 0.81, 0.70, 0.47). With thermal
fluctuations of the fluid the two curves would coincide (fluctuation-dissipation theorem).

**With the centred drag** (same cases, seeds and protocol, with the centred drag). The
temperature that is right for the centred drag is that of the half step (`docs/theory.md`, section 2).

T2, mean temperatures in K, half step / full step (statistical error about 0.5 K):

| $`\Delta t`$ (ps) | $`\tau`$ | $`\gamma`$ = 1/ps | $`\gamma`$ = 5/ps | $`\gamma`$ = 10/ps |
|---|---|---|---|---|
| 0.005 | 0.80 | 297.9 / 297.2 | 290.6 / 287.1 | 283.8 / 277.0 |
| 0.01 | 1.10 | 300.2 / 298.6 | 289.4 / 282.6 | 277.1 / 264.5 |
| 0.02 | 1.70 | 295.9 / 293.0 | 281.3 / 268.9 | 264.7 / 243.1 |

T6, particles of 1000 Da ($`m/m_c`$ = 13.3): the diffusion coefficient is that of the explicit drag
($`D/(k_BT/m\gamma)`$ from the MSD 1.038, 0.984, 1.010 and 0.980 for the four cases), while the half-step
temperature is 274.7, 207.7, 161.1 and 204.2 K: the deficit of a fluid without fluctuations grows with
$`\gamma\Delta t\,m/m_c`$, as predicted (`docs/theory.md`, section 2, Temperature with a fluid without fluctuations).

T7, with the response to a kick (particle of 1000 Da, $`\gamma`$ = 10, $`L`$ = 16 nm) computed with the same drag
on CUDA (with the explicit drag it
reproduces the kick of the reference campaign): ratio of the normalized VACF to the kick response minus its
plateau at 0.05, 0.1, 0.2, 0.3, 0.5 and 1 ps:

| drag | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1 ps |
|---|---|---|---|---|---|---|
| explicit | 0.93 | 0.90 | 0.85 | 0.81 | 0.69 | 0.50 |
| centred | 0.97 | 0.96 | 0.94 | 0.93 | 0.89 | 0.79 |

The centred drag is closer to the fluctuation-dissipation relation for the dynamics.

**With the fluctuating fluid** (same cases and seeds, `setFluidFluctuations(True)`, both drags; the kick
responses of T7 are those above, computed at $`T = 0`$ with the same drag). The fluid temperature, from the kinetic
energy of the fluid at the end of each run, is 294 to 305 K in all cases.

T2, mean temperatures in K, half step / full step (statistical error about 0.5 K); the right one is the full step
for the explicit drag and the half step for the centred drag:

| $`\Delta t`$ (ps) | $`\tau`$ | $`\gamma`$ (1/ps) | explicit | centred |
|---|---|---|---|---|
| 0.005 | 0.80 | 1 | 300.5 / 299.7 | 298.7 / 297.9 |
| 0.005 | 0.80 | 5 | 308.3 / 304.6 | 299.5 / 296.0 |
| 0.005 | 0.80 | 10 | 317.0 / 309.2 | 299.5 / 292.6 |
| 0.01 | 1.10 | 1 | 303.6 / 302.2 | 300.0 / 298.5 |
| 0.01 | 1.10 | 5 | 316.8 / 308.8 | 299.0 / 291.9 |
| 0.01 | 1.10 | 10 | 335.6 / 319.0 | 299.3 / 286.1 |
| 0.02 | 1.70 | 1 | 309.6 / 306.6 | 302.0 / 299.1 |
| 0.02 | 1.70 | 5 | 337.7 / 321.0 | 300.4 / 287.1 |
| 0.02 | 1.70 | 10 | 378.2 / 340.7 | 300.7 / 276.5 |

T6, particles of 1000 Da ($`m/m_c`$ = 13.3), $`L`$ = 16 nm except the last row:

| $`\gamma`$ (1/ps) | $`L`$ (nm) | $`D/(k_BT/m\gamma)`$, MSD, explicit / centred | $`D/(k_BT\,(1/\zeta + y_{\mathrm{centred}}))`$ | full-step $`T`$, explicit | half-step $`T`$, centred |
|---|---|---|---|---|---|
| 1 | 16 | 1.109 / 1.108 | 1.009 / 1.009 | 318.4 K | 298.9 K |
| 5 | 16 | 1.511 / 1.496 | 1.011 / 1.001 | 388.7 K | 299.5 K |
| 10 | 16 | 1.976 / 1.954 | 0.994 / 0.983 | 466.7 K | 300.2 K |
| 5 | 8 | 1.467 / 1.454 | 0.992 / 0.983 | 388.6 K | 299.3 K |

$`\zeta = m\gamma`$, and $`y_{\mathrm{centred}} = y_{\mathrm{explicit}} + \Delta t/(2m_c)`$ is the self-mobility of
the centred drag measured with a constant force (`docs/theory.md`, section 2): 9.59e-5 ps/Da at $`L`$ = 8 nm and
9.88e-5 ps/Da at $`L`$ = 16 nm, with $`y_{\mathrm{explicit}}`$ from the reference campaign at $`L`$ = 16 nm. The
diffusion coefficient now contains the hydrodynamic contribution of the thermal flows, about 2.5e-4 nm²/ps whatever
the friction, and is the same for both drags: the Einstein relation holds with the mobility of the centred drag
(within 2%, the statistical error of D), and not with that of the explicit one
($`D/(k_BT\,(1/\zeta + y_{\mathrm{explicit}}))`$ = 1.07, 1.30, 1.49 and 1.28). The full-step temperature of the
explicit drag agrees with $`T\,(1 + \gamma\Delta t\,m/(2m_c(1 + \zeta y_{\mathrm{explicit}})))`$ within 1% in the nine
T2 cases and in T6 at $`\gamma`$ = 1 and 5/ps, and is 3.6% above it at $`\gamma`$ = 10/ps (`docs/theory.md`, section
7).

T7, ratio of the normalized VACF at half steps to the kick response minus its plateau ($`\gamma`$ = 10/ps,
$`L`$ = 16 nm):

| drag | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1 ps |
|---|---|---|---|---|---|---|
| explicit | 0.97 | 0.96 | 0.95 | 0.93 | 0.89 | 0.79 |
| centred | 1.001 | 1.001 | 1.002 | 1.003 | 0.998 | 0.996 |

With the centred drag and the fluctuating fluid the fluctuation-dissipation theorem holds for the dynamics as
well: the velocity autocorrelation equals the response to a kick.

**SOD1 with the fluctuating fluid** (`examples/cocomo/diffusion.py --fluid-fluctuations`, presets `sod1`:
friction 10/ps, box 15 nm, 200 ns, and `sod1-g30`: friction 30/ps, box 30 nm, 50 ns; one run each, seed 1, the
same as the first run without fluctuations). Temperatures in K (target 298 K) and apparent diffusion coefficient
of the centre of mass, $`\mathrm{MSD}(\mathrm{lag})/(6\,\mathrm{lag})`$, in Å²/ns at lags of 0.1, 1, 2 and 3 ns:

| preset | drag | fluid | full-step $`T`$ | half-step $`T`$ | D at 0.1 / 1 / 2 / 3 ns |
|---|---|---|---|---|---|
| sod1 | explicit | without fluctuations | 293.6 | 309.8 | 2.32 / 2.40 / 2.37 / 2.37 |
| sod1 | centred | without fluctuations | 264.1 | 277.2 | 2.32 / 2.40 / 2.38 / 2.39 |
| sod1 | explicit | fluctuating | 316.4 | 333.7 | 4.91 / 4.91 / 4.76 / 4.77 |
| sod1 | centred | fluctuating | 284.8 | **298.6** | 4.99 / 5.11 / 4.87 / 4.70 |
| sod1-g30 | explicit | without fluctuations | 294.6 | 349.7 | 0.89 / 0.98 / 1.00 / 1.00 |
| sod1-g30 | centred | without fluctuations | 220.5 | 250.2 | 0.86 / 0.93 / 0.96 / 0.99 |
| sod1-g30 | explicit | fluctuating | 355.6 | 419.8 | 3.67 / 3.39 / 3.21 / 3.18 |
| sod1-g30 | centred | fluctuating | 264.6 | **298.7** | 3.68 / 3.34 / 3.25 / 3.29 |

With the fluctuating fluid and the centred drag the protein has the set temperature, and its diffusion coefficient
gains the hydrodynamic contribution of the solvent (twice and three times the value without fluctuations, which is
close to $`k_BT/(M\,\text{friction})`$), the same for both drags. The explicit drag is too hot, by 6% and 19%, as
$`T\,(1 + \text{friction}\cdot\Delta t\,m/(2m_c(1 + \zeta y_{\mathrm{explicit}})))`$ predicts with the mean bead mass
(6% and 18%).

