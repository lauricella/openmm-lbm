# Validation

Tests of openmm-lbm, what each one checks, its tolerance and the values measured. The C++ tests run with
`ctest` (one executable per platform, in `platforms/*/tests`); the Python tests run with `pytest` in
`python/tests`.

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
| `testFluidConservation` | 6x5x4 nodes, $`\tau = 0.7`$, populations perturbed by up to 1%, 100 steps | total mass and momentum unchanged | 1e-13 (relative to the mass) |
| `testBodyForce` | 4x3x5 nodes, $`\tau = 0.9`$, uniform lattice density $`\rho_0`$ = 0.98, 1 or 1.02 at rest, body acceleration $`\mathbf g`$, 20 steps | momentum $`n\rho_0\mathbf g`$ per node (lattice units), density $`\rho_0`$, velocity $`(n + 1/2)\,\mathbf g\,\Delta t`$ | 1e-13 (momentum), 1e-12 nm/ps (velocity) |
| `testFluidMomentumRemoval` | 4x4x4 nodes, $`\tau = 1`$, initial uniform velocity, body force $`\mathbf F`$ per node, removal frequency 3 | total momentum $`((n-1) \bmod 3 + 1)\,\mathbf F`$ per node after $`n = 1,\dots,7`$ steps: removal on step indices 0, 3, 6, before the collision | 1e-13 |
| `testShearWaveViscosity` | 2x64x2 nodes, $`u_x = 10^{-3}\sin(2\pi y/64)`$ in lattice units, $`\tau`$ = 0.6, 1, 1.5; amplitude at steps 200 and 1200 | decay rate $`\nu k^2`$ with $`\nu = (\tau - 1/2)/3`$ | 2e-3 (relative) |
| `testMachNumberCheck` | 4x4x4 nodes, uniform flow at $`\mathrm{Ma} = 0.35`$, check every 10 steps | `getFluidMachNumber()` = 0.35; exception at step 10 with limit 0.3, none with the check off or with limit 0.5 | 1e-12 |
| `testLatticeParameters` | 4x6x8 nodes, $`\tau = 0.9`$ | `getLatticeParametersInContext()` returns $`\Delta x`$, $`\Delta t`$ and $`\tau`$ | 1e-12 |
| `testRelaxationTimeWarning` | $`\tau`$ = 0.503, 1, 2.2 | warning on stderr at Context creation only outside [0.505, 2] | exact |
| `testSolidNodeChecks` | solid node index out of range, repeated, or all nodes solid | exception at Context creation | exact |
| `testPoiseuille` | 2x12x2 nodes, solid plane $`j = 0`$, body force along x, $`\tau`$ = 0.7, 0.875, 1.2, $`4H^2/\nu`$ steps | steady profile equal to the exact solution of the scheme (`docs/theory.md`, solid nodes), zero density and velocity at the solid nodes | 1e-9 (relative to the maximum velocity) |
| `testWallConservation` | 6x5x4 nodes with a solid block of 8 nodes, initial uniform flow, without and with momentum removal | mass of the fluid conserved; after a removal step the momentum of the fluid is zero | 1e-13 |
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

A first version wrote the returned population inside the collision kernel, without a separate pass. It is the
same arithmetic, and it was 3 % faster in a channel and 8 % faster in a porous medium, but the compiler then
rounded the collision differently in some variants of the kernel: with CUDA in single precision the results
differed in the last bit of a float, and with CUDA in double and mixed precision a fluid with fluctuations at
zero temperature and coupled particles was no longer identical bit for bit to a fluid without fluctuations
(`testFluctuationsAtZeroTemperature`). The separate pass copies stored values and does not depend on the
rounding of the compiler, so it was kept.

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
| Repeated force evaluation | 8000 particles with a short-range `CustomNonbondedForce`, compressed into a cube of 2 nm so that the neighbor list overflows and the GPU platforms repeat the force evaluation of the step: the total momentum is conserved in that step (`testRepeatedForceEvaluation`; GPU platforms only) | 1e-11 relative (1.3e-2 before the fix, on CUDA and OpenCL) |

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
(`testFullStepKineticEnergy` checks it on every platform; `docs/theory.md`, section 2). Before the change of the
coupling forces between steps to those of the next step it read 359 K.

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
pass. A first version of the GPU kernel enlarged OpenMM's buffer of random numbers in the first step: with OpenMM
8.3.1 the restart test then failed on CUDA and OpenCL in all precisions (the restarted fluid was a different
realization, with populations such as 5.4e-3 against 5.2e-4), because OpenMM's checkpoint reads the buffer back with
the size it has in the new Context; with 8.6.1 it passed. The buffer is now enlarged when the Context is created
(`docs/theory.md`, section 7). Cost of the fluctuations on the A100 (CUDA, mixed precision): 59.8 → 76.6, 168.9 →
240.4 and 1115.8 → 1590.4 us per step on $`32^3`$, $`64^3`$ and $`128^3`$ nodes. Without fluctuations, and with
fluctuations at zero temperature, the fluid, positions and velocities after 40 steps with walls, body force and six
coupled particles (explicit and centred drag, EM and NVE) are identical, bit for bit, to those of commit `85029a7`
(version 0.2.1 with a change of the build only).

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
shell (6 wave vectors) is 1.5 - 2 %, of the shells from 4 on below 0.6 %. CUDA in single and double precision and
OpenCL in mixed precision draw the same random numbers and give the same values as CUDA mixed to the digits shown; the
Reference platform ($`16^3`$ nodes, its own generator) gives ER per node 1.002, 1.001, 1.003 and spectra within 2.5 %,
within its larger statistical error. The mean velocity is subtracted at every sample: in the run with the body force
it grows from 0 to 0.05 and the spectra are those of the fluid at rest. In the uniform flow and in the accelerated
fluid the ER per node of the velocity along the flow is 0.6 % and 0.2 % above that at rest, an effect of the
second-order equilibrium of D3Q19 at Mach 0.09, which grows as $`u^2`$. Script `velocity_spectra.py` of the campaign;
the test `testVelocitySpectrum` checks the same quantities on $`8^3`$ nodes on every platform (Fluctuating fluid,
above).

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

**Measured** (NVIDIA A100, OpenMM 8.6.1 and 8.3.1): the C++ tests pass on the Reference platform and on CUDA and
OpenCL in the three precisions, and the Python tests pass. The GPU platforms
against the Reference platform with regularized walls and open faces: [below](#gpu-platforms-against-the-reference-platform-fluid-and-walls).

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
$`\tau`$ of water (about 0.6) the bounce-back wall is about four times more accurate. The first version of the
regularized walls of this release (the local regularized condition of Latt, `docs/theory.md`, section 1) had
$`8\lvert\tau - 1\rvert/(3H^2)`$, with the wall on the boundary nodes, and was replaced because of its fluctuations
(below).

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
momentum along the wall is 3 to 4 % low and some stress and ghost modes up to 9 % low. The first version of the
regularized walls (Latt) was much further from equilibrium: on the boundary node the momentum was zero (imposed)
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
  error. With the first version of the regularized walls (Latt), the deficit of the first fluid node depended on
  the wavelength: the normal velocity had ER 0.72 at the longest wavelength (16 nodes) and 0.97 at the shortest,
  the density 0.83 - 0.86 at the longest; imposing the velocity on the boundary node damped the long-wavelength
  fluctuations next to it.
- `Density` faces (16 x 16 x 16, faces YMin and YMax at the fluid density, a solid plane $`x = 0`$): on the nodes of
  the faces the velocity along the face fluctuates at equilibrium (0.96 - 0.99), the velocity across the face 51 %
  more than in equilibrium with bounce-back walls and 38 % more with regularized walls, and the density 0.43 and
  0.57 of the equilibrium variance; the next plane is within 2 % and the following ones within the statistical
  error. With the first version of the faces (Latt), the density and the velocity along the face were imposed on
  the face nodes (ER 0) and the velocity across it fluctuated 21 to 29 % more. Without the solid plane, with x and
  z periodic, the mean flow across the two faces has no restoring force (equal pressures, no friction): the
  fluctuations make it wander like a free Brownian particle, and the run stopped with a Mach number of 0.3 after
  87000 steps (first version of the faces). A duct with walls is stable (mean velocity below 1e-3), because the
  viscous friction at the walls damps the mean flow (`docs/theory.md`, section 1, Open faces).

**Duct driven by a difference of density** (8 x 16 x 8 nodes, bounce-back walls, `Density` faces at 1.01 and 1): the
flow is steady (1e-18 between two steps) with a uniform mass flux, the staggered mode at 1e-16. The density falls
linearly in the middle, but on the face nodes it overshoots the densities of the faces (1.0104 and 0.9996 at
$`\tau = 1`$): next to the faces the flow enters and leaves the duct, and the gradient in the middle is that of a
length of 14.6 nodes instead of the 17 between the nodes beyond the faces. With that gradient the largest error in the
middle cross-section relative to the incompressible solution is 0.9 % at $`\tau = 1`$ (15 % with the gradient of 17
nodes), and 1.3 % at $`\tau = 1.1`$ in the example of the user guide (gradient of a length of 7.3 nm for 8.5 nm
between the nodes beyond the faces). With the first version of the faces (Latt), which imposed the densities on the
face nodes, the gradient was that of the faces, and the errors were 1.3 % at $`\tau = 0.6`$ and 0.5 % at $`\tau = 1`$
(10 x 32 x 10).

**Staggered mode.** With the faces of this release the staggered momentum is at 1e-16 in the duct above with the time
filter; without it, it decayed by itself at $`\tau = 1`$ (to 1e-12 in 3000 steps) and slowly at $`\tau = 0.6`$, and
the `Density` inlets were unstable at $`\tau = 0.6`$ (above). With the first version of the faces (Latt), without the
time filter of the `Density` faces (`docs/theory.md`, section 1, Time filter of the Density faces), the duct between
bounce-back walls kept an oscillation of the velocity from one node to the next and from one step to the next, of 0.6
to 2.9 % of the velocity at $`\tau = 0.8`$ (at $`\tau`$ = 0.6 and 1 the mass flux through neighbouring cross sections
differed by 9 to 18 %): the staggered momentum $`\sum_y (-1)^{y+t} j_y`$, an exact invariant of the bulk (eigenvalue
-1 of the linearized step at $`k = \pi`$, for every $`\tau`$), was excited by the start and then kept constant (-0.047
in lattice units from step 2000 to 16000). With regularized walls, or with `Velocity` faces, it was at the level of
rounding. In that version the time filter ($`\beta = 1/2`$) damped it to rounding (1e-16) in every combination of
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
| Fluid | the `.vti` file has the extent of the lattice, spacing $`\Delta x`$ and origin 0; its density and velocity equal those of `getFluidFields()`; the solid nodes are flagged; the field data array `units` lists the unit of each array | 1e-6 relative in single precision, exact in double |
| Particles | the `.vtp` file has the positions of the State wrapped into the box, its velocities, the masses, the indices, the coupled flags and one vertex per particle, and the array `units`; with `wrap=False` the position outside the box stays outside | as above |
| Series | the `.pvd` file lists the fluid and particle files of each report with the time in ps; with `append=True` a new reporter keeps the files already listed | exact |
| Parts | `fluid=False` and `particles=False` write only the other part | exact |
| No effect on the run | a run with the reporter equals, bit for bit, the run without it | bitwise |

**Measured** (OpenMM 8.6.1): all pass. The files written by the test were also read with the VTK readers of
ParaView 5.13 (`vtkXMLImageDataReader`, `vtkXMLPolyDataReader`, the `.pvd` reader): dimensions, spacing, the
coordinates of the nodes, the fields and the times agree, within 5e-8 relative in single precision and exactly in
double precision, and the readers return the strings of the `units` array of the field data.

## Domain decomposition (`python/tests/mpi_decomposition.py`, Reference, MPI)

The script runs each case on every rank twice, with one domain and with the decomposition given on the command line,
and compares the populations of the fluid nodes that the rank owns (`docs/theory.md`, section 8). Lattice
$`8 \times 6 \times 6`$, $`\tau = 0.8`$, a body force, an initial velocity and populations perturbed by up to
$`10^{-3}`$, 60 steps; cases: periodic; the solid plane $`j = 0`$ and a block of 8 solid nodes with bounce-back walls,
with regularized walls, and with regularized walls and open faces along x (a `Velocity` inlet and a `Density` outlet);
periodic with the removal of the fluid momentum every third step. Run with OpenMPI 4.1.6 on one node, plugin built with
`-DOPENMM_LBM_MPI=ON`, OpenMM 8.6.1, decompositions $`2 \times 1 \times 1`$, $`1 \times 2 \times 1`$,
$`1 \times 1 \times 2`$, $`2 \times 2 \times 1`$, $`4 \times 1 \times 1`$, $`1 \times 2 \times 2`$ and
$`2 \times 2 \times 2`$:

| Case | Fluid nodes of each rank | Force on the walls (relative) |
|---|---|---|
| periodic, bounce-back, regularized walls, open faces | identical bit for bit on every rank and decomposition (104 of 104) | at most $`3.3 \cdot 10^{-13}`$ |
| removal of the fluid momentum | at most $`1.1 \cdot 10^{-19}`$ | |

The force on the walls and the removal differ by rounding because the ranks add their sums in another order. The
slots of the solid nodes are not compared: they hold what the fluid nodes pushed into them, on the rank of each
fluid node, and are not part of the state of the fluid.

**Coupled particles.** Seven particles of 50 Da, with a constant field and a soft pair force, at $`T = 0`$: near the
borders of the blocks, two on the same node, one crossing the periodic boundaries and two moving into the solid
plane $`k = 0`$, where they are reflected; friction 10 ps⁻¹, 60 steps. Cases: explicit drag with bounce-back walls,
centred drag with regularized walls, centred drag with regularized walls and open faces along x. On every rank the
script compares the fluid nodes of the rank and the positions and velocities of all the particles. Decompositions
$`2 	imes 1 	imes 1`$, $`1 	imes 2 	imes 1`$, $`1 	imes 1 	imes 2`$, $`2 	imes 2 	imes 1`$,
$`1 	imes 2 	imes 2`$ and $`2 	imes 2 	imes 2`$ (22 ranks in all):

| Case | Fluid nodes and particles | Force on the walls |
|---|---|---|
| explicit drag, centred drag, open faces | identical bit for bit on every rank and decomposition (66 of 66) | absolute difference at most $`2.9 \cdot 10^{-9}`$ kJ/mol/nm |

Without a body force the force on the walls, 58 to $`2.5 \cdot 10^{4}`$ kJ/mol/nm here, is a small difference between
the pressure forces on the two sides of the solid plane, each about $`ho c_s^2`$ times its area,
$`6.0 \cdot 10^{6}`$ kJ/mol/nm: the difference is $`5 \cdot 10^{-16}`$ of that, the rounding of the sums added in another
order. Relative to the net force it reaches $`3 \cdot 10^{-11}`$, which is why the script compares it with the
pressure force.

With a fluctuating fluid (centred drag, $`T = 300`$ K, 50 steps) the ranks draw different random numbers, so the run
is not compared with one domain; the copies of the particles stay identical on every rank (the plugin compares them
every 10 steps, and the hash of positions and velocities printed by every rank is the same). The checks that must stop
every rank together all do, without any rank waiting: copies of the particles that differ by $`10^{-12}`$ in the
velocities of one rank (stopped at the first step), an `AndersenThermostat` (refused when the Context is created) and,
with $`2 	imes 1 	imes 1`$ and $`2 	imes 2 	imes 1`$, rank 0 on the Reference platform and the others on OpenCL
(50 of 50).

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
  populations of order 0.05 at the ninth digit. Before the populations were stored as deviations from the
  rest equilibrium (`docs/theory.md`, section 4), the drift of openmm-lbm was 3e-5.

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

With the open faces (one particle on the nodes of the `Density` face, measured after the Reference platform took the
velocity of the GPU platforms on those nodes, OpenMM 8.6.1): positions 3e-14, velocities 3e-13 and fluid 2e-16 in
double precision (CUDA and OpenCL, both drags); positions 1e-9, velocities 9e-9 and fluid 1e-12 to 2e-12 in mixed
precision; positions 4e-7, velocities 1.4e-7 and fluid 2e-9 with CUDA in single precision. Before that change the
particle on the face differed by 3e-3 nm after 60 steps.

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

The reference outputs of the old examples (`*.LBLatticeOn.dat`) came from the Langevin scheme of the old plugin, in
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
- **Friction 30/ps** (`--preset fabio-g30`, 50 ns, seed 1): both drags are stable; the apparent diffusion
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
protein (next sections).

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

**With the centred drag** (same cases and seeds; the scripts of the campaign, which are not part of this
repository, were run with the centred drag). The
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

T7, with the kick response T5_m1000_g10_L16 computed with the same drag on CUDA (with the explicit drag it
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
friction 10/ps, box 15 nm, 200 ns, and `fabio-g30`: friction 30/ps, box 30 nm, 50 ns; one run each, seed 1, the
same as the first run without fluctuations). Temperatures in K (target 298 K) and apparent diffusion coefficient
of the centre of mass, $`\mathrm{MSD}(\mathrm{lag})/(6\,\mathrm{lag})`$, in Å²/ns at lags of 0.1, 1, 2 and 3 ns:

| preset | drag | fluid | full-step $`T`$ | half-step $`T`$ | D at 0.1 / 1 / 2 / 3 ns |
|---|---|---|---|---|---|
| sod1 | explicit | without fluctuations | 293.6 | 309.8 | 2.32 / 2.40 / 2.37 / 2.37 |
| sod1 | centred | without fluctuations | 264.1 | 277.2 | 2.32 / 2.40 / 2.38 / 2.39 |
| sod1 | explicit | fluctuating | 316.4 | 333.7 | 4.91 / 4.91 / 4.76 / 4.77 |
| sod1 | centred | fluctuating | 284.8 | **298.6** | 4.99 / 5.11 / 4.87 / 4.70 |
| fabio-g30 | explicit | without fluctuations | 294.6 | 349.7 | 0.89 / 0.98 / 1.00 / 1.00 |
| fabio-g30 | centred | without fluctuations | 220.5 | 250.2 | 0.86 / 0.93 / 0.96 / 0.99 |
| fabio-g30 | explicit | fluctuating | 355.6 | 419.8 | 3.67 / 3.39 / 3.21 / 3.18 |
| fabio-g30 | centred | fluctuating | 264.6 | **298.7** | 3.68 / 3.34 / 3.25 / 3.29 |

With the fluctuating fluid and the centred drag the protein has the set temperature, and its diffusion coefficient
gains the hydrodynamic contribution of the solvent (twice and three times the value without fluctuations, which is
close to $`k_BT/(M\,\text{friction})`$), the same for both drags. The explicit drag is too hot, by 6% and 19%, as
$`T\,(1 + \text{friction}\cdot\Delta t\,m/(2m_c(1 + \zeta y_{\mathrm{explicit}})))`$ predicts with the mean bead mass
(6% and 18%).

