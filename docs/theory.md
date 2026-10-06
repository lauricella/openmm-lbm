# Model, units and conventions

This document describes the physics implemented by `LBMForce`, the conversion between OpenMM and
lattice units, and the conventions every platform must follow. Each section says whether it is
already implemented (version 0.1.0) or still to be implemented; the target scheme is the one of the
CUDA lattice Boltzmann library of the DragOpenMM project, which this plugin reproduces.

## 1. Fluid model (implemented on the Reference platform)

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
- f_i^neq,reg = w_i/(2 cs^4) H2(c_i) : Pi^neq is rebuilt from the non-equilibrium stress
  Pi^neq = sum_i H2(c_i) (f_i - f_i^eq(rho, j/rho)), with the second-order Hermite polynomial
  H2(c) = c c - cs^2 I. In Pi^neq the -cs^2 I part of H2 does not contribute, because the equilibrium
  has the same density as the populations and sum_i (f_i - f_i^eq) = 0: Pi^neq is also the plain
  second moment of f - f^eq, which is how the reference library computes it. In f^neq,reg the -cs^2 I
  part gives the term -cs^2 tr(Pi^neq), without which f^neq,reg would carry mass;
- S_i = w_i [(c_i - u*)/cs^2 + (c_i.u*) c_i/cs^4] . F is the Guo source.

The prefactor of S_i is 1/2, not the (1 - omega/2) of the BGK form. The equilibrium is already
shifted by F/2 and the regularized f^neq has no first-order part, so with 1/2 the momentum increases
by exactly F per step, at every relaxation time [3, 4].

- F = rho g for a body acceleration g (lattice units); the coupling forces are added to it.

**Relaxation and streaming.** The relaxation frequency is omega = 1/tau, with
tau = 3 nu dt/dx^2 + 1/2.

**Velocity of the fluid.** `getFluidFields()` reports u = (j + F/2)/rho, the velocity of the forced
model, with F from the body acceleration.

The update is thread-safe and uses a single copy of the populations:

1. A first kernel computes rho, j and Pi^neq at every node from its populations.
2. A second kernel reads only these moments at a node and writes the 19 post-collision populations
   to the neighbouring nodes (push streaming).

Each population of the destination is written by exactly one thread.

### Solid nodes (implemented on the Reference platform)

`setSolidNodes()` marks lattice nodes as solid walls at run time; an empty list (the default) is a fully
periodic fluid.
- Solid nodes hold no fluid: their populations start at zero, they have no moments and no collision, and
  they do not enter the removal of the fluid momentum.
- After streaming, a population that reached a solid node s from the fluid node s + c_q is sent back to
  that node with the opposite velocity: f_q(s + c_q) = f_opp(q)(s). This halfway bounce-back places the
  wall halfway between the fluid and the solid node and conserves the mass of the fluid. It is the scheme
  of the reference implementation.

**Exact Poiseuille flow.** With the regularized collision, the odd non-hydrodynamic moments relax with
frequency 1 (tau_odd = 1), so the scheme behaves at the walls as a two-relaxation-time scheme with
magic parameter Lambda = (tau - 1/2)(tau_odd - 1/2) = (tau - 1/2)/2 [8]. For a channel between the
walls of the solid plane j = 0 (the lattice is periodic, so the plane bounds the channel on both sides)
driven by a body acceleration g, the steady profile of the scheme is exactly, in lattice units,

  u(y) = g/(2 nu) (y - 1/2)(ny - 1/2 - y) + g (16 Lambda - 3)/(24 nu).

The curvature is the exact one for every tau; the second term is a slip that shifts the effective wall by
(3 - 16 Lambda)/(12 H), with H = ny - 1, and vanishes at Lambda = 3/16, that is tau = 7/8. The
validation tests check this profile to 1e-9 (`docs/validation.md`).

**Force on the walls: momentum exchange.** The momentum that the fluid gives to the solid nodes is
measured with the momentum exchange method of Ladd [9, 10] ([11], section 5.4.3.1, eqs. 5.79 and
5.80).
- On every boundary link, from a fluid node x_f to a solid node x_s = x_f + c, the population f that
  streams into the wall along c comes back along -c. The wall at rest receives the momentum
  f c - f (-c) = 2 f c.
- The sum over all boundary links, times m_c dx/dt, is the momentum given to the walls in one step.
- The coupled particles also exchange momentum with the walls. The reaction -F of a particle whose
  nearest node is solid goes to the wall, and the reflection of a particle gives the wall the momentum
  2 m v.

`getWallForce()` returns the sum of these contributions over the last lattice step, divided by dt. With
it the total momentum of particles, fluid and walls is conserved, and in a steady channel flow the force
on the walls equals the body force on the fluid (`docs/validation.md`).

**Planned: open faces with imposed density or velocity.** Nodes will be of three kinds: fluid, solid,
and wet (fluid nodes with at least one solid neighbour). A wet node rebuilds every population that comes
from a solid neighbour as f^eq(rho, u_bc) + (1 - omega) f^neq,reg(Pi^neq), with the prescribed quantity
(velocity or density) and the unknown one (density or velocity) and Pi^neq taken from the wet node
itself; the wall lies halfway along the link. The equilibrium is the weakly compressible one, with rho
multiplying the whole Hermite expansion.

## 2. Particle-fluid coupling (implemented on the Reference platform)

**Euler-Maruyama scheme.** Each coupled particle k of mass m_k feels

  F_k = -gamma m_k (v_k - u(x_k)) + R_k,   <R_k R_k> = 2 gamma m_k kT/dt (per component).

- u(x_k) = j/rho at the nearest lattice node (section 3, Nearest node).
- The fluid at that node receives -F_k. The reactions of the particles at the same node are summed in
  particle order and added to the body force rho g of the node.
- In lattice units (time step 1): F = -gamma m (v - u) + sqrt(2 gamma m kT) xi, with xi three
  independent N(0, 1) numbers.
- Drag and noise are part of the force, so the System is integrated with `VerletIntegrator`.
- This is the explicit scheme of the reference CUDA library. openmm-lbm reproduces it first, and studies
  changes (time-centred drag, relaxation of the ghost moments) only afterwards.

**Order in the lattice step.** Moments, removal of the fluid momentum, coupling, collision and
streaming, bounce-back. The coupling therefore sees the fluid momentum after the removal.

**Time levels.** OpenMM's Verlet integrator is a leapfrog: during the force evaluation of step t
the velocities are v(t - dt/2). The fluid momentum before the step is j(t - dt/2), because
j(t + dt/2) = j(t - dt/2) + F(t). The drag therefore compares particle and fluid velocities at the
same half step, explicitly.

**Forces once per step.** The coupling forces are computed once per integration step, in the force
evaluation that advances the fluid, and are kept until the next step. Every other force evaluation
(`getState(getForces=True)`, `setVelocitiesToTemperature()`) applies the same forces: it does not
draw new random numbers, so it does not change the trajectory. Before the first step the coupling
forces are zero. The coupling is dissipative and has no energy.

**Random force.** On the Reference platform the random numbers come from a generator owned by the
force (SFMT, Box-Muller transform).
- It is seeded with `setRandomNumberSeed()`, or with a unique seed when the seed is 0.
- Its sequence does not depend on the other forces of the System, and the same seed reproduces a
  simulation.
- Its state is not part of OpenMM checkpoints.

On the GPU platforms the random numbers will come from OpenMM's generator, as in the reference library.

**Walls.** A coupled particle whose nearest node is solid has entered a wall.
- At the start of the step (`updateContextState()`, where OpenMM's `AndersenThermostat` also changes
  velocities) every component of its velocity is reversed, as for a no-slip wall, if the particle moves
  into the wall: v.n > 0.
- n is the gradient of the solid indicator (1 at solid nodes, 0 at fluid nodes), interpolated
  trilinearly between the eight nodes of the lattice cell that contains the particle. It points from the
  fluid into the wall. For a wall one node thick, the cell of the particle tells from which side it came.
- A particle at a solid node that already moves out of the wall keeps its velocity, so it is not sent
  back into the wall by a second reversal.
- In the step the particle feels the drag of the wall at rest (u = 0) and the random force.
- The reaction on the solid node leaves the fluid, since solid nodes do not collide.
- Uncoupled particles do not see the walls.

**Stability of the explicit drag.** In one step the drag multiplies the velocity of a particle
relative to the fluid by 1 - gamma dt. It changes sign at every step for gamma dt > 1, and it grows
without bound for gamma dt >= 2. A warning is printed when a Context is created with gamma dt > 1.

**Self-mobility and relaxation time.** The mobility of a dragged particle is 1/(m gamma) + y, where y is
the hydrodynamic self-mobility from the fluid around its node. y should depend only on the viscosity
(y ~ 1/eta), but with the explicit drag at the nearest node it contains a lattice term that does not
decrease with the viscosity. As a result y eta dx, which should not depend on tau, falls with tau and
changes sign. Measured on the Reference platform (L = 8 nm, dx = 0.5 nm, dt = 0.01 ps, m = 1000 Da,
gamma = 5/ps, protocol of `docs/validation.md`):

| tau | 0.62 | 0.8 | 1.1 | 1.5 | 1.6 | 1.7 | 1.8 | 1.9 | 2.0 | 3.51 |
|---|---|---|---|---|---|---|---|---|---|---|
| y eta dx | 0.0609 | 0.0577 | 0.0442 | 0.0202 | 0.0134 | 0.0065 | -0.0007 | -0.0080 | -0.0154 | -0.1406 |

y vanishes at tau = 1.79 (1.79 also for m = 100 Da, gamma = 10/ps), the value found with the reference
library. Above it, particles move less than Langevin particles with the same friction. A warning is
printed when a Context with coupled particles has tau > 1.7. Time-centred drag and a relaxation of the
ghost moments independent of tau are the candidate corrections, to be studied.

**Kinetic temperature.** OpenMM's leapfrog stores the velocities at half steps.
- For a free particle with fluid at rest, the temperature measured from half-step velocities is
  T/(1 - gamma dt/2). Measured from full-step velocities v(t) = (v(t - dt/2) + v(t + dt/2))/2, it is T.
- The fluid has no thermal fluctuations of its own and takes part of the momentum of the particles, so
  the particles are slightly colder than T. Measured on the Reference platform with 200 free beads of
  100 Da, dx = 0.5 nm, dt = 0.01 ps, tau = 1.10, gamma dt = 0.1 and T = 300 K: 295.8 +- 0.4 K from
  full-step velocities and 311.7 +- 0.4 K from half-step velocities, where T/(1 - gamma dt/2) = 315.8 K.
- **The temperature reported by OpenMM is not valid for coupled particles.** This is the kinetic energy
  of the State, used by `StateDataReporter`. For `VerletIntegrator`, OpenMM shifts the velocities by half
  a step with the forces of the current evaluation, which for the coupling are those of the last step,
  random force included. For a free particle this gives T [(1 - 3a/2)^2/(1 - a/2) + 9a/2], with
  a = gamma dt: 363 K at 300 K and a = 0.1. 359 K was measured in the run above. Compute the
  temperature from full-step velocities instead (user guide, examples).

**Per-cell reaction on the GPU platforms (to be implemented).** The reaction forces of the particles in
the same cell will be summed without atomic operations:

1. the (cell, particle) pairs are sorted with OpenMM's `ComputeSort`, using unique keys;
2. one thread per cell adds the forces of its particles, in particle order.

The result will be bitwise reproducible, and equal to the sum in particle order of the Reference
platform.

**Momentum removal.** When it is enabled, on the steps whose index is a multiple of the removal
frequency, the momentum of the fluid is removed right after the
moments are computed and before the coupling. The index of a step is the step count of the Context
when the step starts: 0 for the first step of a new Context, and restored by checkpoints, so a run
restarted from a checkpoint removes the momentum at the same steps as an uninterrupted one. The
default frequency is 1, every step. The plugin sums rho and j over the lattice, computes u_cm = sum(j)/sum(rho) and applies
j <- j - rho u_cm at every node; Pi^neq is left unchanged. The populations are then rebuilt from
the corrected moments by the collision. The sums use two-stage reductions without atomic
operations.

**Fluid update.** The fluid is advanced once per time step. `VerletIntegrator::step()` calls
`ContextImpl::updateContextState()` before computing the forces of each step; `LBMForceImpl` uses that
call to mark the next force evaluation as the one that advances the fluid. This is how OpenMM's own
forces with internal state (`CMMotionRemover`, `AndersenThermostat`, `MonteCarloBarostat`) act once
per step. Other force evaluations (`getState(getForces=True)`, `setVelocitiesToTemperature()`) do not
advance the fluid and will reuse the coupling forces already computed.

## 3. Units and conversions (implemented)

The public API uses OpenMM units: nm, ps, Da (g/mol), K and kJ/mol. Internally the plugin works in
lattice units only, as is usual for lattice Boltzmann [7]. The conversion is computed in one place,
`LBMForceImpl::computeLatticeParameters()`:

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
| particle mass | m_c | m_lattice = m/m_c, with m from the System |
| friction (to be implemented) | 1/dt | gamma_lattice = gamma dt |
| thermal energy (to be implemented) | m_c dx^2/dt^2 | kT_lattice = kT dt^2/(m_c dx^2) |

With these units the drag and the noise keep their form on the lattice (time step 1):
F_lattice = -gamma_lattice m_lattice (v_lattice - u_lattice) + sqrt(2 gamma_lattice m_lattice
kT_lattice) xi. This is the physical force times dt^2/(m_c dx), as it must be. Forces return to
OpenMM multiplied by m_c dx/dt^2, in Da nm/ps^2 = kJ/mol/nm.

**Example** (water-like fluid used for the SOD1 runs):

- rho0 = 602.2 Da/nm^3, nu = 5.0175 nm^2/ps, box 15 nm with 30 nodes, dt = 0.01 ps;
- this gives dx = 0.5 nm, m_c = 75.275 Da and tau = 1.1021;
- the velocity unit is 50 nm/ps (lattice sound speed 28.9 nm/ps) and the force unit is
  3.764e5 kJ/mol/nm;
- at T = 298 K, kT_lattice = 1.3166e-5;
- gamma = 5 /ps gives gamma_lattice = 0.05.

**Correspondence with the reference implementation.** These conversions coincide with those of the
CUDA lattice Boltzmann plugin of the DragOpenMM project, which uses the same base units dx, dt and
m_c. There are two interface differences:

- That plugin takes dt and the box from explicit arguments. openmm-lbm reads them from the
  integrator and the System.
- Its body force argument is a force per cell divided by the density, that is g dx^3 for an
  acceleration g. openmm-lbm takes the acceleration g directly.

**The time step is set by the molecular dynamics.** In a stand-alone LB simulation dt is a numerical
parameter, chosen to keep the Mach number small [7]. Here the fluid advances once per integrator step,
so dt is the step size of OpenMM. Viscosity and resolution are therefore coupled: tau - 1/2 =
3 nu dt/dx^2. With dt = 0.01 ps:

| dx | water, nu = 1.0035 nm^2/ps | 5 x water |
|---|---|---|
| 0.5 nm | tau = 0.62 | tau = 1.10 |
| 0.25 nm | tau = 0.98 | tau = 2.91 |

Typical velocities are far below the lattice sound speed. A 100 Da bead at 298 K has Ma = 5e-3, so
the fluid is in the quasi-incompressible regime for which the model is accurate.

**Nearest node.** Lattice node (i, j, k) sits at (i dx, j dx, k dx). A particle at x belongs to the
node i = round(x/dx) mod n, after wrapping x into the box. Node i therefore owns the interval
[(i - 1/2) dx, (i + 1/2) dx).

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

## 6. Stability checks (implemented)

The model is accurate only in the quasi-incompressible regime and for relaxation times in a moderate
range. The plugin checks both.

**At Context creation.**
- A relaxation time tau <= 1/2 is an error, as it is today.
- A warning is printed on stderr if tau is outside [0.505, 2], the range used by the reference
  implementation.

**During the simulation.**
- Every N steps the plugin computes the largest Mach number of the fluid, Ma = max |u|/c_s, with
  u = j/rho and c_s = 1/sqrt(3) in lattice units.
- N is set with `setMachCheckFrequency()`; the default is 100 and 0 disables the check.
- If Ma exceeds the limit set with `setMachNumberLimit()` (default 0.3), the plugin throws an
  `OpenMMException` that reports the step and the value.
- `getFluidMachNumber(context)` returns the current value, for monitoring.
- The check runs after the lattice steps whose number is a multiple of N, with the steps numbered by
  the step count of the Context, as for the momentum removal. On the CUDA, OpenCL and HIP
  platforms it will run when the fluid update is ported there, with a two-stage reduction and no atomic
  operations, like the removal of the fluid momentum; `getFluidMachNumber()` already works there.

**Why 0.3.** It is the usual limit of the incompressible approximation: density fluctuations scale
as Ma^2, about 9% at Ma = 0.3. In addition, the second-order equilibrium of D3Q19 lacks the u^3
terms, so the viscous stress has an error of order Ma^3, and the stability margin shrinks quickly
as tau approaches 1/2. Beyond 0.3 the state is no longer physical, so the simulation is stopped.
In the target applications Ma is between 1e-3 and 1e-2: a 100 Da particle at 298 K with dx = 0.5 nm
and dt = 0.01 ps gives Ma = 5e-3.

**Debug builds** (CMake option `-DLBM_DEBUG=ON`, which defines the macro `LBM_DEBUG`):
- a warning is printed on stderr the first time Ma exceeds 0.1, where the accuracy starts to degrade;
- the lattice parameters are printed when a Context is created: dx, dt, m_c, tau,
  kT/(m_c c_s^2).

**Lattice parameters.** `getLatticeParametersInContext(context, dx, dt, tau)` returns the lattice
spacing, the lattice time step and the relaxation time, in the style of
`NonbondedForce::getPMEParametersInContext()`.

## References

1. J. Latt and B. Chopard, Math. Comput. Simul. 72, 165 (2006): regularized collision.
2. Z. Guo, C. Zheng and B. Shi, Phys. Rev. E 65, 046308 (2002): forcing scheme.
3. accLB, arXiv:2505.01126 (2025), eq. 6: f = f^eq + (1 - omega) f^neq + S/2.
4. LBFAST, arXiv:2609.09160 (2026), eq. 3.
5. P. Ahlrichs and B. Dünweg, J. Chem. Phys. 111, 8225 (1999): frictional particle-fluid coupling.
6. B. Dünweg and A. J. C. Ladd, Adv. Polym. Sci. 221, 89 (2009): review of lattice Boltzmann for soft matter.
7. J. Latt, Choice of units in lattice Boltzmann simulations, LBMethod.org (2008).
8. I. Ginzburg, F. Verhaeghe and D. d'Humières, Commun. Comput. Phys. 3, 427 (2008): two-relaxation-time
   scheme, magic parameter Lambda and exact bounce-back solutions.
9. A. J. C. Ladd, J. Fluid Mech. 271, 285 (1994): lattice Boltzmann simulations of particulate
   suspensions, part 1, theoretical foundation.
10. A. J. C. Ladd, J. Fluid Mech. 271, 311 (1994): part 2, numerical results.
11. T. Krüger, H. Kusumaatmaja, A. Kuzmin, O. Shardt, G. Silva and E. M. Viggen, The Lattice Boltzmann
    Method: Principles and Practice (Springer, 2017).
