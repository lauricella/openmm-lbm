# Model, units and conventions

This document describes the physics implemented by `LBMForce`, the conversion between OpenMM and
lattice units, and the conventions every platform must follow. Each section says whether it is
already implemented or still to be implemented. The fluid and the explicit drag reproduce the CUDA
lattice Boltzmann library of the DragOpenMM project; the centred drag (section 2) is an extension of this
plugin.

## 1. Fluid model (implemented on all platforms)

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

Each population of the destination is written by exactly one thread. The removal of the fluid momentum
sits between the two kernels; on the CUDA, OpenCL and HIP platforms its sums are reduced in two stages
(by work group, then over the work groups in a fixed order), without atomic operations, so that runs are
reproducible. The kernels of these platforms (`platforms/common/src/kernels/lbmFluid.cc`) repeat the
arithmetic of the Reference platform; they differ from it only by rounding, since the GPU compilers
contract multiplications and additions into fused multiply-adds.

### Solid nodes and walls

`setSolidNodes()` marks lattice nodes as solid walls at run time; an empty list (the default) is a fully
periodic fluid. `setWallScheme()` chooses the boundary condition of the fluid at the solid nodes: the
halfway bounce-back (`BounceBack`, the default, described first) or the local regularized wall
(`Regularized`, described below).
- Solid nodes hold no fluid: their populations start at zero, they have no moments and no collision, and
  they do not enter the removal of the fluid momentum.
- After streaming, a population that reached a solid node s from the fluid node s + c_q is sent back to
  that node with the opposite velocity: f_q(s + c_q) = f_opp(q)(s). This halfway bounce-back places the
  wall halfway between the fluid and the solid node and conserves the mass of the fluid. It is the scheme
  of the reference implementation.
- Only the links from a solid node to fluid nodes are processed. Nothing streams between two solid nodes,
  and skipping those links makes the result independent of the order in which the solid nodes are
  processed, so that the GPU platforms, which process them in parallel, give the same populations as the
  Reference platform.

**Exact Poiseuille flow with bounce-back.** With the regularized collision, the odd non-hydrodynamic moments relax with
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
- On the CUDA, OpenCL and HIP platforms one thread per solid node computes the part of the deviations,
  -2 sum c_q (f - w), with c_q pointing from the solid node to its fluid neighbour, and the host sums it over
  the solid nodes in the order of the list. The part of the
  weights, -2 sum c w, is the static pressure: it depends only on the geometry and is computed once, in
  double precision. Kept apart, it does not hide the hydrodynamic part in single precision.
- The coupled particles also exchange momentum with the walls. The reaction -F of a particle whose
  nearest node is solid goes to the wall, and the reflection of a particle gives the wall the momentum
  2 m v. On the GPU platforms each particle stores its contribution of the last step (with the centred
  drag the reaction -S of a solid node is stored with the first particle of the node), and the host sums
  them in particle order.

`getWallForce()` returns the sum of these contributions over the last lattice step, divided by dt. With
it the total momentum of particles, fluid and walls is conserved, and in a steady channel flow the force
on the walls equals the body force on the fluid (`docs/validation.md`).

**Regularized walls** (`setWallScheme(Regularized)`, Reference platform; the GPU platforms refuse it for
now). The walls follow the local regularized boundary condition of Latt [20, 21] (section 5.2 of [20];
"BC3" of Malaspinas [22], chapter 4), applied on the fluid nodes next to the solid nodes, the boundary
nodes. A boundary node lies on the wall: the velocity of the fluid there is that of the wall, zero, and
the wall is half a node further into the fluid than with bounce-back (in the channel between the solid
planes j = 0 and j = ny - 1 the walls are on j = 1 and j = ny - 2).
- After the streaming, the populations of a boundary node x that come from a solid node (direction q with
  x - c_q solid) are unknown. All 19 populations of the node are rebuilt as f^eq(rho, v) + f^neq(Pi), the
  regularized form of the bulk, with three quantities:
  - the velocity of the populations v = -g/2, so that the velocity of the fluid (j + F/2)/rho, with the
    body force F = rho g, is zero. The body force acts on the boundary nodes like on the others, and the
    collision keeps the velocity of the fluid zero there;
  - the density rho = sum of the known populations + sum of the populations that x sent into the solid
    nodes in this streaming (stored in the solid node x - c_q, direction opposite to q). The mass that
    enters the wall comes back, as with bounce-back, so the mass of the fluid is conserved exactly. This
    replaces eq. 5.3 of [20] (Zou and He [23]), which does not conserve the mass in unsteady flows: in a
    decaying flow with walls the mass changed by 2e-5, and with fluid fluctuations it drifts at random;
  - the stress Pi = sum_q H2(c_q) f^neq_q, with f^neq_q = f_q - f^eq_q(rho, v) for the known populations and
    the "bounce-back of the non-equilibrium part", f^neq_q = f^neq_opp(q), for the unknown ones; zero if the
    opposite direction is unknown as well (a node between two walls).
- Each boundary node reads only its own populations and the solid slots that it wrote itself in the
  streaming, so the scheme is local and thread-safe. A solid layer one node thick is allowed: the
  boundary nodes on both sides rebuild the populations that would come from it.
- The removal of the fluid momentum leaves the boundary nodes out (their velocity is that of the wall),
  and the reaction of a coupled particle whose nearest node is a boundary node goes to the wall. With the
  centred drag a boundary node is a wall of infinite mass at the velocity of the node.

**Exact Poiseuille flow with regularized walls.** In the channel above, driven by g, the steady profile of
the scheme is exactly, in lattice units,

  u(y) = g/(2 nu) (y - 1)(ny - 1 - y) + g (tau - 1)/(tau - 1/2)   for 1 < y < ny - 1,   u = 0 on the walls,

with the walls on the boundary nodes y = 1 and y = ny - 1 (for the single solid plane j = 0 of a periodic
lattice) and the density uniform. The second term is a slip of the fluid next to the walls that vanishes
at tau = 1. Both walls are accurate to second order: for a channel of width H the largest error relative
to the centre-line velocity is |8 tau - 7|/(3 H^2) with bounce-back and 8 |tau - 1|/(3 H^2) with regularized
walls. Bounce-back is the more accurate below tau = 15/16 (at the tau of water, about 0.6, by 30 %), the
regularized wall above. The validation tests check the profile to 1e-9.

**Which wall to choose.** Both are second order and conserve mass exactly. Differences:
- With fluid fluctuations bounce-back walls are in exact thermal equilibrium with the fluid: bounce-back
  permutes the populations and keeps the Gaussian equilibrium state, so the fluctuations of all modes
  keep their equilibrium variance next to the walls (`docs/validation.md`). The regularized wall rebuilds
  the stress and drops the ghost modes of its nodes: in a test at tau = 0.8 the fluctuations of the first
  fluid node next to the wall were 3 to 10 % below equilibrium (momentum normal to the wall 0.905, stress
  0.92, ghosts 0.91), within 1 % from the second node on.
- With Density faces (below) bounce-back walls let a spurious staggered mode survive, which the Density
  faces damp; regularized walls damp it themselves.
- The regularized wall places the wall on the fluid nodes, which can be convenient to define a geometry;
  the bounce-back wall is halfway between the fluid and the solid nodes.

**Force on the walls with regularized walls.** The walls receive the momentum of the populations that
stream into the solid nodes, minus the momentum that the rebuild adds to the boundary nodes, plus the
reactions of the particles at the boundary nodes. With it the momentum balance of the fluid,
P(t + dt) - P(t) = (M g - F_wall) dt, holds at every step, as with bounce-back.

### Open faces

By default the box is periodic. `setFaceBoundary(face, type)` makes a face of the box open: `Velocity` (the
fluid on the face has the velocity set with `setFaceVelocity()`: an inlet, an outlet or a moving wall) or
`Density` (the fluid on the face has the density, that is the pressure p = c_s^2 rho, set with
`setFaceDensity()`). The two faces perpendicular to an axis must be both periodic or both open, in any
combination of `Velocity` and `Density`. The six faces have independent velocities and densities.
- The nodes on an open face (i = 0 for `XMin`, i = nx - 1 for `XMax`, and so on) are boundary nodes, on
  the face: the populations that would come from beyond the face are unknown, and the populations that
  leave through the face are lost. All 19 populations of a face node are rebuilt as for the regularized
  walls, f^eq(rho, v) + f^neq(Pi), with v = u - g/2 and Pi from the known non-equilibrium parts and their
  bounce-back (Latt [20], section 5.2).
- On a `Velocity` face u is the velocity of the face, and rho follows from eq. 5.3 of [20] generalized to
  any set of unknown directions: rho = [sum of the known f_q + sum over the unknown q of (f_opp(q) +
  6 w_q rho c_q.v), or of f^eq_q(rho, v) if opp(q) is unknown too]. It is solved for rho in closed form.
- On a `Density` face rho is that of the face, the velocity along the face is zero and the velocity across
  it follows from the same balance (note 5.1 of [20]): rho = rho_0 + 2 rho_out + rho v_n, with rho_0 the sum of
  the populations along the face and rho_out the sum of those that leave the domain through it.
- **Staggered mode.** For any lattice whose velocities have components -1, 0 and 1, the staggered
  momentum sum_y (-1)^(y+t) j_y is conserved exactly by the bulk (the collision keeps the momentum of each
  node, the streaming moves a population by one node in one step). In a periodic box it stays zero; open
  faces can excite it. Velocity faces and regularized walls remove it, bounce-back walls keep it, and a
  `Density` face with the velocity above would send it back unchanged: in a duct between bounce-back walls
  driven by two Density faces the velocity then kept an oscillation from one node to the next and from one
  step to the next, with a mass flux that differed by up to 18 % between neighbouring cross sections. It is the lattice Boltzmann form of the pressure oscillations of collocated
  grids that staggered finite-difference grids avoid ([20], chapter 7). The `Density` faces therefore use
  the average of this velocity and of the velocity of the node in the previous step (read from the
  momentum of the node at the start of the step, so nothing more is stored and a restart with
  `setFluidState()` stays exact): the steady state is unchanged, and the staggered mode is damped to
  rounding (`docs/validation.md`).
- Nodes on several open faces (edges and corners): the first `Velocity` face in the order XMin, XMax, YMin,
  YMax, ZMin, ZMax gives the velocity; if all are `Density` faces, the first gives the density and the
  velocity is zero. A face node next to a solid node with regularized walls is a wall (velocity zero);
  with bounce-back walls the bounce-back is applied first and the face then treats the populations it
  returned as known. Links that cross an open face are never bounced back. Malaspinas [22] treats edges
  and corners with finite differences instead; the rule here keeps the scheme local.
- With open faces the fluid exchanges momentum with the outside, so its momentum cannot be removed:
  `setFluidMomentumRemovalFrequency(0)` is required. The reaction of a coupled particle whose nearest node
  is on an open face leaves the fluid through the face. The particles themselves still live in OpenMM's
  periodic box: a particle that crosses an open face reappears on the opposite one, so keep them away from
  the open faces.
- The body acceleration (`setBodyAcceleration()`) acts on the face nodes like on the others, and the
  velocity imposed on a face is the velocity of the fluid (j + F/2)/rho.
- Validation (`docs/validation.md`): a Couette flow between a face at rest and a moving face is linear to
  1e-14; a uniform flow from a Velocity inlet to a Density outlet is steady to rounding; a duct driven by a
  difference of density of 1 % agrees with the incompressible solution within 0.5 to 1.3 %.

## 2. Particle-fluid coupling (implemented on all platforms)

**Euler-Maruyama scheme with the explicit drag** (the default; a frictional coupling at the nearest node,
as in Ahlrichs and Dünweg [5], reviewed in [6]). Each coupled particle k of mass m_k feels

  F_k = -gamma m_k (v_k - u(x_k)) + R_k,   <R_k R_k> = 2 gamma m_k kT/dt (per component).

- u(x_k) = j/rho at the nearest lattice node (section 3, Nearest node).
- The fluid at that node receives -F_k. The reactions of the particles at the same node are summed in
  particle order and added to the body force rho g of the node.
- In lattice units (time step 1): F = -gamma m (v - u) + sqrt(2 gamma m kT) xi, with xi three
  independent N(0, 1) numbers.
- Drag and noise are part of the force, so the System is integrated with `VerletIntegrator`.
- This is the explicit scheme of the reference CUDA library. openmm-lbm reproduces it, and offers a
  time-centred drag as an alternative (Drag schemes, below).

**Coupling schemes** (`setCouplingScheme()`).
- `EulerMaruyama`, the default: friction and random force as above.
- `NVE`: friction only, with no random force whatever the temperature, that is the same scheme at zero
  temperature.
- With either scheme the total momentum of particles, fluid and walls is conserved. The total kinetic
  energy, sum of rho u^2/2 dx^3 over the fluid nodes and m v^2/2 over the particles, is not conserved. The
  drag dissipates about gamma m |v - u|^2 dt per step. The viscosity damps the motion of the fluid, and in
  an isothermal lattice Boltzmann model the energy damped by viscosity leaves the model instead of
  heating the fluid.
- Example: a particle kicked in a fluid at rest ends up moving with the fluid at m v0/(m + M), as in a
  perfectly inelastic collision. The momentum is conserved, and the kinetic energy falls by the factor
  m/(m + M).

**Kinetic energy budget (approximate).** With the NVE scheme the kinetic energy plus the energy dissipated
by viscosity and drag stays constant, but only in the hydrodynamic limit, to O(Ma^2, Kn^2). It is a check
of the model, not an exact conservation law like that of the momentum. In lattice units (dx = dt = 1, mass
in cells m_c):
- the kinetic energy is E = sum over the nodes of rho u^2/2, with u = j/rho, plus sum over the particles
  of m v^2/2;
- the viscous dissipation in one step is sum over the nodes of (tau - 1/2)/(2 tau^2 rho c_s^2)
  Pi_neq:Pi_neq. It follows from the Chapman-Enskog relation Pi_neq = -2 rho c_s^2 tau S and the
  dissipation 2 rho nu S:S with nu = c_s^2 (tau - 1/2), and is computed from Pi_neq of the state without
  finite differences;
- the drag dissipation in one step follows from the discrete update: the particle loses -F.(v_n + v_n+1)/2
  and the fluid, with Guo's forcing, receives -F at the velocity u - F/(2 rho) of the node, so the energy
  dissipated is -F.[(v_n + v_n+1)/2 - u + F/(2 rho)], with F = m (v_n+1 - v_n)/dt the coupling force on
  the particle, which holds for both drags.

Measured on the Reference platform (`python/tests/TestEnergyBudget.py`, tau = 1.1):
- a shear wave without particles closes to -3.1% of the initial energy with 16 nodes per wavelength and to
  -0.77% with 32: the residual falls as k^2, as an O(Kn^2) error should;
- a particle kicked in the fluid closes to -4.4%, the same at Mach numbers 0.035 and 0.10 and with 12^3 to
  24^3 nodes (-5.5% at tau = 0.62). The reaction of the drag acts on a single node, far from the
  hydrodynamic limit (Kn of order 1): part of the energy goes into non-hydrodynamic moments, which the
  regularized collision removes and the hydrodynamic formula does not count.

With the Euler-Maruyama scheme the budget gains the work of the random force; in a steady state its mean
power balances the dissipation of the drag.

**Order in the lattice step.** Moments, removal of the fluid momentum, coupling, collision and
streaming, bounce-back. The coupling therefore sees the fluid momentum after the removal.

**Time levels.** OpenMM's Verlet integrator is a leapfrog: during the force evaluation of step t
the velocities are v(t - dt/2). The fluid momentum before the step is j(t - dt/2), because
j(t + dt/2) = j(t - dt/2) + F(t). The drag therefore compares particle and fluid velocities at the
same half step, explicitly.

**When the coupling forces are computed.** The fluid advances once per integration step, in the force
evaluation of the step, which also computes the coupling forces of that step and their reaction on the
fluid. The force evaluations between steps (`getState(getForces=True)`, `getState(getEnergy=True)`)
return the coupling force of the next step, computed on the current fluid, positions and velocities,
without changing the fluid.
- This is OpenMM's convention for every force. `VerletIntegrator` computes the forces at the current
  positions x(t) and uses them for the next update v(t + dt/2) = v(t - dt/2) + dt F(t)/m, and the force in
  a State at time t is that F(t).
- OpenMM relies on it for the kinetic energy of a State: for `VerletIntegrator` it shifts the velocities
  by half a step with the forces of the State, v(t - dt/2) + dt F(t)/(2m), which is the full-step velocity
  (v(t - dt/2) + v(t + dt/2))/2 only if F(t) is the force of the next step.
- The random numbers of a step are drawn once, by the first evaluation that needs them, and the step
  uses the same ones. The step recomputes the drag with the velocities it finds, so a change of the
  velocities between the evaluations and the step (the reflection at the walls, `setVelocities()`) is
  taken into account. Extra evaluations therefore do not change the trajectory, which is identical with
  and without them.
- The CUDA, OpenCL and HIP platforms repeat all the force evaluations of a step when the neighbor list of
  a nonbonded force with a cutoff has to grow (`ContextImpl::calcForcesAndEnergy()`, `finishComputation()`
  returns valid = false). This needs more than about 1250 atoms: OpenMM first allocates 20 tiles of 32 x 32
  atoms per block of 32 atoms, which covers every tile of a smaller System. The repeated
  evaluation comes before the integrator increments the step count, and it applies the coupling forces of
  the step again. Before version 0.2.0 it applied those of the next step, computed on the fluid that had
  already received the reaction of the step, so in that step particles and fluid received different momenta
  (`testRepeatedForceEvaluation`). With the centred drag the evaluation that will be repeated does nothing
  (Solution of the centred drag, below).
- Before the first step of the Context the coupling forces are zero, so that an energy minimization
  before the dynamics sees only the forces of the other terms.
- The coupling is dissipative and has no energy.

**Random force.** On the Reference platform the random numbers come from a generator owned by the
force (SFMT, Box-Muller transform).
- It is seeded with `setRandomNumberSeed()`, or with a unique seed when the seed is 0.
- Its sequence does not depend on the other forces of the System, and the same seed reproduces a
  simulation.
- Its state is not part of OpenMM checkpoints; `LBMForce::createCheckpoint()` saves it.

On the CUDA, OpenCL and HIP platforms the random numbers come from OpenMM's generator, as in the reference
library (`IntegrationUtilities`, Gaussian numbers in single precision), seeded with `setRandomNumberSeed()`.
OpenMM keeps one generator per Context: another component that uses it with a different seed (an
`AndersenThermostat`, for example) makes OpenMM stop with an error, and the two seeds must then be set equal.
At every step the plugin reserves one Gaussian float4 per (padded) atom of the System and uses element i,
components x, y, z, for coupled particle i. This is how the reference library consumes the generator, so
with the same seed both draw the same numbers and a run with the random force can be compared with it step
by step: the thermal example agrees to 3e-11 over 20000 steps on CUDA in double precision
([examples/README.md](../examples/README.md#comparison-with-the-dragopenmm-plugin)). OpenCL generates the
same sequence with slightly different rounding.
The random forces of the GPU platforms and of the Reference platform are different sequences with the same
statistics; at T = 0 (or with the NVE scheme) the platforms agree to rounding.

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
without bound for gamma dt >= 2. A warning is printed when a Context with coupled particles and the
explicit drag is created with gamma dt > 1. The centred drag is stable for any friction (Which velocity has
the right temperature, below).

**Self-mobility and relaxation time.** The mobility of a dragged particle is 1/(m gamma) + y, where y is
the hydrodynamic self-mobility from the fluid around its node. y should depend only on the viscosity
(y ~ 1/eta), but with the explicit drag at the nearest node it contains a lattice term that does not
decrease with the viscosity. As a result y eta dx, which should not depend on tau, falls with tau and
changes sign. Measured on the Reference platform (L = 8 nm, dx = 0.5 nm, dt = 0.01 ps, m = 1000 Da,
gamma = 5/ps, protocol of `docs/validation.md`):

| tau | 0.62 | 0.8 | 1.1 | 1.5 | 1.6 | 1.7 | 1.8 | 1.9 | 2.0 | 3.51 |
|---|---|---|---|---|---|---|---|---|---|---|
| y eta dx | 0.0609 | 0.0577 | 0.0442 | 0.0202 | 0.0134 | 0.0065 | -0.0007 | -0.0080 | -0.0154 | -0.1406 |

(The scan with both drags below gives 0.0443 at tau = 1.1, a separate run with the same analysis.) y
vanishes at tau = 1.79 (1.79 also for m = 100 Da, gamma = 10/ps), the value found with the reference
library. Above it, particles move less than Langevin particles with the same friction. A warning is
printed when a Context with coupled particles and the explicit drag has tau > 1.7.

With the centred drag (Drag schemes, below) the drag sees the fluid velocity with half of the reaction of
the particle in the same step, h S/m_c, so y grows by dt/(2 m_c), that is by (tau - 1/2)/6 in units of
1/(eta dx). The same protocol, run with both drags, gives exactly that:

| tau | 0.62 | 0.8 | 1.1 | 1.5 | 1.8 | 2.0 | 3.51 |
|---|---|---|---|---|---|---|---|
| y eta dx, explicit | 0.0609 | 0.0577 | 0.0443 | 0.0202 | -0.0007 | -0.0154 | -0.1406 |
| y eta dx, centred | 0.0808 | 0.1077 | 0.1443 | 0.1868 | 0.2160 | 0.2346 | 0.3611 |
| explicit + (tau - 1/2)/6 | 0.0809 | 0.1077 | 0.1443 | 0.1868 | 0.2160 | 0.2346 | 0.3611 |

(m = 1000 Da, gamma = 5/ps; m = 100 Da, gamma = 10/ps gives the same values within 0.003.) With the
centred drag y is positive at every tau, but y eta dx grows with tau: neither drag gives a self-mobility
independent of tau. A relaxation of the ghost moments independent of tau is the next candidate correction.

**Kinetic temperature.** OpenMM's leapfrog stores the velocities at half steps. This paragraph is about
the explicit drag; for the centred drag see Which velocity has the right temperature, below.
- For a free particle with fluid at rest, the temperature measured from half-step velocities is
  T/(1 - gamma dt/2). Measured from full-step velocities v(t) = (v(t - dt/2) + v(t + dt/2))/2, it is T.
- The fluid has no thermal fluctuations of its own and takes part of the momentum of the particles, so
  the particles are slightly colder than T. Measured on the Reference platform with 200 free beads of
  100 Da, dx = 0.5 nm, dt = 0.01 ps, tau = 1.10, gamma dt = 0.1 and T = 300 K: 295.8 +- 0.4 K from
  full-step velocities and 311.7 +- 0.4 K from half-step velocities, where T/(1 - gamma dt/2) = 315.8 K.
- **The temperature reported by OpenMM is the full-step one.** The kinetic energy of a State, used by
  `StateDataReporter`, is computed by OpenMM from v(t - dt/2) + dt F(t)/(2m) with the coupling force of the
  next step (see above), which is the full-step velocity: it equals the temperature from full-step
  velocities to rounding (checked by `testFullStepKineticEnergy` on every platform and with both drags;
  measured once: 293.98 K both, within 4e-12 K, on 200 samples of 200 beads at 300 K on the Reference
  platform, `docs/validation.md`). With the explicit drag this is the right temperature; with the centred
  drag it is lower than the half-step one. Before version 0.1.0 of this plugin the forces between steps were
  those of the last step, and this temperature was wrong for coupled particles: for a free particle
  T [(1 - 3a/2)^2/(1 - a/2) + 9a/2] with a = gamma dt, 363 K at 300 K and a = 0.1.

**Drag schemes** (`setDragScheme()`, fixed when the Context is created). Two time discretizations of the
drag, with the same random force (same variance, numbers drawn in the same order).
- `Explicit`, the default: the scheme above, F_k = -gamma m_k (v_k(t - dt/2) - u(t - dt/2)) + R_k. Among the
  discrete Langevin integrators it is the scheme of Groot and Warren [12] with lambda = 1/2: velocity of the
  particle half a step before the force, new random numbers at every step.
- `Centered`: the velocities of particle and fluid at the time t of the force,

    F_k = -gamma m_k (v_k(t) - u_c(t)) + R_k,

  with v_k(t) = v_k(t - dt/2) + dt (Fc_k + F_k)/(2 m_k), where Fc_k is the sum of the other forces on the
  particle, and u_c(t) = (j_c + G_c/2)/rho_c, the velocity that the collision puts in the equilibrium of the
  node c (Guo forcing, section 1), G_c being the total force on the node, body force minus the forces of its
  particles. For a particle in a fluid at rest it is the scheme of Brünger, Brooks and Karplus [13] (midpoint
  drag, new random numbers at every step).

**Solution of the centred drag.** The system is implicit but linear, and couples only the particles of
the same node. In lattice units (dt = 1, h = 1/2, a = gamma h, masses in cell masses), with
v~_k = v_k(t - h) + h Fc_k/m_k, u~ = (j_c + h rho_c g)/rho_c, m_c = rho_c, and M, P~ and R the sums over the
particles of the node of m_k, m_k v~_k and R_k:

    S = [-gamma (P~ - M u~) + R]/(1 + a + a M/m_c),   u_c(t) = u~ - h S/m_c,
    F_k = [-gamma m_k (v~_k - u_c(t)) + R_k]/(1 + a).

S is the sum of the forces F_k and the node receives -S, so the total momentum is conserved exactly, random
force included. A solid node is a wall at rest of infinite mass: u_c(t) = 0, and the term a M/m_c drops out.
The particles of a node are summed in particle order, as for the reaction of the explicit drag. The force
Fc_k is read from OpenMM's forces after the other forces of the System have been computed, so:
- `LBMForce` must be the last force of the System (checked when the Context is created). On the Reference
  platform the forces are summed in the order of the System.
- The System must not contain virtual sites: OpenMM moves their forces to the particles that define them
  only after all the forces have been computed.
- When `getState()` asks for the forces of a subset of the force groups, the coupling force returned between
  steps is computed with the forces of those groups only.
- Constraints are applied by the integrator after the forces, and the drag does not see them.

On the Reference platform `execute()` finds the other forces already summed in OpenMM's force array, since
`LBMForce` is the last force. On the CUDA, OpenCL and HIP platforms the forces of the System are complete only
at the end of the force evaluation, so the centred drag runs in a `ForcePostComputation`, which OpenMM calls
after all the forces (`finishComputation()`, OpenMM 8.3 to 8.6) and, since `LBMForce` is the last force, after
the post-computations of the other forces, such as the one that joins the separate PME stream of the CUDA
platform (`test_centered_drag_sees_all_forces`).
- Fc is read from OpenMM's fixed point force buffer. The OpenCL platform also has floating point force
  buffers, which it adds to the fixed point one only after the post-computations: they are read too. On one
  device OpenMM's own forces do not write them during the evaluation (OpenMM 8.3 to 8.6), but forces of
  other plugins may.
- When the neighbor list of a nonbonded force has overflowed, OpenMM has already marked the evaluation as
  invalid when the post-computation runs, and will repeat it: the other forces are then incomplete, and the
  post-computation does nothing. The lattice step is done in the repeated evaluation, with the complete
  forces (`testCenteredRepeatedEvaluation`).
- The arithmetic is that of the Reference platform: one thread per particle computes v~_k, its random force
  and its key node*N_p + i; the keys are sorted with OpenMM's `ComputeSort`, in every force evaluation; the
  first key of each node solves the node and writes the forces of its particles and, in a lattice step, the
  reaction -S (to the walls at a solid node).

**Which velocity has the right temperature.** Consider the particles of a node and their cell alone,
without other forces, streaming and viscosity (a closed system with a fluctuation-dissipation balance). Its
dissipative modes have the rates lambda = gamma (relative motion of the particles) and
gamma (1 + M/m_c) (particles against the cell); let z = lambda dt/2.
- With the centred drag the velocities at the half steps, those of the State, are canonical for every
  gamma dt and number of particles, and the update is stable for every gamma dt (amplification factor
  (1 - z)/(1 + z); for z > 1 the relaxation alternates in sign, which is stable but not accurate). The
  full-step velocities have the temperature T/(1 + z) per mode: for one particle, between T/(1 + gamma dt/2)
  when the cell is heavy and T/(1 + gamma dt (1 + m/m_c)/2) when the cell keeps the momentum.
- With the explicit drag the full-step velocities are canonical and the half-step ones have T/(1 - z); the
  closed system is stable only for gamma dt (1 + M/m_c)/2 < 1. In the lattice the streaming carries the
  momentum of the node away, and the bound of a single particle, gamma dt < 2, is the one always required.
- For a harmonic force (frequency w) both have a configurational error of order (w dt)^2: <x^2> is
  1/(1 - (w dt)^2/4) times the exact value for the centred drag, 1/(1 - (w dt)^2/(2 (2 - gamma dt))) for the
  explicit one. In coarse-grained models w dt <~ 0.1, so the error is below 0.3%. Schemes with exact
  configurations exist (the family analysed in [14, 15]), but none of them has an on-site velocity with the
  exact temperature [14].

`StateDataReporter` reports the temperature of the full-step velocity (section 2, Kinetic temperature):
right with the explicit drag, lower with the centred drag. `openmmlbm.LBMTemperatureReporter` reports the
temperature of the coupled particles with the right velocity for each scheme.

**Temperature with a fluid without fluctuations.** The fluid has no thermal fluctuations of its own, so
the coupled particles are colder than T (Kinetic temperature, above). The centred drag couples a particle
to the velocity of its own cell within the step, the cell responding to the force of the particle (the term
h S/m_c), so the deficit is larger than with the explicit drag. Measured on the Reference platform (8^3
nodes, dx = 0.5 nm, dt = 0.01 ps, tau = 0.8, gamma dt = 0.1, T = 300 K, particles of 100 Da, so m/m_c = 1.3;
2000 steps after 500, seed 1):

| | 100 particles | 10 particles | 100 particles, fluid 100 times denser |
|---|---|---|---|
| `Explicit`, full-step T (K) | 285.8 | 293.0 | 296.8 |
| `Centered`, half-step T (K) | 269.6 | 275.9 | 296.7 |

With a heavy fluid both schemes give T to 1%; at the density of water the deficit of the centred drag is
8-10% and does not decrease with fewer particles. It follows from linear response: with F = -zeta (v - u) + R and
the fluid answering the reaction with u = -y F, the force is (-zeta v + R)/(1 + zeta y), and the kinetic
temperature is T/(1 + zeta y). The centred drag has y larger by dt/(2 m_c) (Self-mobility, above), so
zeta (y_centred - y_explicit) = gamma dt m/(2 m_c): measured 0.032, 0.063 and 0.127 at gamma dt = 0.05, 0.1
and 0.2 with 10 particles, against 0.033, 0.066 and 0.133. With a fluid that does not respond (10^4 times
denser) both drags give the exact values of their discretization, at gamma dt = 0.1, 0.5 and 1.5, within
2e-4, the statistical error (`docs/validation.md`). With a fluid that has thermal fluctuations of its own the
same argument gains a term, and only the centred drag gives the right temperature (section 7, Particles in the
fluctuating fluid).

The extra deficit of the centred drag grows with gamma dt m/m_c, so it is large for heavy particles or large
friction, while the diffusion coefficient does not change. Measured on CUDA (mixed precision; stochastic
tests T2, T6, T7 and examples, `docs/validation.md`), with the temperature that is right for each drag:

| System | m/m_c | gamma dt | `Explicit`, full step | `Centered`, half step | centred predicted from the explicit value |
|---|---|---|---|---|---|
| 64 free particles of 1000 Da (T6), 300 K | 13.3 | 0.05 | 270.4 K | 207.7 K | 208 K |
| same, gamma = 10/ps | 13.3 | 0.1 | 254.0 K | 161.1 K | 163 K |
| SOD1, COCOMO2 (110 beads), 298 K, three runs of 200 ns | 1.32 (mean) | 0.1 | 294.0 K | 277.5 K | 276 K |
| SOD1 with friction 30/ps, 50 ns | 1.32 (mean) | 0.3 | 294.6 K | 250.2 K | 246 K |

The prediction is 1/T_centred = 1/T_explicit + gamma dt m/(2 m_c T), for the protein with the mean bead
mass. In T6 the diffusion coefficient of the particles is the same
with both drags (D/(kT/(m gamma)) = 0.984 and 1.010 at gamma = 5 and 10/ps, for both), and so is the
diffusion of the centre of mass of SOD1 (`docs/validation.md`),
and the normalized velocity autocorrelation of the centred drag is closer to the response to a kick computed
with the same drag (fluctuation-dissipation for the dynamics: ratio 0.97 to 0.79 between 0.05 and 1 ps,
against 0.93 to 0.50 with the explicit drag). So, without thermal fluctuations of the fluid, the centred drag
keeps the diffusion and is stable for any friction, but its kinetic temperature is lower. With the fluctuating
fluid of section 7 the centred drag gives T, as the local fluctuation-dissipation balance of the centred drag
(Which velocity has the right temperature, above) suggests for particles and cells together, while the explicit
drag makes the particles too hot (section 7, Particles in the fluctuating fluid).

**Per-cell reaction on the GPU platforms.** The reaction forces of the particles in the same cell are
summed without atomic operations (`platforms/common/src/kernels/lbmCoupling.cc`), by sorting keys and
reducing segments with one writer per cell, as in the parallel spreading of the immersed boundary method of
Kassen et al. [16] (their case of a single node per point):

1. one thread per particle computes its force and the key node*N_p + i, unique, with N_p the number of
   coupled particles and i the index of the particle in the list of the force;
2. the keys are sorted with OpenMM's `ComputeSort`;
3. the first entry of each node adds the reactions of its particles, in particle order, and writes the sum
   to the node: one writer per node;
4. after the collision the nodes that received a reaction are set back to zero.

With the centred drag, step 1 computes v~_k and the random force R_k, the keys are sorted in every force
evaluation, and in step 3 the first entry solves the node, writes the forces of its particles and, in a
lattice step, the reaction -S. The result is reproducible bit for bit, and the sum is taken in the order of
the Reference platform.

**Forces on the particles.** The coupling forces (in the evaluation of a step those of the step, between
steps those of the next step: When the coupling forces are computed, above) are added to OpenMM's force buffer, in fixed point
(resolution 2^-32 kJ/mol/nm), at every force evaluation; with the centred drag in the post-computation. OpenMM handles the forces in its
"real" type: on the OpenCL platform, in single and mixed precision, the total force on a particle is
rounded to single precision, while the fluid receives the exact reaction, so the total momentum is
conserved there to about 1e-8. With CUDA in mixed or double precision, and with OpenCL in double
precision, it is conserved to 1e-13 (`docs/validation.md`).

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
per step. Other force evaluations (`getState(getForces=True)`, for example) do not advance the fluid: they
compute the coupling force of the next step without its reaction on the fluid, with the random numbers of
that step drawn once (When the coupling forces are computed, above).

## 3. Units and conversions (implemented)

The public API uses OpenMM units: nm, ps, Da (g/mol), K and kJ/mol. Internally the plugin works in
lattice units only, as is usual for lattice Boltzmann [7]. `LBMForceImpl::computeLatticeParameters()`
computes the lattice parameters of the fluid (tau, omega, the initial velocity and the body acceleration)
and passes dx, dt, rho0, the friction and kT, from which every kernel converts the quantities of the
particles with the same factors:

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
| friction | 1/dt | gamma_lattice = gamma dt |
| thermal energy | m_c dx^2/dt^2 | kT_lattice = kT dt^2/(m_c dx^2) |

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
- **Populations.** Stored as deviations from the rest equilibrium, df_q = f_q - w_q, at [q numNodes + node],
  with the velocity set of
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

- **Deviations from the rest equilibrium.** The rest equilibrium at lattice density 1 is f_q = w_q, so the
  populations are close to the weights, and the hydrodynamic signal is a small difference
  df_q = f_q - w_q.
  - Storing f_q itself would keep that signal only in the last digits of numbers of order 0.05. With a
    slow flow on a large lattice the total momentum then drifts by rounding: 3e-5 of the momentum with
    64^3 nodes, 20000 steps and a fluid velocity of 5e-7 nm/ps, in double precision.
  - Storing df_q keeps the full precision of the type for the signal.
  - The moments follow from the sums of the weights: rho = 1 + sum df, j = sum c df (sum w = 1,
    sum w c = 0). The non-equilibrium part df - dfeq is unchanged, with dfeq = feq - w computed directly
    (`D3Q19::equilibriumDeviation`).
  - Bounce-back copies df unchanged, since opposite directions have the same weight. The momentum
    exchange uses the full f = df + w, whose part w carries the static pressure on the walls.
- **Checkpoints.** A plugin cannot add data to OpenMM checkpoints. `LBMForce::createCheckpoint()` writes
  what they miss: the populations, the random numbers already drawn for the next step, the momentum given
  to the walls in the last step and, on the Reference platform, the random number generator of the force
  (on the GPU platforms it is OpenMM's, which the OpenMM checkpoint contains). With both checkpoints a run
  continues bit for bit (`testCheckpointWithRandomForce`). `LBMForceImpl` writes a header before the data
  of the kernel: a tag, the format version (3), the platform, the grid size, the number of coupled particles,
  the drag scheme and whether the fluid fluctuates. A checkpoint is refused on another platform, with another
  grid, number of coupled particles, drag scheme or switch of the fluid fluctuations, and on the GPU platforms
  with another precision; a checkpoint of version 1 (plugin version 0.1.0) is read as one of the explicit drag,
  and those of versions 1 and 2 (plugin versions 0.1 and 0.2) as ones without fluid fluctuations.
- **Fluid state.** `getFluidState()` returns the deviations df_q in this layout, in lattice units: a
  population is the value plus w_q. Saving and restoring them is exact, so a restarted run is identical
  to an uninterrupted one.
- **Initial state.** A new Context starts from the equilibrium at lattice density 1 and the initial
  velocity.

## 5. Precision (implemented)

The fluid (populations and moments) uses the "mixed" type of the platform:

- float when the platform `Precision` is `single`;
- double when it is `mixed` or `double`.

The Reference platform always uses double precision.

With the deviations df_q the resolution of single precision applies to the signal itself rather than to
the populations of order 0.05, which matters most for weak forces and slow flows. `mixed` is still
recommended for production.

## 6. Stability checks (implemented)

The model is accurate only in the quasi-incompressible regime and for relaxation times in a moderate
range. The plugin checks both.

**At Context creation.**
- tau > 1/2 holds by construction, since a viscosity that is not positive is an error.
- A warning is printed on stderr if tau is outside [0.505, 2], the range used by the reference
  implementation.
- With coupled particles and the explicit drag, warnings for tau > 1.7 and friction*dt > 1 (section 2), and,
  with fluid fluctuations and the EM scheme at T > 0, a warning that the particles will be too hot, with the
  bound friction*dt*m/(2 m_c) for the heaviest one (section 7);
  with the centred drag, an error if `LBMForce` is not the last force or the System has virtual sites
  (section 2, Solution of the centred drag).

**During the simulation.**
- Every N steps the plugin computes the largest Mach number of the fluid, Ma = max |u|/c_s, with
  u = j/rho and c_s = 1/sqrt(3) in lattice units.
- N is set with `setMachCheckFrequency()`; the default is 100 and 0 disables the check.
- If Ma exceeds the limit set with `setMachNumberLimit()` (default 0.3), the plugin throws an
  `OpenMMException` that reports the step and the value.
- `getFluidMachNumber(context)` returns the current value, for monitoring.
- The check runs after a lattice step that brings the step count of the Context to a multiple of N (the
  momentum removal, instead, uses the step count at the start of the step). On the CUDA, OpenCL and HIP
  platforms the maximum is taken by work group on the device and then over the work groups on the host,
  without atomic operations.

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

## 7. Fluctuating fluid (implemented on all platforms)

Without fluctuations the fluid receives thermal energy only from the reaction to the random forces on the
coupled particles. The particles then miss the thermal motion of the solvent: their diffusion coefficient is
about kT/(m gamma) instead of containing the hydrodynamic contribution, and they are colder than the bath, more
so with the centred drag (section 2). `setFluidFluctuations(true)` gives the fluid thermal fluctuations of its
own, at the temperature of `setTemperature()`: particles and fluid share one heat bath. It works with both
coupling schemes; with `NVE` the particles have no random force and are thermalized only by the fluid. The
default is off, and then nothing changes: the run is identical, bit for bit, to that of version 0.2.

**Model: ghost-mode filtered fluctuating lattice Boltzmann (GMF-FLBM, reference 17).** The regularized
collision of section 1 is already its deterministic part: it rebuilds the non-equilibrium populations from
the second moment alone, so the ghost moments (those above the stress) leave the collision at zero, as if they
relaxed with rate 1, and they carry no deterministic memory. The fluctuating model adds to the post-collision
populations of each fluid node, before streaming, the random part

    xi_q = w_q sum_{k=4..18} e_k(c_q)/b_k phi_k r_k,
    phi_k = sqrt(mu rho b_k omega (2 - omega))   for the six stress modes, k = 4...9,
    phi_k = sqrt(mu rho b_k)                     for the nine ghost modes, k = 10...18,

with r_k fifteen independent normal numbers N(0, 1), rho the density of the node, omega = 1/tau and
mu = kT/c_s^2 in lattice units. The e_k are the polynomials of the orthogonal D3Q19 basis of Lulli et al.
(reference 18), orthogonal with the weights w_q, sum_q w_q e_k(c_q) e_l(c_q) = b_k delta_kl:

| k | e_k(c) | b_k |
|---|---|---|
| 0 | 1 | 1 |
| 1-3 | c_x, c_y, c_z | 1/3 |
| 4-6 | c_x^2 - 1/3, c_y^2 - 1/3, c_z^2 - 1/3 | 2/9 |
| 7-9 | c_x c_y, c_x c_z, c_y c_z | 1/9 |
| 10-12 | c_y (c_x^2 - 1/3), c_z (c_x^2 - 1/3), c_x (c_y^2 - 1/3) | 2/27 |
| 13 | c_x (c_y^2 + 2 c_z^2 - 1)/2 | 1/18 |
| 14 | c_y (c_x^2 + 2 c_z^2 - 1)/2 | 1/18 |
| 15 | c_z (c_x^2 + 2 c_y^2 - 1)/2 | 1/18 |
| 16 | c_x^2 c_y^2 - c_x^2/3 - c_y^2/3 + c_z^2/6 + 1/18 | 7/162 |
| 17 | 2/7 c_x^2 c_y^2 + c_x^2 c_z^2 - 3/7 c_x^2 + c_y^2/14 - 2/7 c_z^2 + 1/14 | 5/126 |
| 18 | 2/5 c_x^2 c_y^2 + 2/5 c_x^2 c_z^2 + c_y^2 c_z^2 - c_x^2/10 - 2/5 c_y^2 - 2/5 c_z^2 + 1/10 | 1/30 |

Each moment m_k = sum_q e_k(c_q) f_q receives phi_k r_k. The polynomials are evaluated on the velocities, so
they do not depend on the ordering of the populations (`D3Q19::mode()` and `D3Q19::fluctuation()` in
`openmmapi/include/internal/D3Q19.h`).

**Why these amplitudes.** In equilibrium the populations of a node fluctuate independently, with
<delta f_q delta f_r> = mu rho w_q delta_qr, so a moment has the variance <delta m_k^2> = mu rho b_k: density
mu rho, momentum rho kT per component (equipartition for the mass rho of the node). A stress mode relaxes as
m_k -> (1 - omega) m_k + phi_k r_k, whose stationary variance is mu rho b_k only if
phi_k^2 = mu rho b_k [1 - (1 - omega)^2] = mu rho b_k omega (2 - omega); a ghost mode is replaced by phi_k r_k,
so phi_k^2 = mu rho b_k (Dünweg, Schiller and Ladd, reference 19). The density and the momentum (k = 0...3)
receive nothing: by orthogonality sum_q xi_q = 0 and sum_q c_q xi_q = 0, so the noise conserves the mass and
the momentum of every node exactly, and the removal of the fluid momentum and the Mach number check are
unchanged. They fluctuate through the stress, which streaming couples to them.

**Units.** mu = kT/c_s^2 = 3 kT, with kT in lattice units kT dt^2/(m_c dx^2) and m_c = rho0 dx^3 the mass of a
cell (section 3). For water at 300 K, dx = 0.5 nm and dt = 0.01 ps: m_c = 75.3 Da, kT = 1.3e-5 in lattice units
and a thermal Mach number sqrt(kT)/c_s = 0.006. The populations are stored as deviations f - w (section 4), so
fluctuations of order 1e-3 keep their precision.

**Random numbers.** On the Reference platform the fifteen normal numbers of a node come from the generator of
the force (OpenMM's SFMT, seeded with `setRandomNumberSeed()`, which also draws the random forces on the
particles), node after node in index order, after the coupling of the step. The force evaluations between
steps draw only the random forces of the next step, which come after the fluid numbers of the current step in
any case, so they do not change the sequence. The checkpoint of the force (`createCheckpoint()`) contains the
state of the generator, so a restarted run is identical to an uninterrupted one. The CUDA, OpenCL and HIP
platforms draw them from OpenMM's random numbers of the Context, as they do for the particles: in every step,
after those of the particles, four float4 per node (`IntegrationUtilities::prepareRandomNumbers(4*numNodes)`,
16 normal numbers of which 15 are used, in single precision whatever the precision of the platform), with the
coefficients w_q e_k(c_q)/sqrt(b_k) in an array of the kernel; OpenMM's checkpoints contain the state of that
generator. The buffer of the random numbers grows to 64 bytes per node, already when the Context is created:
OpenMM's checkpoints write the buffer as it is but read it back with the size it has in the Context that loads
them, so a Context created for a restart must have the full size before the first step (with OpenMM 8.3.1, a
buffer grown only in the first step made the restart differ from the uninterrupted run; with 8.6.1 it did not).
At zero temperature no numbers are drawn and nothing is added. Cost on an NVIDIA A100 (CUDA, mixed precision,
fluid with one coupled particle): 60, 169 and 1116 us per step without fluctuations and 77, 240 and 1590 us with
them on lattices of 32^3, 64^3 and 128^3 nodes, that is 28% to 43% more; most of it is the generation, writing
and reading of the random numbers in OpenMM's buffer. The platforms therefore draw different numbers, as for the particles: they agree
with each other through the statistics of the tests below, and without fluctuations or at zero temperature to
rounding.

**Stability near tau = 1/2.** Without fluid velocity the linearized model is a contraction: collision multiplies
every non-conserved mode by |1 - omega_k| <= 1 and streaming permutes the populations, so the norm
sum_q delta f_q^2/w_q cannot grow. The fluctuating fluid nevertheless becomes unstable close to tau = 1/2,
through the nonlinear terms of the equilibrium: the thermal velocities act as local flows at a grid Reynolds
number u_th dx/nu, with u_th = sqrt(3 kT) in lattice units. On a 64^3 lattice over 10^5 steps (CUDA, mixed
precision; the same in double precision and on the Reference platform) the Mach number exceeds the limit within
a few thousand steps for tau <= 0.501 at kT = 1/3000 (the value of reference 17), tau <= 0.5005 at kT = 1e-4 and
tau = 0.5001 at kT = 1.3e-5 (water at dx = 0.5 nm and dt = 0.01 ps); above these values it is stable. The limit
is a grid Reynolds number between about 50 and 100. The same initial thermal state without noise is unstable only
at tau = 0.5001 and 0.5002 (kT = 1/3000), because its velocities decay. Reference 17 uses the D3Q27 lattice,
which stays stable down to tau = 0.5001 at kT = 1/3000; on D3Q19 the regularized collision has less margin. Close
to the limit the fluctuations are also too large at long wavelengths (`docs/validation.md`, Fluctuating fluid).
Molecular simulations of water have tau >= 0.52 and kT ~ 1e-5, far from the limit, and the warning for tau
below 0.505 (section 6) covers this range.

**Particles in the fluctuating fluid: use the centred drag.** The linear-response argument of section 2
(Temperature with a fluid without fluctuations) gains a term. Let the velocity of the node be
u = u_th - y F, with u_th the thermal velocity of the fluid and y the self-mobility with which the node answers
the force F of the particle within the drag (section 2, Self-mobility). The thermal velocity adds the random
force zeta u_th, whose strength the fluctuation-dissipation theorem of the fluid sets through the self-mobility
y_th that the thermal flows carry, and the kinetic temperature becomes

    T_p = T (1 + zeta y_th)/(1 + zeta y).

It is T only for the drag whose y equals y_th. The measurements show that this is the centred drag: its particles
have the right temperature at half steps for every friction and time step tried, their velocity autocorrelation
equals the response to a kick computed with the same drag (fluctuation-dissipation theorem for the dynamics), and
their diffusion coefficient follows the Einstein relation with the mobility of the same drag,
D = kT (1/zeta + y_centred). The explicit drag misses the response of the cell within the step,
y_explicit = y_centred - dt/(2 m_c), so its particles are too hot:

    T_explicit/T = 1 + gamma dt m/(2 m_c (1 + zeta y_explicit)),

the mirror image of the deficit without fluctuations. Measured on CUDA (mixed precision; stochastic tests T2
and T6 with the fluctuating fluid, `docs/validation.md`), with y from section 2 and from the reference campaign:

| System | m/m_c | gamma dt | `Explicit`, full step | predicted | `Centered`, half step |
|---|---|---|---|---|---|
| 100 free particles of 100 Da (T2), gamma = 10/ps, dt = 0.005 ps | 1.33 | 0.05 | 309.2 K | 309.6 K | 299.5 K |
| same, dt = 0.01 ps | 1.33 | 0.1 | 319.0 K | 319.4 K | 299.3 K |
| same, dt = 0.02 ps | 1.33 | 0.2 | 340.7 K | 339.7 K | 300.7 K |
| 64 free particles of 1000 Da (T6), gamma = 1/ps, dt = 0.01 ps | 13.3 | 0.01 | 318.4 K | 319.3 K | 298.9 K |
| same, gamma = 5/ps | 13.3 | 0.05 | 388.7 K | 385.8 K | 299.5 K |
| same, gamma = 10/ps | 13.3 | 0.1 | 466.7 K | 450.5 K | 300.2 K |

The diffusion coefficient now contains the hydrodynamic contribution of the thermal flows and is the same for
both drags, D = kT (1/zeta + y_centred) within 2%: the thermal flows move the particle with the self-mobility of
the centred drag. With `setFluidFluctuations(true)` the centred drag is therefore the scheme to use, with the
temperature of the half steps (`LBMTemperatureReporter`); the explicit drag is accurate only when
gamma dt m/(2 m_c) is small. When a Context is created with fluid fluctuations, the explicit drag, coupled
particles, the EM scheme, T > 0 and a friction that is not zero, a warning on stderr gives that bound for the
heaviest coupled particle.

**Walls.** The halfway bounce-back of the solid nodes (section 1) is a permutation of populations: it neither
dissipates nor needs noise, and it is unchanged. Walls that also exchange thermal fluctuations with the fluid
(walls at the temperature of the bath, with a thermal accommodation coefficient) are a planned extension.

**Tests** (`tests/TestLBMFluctuations.h`, all platforms and precisions).
- A lattice of a single node, which streams every population back to itself: mass and momentum stay those of the
  start to 1e-15 (2e-6 in single precision, the resolution of the stored populations), and over 20000 steps the moments k = 4...18 have the variance mu b_k within 5% and are
  uncorrelated with each other, for tau = 1 (new values at every step) and tau = 0.8 (stress modes with
  memory).
- A fluid at rest on an 8x8x8 lattice, after 300 steps: the variances of the density, of the momentum and of
  the moments k = 4...18 of a node are mu rho, rho kT and mu rho b_k within 5%, for tau = 0.8 and 2.5; the total
  mass and momentum are conserved to 1e-13 (1e-5 in single precision).
- With the fluctuations switched on at zero temperature the run is identical, bit for bit, to the run without
  them; runs with the same seed are identical, also with force evaluations between steps; a different seed
  gives a different run; a run restarted from the checkpoints is identical to the uninterrupted one; with the
  `NVE` scheme a fluctuating fluid sets particles at rest in motion.
- Validation on an NVIDIA A100 (`docs/validation.md`, Fluctuating fluid and Stochastic tests): equilibrium
  spectra of all 19 moments on 64^3 nodes for tau from 0.505 to 100, with the protocol of reference 17 (for
  tau >= 0.55 the ER per node within 0.5% of 1, and ER(|k|) within 1.6% from the fourth shell on); decay of the
  thermal shear and
  sound modes as in the deterministic model; stability near tau = 1/2 (above); coupled particles with both drags
  (above).

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
12. R. D. Groot and P. B. Warren, J. Chem. Phys. 107, 4423 (1997): dissipative particle dynamics, the
    modified velocity-Verlet integrator with lambda.
13. A. Brünger, C. L. Brooks III and M. Karplus, Chem. Phys. Lett. 105, 495 (1984): stochastic boundary
    conditions, the midpoint Langevin integrator (BBK).
14. N. Grønbech-Jensen, J. Stat. Phys. 191, 137 (2024): on the definition of velocity in discrete-time,
    stochastic Langevin simulations.
15. N. Grønbech-Jensen, J. Stat. Phys. 193, 12 (2026): linear analysis of stochastic Verlet-type
    integrators for Langevin equations.
16. A. Kassen, V. Shankar and A. L. Fogelson, Int. J. High Perform. Comput. Appl. 36, 443 (2022): a
    fine-grained parallelization of the immersed boundary method.
17. M. Lauricella, A. Montessori, A. Tiribocchi and S. Succi, J. Chem. Phys. 164, 194905 (2026): ghost-mode
    filtered fluctuating lattice Boltzmann method.
18. M. Lulli, L. Biferale, G. Falcucci, M. Sbragaglia, D. Yang and X. Shan, Phys. Rev. E 109, 045304 (2024):
    orthogonal basis of D3Q19.
19. B. Dünweg, U. D. Schiller and A. J. C. Ladd, Phys. Rev. E 76, 036704 (2007): statistical mechanics of the
    fluctuating lattice Boltzmann equation.
20. J. Latt, Hydrodynamic limit of lattice Boltzmann equations, PhD thesis, University of Geneva (2007),
    doi:10.13097/archive-ouverte/unige:464: section 5.2, local regularized boundary condition.
21. J. Latt, B. Chopard, O. Malaspinas, M. Deville and A. Michler, Phys. Rev. E 77, 056703 (2008): straight
    velocity boundaries in the lattice Boltzmann method.
22. O. Malaspinas, Lattice Boltzmann method for the simulation of viscoelastic fluid flows, PhD thesis 4505,
    EPFL (2009): chapter 4, velocity boundary conditions.
23. Q. Zou and X. He, Phys. Fluids 9, 1591 (1997): on pressure and velocity boundary conditions for the lattice
    Boltzmann BGK model.
