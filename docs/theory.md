# Model, units and conventions

This document describes the physics implemented by `LBMForce`, the conversion between OpenMM and
lattice units, and the conventions every platform must follow. Each section says whether it is
already implemented or still to be implemented. The fluid and the explicit drag reproduce the CUDA
lattice Boltzmann library of the DragOpenMM project; the centred drag (section 2), the regularized walls and
the open faces (section 1) and the fluctuating fluid (section 7) are
extensions of this plugin.

## 1. Fluid model (implemented on all platforms)

The fluid is a D3Q19 lattice Boltzmann model, weakly compressible:

- moment 0 is the density, $`\rho = \sum_i f_i`$;
- moment 1 is the momentum, $`\mathbf j = \rho\mathbf u = \sum_i \mathbf c_i f_i`$ (the arrays of the plugin store
  $`\mathbf j`$, not $`\mathbf u`$);
- the equilibrium is the second-order Hermite expansion

  $`\displaystyle f_i^{\mathrm{eq}}(\rho, \mathbf u) = w_i\,\rho\left[1 + \frac{\mathbf c_i\cdot\mathbf u}{c_s^2} +
  \frac{(\mathbf c_i\cdot\mathbf u)^2}{2c_s^4} - \frac{\mathbf u\cdot\mathbf u}{2c_s^2}\right],\qquad c_s^2 = \frac13.`$

  Here $`c_s`$ is the speed of sound of the lattice: $`c_s^2 = 1/3`$ in lattice units, everywhere in this document.

**Collision: regularized, with Guo forcing [1, 2].** With the force density $`\mathbf F`$ acting on a node
during the step, the post-collision populations are

```math
f_i^{*} = f_i^{\mathrm{eq}}(\rho, \mathbf u^{*}) + (1 - \omega)\, f_i^{\mathrm{neq,reg}} + \tfrac12 S_i,
```

where:

- $`\mathbf u^{*} = (\mathbf j + \mathbf F/2)/\rho`$ is the velocity shifted by half a force;
- $`f_i^{\mathrm{neq,reg}} = \frac{w_i}{2c_s^4}\, H^{(2)}(\mathbf c_i) : \Pi^{\mathrm{neq}}`$ is rebuilt from the
  non-equilibrium stress
  $`\Pi^{\mathrm{neq}} = \sum_i H^{(2)}(\mathbf c_i)\,\bigl(f_i - f_i^{\mathrm{eq}}(\rho, \mathbf j/\rho)\bigr)`$,
  with the second-order Hermite polynomial $`H^{(2)}(\mathbf c) = \mathbf c\,\mathbf c - c_s^2 I`$. In
  $`\Pi^{\mathrm{neq}}`$ the $`-c_s^2 I`$ part of $`H^{(2)}`$ does not contribute, because the equilibrium has the
  same density as the populations and $`\sum_i (f_i - f_i^{\mathrm{eq}}) = 0`$: $`\Pi^{\mathrm{neq}}`$ is also the
  plain second moment of $`f - f^{\mathrm{eq}}`$, which is how the reference library computes it. In
  $`f^{\mathrm{neq,reg}}`$ the $`-c_s^2 I`$ part gives the term $`-c_s^2\,\mathrm{tr}(\Pi^{\mathrm{neq}})`$, without
  which $`f^{\mathrm{neq,reg}}`$ would carry mass;
- $`S_i = w_i\left[(\mathbf c_i - \mathbf u^{*})/c_s^2 +
  (\mathbf c_i\cdot\mathbf u^{*})\,\mathbf c_i/c_s^4\right]\cdot\mathbf F`$ is the Guo source.

The prefactor of $`S_i`$ is 1/2, not the $`(1 - \omega/2)`$ of the BGK form. The equilibrium is already shifted by
$`\mathbf F/2`$ and the regularized $`f^{\mathrm{neq}}`$ has no first-order part, so with 1/2 the momentum increases
by exactly $`\mathbf F`$ per step, at every relaxation time [3, 4].

- $`\mathbf F = \rho\mathbf g`$ for a body acceleration $`\mathbf g`$ (lattice units); the coupling forces are added
  to it.

**Relaxation and streaming.** The relaxation frequency is $`\omega = 1/\tau`$, with
$`\tau = 3\nu\,\Delta t/\Delta x^2 + 1/2`$.

**Velocity of the fluid.** `getFluidFields()` reports $`\mathbf u = (\mathbf j + \mathbf F/2)/\rho`$, the velocity
of the forced model, with $`\mathbf F`$ from the body acceleration.

The update is thread-safe and uses a single copy of the populations:

1. A first kernel computes $`\rho`$, $`\mathbf j`$ and $`\Pi^{\mathrm{neq}}`$ at every node from its populations.
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
halfway bounce-back (`BounceBack`, the default, described first) or the regularized wall
(`Regularized`, described below).
- Solid nodes hold no fluid: their populations start at zero, they have no moments and no collision, and
  they do not enter the removal of the fluid momentum.
- A population that the fluid node $`\mathbf x`$ sends towards a solid node $`\mathbf x + \mathbf c_q`$ comes back
  to $`\mathbf x`$ with the opposite velocity: $`f_{\mathrm{opp}(q)}(\mathbf x, t + 1) = f_q^{*}(\mathbf x, t)`$,
  where $`f_q^{*} = f_q^{\mathrm{eq}}(\rho, \mathbf u^{*}) + (1 - \omega)\, f_q^{\mathrm{neq,reg}} + S_q/2`$, plus
  the random part $`\xi_q`$ with a fluctuating fluid (section 7), is the post-collision population of $`\mathbf x`$.
  This halfway bounce-back places the wall halfway between the fluid and the solid node and conserves the mass of
  the fluid. It is the scheme of the reference implementation.
- With the regularized collision $`f_q^{*}`$ depends only on the moments $`\rho`$, $`\mathbf j`$ and
  $`\Pi^{\mathrm{neq}}`$ of $`\mathbf x`$, its force and its random part, so the population that arrives from the
  wall is the population that $`\mathbf x`$ rebuilds from its own moments for the direction towards the wall, put in
  the opposite direction. The code does exactly this at the fluid nodes next to the walls: the collision and the
  streaming write $`f_q^{*}(\mathbf x)`$ into the solid node $`\mathbf x + \mathbf c_q`$, then each of these nodes
  copies it into its own population along $`-\mathbf c_q`$. A node reads only the populations that it wrote into the
  solid nodes and writes only its own populations that come from the wall, so on the parallel platforms no two
  threads touch the same value. The copy left in the solid node gives the momentum exchanged with the wall.
- Only the links from fluid nodes to solid nodes that do not cross an open face are used (an open face is not a
  wall: section 1, Open faces). Nothing streams between two solid nodes, and the momentum exchange is summed over
  the solid nodes in the order of the list, so that the GPU platforms, which process the nodes in parallel, give
  the same populations and wall force as the Reference platform.
- Up to version 0.2.1 the copy was done by the solid nodes, one thread per solid node writing into its fluid
  neighbours. The two forms move the same values, and they give the same results bit for bit on all platforms
  and precisions (`docs/validation.md`, Walls).

**Exact Poiseuille flow with bounce-back.** With the regularized collision, the odd non-hydrodynamic moments relax
with frequency 1 ($`\tau_{\mathrm{odd}} = 1`$), so the scheme behaves at the walls as a two-relaxation-time scheme
with magic parameter $`\Lambda = (\tau - 1/2)(\tau_{\mathrm{odd}} - 1/2) = (\tau - 1/2)/2`$ [8]. For a channel
between the walls of the solid plane $`j = 0`$ (the lattice is periodic, so the plane bounds the channel on both
sides) driven by a body acceleration $`g`$, the steady profile of the scheme is exactly, in lattice units,

```math
u(y) = \frac{g}{2\nu}\left(y - \tfrac12\right)\left(n_y - \tfrac12 - y\right) + \frac{g\,(16\Lambda - 3)}{24\nu}.
```

The curvature is the exact one for every $`\tau`$; the second term is a slip that shifts the effective wall by
$`(3 - 16\Lambda)/(12H)`$, with $`H = n_y - 1`$, and vanishes at $`\Lambda = 3/16`$, that is $`\tau = 7/8`$. The
validation tests check this profile to 1e-9 (`docs/validation.md`).

**Force on the walls: momentum exchange.** The momentum that the fluid gives to the solid nodes is
measured with the momentum exchange method of Ladd [9, 10] ([11], section 5.4.3.1, eqs. 5.79 and
5.80).
- On every boundary link, from a fluid node $`\mathbf x_f`$ to a solid node
  $`\mathbf x_s = \mathbf x_f + \mathbf c`$, the population $`f`$ that streams into the wall along $`\mathbf c`$
  comes back along $`-\mathbf c`$. The wall at rest receives the momentum
  $`f\mathbf c - f(-\mathbf c) = 2f\mathbf c`$.
- The sum over all boundary links, times $`m_c\,\Delta x/\Delta t`$, is the momentum given to the walls in one step.
- On the CUDA, OpenCL and HIP platforms one thread per solid node computes the part of the deviations,
  $`-2\sum \mathbf c_q (f - w)`$, with $`\mathbf c_q`$ pointing from the solid node to its fluid neighbour, and the
  host sums it over the solid nodes in the order of the list. The part of the weights, $`-2\sum \mathbf c\, w`$, is
  the static pressure: it depends only on the geometry and is computed once, in double precision. Kept apart, it
  does not hide the hydrodynamic part in single precision.
- The coupled particles also exchange momentum with the walls. The reaction $`-\mathbf F`$ of a particle whose
  nearest node is solid goes to the wall, and the reflection of a particle gives the wall the momentum
  $`2m\mathbf v`$. On the GPU platforms each particle stores its contribution of the last step (with the centred
  drag the reaction $`-\mathbf S`$ of a solid node is stored with the first particle of the node), and the host sums
  them in particle order.

`getWallForce()` returns the sum of these contributions over the last lattice step, divided by $`\Delta t`$. With
it the total momentum of particles, fluid and walls is conserved, and in a steady channel flow the force
on the walls equals the body force on the fluid (`docs/validation.md`).

**Regularized walls** (`setWallScheme(Regularized)`). The fluid nodes next to the solid nodes, the
boundary nodes, rebuild the populations that would come from the solid nodes as those of a node of fluid at
rest placed on the solid node.
- After the streaming, the populations of a boundary node $`\mathbf x`$ that come from a solid node (direction $`q`$
  with $`\mathbf x - \mathbf c_q`$ solid) are unknown. Each of them is set to the post-collision population that a
  node at $`\mathbf x - \mathbf c_q`$ with the moments of $`\mathbf x`$, except a velocity of zero, would send along
  $`\mathbf c_q`$:

  $`\displaystyle f_q(\mathbf x) = f_q^{\mathrm{eq}}(\rho_b, \mathbf 0) + (1 - \omega)\,
  f_q^{\mathrm{neq,reg}}\bigl(\Pi^{\mathrm{neq}}(\mathbf x)\bigr) + \tfrac12 S_q(\mathbf 0, \rho_b\mathbf g) + \xi_q,`$

  with $`\Pi^{\mathrm{neq}}(\mathbf x)`$ the non-equilibrium stress of $`\mathbf x`$ in this step, $`S_q`$ the Guo
  term of the body force and, with a fluctuating fluid, $`\xi_q`$ the random part of a node with the density of
  $`\mathbf x`$, drawn for $`\mathbf x`$ on its own before the mass balance below (section 7). The known populations
  of $`\mathbf x`$, and its collision, are those of any fluid node.
- This is the thread-safe boundary condition of Lauricella et al. [24] (appendix, eqs. A4 and A5; introduced in
  [25]): the non-equilibrium extrapolation of Guo, Zheng and Shi [26] written for the post-collision populations,
  with the equilibrium at the imposed values on the node beyond the boundary and $`(1 - \omega)`$ times the
  non-equilibrium part of the neighbouring fluid node, rebuilt from its stress by Hermite projection. Here it also
  carries the Guo term of the body force and, with a fluctuating fluid, a random part, and only the unknown
  populations are rebuilt, with $`\rho_b`$ from the mass balance below. As every scheme of this family (wet-node
  schemes, section 5.3.4 of [11]) it puts the boundary on the node where the imposed values are taken: it solves a
  Couette flow exactly for every $`\tau`$, and a Poiseuille flow exactly only at $`\tau = 1`$ (below).
- The density $`\rho_b`$ follows from the mass balance of the rebuilt links: their populations carry, in total, the
  mass that $`\mathbf x`$ sent into the solid nodes in this streaming (stored in the solid node
  $`\mathbf x - \mathbf c_q`$, direction opposite to $`q`$). The mass of the fluid is therefore conserved exactly,
  as with bounce-back. The balance is linear in $`\rho_b`$ and is solved for $`\rho_b - 1`$, so that small
  deviations keep their precision.
- With the density of $`\mathbf x`$ in place of $`\rho_b`$ (the reconstruction alone) the mass changes in unsteady
  flows, by 2e-5 in a decaying flow around a block, and with fluid fluctuations it falls steadily: the populations
  that leave towards the wall carry the mean of the quadratic term
  $`\rho\,\bigl(4.5\,(\mathbf c\cdot\mathbf u)^2 - 1.5\,u^2\bigr)`$ of the velocity fluctuations, about $`k_BT/2`$
  per node and step, which the rebuilt populations, at zero velocity, do not return. In a test at $`k_BT = 1/3000`$
  the fluid lost 5 % of its mass in 1000 steps and 49 % in 10000 (`docs/validation.md`). The mass balance removes
  this.
- The wall lies on the solid nodes: the populations that arrive at $`\mathbf x`$ are those of a node at rest on the
  solid node, while with bounce-back they are the populations of $`\mathbf x`$ itself, sent back, and the wall is
  halfway. In the channel of the solid plane $`j = 0`$ of a periodic lattice the walls are on $`j = 0`$ and
  $`j = n_y`$, a full node further out than with bounce-back on each side. Starting from either of the two parabolas
  (zero on the solid nodes or halfway), the fluid next to the wall moves to the profile of its scheme in about a
  thousand steps.
- Each boundary node reads only its own populations and moments and the solid slots that it wrote itself
  in the streaming, and writes only its own unknown populations, so the scheme is local and thread-safe: on
  the CUDA, OpenCL and HIP platforms one thread per boundary node rebuilds it (kernel `applyBoundaries`),
  after the streaming, with the arithmetic of the Reference platform, and every platform finds the boundary
  nodes with the same code (`openmmapi/include/internal/LBMBoundaries.h`). A solid layer one node thick is
  allowed: the boundary nodes on both sides rebuild the populations that would come from it.
- The boundary nodes are fluid nodes in everything else: they enter the removal of the fluid momentum, and
  the reaction of a coupled particle whose nearest node is a boundary node acts on the fluid.
- An alternative, tried and left out, is the local regularized boundary condition of Latt [20, 21] (section 5.2 of [20];
  "BC3" of Malaspinas [22]), which rebuilds all 19 populations of the boundary node with the velocity of the wall
  imposed on the node. It put the wall on the boundary nodes, and next to it the fluctuations were 3 to 9 % below
  equilibrium (momentum normal to the wall 0.910 on the first node): imposing $`\mathbf u = 0`$ on the node removes its
  momentum fluctuations in every step without returning them. The scheme above is used instead.

**Exact Poiseuille flow with regularized walls.** In the channel above, driven by $`g`$, the steady profile of the
scheme is exactly, in lattice units,

```math
u(y) = \frac{g}{2\nu}\, y\,(n_y - y) + \frac{3g\,(\tau - 1)}{\tau - 1/2}
```

at every fluid node, with the walls on the solid nodes $`y = 0`$ and $`y = n_y`$, and the density uniform. The
second term is a slip that vanishes at $`\tau = 1`$: there the populations rebuilt from the moments of $`\mathbf x`$
are exactly those of the node on the wall. Otherwise it comes from the stress of $`\mathbf x`$ copied to the node on
the wall, and shifts the wall by 0.06 nodes at $`\tau = 0.6`$ and by -0.08 at $`\tau = 1.5`$ in a channel of 13
nodes. The validation tests check the profile to 1e-9.

**Which wall to choose.** Both are second order and conserve mass exactly. Differences:
- Where the wall is: halfway between the fluid and the solid nodes with bounce-back (exact Poiseuille flow at
  $`\tau = 7/8`$), on the solid nodes with regularized walls (exact at $`\tau = 1`$).
- With fluid fluctuations bounce-back walls are in exact thermal equilibrium with the fluid: bounce-back
  permutes the populations and keeps the Gaussian equilibrium state, so the fluctuations of all modes keep
  their equilibrium variance next to the walls. Next to regularized walls the fluctuations are at
  equilibrium within the statistics from the second node on; on the boundary node the density and the
  momentum normal to the wall are at equilibrium, the momentum along the wall is 3 to 4 % low and some
  stress and ghost modes up to 9 % low ($`\tau = 0.8`$, `docs/validation.md`).
- With Density faces (below) bounce-back walls let a spurious staggered mode survive, which the Density
  faces damp.

**Force on the walls with regularized walls.** On each link from a boundary node $`\mathbf x`$ to a solid node the
wall receives the momentum of the population that $`\mathbf x`$ sent into it and gives that of the population
rebuilt in its place: $`-\mathbf c_q\,(f_{\mathrm{sent}} + f_q)`$ for the direction $`q`$ with
$`\mathbf x - \mathbf c_q`$ solid, with the full populations $`f = (f - w) + w`$. The reactions of particles whose
nearest node is solid are added as with bounce-back. With it the momentum balance of the fluid,
$`\mathbf P(t + \Delta t) - \mathbf P(t) = (M\mathbf g - \mathbf F_{\mathrm{wall}})\,\Delta t`$, holds at every
step, as with bounce-back. On the CUDA, OpenCL and HIP platforms each boundary node stores the part of the
deviations $`f - w`$, the host sums it in the order of the boundary nodes, and the part of the weights $`w`$,
$`-2\,\mathbf c_q w_q`$ per link, which depends only on the geometry, is computed once in double precision.

### Open faces

By default the box is periodic. `setFaceBoundary(face, type)` makes a face of
the box open: `Velocity` (the fluid beyond the face has the velocity set with `setFaceVelocity()`: an inlet,
an outlet or a moving wall) or `Density` (the fluid beyond the face has the density, that is the pressure
$`p = c_s^2\rho`$, set with `setFaceDensity()`). The two faces perpendicular to an axis must be both periodic or
both open, in any combination of `Velocity` and `Density`, and an open axis needs at least 3 nodes. The six
faces have independent velocities and densities.
- The nodes on an open face ($`i = 0`$ for `XMin`, $`i = n_x - 1`$ for `XMax`, and so on) are boundary nodes: the
  populations that would come from beyond the face are unknown, and the populations that leave through the face are
  lost. As for the regularized walls (the thread-safe boundary condition of [24, 25]), each unknown population is
  the one that a node beyond the face, at $`\mathbf x - \mathbf c_q`$, would send, with the moments of $`\mathbf x`$
  except the one that the face imposes:

  $`\displaystyle f_q(\mathbf x) = f_q^{\mathrm{eq}}(\rho_b, \mathbf u_b) + (1 - \omega)\,
  f_q^{\mathrm{neq,reg}}\bigl(\Pi^{\mathrm{neq}}(\mathbf x)\bigr) + \tfrac12 S_q(\mathbf u_b, \rho_b\mathbf g) +
  \xi_q,`$

  with, for a fluctuating fluid, a random part drawn for the node on its own. The known populations, and the
  collision of the face node, are those of any fluid node.
- The imposed velocity or density holds on the nodes beyond the face, one node outside the box (as the wall of the
  regularized walls lies on the solid nodes): a Couette flow between a face at rest and a face moving with $`U`$ is
  $`u(z) = U\,(z + 1)/(n_z + 1)`$, exactly, and a Poiseuille flow between two `Velocity` faces at rest vanishes at
  $`z = -1`$ and $`z = n_z`$ at $`\tau = 1`$, with the slip of the regularized walls otherwise.
- On a `Velocity` face $`\mathbf u_b`$ is the velocity of the face, and $`\rho_b`$ follows from the mass balance of
  the rebuilt links: their populations carry the populations that arrived at $`\mathbf x`$ moving out of the face,
  plus the inflow $`6\,w_q\,\rho_b\,\mathbf c_q\cdot\mathbf u_b`$ (Zou and He [23]; eq. 5.3 of Latt [20]); links
  whose opposite direction is unknown too (edges and corners) do not count. With the density of $`\mathbf x`$
  instead, a fluctuating fluid between two `Velocity` faces lost its mass as next to the walls (the mean density
  fell to 0.40 in 10000 steps); with the balance it stays within 2e-4.
- On a `Density` face $`\rho_b`$ is the density of the face, the velocity along the face is that of $`\mathbf x`$,
  and the velocity across it is filtered in time (below). The velocity of $`\mathbf x`$ is
  $`(\mathbf j + \rho\mathbf g/2)/\rho`$, the one of `getFluidFields()`: it leaves out the reaction of the coupled
  particles whose nearest node is $`\mathbf x`$, since the rebuilt populations are those of a node beyond the face,
  on which the particles do not act (their reaction acts on $`\mathbf x`$ through its own collision), and the Guo
  term of the rebuilt populations has the body force only. Up to version 0.3.0 the Reference platform included that
  reaction, unlike the other platforms; the two agree since then (`test_coupling_agrees_with_reference`, case
  `faces`).
- **Staggered mode.** For any lattice whose velocities have components -1, 0 and 1, the staggered momentum
  $`\sum_y (-1)^{y+t} j_y`$ is conserved exactly by the bulk (the collision keeps the momentum of each node, the
  streaming moves a population by one node in one step). In a periodic box it stays zero; open faces can excite it,
  and bounce-back walls keep it. It is the lattice Boltzmann form of the pressure oscillations of collocated grids
  that staggered finite-difference grids avoid ([20], chapter 7). The `Density` faces damp it (below).
- **Density faces as inlets.** A `Density` face through which the fluid enters is less robust than a `Velocity`
  inlet: in a channel between bounce-back walls driven by two `Density` faces (a difference of density of 1 %,
  inflow velocity about 0.03 in lattice units) the flow is steady for $`\tau \ge 0.6`$ and unstable for
  $`\tau \le 0.55`$ (also with a difference of 0.3 %), while a `Velocity` inlet with a `Density` outlet is stable
  down to $`\tau = 0.52`$ with an inflow of 0.05. Without the time filter below the `Density` inlet was unstable
  already at $`\tau = 0.6`$. Prefer a `Velocity` inlet and a `Density` outlet at small $`\tau`$.
- **Entrance and exit regions.** Next to the faces the flow enters and leaves: in a duct driven by two
  `Density` faces the density on the face nodes overshoots the densities of the faces (1.0104 for 1.01 and
  0.9996 for 1.0 in a duct of 16 nodes), and the gradient in the middle is that of a length of 14.6 nodes
  rather than of the 17 between the nodes beyond the faces. In the middle the flow is that of the
  incompressible duct for that gradient (within 0.9 %).
- Nodes on several open faces (edges and corners): the first `Velocity` face in the order XMin, XMax, YMin,
  YMax, ZMin, ZMax gives the velocity; if all are `Density` faces, the first gives the density and the
  velocity is zero. A face node next to a solid node with regularized walls is a wall node: velocity zero,
  and $`\rho_b`$ from the mass balance of all its rebuilt links, the solid links as on a wall and the links across
  the face as on a `Velocity` face at rest. With bounce-back walls the bounce-back returns the
  populations from the solid nodes, and the face rebuilds those from beyond the face. Links that cross an
  open face are never bounced back. Malaspinas [22] treats edges and corners with finite differences
  instead; the rule here keeps the scheme local.
- With two `Density` faces at the same density and no walls (the other axes periodic) the mean flow across
  the faces has no restoring force: equal pressures, no friction. Any disturbance moves the whole fluid, and
  with fluid fluctuations the mean flow wanders like a free Brownian particle until the Mach number check stops
  the run (`docs/validation.md`). Walls along the flow damp it by viscous friction; a `Velocity` face fixes it.
- With open faces the fluid exchanges momentum with the outside, so its momentum cannot be removed:
  `setFluidMomentumRemovalFrequency(0)` is required. The reaction of a coupled particle acts on the face nodes
  as on any fluid node. The particles themselves still live in OpenMM's periodic box: a particle that crosses
  an open face reappears on the opposite one, so keep them away from the open faces.
- The body acceleration (`setBodyAcceleration()`) acts on the face nodes like on the others, and the velocity of a
  face is the velocity of the fluid $`(\mathbf j + \mathbf F/2)/\rho`$.
- The local regularized boundary condition of Latt [20] (all 19 populations of the face node rebuilt, with the
  velocity or the density imposed on the face node itself) was tried for the faces too, and left out with the one
  of the walls.
- Validation (`docs/validation.md`): a Couette flow between a face at rest and a moving face is linear to
  1e-14; a uniform flow from a Velocity inlet to a Density outlet is steady to rounding; a duct driven by a
  difference of density of 1 % agrees in the middle with the incompressible solution within 0.9 %.

**Time filter of the Density faces.** On a node of a `Density` face the velocity across the face, $`v_n`$, used for
the rebuilt populations is

```math
v_n = \tfrac12\,(v_{\mathrm{ZH}} + v_x),
```

the mean of $`v_{\mathrm{ZH}}`$, the velocity that gives the node the density of the face from the populations that
have arrived (the unknown ones replaced by the bounce-back of their opposite directions:
$`\rho_{\mathrm{face}} = \rho_0 + 2\rho_{\mathrm{out}} + \rho_{\mathrm{face}}\, v_n`$, with $`v_n`$ the component of
the velocity that points into the box, $`\rho_0`$ the sum of the populations along the face and
$`\rho_{\mathrm{out}}`$ that of those that leave through it; Zou and He [23], note 5.1 of Latt [20]), and of
$`v_x`$, the velocity of the node at the start of the step, from its moments and the body force (above). $`v_x`$ is
the outcome of the previous step, so the filter acts as a first-order recursive low-pass filter (an exponential
moving average with weight 1/2) on the velocity of the face:
- in a steady state $`v_{\mathrm{ZH}} = v_x`$, so steady flows are those without the filter;
- at the frequency of the staggered mode, a period of two steps, the mode is reduced at every return to the
  face instead of being sent back unchanged, so it decays (to 1e-16 in the tests);
- it is local, needs no extra memory, and a restart from `setFluidState()` or a checkpoint is exact, since $`v_x`$
  is contained in the populations. Without the filter ($`v_n = v_x`$, the velocity of the node) the staggered mode
  decays slowly, and the `Density` inlets above become unstable already at $`\tau = 0.6`$. The `Velocity` faces need
  no filter, since they impose $`v_n`$.

## 2. Particle-fluid coupling (implemented on all platforms)

**Euler-Maruyama scheme with the explicit drag** (the default; a frictional coupling at the nearest node,
as in Ahlrichs and Dünweg [5], reviewed in [6]). Each coupled particle $`k`$ of mass $`m_k`$ feels

```math
\mathbf F_k = -\gamma m_k\,\bigl(\mathbf v_k - \mathbf u(\mathbf x_k)\bigr) + \mathbf R_k,\qquad
\langle R_k R_k\rangle = 2\gamma m_k k_BT/\Delta t \quad\text{(per component)}.
```

- $`\mathbf u(\mathbf x_k) = \mathbf j/\rho`$ at the nearest lattice node (section 3, Nearest node).
- The fluid at that node receives $`-\mathbf F_k`$. The reactions of the particles at the same node are summed in
  particle order and added to the body force $`\rho\mathbf g`$ of the node.
- In lattice units (time step 1):
  $`\mathbf F = -\gamma m\,(\mathbf v - \mathbf u) + \sqrt{2\gamma m k_BT}\,\boldsymbol\xi`$, with
  $`\boldsymbol\xi`$ three independent $`N(0, 1)`$ numbers.
- Drag and noise are part of the force, so the System is integrated with `VerletIntegrator`.
- This is the explicit scheme of the reference CUDA library. openmm-lbm reproduces it, and offers a
  time-centred drag as an alternative (Drag schemes, below).

**Coupling schemes** (`setCouplingScheme()`).
- `EulerMaruyama`, the default: friction and random force as above.
- `NVE`: friction only, with no random force whatever the temperature, that is the same scheme at zero
  temperature.
- With either scheme the total momentum of particles, fluid and walls is conserved. The total kinetic energy, sum of
  $`\rho u^2\Delta x^3/2`$ over the fluid nodes and $`m v^2/2`$ over the particles, is not conserved. The drag
  dissipates about $`\gamma m\,\lvert\mathbf v - \mathbf u\rvert^2\,\Delta t`$ per step. The viscosity damps the
  motion of the fluid, and in an isothermal lattice Boltzmann model the energy damped by viscosity leaves the model
  instead of heating the fluid.
- Example: a particle kicked in a fluid at rest ends up moving with the fluid at $`m v_0/(m + M)`$, as in a
  perfectly inelastic collision. The momentum is conserved, and the kinetic energy falls by the factor
  $`m/(m + M)`$.

**Kinetic energy budget (approximate).** With the NVE scheme the kinetic energy plus the energy dissipated by
viscosity and drag stays constant, but only in the hydrodynamic limit, to $`O(\mathrm{Ma}^2, \mathrm{Kn}^2)`$. It is
a check of the model, not an exact conservation law like that of the momentum. In lattice units
($`\Delta x = \Delta t = 1`$, mass in cells $`m_c`$):
- the kinetic energy is $`E = \sum_{\mathrm{nodes}} \rho u^2/2`$, with $`\mathbf u = \mathbf j/\rho`$, plus
  $`\sum_{\mathrm{particles}} m v^2/2`$;
- the viscous dissipation in one step is
  $`\sum_{\mathrm{nodes}} \frac{\tau - 1/2}{2\tau^2\rho c_s^2}\, \Pi^{\mathrm{neq}}:\Pi^{\mathrm{neq}}`$. It follows
  from the Chapman-Enskog relation $`\Pi^{\mathrm{neq}} = -2\rho c_s^2\tau S`$ and the dissipation
  $`2\rho\nu\, S:S`$ with $`\nu = c_s^2\,(\tau - 1/2)`$, and is computed from $`\Pi^{\mathrm{neq}}`$ of the state
  without finite differences;
- the drag dissipation in one step follows from the discrete update: the particle loses
  $`-\mathbf F\cdot(\mathbf v_n + \mathbf v_{n+1})/2`$ and the fluid, with Guo's forcing, receives $`-\mathbf F`$ at
  the velocity $`\mathbf u - \mathbf F/(2\rho)`$ of the node, so the energy dissipated is
  $`-\mathbf F\cdot\left[(\mathbf v_n + \mathbf v_{n+1})/2 - \mathbf u + \mathbf F/(2\rho)\right]`$, with
  $`\mathbf F = m\,(\mathbf v_{n+1} - \mathbf v_n)/\Delta t`$ the coupling force on the particle, which holds for
  both drags.

Measured on the Reference platform (`python/tests/TestEnergyBudget.py`, $`\tau = 1.1`$):
- a shear wave without particles closes to -3.1% of the initial energy with 16 nodes per wavelength and to -0.77%
  with 32: the residual falls as $`k^2`$, as an $`O(\mathrm{Kn}^2)`$ error should;
- a particle kicked in the fluid closes to -4.4%, the same at Mach numbers 0.035 and 0.10 and with $`12^3`$ to
  $`24^3`$ nodes (-5.5% at $`\tau = 0.62`$). The reaction of the drag acts on a single node, far from the
  hydrodynamic limit ($`\mathrm{Kn}`$ of order 1): part of the energy goes into non-hydrodynamic moments, which the
  regularized collision removes and the hydrodynamic formula does not count.

With the Euler-Maruyama scheme the budget gains the work of the random force; in a steady state its mean
power balances the dissipation of the drag.

**Order in the lattice step.** Moments, removal of the fluid momentum, coupling, collision and
streaming, the bounce-back at the fluid nodes next to the solid nodes (with the `BounceBack` wall scheme), then
the rebuild of the boundary nodes of regularized walls and open faces (section 1). The coupling therefore sees the fluid
momentum after the removal.

**Time levels.** OpenMM's Verlet integrator is a leapfrog: during the force evaluation of step $`t`$ the velocities
are $`\mathbf v(t - \Delta t/2)`$. The fluid momentum before the step is $`\mathbf j(t - \Delta t/2)`$, because
$`\mathbf j(t + \Delta t/2) = \mathbf j(t - \Delta t/2) + \mathbf F(t)`$. The drag therefore compares particle and
fluid velocities at the same half step, explicitly.

**When the coupling forces are computed.** The fluid advances once per integration step, in the force
evaluation of the step, which also computes the coupling forces of that step and their reaction on the
fluid. The force evaluations between steps (`getState(getForces=True)`, `getState(getEnergy=True)`)
return the coupling force of the next step, computed on the current fluid, positions and velocities,
without changing the fluid.
- This is OpenMM's convention for every force. `VerletIntegrator` computes the forces at the current positions
  $`\mathbf x(t)`$ and uses them for the next update
  $`\mathbf v(t + \Delta t/2) = \mathbf v(t - \Delta t/2) + \Delta t\,\mathbf F(t)/m`$, and the force in a State at
  time $`t`$ is that $`\mathbf F(t)`$.
- OpenMM relies on it for the kinetic energy of a State: for `VerletIntegrator` it shifts the velocities by half a
  step with the forces of the State, $`\mathbf v(t - \Delta t/2) + \Delta t\,\mathbf F(t)/(2m)`$, which is the
  full-step velocity $`\bigl(\mathbf v(t - \Delta t/2) + \mathbf v(t + \Delta t/2)\bigr)/2`$ only if
  $`\mathbf F(t)`$ is the force of the next step.
- The random numbers of a step are drawn once, by the first evaluation that needs them, and the step
  uses the same ones. The step recomputes the drag with the velocities it finds, so a change of the
  velocities between the evaluations and the step (the reflection at the walls, `setVelocities()`) is
  taken into account. Extra evaluations therefore do not change the trajectory, which is identical with
  and without them.
- The CUDA, OpenCL and HIP platforms repeat all the force evaluations of a step when the neighbor list of
  a nonbonded force with a cutoff has to grow (`ContextImpl::calcForcesAndEnergy()`, `finishComputation()`
  returns valid = false). This needs more than about 1250 atoms: OpenMM first allocates 20 tiles of 32 × 32
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

On the CUDA, OpenCL and HIP platforms the random numbers come from OpenMM's generator, as in the reference library
(`IntegrationUtilities`, Gaussian numbers in single precision), seeded with `setRandomNumberSeed()`. OpenMM keeps
one generator per Context: another component that uses it with a different seed (an `AndersenThermostat`, for
example) makes OpenMM stop with an error, and the two seeds must then be set equal. At every step the plugin
reserves one Gaussian float4 per (padded) atom of the System and uses element $`i`$, components $`x`$, $`y`$, $`z`$,
for coupled particle $`i`$. This is how the reference library consumes the generator, so with the same seed both
draw the same numbers and a run with the random force can be compared with it step by step: the thermal example
agrees to 3e-11 over 20000 steps on CUDA in double precision
([examples/README.md](../examples/README.md#comparison-with-the-dragopenmm-plugin)). OpenCL generates the same
sequence with slightly different rounding. The random forces of the GPU platforms and of the Reference platform are
different sequences with the same statistics; at $`T = 0`$ (or with the NVE scheme) the platforms agree to rounding.

**Walls.** A coupled particle whose nearest node is solid has entered a wall.
- At the start of the step (`updateContextState()`, where OpenMM's `AndersenThermostat` also changes velocities)
  every component of its velocity is reversed, as for a no-slip wall, if the particle moves into the wall:
  $`\mathbf v\cdot\mathbf n > 0`$.
- $`\mathbf n`$ is the gradient of the solid indicator (1 at solid nodes, 0 at fluid nodes), interpolated
  trilinearly between the eight nodes of the lattice cell that contains the particle. It points from the
  fluid into the wall. For a wall one node thick, the cell of the particle tells from which side it came.
- A particle at a solid node that already moves out of the wall keeps its velocity, so it is not sent
  back into the wall by a second reversal.
- In the step the particle feels the drag of the wall at rest ($`\mathbf u = 0`$) and the random force.
- The reaction on the solid node leaves the fluid, since solid nodes do not collide.
- Uncoupled particles do not see the walls.

**Stability of the explicit drag.** In one step the drag multiplies the velocity of a particle relative to the fluid
by $`1 - \gamma\Delta t`$. It changes sign at every step for $`\gamma\Delta t > 1`$, and it grows without bound for
$`\gamma\Delta t \ge 2`$. A warning is printed when a Context with coupled particles and the explicit drag is
created with $`\gamma\Delta t > 1`$. The centred drag is stable for any friction (Which velocity has the right
temperature, below).

**Self-mobility and relaxation time.** The mobility of a dragged particle is $`1/(m\gamma) + y`$, where $`y`$ is the
hydrodynamic self-mobility from the fluid around its node. $`y`$ should depend only on the viscosity
($`y \sim 1/\eta`$), but with the explicit drag at the nearest node it contains a lattice term that does not
decrease with the viscosity. As a result $`y\,\eta\,\Delta x`$, which should not depend on $`\tau`$, falls with
$`\tau`$ and changes sign. Measured on the Reference platform ($`L`$ = 8 nm, $`\Delta x`$ = 0.5 nm, $`\Delta t`$ =
0.01 ps, $`m`$ = 1000 Da, $`\gamma`$ = 5/ps, protocol of `docs/validation.md`):

| $`\tau`$ | 0.62 | 0.8 | 1.1 | 1.5 | 1.6 | 1.7 | 1.8 | 1.9 | 2.0 | 3.51 |
|---|---|---|---|---|---|---|---|---|---|---|
| $`y\,\eta\,\Delta x`$ | 0.0609 | 0.0577 | 0.0442 | 0.0202 | 0.0134 | 0.0065 | -0.0007 | -0.0080 | -0.0154 | -0.1406 |

(The scan with both drags below gives 0.0443 at $`\tau = 1.1`$, a separate run with the same analysis.) $`y`$
vanishes at $`\tau = 1.79`$ (1.79 also for $`m`$ = 100 Da, $`\gamma`$ = 10/ps), the value found with the reference
library. Above it, particles move less than Langevin particles with the same friction. A warning is printed when a
Context with coupled particles and the explicit drag has $`\tau > 1.7`$.

With the centred drag (Drag schemes, below) the drag sees the fluid velocity with half of the reaction of the
particle in the same step, $`h\mathbf S/m_c`$, so $`y`$ grows by $`\Delta t/(2m_c)`$, that is by $`(\tau - 1/2)/6`$
in units of $`1/(\eta\,\Delta x)`$. The same protocol, run with both drags, gives exactly that:

| $`\tau`$ | 0.62 | 0.8 | 1.1 | 1.5 | 1.8 | 2.0 | 3.51 |
|---|---|---|---|---|---|---|---|
| $`y\,\eta\,\Delta x`$, explicit | 0.0609 | 0.0577 | 0.0443 | 0.0202 | -0.0007 | -0.0154 | -0.1406 |
| $`y\,\eta\,\Delta x`$, centred | 0.0808 | 0.1077 | 0.1443 | 0.1868 | 0.2160 | 0.2346 | 0.3611 |
| explicit + $`(\tau - 1/2)/6`$ | 0.0809 | 0.1077 | 0.1443 | 0.1868 | 0.2160 | 0.2346 | 0.3611 |

($`m`$ = 1000 Da, $`\gamma`$ = 5/ps; $`m`$ = 100 Da, $`\gamma`$ = 10/ps gives the same values within 0.003.) With
the centred drag $`y`$ is positive at every $`\tau`$, but $`y\,\eta\,\Delta x`$ grows with $`\tau`$: neither drag
gives a self-mobility independent of $`\tau`$. A relaxation of the ghost moments independent of $`\tau`$ is the next
candidate correction.

**Kinetic temperature.** OpenMM's leapfrog stores the velocities at half steps. This paragraph is about
the explicit drag; for the centred drag see Which velocity has the right temperature, below.
- For a free particle with fluid at rest, the temperature measured from half-step velocities is
  $`T/(1 - \gamma\Delta t/2)`$. Measured from full-step velocities
  $`\mathbf v(t) = \bigl(\mathbf v(t - \Delta t/2) + \mathbf v(t + \Delta t/2)\bigr)/2`$, it is $`T`$.
- The fluid has no thermal fluctuations of its own and takes part of the momentum of the particles, so the particles
  are slightly colder than $`T`$. Measured on the Reference platform with 200 free beads of 100 Da, $`\Delta x`$ =
  0.5 nm, $`\Delta t`$ = 0.01 ps, $`\tau = 1.10`$, $`\gamma\Delta t = 0.1`$ and $`T`$ = 300 K: 295.8 ± 0.4 K from
  full-step velocities and 311.7 ± 0.4 K from half-step velocities, where $`T/(1 - \gamma\Delta t/2)`$ = 315.8 K.
- **The temperature reported by OpenMM is the full-step one.** The kinetic energy of a State, used by
  `StateDataReporter`, is computed by OpenMM from $`\mathbf v(t - \Delta t/2) + \Delta t\,\mathbf F(t)/(2m)`$ with
  the coupling force of the next step (see above), which is the full-step velocity: it equals the temperature from
  full-step velocities to rounding (checked by `testFullStepKineticEnergy` on every platform and with both drags;
  measured once: 293.98 K both, within 4e-12 K, on 200 samples of 200 beads at 300 K on the Reference platform,
  `docs/validation.md`). With the explicit drag this is the right temperature; with the centred drag it is lower
  than the half-step one. Before version 0.1.0 of this plugin the forces between steps were those of the last step,
  and this temperature was wrong for coupled particles: for a free particle
  $`T\,\bigl[(1 - 3a/2)^2/(1 - a/2) + 9a/2\bigr]`$ with $`a = \gamma\Delta t`$, 363 K at 300 K and $`a = 0.1`$.

**Drag schemes** (`setDragScheme()`, fixed when the Context is created). Two time discretizations of the
drag, with the same random force (same variance, numbers drawn in the same order).
- `Explicit`, the default: the scheme above,
  $`\mathbf F_k = -\gamma m_k\,\bigl(\mathbf v_k(t - \Delta t/2) - \mathbf u(t - \Delta t/2)\bigr) + \mathbf R_k`$.
  Among the discrete Langevin integrators it is the scheme of Groot and Warren [12] with $`\lambda = 1/2`$: velocity
  of the particle half a step before the force, new random numbers at every step.
- `Centered`: the velocities of particle and fluid at the time $`t`$ of the force,

  $`\displaystyle \mathbf F_k = -\gamma m_k\,\bigl(\mathbf v_k(t) - \mathbf u_c(t)\bigr) + \mathbf R_k,`$

  with $`\mathbf v_k(t) = \mathbf v_k(t - \Delta t/2) + \Delta t\,(\mathbf F^{\mathrm c}_k + \mathbf F_k)/(2m_k)`$,
  where $`\mathbf F^{\mathrm c}_k`$ is the sum of the other forces on the particle, and
  $`\mathbf u_c(t) = (\mathbf j_c + \mathbf G_c/2)/\rho_c`$, the velocity that the collision puts in the equilibrium
  of the node $`c`$ (Guo forcing, section 1), $`\mathbf G_c`$ being the total force on the node, body force minus
  the forces of its particles. For a particle in a fluid at rest it is the scheme of Brünger, Brooks and Karplus
  [13] (midpoint drag, new random numbers at every step).

**Solution of the centred drag.** The system is implicit but linear, and couples only the particles of the same
node. In lattice units ($`\Delta t = 1`$, $`h = 1/2`$, $`a = \gamma h`$, masses in cell masses), with
$`\tilde{\mathbf v}_k = \mathbf v_k(t - h) + h\,\mathbf F^{\mathrm c}_k/m_k`$,
$`\tilde{\mathbf u} = (\mathbf j_c + h\rho_c\mathbf g)/\rho_c`$, $`m_c = \rho_c`$, and $`M`$, $`\tilde{\mathbf P}`$
and $`\mathbf R`$ the sums over the particles of the node of $`m_k`$, $`m_k\tilde{\mathbf v}_k`$ and
$`\mathbf R_k`$:

```math
\begin{aligned}
\mathbf S &= \frac{-\gamma\,(\tilde{\mathbf P} - M\tilde{\mathbf u}) + \mathbf R}{1 + a + aM/m_c},\qquad
\mathbf u_c(t) = \tilde{\mathbf u} - h\,\mathbf S/m_c,\\
\mathbf F_k &= \frac{-\gamma m_k\,\bigl(\tilde{\mathbf v}_k - \mathbf u_c(t)\bigr) + \mathbf R_k}{1 + a}.
\end{aligned}
```

$`\mathbf S`$ is the sum of the forces $`\mathbf F_k`$ and the node receives $`-\mathbf S`$, so the total momentum
is conserved exactly, random force included. A solid node is a wall at rest of infinite mass:
$`\mathbf u_c(t) = 0`$, and the term $`aM/m_c`$ drops out. The particles of a node are summed in particle order, as
for the reaction of the explicit drag. The force $`\mathbf F^{\mathrm c}_k`$ is read from OpenMM's forces after the
other forces of the System have been computed, so:
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
- $`\mathbf F^{\mathrm c}`$ is read from OpenMM's fixed point force buffer. The OpenCL platform also has floating
  point force buffers, which it adds to the fixed point one only after the post-computations: they are read too. On
  one device OpenMM's own forces do not write them during the evaluation (OpenMM 8.3 to 8.6), but forces of other
  plugins may.
- When the neighbor list of a nonbonded force has overflowed, OpenMM has already marked the evaluation as
  invalid when the post-computation runs, and will repeat it: the other forces are then incomplete, and the
  post-computation does nothing. The lattice step is done in the repeated evaluation, with the complete
  forces (`testCenteredRepeatedEvaluation`).
- The arithmetic is that of the Reference platform: one thread per particle computes $`\tilde{\mathbf v}_k`$, its
  random force and its key $`\mathrm{node}\cdot N_p + i`$; the keys are sorted with OpenMM's `ComputeSort`, in every
  force evaluation; the first key of each node solves the node and writes the forces of its particles and, in a
  lattice step, the reaction $`-\mathbf S`$ (to the walls at a solid node).

**Which velocity has the right temperature.** Consider the particles of a node and their cell alone, without other
forces, streaming and viscosity (a closed system with a fluctuation-dissipation balance). Its dissipative modes have
the rates $`\lambda = \gamma`$ (relative motion of the particles) and $`\gamma\,(1 + M/m_c)`$ (particles against the
cell); let $`z = \lambda\Delta t/2`$.
- With the centred drag the velocities at the half steps, those of the State, are canonical for every
  $`\gamma\Delta t`$ and number of particles, and the update is stable for every $`\gamma\Delta t`$ (amplification
  factor $`(1 - z)/(1 + z)`$; for $`z > 1`$ the relaxation alternates in sign, which is stable but not accurate).
  The full-step velocities have the temperature $`T/(1 + z)`$ per mode: for one particle, between
  $`T/(1 + \gamma\Delta t/2)`$ when the cell is heavy and $`T/\bigl(1 + \gamma\Delta t\,(1 + m/m_c)/2\bigr)`$ when
  the cell keeps the momentum.
- With the explicit drag the full-step velocities are canonical and the half-step ones have $`T/(1 - z)`$; the
  closed system is stable only for $`\gamma\Delta t\,(1 + M/m_c)/2 < 1`$. In the lattice the streaming carries the
  momentum of the node away, and the bound of a single particle, $`\gamma\Delta t < 2`$, is the one always required.
- For a harmonic force (frequency $`w`$) both have a configurational error of order $`(w\Delta t)^2`$:
  $`\langle x^2\rangle`$ is $`1/\bigl(1 - (w\Delta t)^2/4\bigr)`$ times the exact value for the centred drag,
  $`1/\bigl(1 - (w\Delta t)^2/(2\,(2 - \gamma\Delta t))\bigr)`$ for the explicit one. In coarse-grained models
  $`w\Delta t \lesssim 0.1`$, so the error is below 0.3%. Schemes with exact configurations exist (the family
  analysed in [14, 15]), but none of them has an on-site velocity with the exact temperature [14].

`StateDataReporter` reports the temperature of the full-step velocity (section 2, Kinetic temperature):
right with the explicit drag, lower with the centred drag. `openmmlbm.LBMTemperatureReporter` reports the
temperature of the coupled particles with the right velocity for each scheme.

**Temperature with a fluid without fluctuations.** The fluid has no thermal fluctuations of its own, so the coupled
particles are colder than $`T`$ (Kinetic temperature, above). The centred drag couples a particle to the velocity of
its own cell within the step, the cell responding to the force of the particle (the term $`h\mathbf S/m_c`$), so the
deficit is larger than with the explicit drag. Measured on the Reference platform ($`8^3`$ nodes, $`\Delta x`$ = 0.5
nm, $`\Delta t`$ = 0.01 ps, $`\tau = 0.8`$, $`\gamma\Delta t = 0.1`$, $`T`$ = 300 K, particles of 100 Da, so
$`m/m_c = 1.3`$; 2000 steps after 500, seed 1):

| | 100 particles | 10 particles | 100 particles, fluid 100 times denser |
|---|---|---|---|
| `Explicit`, full-step $`T`$ (K) | 285.8 | 293.0 | 296.8 |
| `Centered`, half-step $`T`$ (K) | 269.6 | 275.9 | 296.7 |

With a heavy fluid both schemes give $`T`$ to 1%; at the density of water the deficit of the centred drag is 8-10%
and does not decrease with fewer particles. It follows from linear response: with
$`\mathbf F = -\zeta\,(\mathbf v - \mathbf u) + \mathbf R`$ and the fluid answering the reaction with
$`\mathbf u = -y\mathbf F`$, the force is $`(-\zeta\mathbf v + \mathbf R)/(1 + \zeta y)`$, and the kinetic
temperature is $`T/(1 + \zeta y)`$. The centred drag has $`y`$ larger by $`\Delta t/(2m_c)`$ (Self-mobility, above),
so $`\zeta\,(y_{\mathrm{centred}} - y_{\mathrm{explicit}}) = \gamma\Delta t\, m/(2m_c)`$: measured 0.032, 0.063 and
0.127 at $`\gamma\Delta t`$ = 0.05, 0.1 and 0.2 with 10 particles, against 0.033, 0.066 and 0.133. With a fluid that
does not respond ($`10^4`$ times denser) both drags give the exact values of their discretization, at
$`\gamma\Delta t`$ = 0.1, 0.5 and 1.5, within 2e-4, the statistical error (`docs/validation.md`). With a fluid that
has thermal fluctuations of its own the same argument gains a term, and only the centred drag gives the right
temperature (section 7, Particles in the fluctuating fluid).

The extra deficit of the centred drag grows with $`\gamma\Delta t\, m/m_c`$, so it is large for heavy particles or large
friction, while the diffusion coefficient does not change. Measured on CUDA (mixed precision; stochastic
tests T2, T6, T7 and examples, `docs/validation.md`), with the temperature that is right for each drag:

| System | $`m/m_c`$ | $`\gamma\Delta t`$ | `Explicit`, full step | `Centered`, half step | centred predicted from the explicit value |
|---|---|---|---|---|---|
| 64 free particles of 1000 Da (T6), 300 K | 13.3 | 0.05 | 270.4 K | 207.7 K | 208 K |
| same, $`\gamma`$ = 10/ps | 13.3 | 0.1 | 254.0 K | 161.1 K | 163 K |
| SOD1, COCOMO2 (110 beads), 298 K, three runs of 200 ns | 1.32 (mean) | 0.1 | 294.0 K | 277.5 K | 276 K |
| SOD1 with friction 30/ps, 50 ns | 1.32 (mean) | 0.3 | 294.6 K | 250.2 K | 246 K |

The prediction is $`1/T_{\mathrm{centred}} = 1/T_{\mathrm{explicit}} + \gamma\Delta t\, m/(2m_cT)`$, for the protein
with the mean bead mass. In T6 the diffusion coefficient of the particles is the same with both drags
($`D/\bigl(k_BT/(m\gamma)\bigr)`$ = 0.984 and 1.010 at $`\gamma`$ = 5 and 10/ps, for both), and so is the diffusion
of the centre of mass of SOD1 (`docs/validation.md`), and the normalized velocity autocorrelation of the centred
drag is closer to the response to a kick computed with the same drag (fluctuation-dissipation for the dynamics:
ratio 0.97 to 0.79 between 0.05 and 1 ps, against 0.93 to 0.50 with the explicit drag). So, without thermal
fluctuations of the fluid, the centred drag keeps the diffusion and is stable for any friction, but its kinetic
temperature is lower. With the fluctuating fluid of section 7 the centred drag gives $`T`$, as the local
fluctuation-dissipation balance of the centred drag (Which velocity has the right temperature, above) suggests for
particles and cells together, while the explicit drag makes the particles too hot (section 7, Particles in the
fluctuating fluid).

**Per-cell reaction on the GPU platforms.** The reaction forces of the particles in the same cell are
summed without atomic operations (`platforms/common/src/kernels/lbmCoupling.cc`), by sorting keys and
reducing segments with one writer per cell, as in the parallel spreading of the immersed boundary method of
Kassen et al. [16] (their case of a single node per point):

1. one thread per particle computes its force and the key $`\mathrm{node}\cdot N_p + i`$, unique, with $`N_p`$ the
   number of coupled particles and $`i`$ the index of the particle in the list of the force;
2. the keys are sorted with OpenMM's `ComputeSort`;
3. the first entry of each node adds the reactions of its particles, in particle order, and writes the sum
   to the node: one writer per node;
4. after the collision the nodes that received a reaction are set back to zero.

With the centred drag, step 1 computes $`\tilde{\mathbf v}_k`$ and the random force $`\mathbf R_k`$, the keys are
sorted in every force evaluation, and in step 3 the first entry solves the node, writes the forces of its particles
and, in a lattice step, the reaction $`-\mathbf S`$. The result is reproducible bit for bit, and the sum is taken in
the order of the Reference platform.

**Forces on the particles.** The coupling forces (in the evaluation of a step those of the step, between steps those
of the next step: When the coupling forces are computed, above) are added to OpenMM's force buffer, in fixed point
(resolution $`2^{-32}`$ kJ/mol/nm), at every force evaluation; with the centred drag in the post-computation. OpenMM
handles the forces in its "real" type: on the OpenCL platform, in single and mixed precision, the total force on a
particle is rounded to single precision, while the fluid receives the exact reaction, so the total momentum is
conserved there to about 1e-8. With CUDA in mixed or double precision, and with OpenCL in double precision, it is
conserved to 1e-13 (`docs/validation.md`).

**Momentum removal.** When it is enabled, on the steps whose index is a multiple of the removal frequency, the
momentum of the fluid is removed right after the moments are computed and before the coupling. The index of a step
is the step count of the Context when the step starts: 0 for the first step of a new Context, and restored by
checkpoints, so a run restarted from a checkpoint removes the momentum at the same steps as an uninterrupted one.
The default frequency is 1, every step. The plugin sums $`\rho`$ and $`\mathbf j`$ over the lattice, computes
$`\mathbf u_{\mathrm{cm}} = \sum\mathbf j/\sum\rho`$ and applies
$`\mathbf j \leftarrow \mathbf j - \rho\,\mathbf u_{\mathrm{cm}}`$ at every node; $`\Pi^{\mathrm{neq}}`$ is left
unchanged. The populations are then rebuilt from the corrected moments by the collision. The sums use two-stage
reductions without atomic operations.

**Fluid update.** The fluid is advanced once per time step. `VerletIntegrator::step()` calls
`ContextImpl::updateContextState()` before computing the forces of each step; `LBMForceImpl` uses that
call to mark the next force evaluation as the one that advances the fluid. This is how OpenMM's own
forces with internal state (`CMMotionRemover`, `AndersenThermostat`, `MonteCarloBarostat`) act once
per step. Other force evaluations (`getState(getForces=True)`, for example) do not advance the fluid: they
compute the coupling force of the next step without its reaction on the fluid, with the random numbers of
that step drawn once (When the coupling forces are computed, above).

## 3. Units and conversions (implemented)

The public API uses OpenMM units: nm, ps, Da (g/mol), K and kJ/mol. Internally the plugin works in lattice units
only, as is usual for lattice Boltzmann [7]. `LBMForceImpl::computeLatticeParameters()` computes the lattice
parameters of the fluid ($`\tau`$, $`\omega`$, the initial velocity, the body acceleration and the velocities and
densities of the faces) and passes $`\Delta x`$, $`\Delta t`$, $`\rho_0`$, the friction and $`k_BT`$, from which
every kernel converts the quantities of the particles with the same factors:

| Quantity | Lattice unit | OpenMM value |
|---|---|---|
| length | $`\Delta x`$ | box length / number of nodes (cells must be cubic) |
| time | $`\Delta t`$ | integrator step size |
| mass | $`m_c = \rho_0\,\Delta x^3`$ | $`\rho_0`$ = `setFluidDensity()`, in Da/nm³ |
| density | $`\rho_0`$ | the fluid at rest has lattice density 1 |
| velocity | $`\Delta x/\Delta t`$ | $`u_{\mathrm{lattice}} = u\,\Delta t/\Delta x`$ |
| acceleration | $`\Delta x/\Delta t^2`$ | $`g_{\mathrm{lattice}} = g\,\Delta t^2/\Delta x`$ |
| force on a cell | $`m_c\,\Delta x/\Delta t^2`$ | $`F_{\mathrm{lattice}} = F\,\Delta t^2/(m_c\,\Delta x)`$ |
| kinematic viscosity | $`\Delta x^2/\Delta t`$ | $`\tau = 3\nu\,\Delta t/\Delta x^2 + 1/2 > 1/2`$ |
| particle mass | $`m_c`$ | $`m_{\mathrm{lattice}} = m/m_c`$, with $`m`$ from the System |
| friction | $`1/\Delta t`$ | $`\gamma_{\mathrm{lattice}} = \gamma\,\Delta t`$ |
| thermal energy | $`m_c\,\Delta x^2/\Delta t^2`$ | $`(k_BT)_{\mathrm{lattice}} = k_BT\,\Delta t^2/(m_c\,\Delta x^2)`$ |

With these units the drag and the noise keep their form on the lattice (time step 1):

```math
F_{\mathrm{lattice}} = -\gamma_{\mathrm{lattice}}\, m_{\mathrm{lattice}}\,
(v_{\mathrm{lattice}} - u_{\mathrm{lattice}}) +
\sqrt{2\gamma_{\mathrm{lattice}}\, m_{\mathrm{lattice}}\,(k_BT)_{\mathrm{lattice}}}\;\xi.
```

This is the physical force times $`\Delta t^2/(m_c\,\Delta x)`$, as it must be. Forces return to OpenMM multiplied
by $`m_c\,\Delta x/\Delta t^2`$, in Da nm/ps² = kJ/mol/nm.

**Example** (water-like fluid used for the SOD1 runs):

- $`\rho_0`$ = 602.2 Da/nm³, $`\nu`$ = 5.0175 nm²/ps, box 15 nm with 30 nodes, $`\Delta t`$ = 0.01 ps;
- this gives $`\Delta x`$ = 0.5 nm, $`m_c`$ = 75.275 Da and $`\tau = 1.1021`$;
- the velocity unit is 50 nm/ps (lattice sound speed 28.9 nm/ps) and the force unit is 3.764e5 kJ/mol/nm;
- at $`T`$ = 298 K, $`(k_BT)_{\mathrm{lattice}} = 1.3166\cdot 10^{-5}`$;
- $`\gamma`$ = 5 /ps gives $`\gamma_{\mathrm{lattice}} = 0.05`$.

**Correspondence with the reference implementation.** These conversions coincide with those of the CUDA lattice
Boltzmann plugin of the DragOpenMM project, which uses the same base units $`\Delta x`$, $`\Delta t`$ and $`m_c`$.
There are two interface differences:

- That plugin takes $`\Delta t`$ and the box from explicit arguments. openmm-lbm reads them from the integrator and
  the System.
- Its body force argument is a force per cell divided by the density, that is $`g\,\Delta x^3`$ for an acceleration
  $`g`$. openmm-lbm takes the acceleration $`g`$ directly.

**The time step is set by the molecular dynamics.** In a stand-alone LB simulation $`\Delta t`$ is a numerical
parameter, chosen to keep the Mach number small [7]. Here the fluid advances once per integrator step, so
$`\Delta t`$ is the step size of OpenMM. Viscosity and resolution are therefore coupled:
$`\tau - 1/2 = 3\nu\,\Delta t/\Delta x^2`$. With $`\Delta t`$ = 0.01 ps:

| $`\Delta x`$ | water, $`\nu`$ = 1.0035 nm²/ps | 5 × water |
|---|---|---|
| 0.5 nm | $`\tau = 0.62`$ | $`\tau = 1.10`$ |
| 0.25 nm | $`\tau = 0.98`$ | $`\tau = 2.91`$ |

Typical velocities are far below the lattice sound speed. A 100 Da bead at 298 K has $`\mathrm{Ma}`$ = 5e-3, so the
fluid is in the quasi-incompressible regime for which the model is accurate.

**Nearest node.** Lattice node $`(i, j, k)`$ sits at $`(i\,\Delta x, j\,\Delta x, k\,\Delta x)`$. A particle at
$`x`$ belongs to the node $`i = \mathrm{round}(x/\Delta x) \bmod n`$, after wrapping $`x`$ into the box. Node $`i`$
therefore owns the interval $`[(i - 1/2)\,\Delta x, (i + 1/2)\,\Delta x)`$.

The thermal energy is $`k_BT`$, with $`k_B`$ = `BOLTZ` of OpenMM, in kJ/mol. The body force on the fluid is set as
an acceleration (`setBodyAcceleration()`); the force density on a node is $`\rho\, g_{\mathrm{lattice}}`$.

Changing the integrator step size after creating the Context would change the lattice time step,
so it is rejected.

## 4. Storage and ordering (implemented)

- **Node index.** Node $`(i, j, k)`$ has index $`i + n_x\,(j + n_y k)`$.
- **Populations.** Stored as deviations from the rest equilibrium, $`\delta f_q = f_q - w_q`$, at
  `[q*numNodes + node]`, with the velocity set of `openmmapi/include/internal/D3Q19.h`:

| $`q`$ | $`\mathbf c_q`$ | weight |
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

- **Deviations from the rest equilibrium.** The rest equilibrium at lattice density 1 is $`f_q = w_q`$, so the
  populations are close to the weights, and the hydrodynamic signal is a small difference
  $`\delta f_q = f_q - w_q`$.
  - Storing $`f_q`$ itself would keep that signal only in the last digits of numbers of order 0.05. With a slow flow
    on a large lattice the total momentum then drifts by rounding: 3e-5 of the momentum with $`64^3`$ nodes, 20000
    steps and a fluid velocity of 5e-7 nm/ps, in double precision.
  - Storing $`\delta f_q`$ keeps the full precision of the type for the signal.
  - The moments follow from the sums of the weights: $`\rho = 1 + \sum_q \delta f_q`$,
    $`\mathbf j = \sum_q \mathbf c_q\,\delta f_q`$ ($`\sum_q w_q = 1`$, $`\sum_q w_q\mathbf c_q = 0`$). The
    non-equilibrium part $`\delta f - \delta f^{\mathrm{eq}}`$ is unchanged, with
    $`\delta f^{\mathrm{eq}} = f^{\mathrm{eq}} - w`$ computed directly (`D3Q19::equilibriumDeviation`).
  - Bounce-back copies $`\delta f`$ unchanged, since opposite directions have the same weight. The momentum exchange
    uses the full $`f = \delta f + w`$, whose part $`w`$ carries the static pressure on the walls.
- **Checkpoints.** A plugin cannot add data to OpenMM checkpoints. `LBMForce::createCheckpoint()` writes
  what they miss: the populations, the random numbers already drawn for the next step, the momentum given
  to the walls in the last step and, on the Reference platform, the random number generator of the force
  (on the GPU platforms it is OpenMM's, which the OpenMM checkpoint contains). With both checkpoints a run
  continues bit for bit (`testCheckpointWithRandomForce`). `LBMForceImpl` writes a header before the data
  of the kernel: a tag, the format version (5), the platform, the grid size, the number of coupled particles,
  the drag scheme, whether the fluid fluctuates, the wall scheme and the boundary types of the six faces. A
  checkpoint is refused on another platform, with another grid, number of coupled particles, drag scheme,
  switch of the fluid fluctuations, wall scheme or types of the faces, and on the GPU platforms with another
  precision; a checkpoint of version 1 (plugin version 0.1.0) is read as one of the explicit drag, those of
  versions 1 and 2 (plugin versions 0.1 and 0.2) as ones without fluid fluctuations, those of versions 1 to 3
  as ones with bounce-back walls and those of versions 1 to 4 as ones with periodic faces. The boundary nodes
  have no state of their own beyond the populations (the time filter of the Density faces reads the velocity
  of the previous step from them), so the restart is exact with walls and open faces too
  (`testFluctuationsRestart`, `testFluidStateRestartWithBoundaries`).
- **Fluid state.** `getFluidState()` returns the deviations $`\delta f_q`$ in this layout, in lattice units: a
  population is the value plus $`w_q`$. Saving and restoring them is exact, so a restarted run is identical to an
  uninterrupted one.
- **Initial state.** A new Context starts from the equilibrium at lattice density 1 and the initial
  velocity, at every fluid node.

## 5. Precision (implemented)

The fluid (populations and moments) uses the "mixed" type of the platform:

- float when the platform `Precision` is `single`;
- double when it is `mixed` or `double`.

The Reference platform always uses double precision.

With the deviations $`\delta f_q`$ the resolution of single precision applies to the signal itself rather than to
the populations of order 0.05, which matters most for weak forces and slow flows. `mixed` is still
recommended for production.

## 6. Stability checks (implemented)

The model is accurate only in the quasi-incompressible regime and for relaxation times in a moderate
range. The plugin checks both.

**At Context creation.**
- $`\tau > 1/2`$ holds by construction, since a viscosity that is not positive is an error.
- A warning is printed on stderr if $`\tau`$ is outside [0.505, 2], the range used by the reference implementation.
- With coupled particles and the explicit drag, warnings for $`\tau > 1.7`$ and $`\gamma\Delta t > 1`$ (section 2),
  and, with fluid fluctuations and the EM scheme at $`T > 0`$, a warning that the particles will be too hot, with
  the estimate $`\gamma\Delta t\, m/(2m_c)`$ for the heaviest one (section 7); with the centred drag, an error if
  `LBMForce` is not the last force or the System has virtual sites (section 2, Solution of the centred drag).
- With fluid fluctuations at $`T > 0`$, a line with $`k_BT`$ in lattice units and the thermal Mach number, and a
  warning if $`k_BT`$ exceeds 1/3000, the largest value validated (section 7, Size of the fluctuations).

**During the simulation.**
- Every $`N`$ steps the plugin computes the largest Mach number of the fluid,
  $`\mathrm{Ma} = \max\lvert\mathbf u\rvert/c_s`$, with $`\mathbf u = \mathbf j/\rho`$ and $`c_s = 1/\sqrt3`$ in
  lattice units.
- $`N`$ is set with `setMachCheckFrequency()`; the default is 100 and 0 disables the check.
- If $`\mathrm{Ma}`$ exceeds the limit set with `setMachNumberLimit()` (default 0.3), the plugin throws an
  `OpenMMException` that reports the step and the value.
- `getFluidMachNumber(context)` returns the current value, for monitoring.
- The check runs after a lattice step that brings the step count of the Context to a multiple of $`N`$ (the
  momentum removal, instead, uses the step count at the start of the step). On the CUDA, OpenCL and HIP
  platforms the maximum is taken by work group on the device and then over the work groups on the host,
  without atomic operations.

**Why 0.3.** It is the usual limit of the incompressible approximation: density fluctuations scale as
$`\mathrm{Ma}^2`$, about 9% at $`\mathrm{Ma} = 0.3`$. In addition, the second-order equilibrium of D3Q19 lacks the
$`u^3`$ terms, so the viscous stress has an error of order $`\mathrm{Ma}^3`$, and the stability margin shrinks
quickly as $`\tau`$ approaches 1/2. Beyond 0.3 the state is no longer physical, so the simulation is stopped. In the
target applications $`\mathrm{Ma}`$ is between 1e-3 and 1e-2: a 100 Da particle at 298 K with $`\Delta x`$ = 0.5 nm
and $`\Delta t`$ = 0.01 ps gives $`\mathrm{Ma}`$ = 5e-3.

**Debug builds** (CMake option `-DLBM_DEBUG=ON`, which defines the macro `LBM_DEBUG`):
- a warning is printed on stderr the first time $`\mathrm{Ma}`$ exceeds 0.1, where the accuracy starts to degrade;
- the lattice parameters are printed when a Context is created: $`\Delta x`$, $`\Delta t`$, $`m_c`$, $`\tau`$,
  $`k_BT/(m_c c_s^2)`$.

**Lattice parameters.** `getLatticeParametersInContext(context, dx, dt, tau)` returns the lattice
spacing, the lattice time step and the relaxation time, in the style of
`NonbondedForce::getPMEParametersInContext()`.

## 7. Fluctuating fluid (implemented on all platforms)

Without fluctuations the fluid receives thermal energy only from the reaction to the random forces on the
coupled particles. The particles then miss the thermal motion of the solvent: their diffusion coefficient is
about $`k_BT/(m\gamma)`$ instead of containing the hydrodynamic contribution, and they are colder than the bath, more
so with the centred drag (section 2). `setFluidFluctuations(true)` gives the fluid thermal fluctuations of its
own, at the temperature of `setTemperature()`: particles and fluid share one heat bath. It works with both
coupling schemes; with `NVE` the particles have no random force and are thermalized only by the fluid. The
default is off, and then nothing changes: the run is identical, bit for bit, to that of version 0.2.

**Model: ghost-mode filtered fluctuating lattice Boltzmann (GMF-FLBM, reference 17).** The regularized
collision of section 1 is already its deterministic part: it rebuilds the non-equilibrium populations from
the second moment alone, so the ghost moments (those above the stress) leave the collision at zero, as if they
relaxed with rate 1, and they carry no deterministic memory. The fluctuating model adds to the post-collision
populations of each fluid node, before streaming, the random part

```math
\begin{aligned}
\xi_q &= w_q \sum_{k=4}^{18} \frac{e_k(\mathbf c_q)}{b_k}\,\phi_k r_k,\\
\phi_k &= \sqrt{\mu\rho\, b_k\,\omega\,(2 - \omega)} &&\text{for the six stress modes, } k = 4\dots 9,\\
\phi_k &= \sqrt{\mu\rho\, b_k} &&\text{for the nine ghost modes, } k = 10\dots 18,
\end{aligned}
```

with $`r_k`$ fifteen independent normal numbers $`N(0, 1)`$, $`\rho`$ the density of the node, $`\omega = 1/\tau`$
and $`\mu = k_BT/c_s^2`$ in lattice units. The $`e_k`$ are the polynomials of the orthogonal D3Q19 basis of Lulli et
al. (reference 18), orthogonal with the weights $`w_q`$,
$`\sum_q w_q\, e_k(\mathbf c_q)\, e_l(\mathbf c_q) = b_k\,\delta_{kl}`$, with $`c_s^2 = 1/3`$:

| $`k`$ | $`e_k(\mathbf c)`$ | $`b_k`$ |
|---|---|---|
| 0 | $`1`$ | 1 |
| 1-3 | $`c_x`$, $`c_y`$, $`c_z`$ | 1/3 |
| 4-6 | $`c_x^2 - c_s^2`$, $`c_y^2 - c_s^2`$, $`c_z^2 - c_s^2`$ | 2/9 |
| 7-9 | $`c_x c_y`$, $`c_x c_z`$, $`c_y c_z`$ | 1/9 |
| 10-12 | $`c_y\,(c_x^2 - c_s^2)`$, $`c_z\,(c_x^2 - c_s^2)`$, $`c_x\,(c_y^2 - c_s^2)`$ | 2/27 |
| 13 | $`c_x\,(c_y^2 + 2c_z^2 - 3c_s^2)/2`$ | 1/18 |
| 14 | $`c_y\,(c_x^2 + 2c_z^2 - 3c_s^2)/2`$ | 1/18 |
| 15 | $`c_z\,(c_x^2 + 2c_y^2 - 3c_s^2)/2`$ | 1/18 |
| 16 | $`c_x^2 c_y^2 - c_s^2\,(c_x^2 + c_y^2) + \tfrac12 c_s^2 c_z^2 + \tfrac12 c_s^4`$ | 7/162 |
| 17 | $`\tfrac27 c_x^2 c_y^2 + c_x^2 c_z^2 - \tfrac97 c_s^2 c_x^2 + \tfrac3{14} c_s^2 c_y^2 - \tfrac67 c_s^2 c_z^2 + \tfrac9{14} c_s^4`$ | 5/126 |
| 18 | $`\tfrac25 c_x^2 c_y^2 + \tfrac25 c_x^2 c_z^2 + c_y^2 c_z^2 - \tfrac3{10} c_s^2 c_x^2 - \tfrac65 c_s^2\,(c_y^2 + c_z^2) + \tfrac9{10} c_s^4`$ | 1/30 |

The same polynomials, written with the Hermite polynomials of the lattice velocities ($`c_s^2 = 1/3`$; for
$`a \ne b`$, without sums over repeated indices):
$`H^{(1)}_a = c_a`$, $`H^{(2)}_{aa} = c_a^2 - c_s^2`$, $`H^{(2)}_{ab} = c_a c_b`$,
$`H^{(3)}_{aab} = (c_a^2 - c_s^2)\,c_b`$ and $`H^{(4)}_{aabb} = (c_a^2 - c_s^2)(c_b^2 - c_s^2)`$:

| $`k`$ | Sector | $`e_k`$ with Hermite polynomials | $`b_k`$ |
|---|---|---|---|
| 0 | density | $`1`$ | 1 |
| 1-3 | momentum | $`H^{(1)}_x`$, $`H^{(1)}_y`$, $`H^{(1)}_z`$ | 1/3 |
| 4-6 | stress | $`H^{(2)}_{xx}`$, $`H^{(2)}_{yy}`$, $`H^{(2)}_{zz}`$ | 2/9 |
| 7-9 | stress | $`H^{(2)}_{xy}`$, $`H^{(2)}_{xz}`$, $`H^{(2)}_{yz}`$ | 1/9 |
| 10-12 | ghost | $`H^{(3)}_{xxy}`$, $`H^{(3)}_{xxz}`$, $`H^{(3)}_{yyx}`$ | 2/27 |
| 13 | ghost | $`\tfrac12 H^{(3)}_{yyx} + H^{(3)}_{zzx}`$ | 1/18 |
| 14 | ghost | $`\tfrac12 H^{(3)}_{xxy} + H^{(3)}_{zzy}`$ | 1/18 |
| 15 | ghost | $`\tfrac12 H^{(3)}_{xxz} + H^{(3)}_{yyz}`$ | 1/18 |
| 16 | ghost | $`H^{(4)}_{xxyy} + \tfrac16 H^{(2)}_{zz}`$ | 7/162 |
| 17 | ghost | $`\tfrac27 H^{(4)}_{xxyy} + H^{(4)}_{xxzz} + \tfrac16 H^{(2)}_{yy} + \tfrac1{21} H^{(2)}_{zz}`$ | 5/126 |
| 18 | ghost | $`\tfrac25 H^{(4)}_{xxyy} + \tfrac25 H^{(4)}_{xxzz} + H^{(4)}_{yyzz} + \tfrac16 H^{(2)}_{xx} + \tfrac1{15} H^{(2)}_{yy} + \tfrac1{15} H^{(2)}_{zz}`$ | 1/30 |

The two tables give the same values on the 19 velocities (checked in exact rational arithmetic, with the norms
$`b_k`$ and the orthogonality). Up to $`k = 12`$ each polynomial is one component of a Hermite tensor; from
$`k = 13`$ they are combinations orthogonalized on the lattice, because the Hermite polynomials of third and fourth
order are not all orthogonal to each other on D3Q19, and $`H^{(3)}_{xyz} = c_x c_y c_z`$ vanishes on every D3Q19
velocity. So the norms must be computed from the discrete scalar product: $`b_k = n!\,c_s^{2n}/\mu`$ (order
$`n`$, multiplicity $`\mu`$ of the component) holds up to $`k = 12`$, not for the combinations ($`b_{13} = 1/18`$,
where one cubic component with multiplicity 3 would have $`2/27`$). All the third-order modes are ghost modes.

Each moment $`m_k = \sum_q e_k(\mathbf c_q)\, f_q`$ receives $`\phi_k r_k`$. The polynomials are evaluated on the
velocities, so they do not depend on the ordering of the populations (`D3Q19::mode()` and `D3Q19::fluctuation()` in
`openmmapi/include/internal/D3Q19.h`).

**Why these amplitudes.** In equilibrium the populations of a node fluctuate independently, with
$`\langle\delta f_q\,\delta f_r\rangle = \mu\rho\, w_q\,\delta_{qr}`$, so a moment has the variance
$`\langle\delta m_k^2\rangle = \mu\rho\, b_k`$: density $`\mu\rho`$, momentum $`\rho k_BT`$ per component
(equipartition for the mass $`\rho`$ of the node). A stress mode relaxes as
$`m_k \to (1 - \omega)\, m_k + \phi_k r_k`$, whose stationary variance is $`\mu\rho\, b_k`$ only if
$`\phi_k^2 = \mu\rho\, b_k\,[1 - (1 - \omega)^2] = \mu\rho\, b_k\,\omega\,(2 - \omega)`$; a ghost mode is replaced
by $`\phi_k r_k`$, so $`\phi_k^2 = \mu\rho\, b_k`$ (Dünweg, Schiller and Ladd, reference 19). The density and the
momentum ($`k = 0\dots 3`$) receive nothing: by orthogonality $`\sum_q \xi_q = 0`$ and
$`\sum_q \mathbf c_q\,\xi_q = 0`$, so the noise conserves the mass and the momentum of every node exactly, and the
removal of the fluid momentum and the Mach number check are unchanged. They fluctuate through the stress, which
streaming couples to them.

**Units.** $`\mu = k_BT/c_s^2 = 3k_BT`$, with $`k_BT`$ in lattice units $`k_BT\,\Delta t^2/(m_c\,\Delta x^2)`$ and
$`m_c = \rho_0\,\Delta x^3`$ the mass of a cell (section 3). For water at 300 K, $`\Delta x`$ = 0.5 nm and
$`\Delta t`$ = 0.01 ps: $`m_c`$ = 75.3 Da, $`k_BT`$ = 1.3e-5 in lattice units and a thermal Mach number
$`\sqrt{k_BT}/c_s`$ = 0.006. The populations are stored as deviations $`f - w`$ (section 4), so fluctuations of
order 1e-3 keep their precision.

**Random numbers.** On the Reference platform the fifteen normal numbers of a node come from the generator of the
force (OpenMM's SFMT, seeded with `setRandomNumberSeed()`, which also draws the random forces on the particles),
node after node in index order, after the coupling of the step. The force evaluations between steps draw only the
random forces of the next step, which come after the fluid numbers of the current step in any case, so they do not
change the sequence. The checkpoint of the force (`createCheckpoint()`) contains the state of the generator, so a
restarted run is identical to an uninterrupted one. The CUDA, OpenCL and HIP platforms draw them from OpenMM's
random numbers of the Context, as they do for the particles: in every step, after those of the particles, four
float4 per node (`IntegrationUtilities::prepareRandomNumbers(4*numNodes)` without boundary nodes, 16 normal numbers
of which 15 are used, in single precision whatever the precision of the platform), with the coefficients
$`w_q\, e_k(\mathbf c_q)/\sqrt{b_k}`$ in an array of the kernel; OpenMM's checkpoints contain the state of that
generator. The buffer of the random numbers grows to 64 bytes per node (and per boundary node, below), already when
the Context is created: OpenMM's checkpoints write the buffer as it is but read it back with the size it has in the
Context that loads them, so a Context created for a restart must have the full size before the first step (with
OpenMM 8.3.1, a buffer grown only in the first step made the restart differ from the uninterrupted run; with 8.6.1
it did not). The boundary nodes of regularized walls and open faces (section 1) draw fifteen normal numbers more
each for the random part of their rebuilt populations: on the Reference platform after those of the collision, in
the order of the boundary nodes; on the other platforms four float4 per boundary node after those of all the nodes,
in the same buffer (`prepareRandomNumbers(4*(numNodes + numBoundaryNodes))`). At zero temperature no numbers are
drawn and nothing is added. Cost on an NVIDIA A100 (CUDA, mixed precision, fluid with one coupled particle): 60, 169
and 1116 us per step without fluctuations and 77, 240 and 1590 us with them on lattices of $`32^3`$, $`64^3`$ and
$`128^3`$ nodes, that is 28% to 43% more; most of it is the generation, writing and reading of the random numbers in
OpenMM's buffer. The platforms therefore draw different numbers, as for the particles: they agree with each other
through the statistics of the tests below, and without fluctuations or at zero temperature to rounding.

**Stability near $`\tau = 1/2`$.** Without fluid velocity the linearized model is a contraction: collision
multiplies every non-conserved mode by $`\lvert 1 - \omega_k\rvert \le 1`$ and streaming permutes the populations,
so the norm $`\sum_q \delta f_q^2/w_q`$ cannot grow. The fluctuating fluid nevertheless becomes unstable close to
$`\tau = 1/2`$, through the nonlinear terms of the equilibrium: the thermal velocities act as local flows at a grid
Reynolds number $`u_{\mathrm{th}}\,\Delta x/\nu`$, with $`u_{\mathrm{th}} = \sqrt{3k_BT}`$ in lattice units. On a
$`64^3`$ lattice over $`10^5`$ steps (CUDA, mixed precision; the same in double precision and on the Reference
platform) the Mach number exceeds the limit within a few thousand steps for $`\tau \le 0.501`$ at $`k_BT = 1/3000`$
(the value of reference 17), $`\tau \le 0.5005`$ at $`k_BT = 10^{-4}`$ and $`\tau = 0.5001`$ at
$`k_BT = 1.3\cdot 10^{-5}`$ (water at $`\Delta x`$ = 0.5 nm and $`\Delta t`$ = 0.01 ps); above these values it is
stable. The limit is a grid Reynolds number between about 50 and 100. The same initial thermal state without noise
is unstable only at $`\tau`$ = 0.5001 and 0.5002 ($`k_BT = 1/3000`$), because its velocities decay. Reference 17
uses the D3Q27 lattice, which stays stable down to $`\tau = 0.5001`$ at $`k_BT = 1/3000`$; on D3Q19 the regularized
collision has less margin. Close to the limit the fluctuations are also too large at long wavelengths
(`docs/validation.md`, Fluctuating fluid). Molecular simulations of water have $`\tau \ge 0.52`$ and
$`k_BT \sim 10^{-5}`$, far from the limit, and the warning for $`\tau`$ below 0.505 (section 6) covers this range.

**Size of the fluctuations.** The fluctuating lattice Boltzmann method holds only small fluctuations: with large ones
the nonlinear terms of the equilibrium make it unstable (above). Their size on the lattice is $`k_BT`$ in lattice
units, $`k_BT\,\Delta t^2/(m_c\,\Delta x^2)`$, which at a given temperature and density goes as
$`\Delta t^2/\Delta x^5`$, or the thermal Mach number

```math
\mathrm{Ma}_{\mathrm{th}} = \frac{\sqrt{k_BT/m_c}}{c_s} = \sqrt{3k_BT}\ \text{(lattice units)},
\qquad c_s = \frac{\Delta x}{\sqrt3\,\Delta t},
```

the r.m.s. velocity of a node along one axis over the speed of sound of the lattice, which is also the r.m.s. relative
fluctuation of the density of a node, $`\langle\delta\rho^2\rangle/\rho_0^2 = k_BT/(m_c c_s^2)`$. Both depend on
the temperature, the density, $`\Delta x`$ and $`\Delta t`$, not on the speed of sound of the real fluid. The lattice
fluid has the speed of sound of the lattice, 28.9 nm/ps with $`\Delta x`$ = 0.5 nm and $`\Delta t`$ = 0.01 ps,
against about 1.5 nm/ps for water: its velocity fluctuations are those of the temperature (equipartition,
$`\langle u_a^2\rangle = k_BT/m_c`$), its density fluctuations those of a fluid much less compressible than water,
smaller by $`(1.5/28.9)^2`$ = 0.0027. Giving the lattice the speed of sound of water would need
$`\Delta t`$ = 0.19 ps and $`k_BT = 4.9\cdot 10^{-3}`$ in lattice units, beyond the validated range; what matters is
that the fluctuations stay small enough for the method. When a Context with fluid fluctuations is created at
$`T > 0`$, the force prints $`k_BT`$ in lattice units and $`\mathrm{Ma}_{\mathrm{th}}`$ (with the domain
decomposition rank 0 only), and a warning if $`k_BT`$ exceeds 1/3000 ($`\mathrm{Ma}_{\mathrm{th}}`$ = 0.032), the
largest value validated (`docs/validation.md`, Fluctuating fluid), at which the fluid is already unstable for
$`\tau \le 0.501`$. Water at 300 K:

| $`\Delta x`$ | $`\Delta t`$ | $`m_c`$ | $`k_BT`$ (lattice units) | $`\mathrm{Ma}_{\mathrm{th}}`$ |
|---|---|---|---|---|
| 1 nm | 0.284 ps | 602 Da | 1/3000 | 0.032 |
| 0.5 nm | 0.01 ps | 75.3 Da | $`1.3\cdot 10^{-5}`$ | 0.0063 |
| 0.25 nm | 0.01 ps | 9.4 Da | $`4.2\cdot 10^{-4}`$ (warning) | 0.036 |

**Particles in the fluctuating fluid: use the centred drag.** The linear-response argument of section 2 (Temperature
with a fluid without fluctuations) gains a term. Let the velocity of the node be
$`\mathbf u = \mathbf u_{\mathrm{th}} - y\mathbf F`$, with $`\mathbf u_{\mathrm{th}}`$ the thermal velocity of the
fluid and $`y`$ the self-mobility with which the node answers the force $`\mathbf F`$ of the particle within the
drag (section 2, Self-mobility). The thermal velocity adds the random force $`\zeta\mathbf u_{\mathrm{th}}`$, whose
strength the fluctuation-dissipation theorem of the fluid sets through the self-mobility $`y_{\mathrm{th}}`$ that
the thermal flows carry, and the kinetic temperature becomes

```math
T_p = T\,\frac{1 + \zeta y_{\mathrm{th}}}{1 + \zeta y}.
```

It is $`T`$ only for the drag whose $`y`$ equals $`y_{\mathrm{th}}`$. The measurements show that this is the centred
drag: its particles have the right temperature at half steps for every friction and time step tried, their velocity
autocorrelation equals the response to a kick computed with the same drag (fluctuation-dissipation theorem for the
dynamics), and their diffusion coefficient follows the Einstein relation with the mobility of the same drag,
$`D = k_BT\,(1/\zeta + y_{\mathrm{centred}})`$. The explicit drag misses the response of the cell within the step,
$`y_{\mathrm{explicit}} = y_{\mathrm{centred}} - \Delta t/(2m_c)`$, so its particles are too hot:

```math
\frac{T_{\mathrm{explicit}}}{T} = 1 + \frac{\gamma\Delta t\, m}{2m_c\,(1 + \zeta y_{\mathrm{explicit}})},
```

the mirror image of the deficit without fluctuations. Measured on CUDA (mixed precision; stochastic tests T2
and T6 with the fluctuating fluid, `docs/validation.md`), with $`y`$ from section 2 and from the reference campaign:

| System | $`m/m_c`$ | $`\gamma\Delta t`$ | `Explicit`, full step | predicted | `Centered`, half step |
|---|---|---|---|---|---|
| 100 free particles of 100 Da (T2), $`\gamma`$ = 10/ps, $`\Delta t`$ = 0.005 ps | 1.33 | 0.05 | 309.2 K | 309.6 K | 299.5 K |
| same, $`\Delta t`$ = 0.01 ps | 1.33 | 0.1 | 319.0 K | 319.4 K | 299.3 K |
| same, $`\Delta t`$ = 0.02 ps | 1.33 | 0.2 | 340.7 K | 339.7 K | 300.7 K |
| 64 free particles of 1000 Da (T6), $`\gamma`$ = 1/ps, $`\Delta t`$ = 0.01 ps | 13.3 | 0.01 | 318.4 K | 319.3 K | 298.9 K |
| same, $`\gamma`$ = 5/ps | 13.3 | 0.05 | 388.7 K | 385.8 K | 299.5 K |
| same, $`\gamma`$ = 10/ps | 13.3 | 0.1 | 466.7 K | 450.5 K | 300.2 K |

The diffusion coefficient now contains the hydrodynamic contribution of the thermal flows and is the same for both
drags, $`D = k_BT\,(1/\zeta + y_{\mathrm{centred}})`$ within 2%: the thermal flows move the particle with the
self-mobility of the centred drag. With `setFluidFluctuations(true)` the centred drag is therefore the scheme to
use, with the temperature of the half steps (`LBMTemperatureReporter`); the explicit drag is accurate only when
$`\gamma\Delta t\, m/(2m_c)`$ is small. When a Context is created with fluid fluctuations, the explicit drag,
coupled particles, the EM scheme, $`T > 0`$ and a friction that is not zero, a warning on stderr gives that bound
for the heaviest coupled particle.

**Walls.** The halfway bounce-back (section 1), done by the fluid nodes next to the walls, is a permutation of
populations: it neither dissipates nor needs noise, and it is unchanged. A population comes back from the wall with
the random part that the fluid node drew for it in its collision, so the wall returns the fluctuations it receives
and adds none of its own: in the language of kinetic theory its thermal accommodation coefficient is zero
($`\alpha = 0`$), while the velocity stays no-slip. Such a wall has no temperature of its own and exchanges no
thermal energy with the fluid, and since the permutation keeps the equilibrium distribution of the populations, the
fluctuations next to it are those of the bulk at every distance (`docs/validation.md`, Walls and open faces,
Fluctuations next to the walls). A wall that forgot what arrives and emitted new fluctuations at a temperature of
its own ($`\alpha = 1`$, as the diffuse re-emission of Maxwell, or a partial accommodation $`0 < \alpha < 1`$) would
give the same equilibrium with the single temperature of the force. It would differ only in the correlations of the
populations over the step from the fluid node to the wall and back, since the ghost modes relax with rate 1 in every
collision, and it would matter only for walls at a temperature different from that of the fluid, which the force
does not have. The plugin therefore has no accommodation parameter: its bounce-back walls are those with
$`\alpha = 0`$. The regularized walls of section 1, and the open faces, rebuild the populations that come from the
solid nodes (or from beyond the face) with a random part drawn on purpose and with the stress of $`\mathbf x`$,
returning only the mass that arrives: for the fluctuations of the stress and ghost modes they behave as a wall with
$`\alpha = 1`$; next to them the fluctuations are at equilibrium from the second node on (section 1, Which wall to
choose).

**Tests** (`tests/TestLBMFluctuations.h`, all platforms and precisions).
- A lattice of a single node, which streams every population back to itself: mass and momentum stay those of the
  start to 1e-15 (2e-6 in single precision, the resolution of the stored populations), and over 20000 steps the
  moments $`k = 4\dots 18`$ have the variance $`\mu b_k`$ within 5% and are uncorrelated with each other, for
  $`\tau = 1`$ (new values at every step) and $`\tau = 0.8`$ (stress modes with memory).
- A fluid at rest on an 8x8x8 lattice, after 300 steps: the variances of the density, of the momentum and of the
  moments $`k = 4\dots 18`$ of a node are $`\mu\rho`$, $`\rho k_BT`$ and $`\mu\rho\, b_k`$ within 5%, for $`\tau`$ =
  0.8 and 2.5; the total mass and momentum are conserved to 1e-13 (1e-5 in single precision).
- The spectrum of the velocity fluctuations on an 8x8x8 lattice at $`\tau = 1`$: the velocity of `getFluidFields()`,
  minus the mean velocity of each sample, is Fourier transformed, and its longitudinal and transverse parts are
  white, $`k_BT/\rho`$ and $`2k_BT/\rho`$ within 5% at long, medium and short wavelengths, at rest, in a uniform
  flow and in a fluid accelerated by a body force (the mean velocity, which grows with time, is subtracted).
- With the fluctuations switched on at zero temperature the run is identical, bit for bit, to the run without
  them; runs with the same seed are identical, also with force evaluations between steps; a different seed
  gives a different run; a run restarted from the checkpoints is identical to the uninterrupted one; with the
  `NVE` scheme a fluctuating fluid sets particles at rest in motion.
- Validation on an NVIDIA A100 (`docs/validation.md`, Fluctuating fluid and Stochastic tests): equilibrium spectra
  of all 19 moments on $`64^3`$ nodes for $`\tau`$ from 0.505 to 100, with the protocol of reference 17 (for
  $`\tau \ge 0.55`$ the ER per node within 0.5% of 1, and $`\mathrm{ER}(\lvert\mathbf k\rvert)`$ within 1.6% from
  the fourth shell on); decay of the thermal shear and sound modes as in the deterministic model; stability near
  $`\tau = 1/2`$ (above); coupled particles with both drags (above).

## 8. Domain decomposition (implemented on all platforms)

`setDomainDecomposition(px, py, pz)` divides the lattice into $`p_x p_y p_z`$ blocks, one per MPI rank of
`MPI_COMM_WORLD`, for runs of the same script in several processes. Rank $`r = c_x + p_x(c_y + p_y c_z)`$ owns the
block with coordinates $`(c_x, c_y, c_z)`$; along an axis of $`n`$ nodes block $`c`$ holds the nodes
$`[\lfloor nc/p\rfloor, \lfloor n(c + 1)/p\rfloor)`$, so blocks differ by at most one node. A 0 lets
`MPI_Dims_create` choose, with the most domains along z, then y (dividing x costs more on the GPUs,
`docs/validation.md`); the product must be the number of ranks, and an axis cannot have more domains than nodes. All
MPI calls are in `openmmapi/src/LBMDecomposition.cpp`, the only file that includes `mpi.h`, whose MPI code is compiled
only with the CMake option `OPENMM_LBM_MPI`; without it, or with one domain, nothing changes.

**Reference platform** (the fluid, walls, open faces and coupled particles).
- The arrays cover the whole lattice on every rank, and each rank advances only the nodes it owns: moments,
  collision and streaming, bounce-back, rebuilt boundary nodes. The Reference platform is the platform of
  correctness, so it keeps the global indices, the periodic wrap and the arithmetic of one domain; the GPU
  platforms keep only their block and a layer of halo nodes (below).
- Each rank sends to rank $`r`$ the populations it pushed into fluid nodes of $`r`$ and receives those that the
  other ranks pushed into its nodes, which is all the communication of a step. Both sides list the slots in the
  order of (node, $`q`$), built once when the Context is created. The communication overlaps the computation: the
  rank first collides and streams the frame of its block (the nodes that push into other ranks), starts the sends
  and receives without blocking (`MPI_Isend`, `MPI_Irecv`), collides and streams the interior, and then waits
  (`MPI_Waitall`) before the walls and the boundary nodes, which read the received populations. With a fluctuating
  fluid the normal numbers of all the nodes of the rank are drawn first, in node order, so that the order of the two
  groups does not change them.
- The populations pushed into solid nodes are not exchanged: the fluid node next to a wall finds, on its own rank,
  what it pushed into the solid node, also when the solid node belongs to another rank, so the bounce-back and the
  rebuilt boundary nodes need no communication (section 1: they use only the populations of the node and the solid
  slots it wrote). The slots of a solid node therefore differ between ranks; they are not part of the state of
  the fluid. Populations that leave through an open face are not exchanged either: the face node of the other
  side rebuilds them.
- The sums of the removal of the fluid momentum and of the force on the walls are added on each rank and then in
  rank order (`MPI_Allgather`), the Mach number is the largest over the ranks (`MPI_Allreduce`): the result is the
  same on every rank and in every run with the same decomposition. `getWallForce()` and `getFluidMachNumber()`
  are therefore collective: every rank must call them.
- Each rank draws its own random numbers (fluctuating fluid, rebuilt boundary nodes) from the generator of the
  force, with the seed of rank 0 plus $`1000003\,r`$: the same seed on every rank would give the nodes of every
  block the same random numbers.
- Without the removal of the fluid momentum and without random numbers (no fluid fluctuations, and no random coupling
  force: temperature zero or the `NVE` coupling scheme) the fluid nodes are identical, bit for bit, to those of one
  domain, for any decomposition (`docs/validation.md`, Domain decomposition). With the removal they agree to rounding,
  because the sum over the nodes is added in another order; with fluctuations the random numbers differ, so the
  agreement is statistical: the variances and spectra of the fluctuations, the planes at the borders of the blocks, the
  independence of the ranks and the temperature of particles that cross the domains agree with one domain within the
  statistical errors (`docs/validation.md`, Fluctuating fluid and particles across the domains).

**CUDA, OpenCL and HIP platforms** (everything that the Reference platform decomposes).
- Each rank stores only its block, plus one layer of halo nodes on each side along the divided axes: for a block of
  $`n_x \times n_y \times n_z`$ nodes divided along the three axes the arrays hold $`(n_x + 2)(n_y + 2)(n_z + 2)`$
  nodes, in the layout of one domain (`platforms/common/src/kernels/lbmFluid.cc`). Along an axis that is not divided
  the block is the whole axis and the streaming wraps around within it, as with one domain. The memory of the fluid
  is divided among the ranks, and each node is computed with the arithmetic of one domain.
- The streaming pushes the populations that leave the block into the halo. After the collision of the frame of the
  block a kernel packs those that go to fluid nodes of other ranks into a buffer, in the precision of the populations
  (4 bytes each in single precision, 8 in mixed and double precision). On the CUDA platform, when the MPI library of
  every rank reads and writes the memory of the devices (CUDA-aware MPI, which Open MPI reports with
  `MPIX_Query_cuda_support()`), the populations for the ranks of the same node are sent from the device (`MPI_Isend`)
  as soon as they are packed, while the interior of the block collides, and those of these ranks are received into the
  buffer of the device: they go from GPU to GPU, over NVLink where the GPUs have it. The populations for and from the
  ranks of other nodes always go through the host, which was faster over InfiniBand for blocks of $`128^3`$ nodes
  (`docs/validation.md`): the buffer is copied to the host and sent, the interior collides meanwhile, and after
  `MPI_Waitall` the populations received are copied to the device. They are then unpacked at their slots, before the
  walls and the boundary nodes. The MPI library (UCX, under Open MPI) binds its transfers between devices to the CUDA
  context of the first one, while OpenMM gives every Context its own CUDA context and destroys it with the Context;
  with a second Context the transfers failed ("context is destroyed"). So only the first Context of a process that
  exchanges device memory does so, and the later ones go through the host, which a run with one Simulation never
  needs. The environment variable `OPENMM_LBM_DEVICE_MPI=0` makes every Context go through the host; OpenCL and HIP
  always do. The slots are listed in the order of (node, $`q`$) of the Reference platform, with the same rules for the
  solid nodes and the open faces: a population pushed into a solid node of the halo stays there, where the
  bounce-back of the node reads it. A rank sends one message to each rank it shares links with, at most 18 (6 faces
  and 12 edges: the D3Q19 lattice has no velocity along the diagonals of the cube), rather than exchanging along
  $`x`$, $`y`$ and $`z`$ in turn and forwarding the populations of the edges; the volume is the same.
- OpenMM computes the size in bytes of an upload to the device as an `int` (`ComputeArray::upload()` and
  `uploadSubArray()`, OpenMM 8.3 to 8.6), which overflows beyond 2 GB: with one domain or with the decomposition, the
  populations of more than 14.1 million stored nodes per GPU (the block with its halo) in mixed and double precision
  (28.3 million in single precision) could not be uploaded, and the Context failed with `CUDA_ERROR_INVALID_VALUE`.
  They are uploaded in parts of 256 MB into a smaller array of the device, and each part copied into its place by a
  kernel. The plugin itself indexes the nodes and the populations with 32-bit integers, on every platform: the
  lattice must have fewer than $`2^{31}`$ nodes, and the 19 populations of a domain with its halo fewer than
  $`2^{31}`$ entries (at most 113025455 nodes, for example $`480^3`$ but not $`490^3`$ in one domain). Creating a
  Context beyond these limits is an error; dividing the lattice into more domains lifts the second one.
- The sums of the removal of the fluid momentum and of the force on the walls, and the Mach number, are reduced over
  the block (on the device by work group or per node, then on the host) and then over the ranks as on the Reference
  platform. A fluctuating fluid draws its
  numbers from OpenMM's generator of the Context of each rank, seeded with the seed of rank 0 plus $`1000003\,r`$.
- The coupled particles follow the rules of the Reference platform (below). The coupling kernels find the nearest node
  in the whole lattice; the rank that owns it computes the coupling force and the reaction on the node, the other ranks
  set the force to zero and give the particle a sort key past all the nodes of the block, so that it enters no segment;
  the forces are summed over the ranks on the host. The sort key of a particle of another rank follows its nearest node
  in the lattice, $`(N_{\mathrm{block}} + \mathrm{node})\,N_p + i`$ with $`N_{\mathrm{block}}`$ the nodes stored by the
  rank, rather than being one value for all those particles: OpenMM's `ComputeSort` puts the keys into buckets by value
  and sorts a bucket larger than a work group with a single work group, so with one value for the particles of the other
  ranks, most of them, the sort took most of a step (with $`10^5`$ particles on four GPUs, 8.7 ms per step, more than on
  one GPU). Summing the forces without blocking (`MPI_Iallreduce`) while the fluid advances was slower with many
  particles on several nodes, and was left out. Every rank reflects its copy of the particles with the mask of the whole
  lattice. The copies stay identical only if every rank computes the same forces of OpenMM: on the CUDA and HIP
  platforms some sums may depend on the order of the threads unless the platform property `DeterministicForces` is
  `true`, which a Context with more than one domain and coupled particles requires (checked when it is created). The
  OpenCL platform has no such property; the comparison of the copies would stop a run whose copies drift apart.
- The exchange of the halo of the density and the velocity (below): at the end of the step a kernel computes the moments
  of the populations (`computeFluidMoments`) and another (`packFields`) copies $`\rho - 1`$ and $`\mathbf j`$ of the
  nodes of the block that are in the halo of other ranks into a buffer; the host computes their fields as
  `getFluidFields()` does and sends them, so the copies are identical bit for bit to what the owner returns. It costs
  one more pass over the populations per step, and the transfer of the halo through the host, which keeps the fields of
  the nodes of the halo only.
- With the same platform and precision, the fluid nodes and the particles are identical bit for bit to those of one
  domain without the removal of the fluid momentum and without random numbers (`docs/validation.md`, Domain
  decomposition).

**Particles, errors, fields and state** (every platform).
- **Particles: replicated data.** Every rank has a Context with all the particles and integrates all of them; only
  the fluid is divided. The copies must stay identical on every rank. The rank that owns the nearest node of a
  coupled particle computes its coupling force (explicit or centred drag, section 2) and the reaction on that node;
  the other ranks set it to zero, and the forces are summed over the ranks (`MPI_Allreduce`). Each force has one
  contribution different from zero, so the sum is exact and the same on every rank. The particles of a node all
  belong to the rank of the node, so the centred drag, which couples them, needs nothing else; the other forces on a
  particle, which the centred drag reads, are the same on every rank. Every rank draws the random numbers of all the
  coupled particles from its own generator, and the force uses those of the owner. Every rank reflects its copy of
  the particles at the walls (the solid nodes are known everywhere), and rank 0 alone counts the momentum given to
  the wall.
- The copies stay identical only if nothing else draws different random numbers on different ranks. The integrator is a
  `VerletIntegrator`, which draws none (section 2); a System with an `AndersenThermostat` or a Monte Carlo barostat is
  refused with more than one domain. The positions and velocities must also start the same on every rank: velocities
  drawn at random need the same seed on every rank, for example `setVelocitiesToTemperature(T, seed)`. Copies that
  differ would give wrong results without any sign, so the plugin compares a hash of the bits of all positions and
  velocities over the ranks at the first lattice step and then with the period of the Mach number check
  (`setMachCheckFrequency()`), and stops with an error if they differ. The Reference platform, whose forces are always
  deterministic, checks every System; the CUDA, OpenCL and HIP platforms check when there are coupled particles, the
  only particles whose copies act on the fluid. The check costs about 7 ns per particle and one `MPI_Allreduce` of two
  numbers; `setParticleCopiesCheck(False)` turns it off, for timings.
- The coupling forces of the next step, which OpenMM evaluates between steps when a script asks for the forces
  (`getState(getForces=True)`, and `getState(getEnergy=True)`, because the `VerletIntegrator` needs the forces for the
  kinetic energy), are summed over the ranks too: like `getWallForce()`, such a call is collective.
- **Errors.** An error found by one rank only would leave the others waiting forever in the next communication.
  The checks that depend on the rank are collective: all ranks must run on the same platform with the same
  precision (the replicated particles would drift apart otherwise), which is checked when the Context is created;
  if a rank differs, every rank stops with the same error. The Mach number check uses the largest value over the
  ranks, and the comparison of the particles a hash compared over the ranks, so every rank stops at the same step.
  The other checks depend only on the System, the same on every rank. An exception in the script on one rank only
  (an error of the script, for example) cannot be made collective: the Python module `openmmlbm` then prints it and
  aborts every rank with `MPI_Abort` (its `sys.excepthook` prints the exception with the previous hook, then calls
  `LBMForce.abortMPI(1)`), as `python -m mpi4py` does, instead of leaving the others waiting.
- **Start and end of MPI.** The plugin initializes MPI on first use (`MPI_Init_thread` with `MPI_THREAD_FUNNELED`:
  only the main thread calls MPI), unless the program has done it, and then finalizes it. The Python module finalizes
  it as soon as the script ends (`atexit`), before the objects of the script, and with them the Contexts, are deleted;
  a C++ program finalizes it at the exit of the process. The order matters on several nodes: the MPI library (UCX,
  under Open MPI) registers the host buffers of the exchanges between nodes in the CUDA context of the Context, which
  OpenMM destroys with the Context, and finalizing MPI after that made UCX print hundreds of errors at the end of every
  run (`docs/validation.md`). MPI initialized by the program, for example by mpi4py, is finalized by the program.
- The MD part is not divided: every rank computes all the other forces of the System. This suits coarse-grained
  systems, in which the fluid dominates the cost.
- **Fields and state of the fluid.** `getFluidFields()`, `getFluidState()` and `setFluidState()` work on the domain of
  the rank (`getLocalDomain()`). The first two need no communication; `setFluidState()` is collective with more than one
  domain (it checks the size of the state on every rank and exchanges the halo, if on). The default keeps the data where
  they are, since gathering them would move the whole lattice to one rank (320 MB of populations for $`128^3`$ nodes, 20
  GB for $`512^3`$). The whole lattice is gathered on rank 0, or set from it, only on request (`gather=True`,
  `scatter=True`, with `MPI_Gatherv` and `MPI_Scatterv`), for tests and small lattices. With one domain the domain is
  the whole lattice and nothing changes. The kernels give the fields and the state of the domain of the rank only, with
  its halo on request, following the same rules on every platform (`CalcLBMForceKernel` in
  `openmmapi/include/LBMKernels.h`, the nodes of the extended domain from `LBMDecomposition::getDomainNodes()`);
  `LBMForceImpl` gathers or scatters them, so no rank holds the fields or the state of the whole lattice unless they are
  gathered on it.
- **Halo of the density and the velocity.** Observables that need the neighbours of a node, such as the gradient of the
  density or the vorticity, need on each rank the fields of the layer one node thick around its domain (the halo, 26
  neighbours with edges and corners). Two switches, off by default, exchange them at the end of every step, after the
  walls and the open faces, when the Context is created and when the state is set: `setDensityHaloExchange()` and
  `setVelocityHaloExchange()`. Each rank sends to each neighbouring rank the fields of its nodes in the halo of that
  rank, in index order, computed as for its own nodes, so the copies are identical bit for bit;
  `getFluidFields(halo=True)` returns them around the domain, without communication. Across periodic boundaries the halo
  wraps; beyond open faces it does not exist (NaN). The fields between steps are those of the populations at the end of
  the step, which are also those the next step starts from. With one domain the halo of an exchanged field comes from
  the lattice itself; the halo of a field that is not exchanged holds NaN. No observable of the plugin uses the halo
  yet.
- **Checkpoints and files of the fluid.** The files are independent of the decomposition: an array of the lattice is
  stored in the order of the node index $`i + n_x(j + n_y k)`$, and each rank writes or reads the nodes of its domain at
  their place in the one file, with collective MPI-IO (`LBMParallelFile` in `LBMDecomposition.cpp`: the file is seen
  through an `MPI_Type_create_subarray` of the domain, so that the rows of the domain along $`x`$ are contiguous both in
  the file and in memory); rank 0 writes the parts that every rank has, such as a header. Nothing is gathered on one
  rank, so the files work for lattices that do not fit in the memory of one rank. With one domain the same code writes
  with the standard library. A checkpoint file (`saveCheckpointFile()`) holds the populations (doubles, value by value:
  $`f_q`$ of every node, then $`f_{q+1}`$), the particles, which are the same on every rank (time, step count, box,
  positions, velocities, global parameters), and for every rank its OpenMM checkpoint and its own part of the state of
  the force: the random numbers already drawn for the next step, the force on the walls of the last step and, on the
  Reference platform, the state of the generator of the force. The random number generators belong to the Context of
  each rank and cannot be divided among the domains of another decomposition: OpenMM's generator on the GPU platforms
  keeps one state per thread, sized by the number of nodes of the domain, and can be seeded only once. So a checkpoint
  loaded with the same decomposition continues the run bit for bit, rank by rank; with another decomposition the
  populations are restored exactly, the particles from their State, which needs no generator (exactly in double
  precision; in mixed and single precision a position outside the box is rounded to float when it is set again), every
  rank keeps the generator of its new Context, and the run continues with new random numbers. Without random numbers (no
  fluctuations, temperature zero) the run does not depend on the decomposition, so it continues as the uninterrupted
  one: exactly in double precision; in mixed and single precision to rounding, because OpenMM keeps the positions in
  float wrapped into the box (with a correction in mixed precision) and rebuilds that representation from the positions
  it is given, so its last bits may differ from those of the uninterrupted run. The slots of the solid nodes are not
  saved beyond what the owner of each node holds: as above, they are not part of the state of the fluid. The VTK files
  of `LBMVTKReporter` follow the same scheme (`writeFluidFile()`): the head and the tail of the XML from rank 0, the
  arrays of the density, the velocity and the solid nodes from every rank, so they are the same files, byte for byte, as
  with one domain.

## References

1. J. Latt and B. Chopard, Math. Comput. Simul. 72, 165 (2006): regularized collision.
2. Z. Guo, C. Zheng and B. Shi, Phys. Rev. E 65, 046308 (2002): forcing scheme.
3. accLB, Procedia Comput. Sci. 267, 40-51 (2025), doi:10.1016/j.procs.2025.08.231, eq. 6:
   $`f = f^{\mathrm{eq}} + (1 - \omega)\, f^{\mathrm{neq}} + S/2`$.
4. LBFAST, Procedia Comput. Sci. 286, 34-47 (2026), doi:10.1016/j.procs.2026.08.017, eq. 3.
5. P. Ahlrichs and B. Dünweg, J. Chem. Phys. 111, 8225 (1999): frictional particle-fluid coupling.
6. B. Dünweg and A. J. C. Ladd, Adv. Polym. Sci. 221, 89 (2009): review of lattice Boltzmann for soft matter.
7. J. Latt, Choice of units in lattice Boltzmann simulations, LBMethod.org (2008).
8. I. Ginzburg, F. Verhaeghe and D. d'Humières, Commun. Comput. Phys. 3, 427 (2008): two-relaxation-time
   scheme, magic parameter $`\Lambda`$ and exact bounce-back solutions.
9. A. J. C. Ladd, J. Fluid Mech. 271, 285 (1994): lattice Boltzmann simulations of particulate
   suspensions, part 1, theoretical foundation.
10. A. J. C. Ladd, J. Fluid Mech. 271, 311 (1994): part 2, numerical results.
11. T. Krüger, H. Kusumaatmaja, A. Kuzmin, O. Shardt, G. Silva and E. M. Viggen, The Lattice Boltzmann
    Method: Principles and Practice (Springer, 2017).
12. R. D. Groot and P. B. Warren, J. Chem. Phys. 107, 4423 (1997): dissipative particle dynamics, the
    modified velocity-Verlet integrator with $`\lambda`$.
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
24. M. Lauricella, A. Tiribocchi, S. Succi, L. Brandt, A. Mukherjee, M. La Rocca and A. Montessori, Phys. Fluids
    37, 072111 (2025), doi:10.1063/5.0271706: thread-safe multiphase lattice Boltzmann model for droplet and
    bubble dynamics at high density and viscosity contrasts; appendix, thread-safe boundary conditions.
25. A. Montessori, M. La Rocca, G. Amati, M. Lauricella, A. Tiribocchi and S. Succi, Phys. Fluids 36, 035171
    (2024): high-order thread-safe lattice Boltzmann model for high performance computing turbulent flow
    simulations.
26. Z. Guo, C. Zheng and B. Shi, Chin. Phys. 11, 366 (2002): non-equilibrium extrapolation method for velocity
    and pressure boundary conditions in the lattice Boltzmann method.
