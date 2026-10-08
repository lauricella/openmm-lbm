# API reference

`openmmlbm.LBMForce` is an OpenMM `Force`. It is configured with setters before the Context is created,
added to the System with `System.addForce()`, and used with a `VerletIntegrator`. Methods that take a
`context` act on the fluid of that Context.

Values are in OpenMM units (nm, ps, Da, K). Setters accept plain numbers in these units or `Quantity`
objects in units that convert to them (see [units](getting_started.md#units)); a `Quantity` in a unit that
does not convert, such as a density in g/cm³ without the Avogadro constant, raises `TypeError`. Getters
return `Quantity` objects where the quantity has units.

- [Summary](#summary)
- [Lattice](#lattice)
- [Fluid properties](#fluid-properties)
- [Driving and initial state](#driving-and-initial-state)
- [Removal of the fluid momentum](#removal-of-the-fluid-momentum)
- [Stability checks](#stability-checks)
- [Solid nodes](#solid-nodes)
- [Open faces](#open-faces)
- [Coupled particles](#coupled-particles)
- [Reading and writing the fluid of a Context](#reading-and-writing-the-fluid-of-a-context)
- [Checkpoints](#checkpoints)
- [Changing parameters in a Context](#changing-parameters-in-a-context)
- [Other methods](#other-methods)
- [Serialization](#serialization)
- [Errors](#errors)

## Summary

| Method | Unit | Default | Change in a Context |
|---|---|---|---|
| `setGridSize(nx, ny, nz)` | nodes | none, required | no |
| `setFluidDensity(density)` | Da/nm³ | 602.214 (water) | no |
| `setKinematicViscosity(viscosity)` | nm²/ps | 1.0035 (water) | no |
| `setBodyAcceleration(acceleration)` | nm/ps² | (0, 0, 0) | yes |
| `setInitialFluidVelocity(velocity)` | nm/ps | (0, 0, 0) | used only at creation |
| `setFluidMomentumRemovalFrequency(frequency)` | steps | 1 | yes |
| `setMachCheckFrequency(frequency)` | steps | 100 | yes |
| `setMachNumberLimit(limit)` | dimensionless | 0.3 | yes |
| `setSolidNodes(nodes)` | node indices | none | no |
| `setWallScheme(scheme)` | | `BounceBack` | no |
| `setFaceBoundary(face, type)` | | `Periodic` | no |
| `setFaceVelocity(face, velocity)` | nm/ps | (0, 0, 0) | yes |
| `setFaceDensity(face, density)` | Da/nm³ | 0, meaning the fluid density | yes |
| `addParticle(particle)`, `setParticle(index, particle)` | particle indices | none | no |
| `setCouplingScheme(scheme)` | | `EulerMaruyama` | yes |
| `setDragScheme(scheme)` | | `Explicit` | no |
| `setFluidFluctuations(fluctuations)` | | `False` | no |
| `setFriction(friction)` | 1/ps | 1.0 | yes |
| `setTemperature(temperature)` | K | 300 | yes |
| `setRandomNumberSeed(seed)` | | 0 | used only at creation |

All parameters are read when the Context is created. "Change in a Context" tells whether a later
change can be applied with [`updateParametersInContext()`](#changing-parameters-in-a-context); the
others need a new Context.

## Lattice

### `setGridSize(nx, ny, nz)`, `getGridSize()`

Number of lattice nodes along x, y and z. It must be set: the default 0 raises an error when the Context is created.
The lattice spans the periodic box of the System, which must be rectangular, with cubic cells:
$`\Delta x = L_x/n_x = L_y/n_y = L_z/n_z`$. `getGridSize()` returns the list `[nx, ny, nz]`.

```python
force.setGridSize(30, 30, 30)
nx, ny, nz = force.getGridSize()
```

See [the lattice](lattice.md#geometry) for the geometry and the numbering of the nodes.

### `setDomainDecomposition(px, py, pz)`, `getDomainDecomposition()`

**In development for version 0.4.0.** The decomposition of the lattice into $`p_x \times p_y \times p_z`$ domains,
one per MPI rank, for runs of the same script in several processes (`srun` or `mpirun`). The default, 1, 1, 1, is
one domain: the whole lattice in one process, without MPI. A 0 lets MPI choose the number of domains along that axis.
More than one domain needs the plugin built with MPI (`-DOPENMM_LBM_MPI=ON`, [installation](installation.md)), and
the product must be the number of MPI ranks; otherwise creating the Context raises an error. It is fixed when the
Context is created. In this version the Reference platform decomposes the fluid, the walls, the open faces and the
coupling of the particles, and the CUDA, OpenCL and HIP platforms the fluid, the walls and the open faces, each rank
on its own GPU ([theory](../theory.md#8-domain-decomposition-in-development-for-version-040)); a Context with more
than one domain refuses checkpoints, and on the CUDA, OpenCL and HIP platforms coupled particles and the exchange of
the halo. To give each rank its own GPU, launch with one GPU per task (`srun --gpus-per-task=1`) or set the platform
property `DeviceIndex` to `openmmlbm.mpiLocalRank()`.
`getFluidFields()`, `getFluidState()` and `setFluidState()` work on the domain of each rank, or gather the whole
lattice on rank 0 ([reading and writing the fluid](#reading-and-writing-the-fluid-of-a-context)).
`getWallForce()`, `getFluidMachNumber()`, `getState(getForces=True)`, `setFluidState()` and the calls with `gather`
must then be called by every rank.

The particles are replicated: every rank builds the same System and integrates all the particles, and the copies must
stay identical. Set positions and velocities in the same way on every rank (velocities drawn at random need a fixed
seed, `setVelocitiesToTemperature(T, seed)`); the plugin compares the copies at the first step and then with the
period of the Mach number check (`setMachCheckFrequency()`), and stops with an error if they differ
(`setParticleCopiesCheck()`, below). With more than one domain the System cannot contain an
`AndersenThermostat` or a Monte Carlo barostat, and every rank must use the same platform and precision.

`LBMForce.isMPIAvailable()` says whether the plugin was built with MPI. `openmmlbm.mpiRank()`, `openmmlbm.mpiSize()`
and `openmmlbm.mpiLocalRank()` (also `LBMForce.getMPIRank()`, `getMPISize()`, `getMPILocalRank()`) give the rank of
the process, the number of ranks and the rank among the processes on the same node, to print from one rank or to
choose the GPU; without MPI they return 0, 1 and 0. The first call initializes MPI if the program has not done it.

An exception that stops the script on one rank only would leave the other ranks waiting forever in the next
communication. When MPI runs with more than one rank, the `openmmlbm` module therefore prints an uncaught exception
and then aborts every rank (`LBMForce.abortMPI()`, which calls `MPI_Abort`), as `python -m mpi4py` does; it installs
this as `sys.excepthook` when it is imported. With one process, or without MPI, nothing changes.

```python
force.setDomainDecomposition(1, 1, 1)
px, py, pz = force.getDomainDecomposition()
print(LBMForce.getMPIRank(), LBMForce.getMPISize())   # 0 1 in one process
```

### `setParticleCopiesCheck(check)`, `getParticleCopiesCheck()`

**In development for version 0.4.0.** With more than one domain, whether the copies of the particles are compared over
the MPI ranks; `True` by default. A check computes a hash of the positions and velocities of all the particles, about
7 ns per particle (0.7 ms for $`10^5`$ particles on one core), and compares it over the ranks with one small
`MPI_Allreduce`; at the default period of 100 steps it costs well under 1 % of a step. Turn it off only to time runs
whose copies are known to be identical, since copies that differ give wrong results without any other sign. With one
domain it does nothing. It is serialized with the force.

### `setDensityHaloExchange(exchange)`, `setVelocityHaloExchange(exchange)`

**In development for version 0.4.0.** Whether the density, and the velocity, of the halo of each domain is exchanged:
the layer one node thick around the domain, edges and corners included, whose nodes belong to the neighbouring ranks.
When it is on, every rank receives at the end of every step the fields of its halo from their owners, so that
`getFluidFields(context, halo=True)` returns them, for example to compute gradients on the domain of the rank without
other communication. Both are off by default: each costs one message to every neighbouring rank per step. With one
domain nothing is exchanged, and the halo comes from the nodes of the lattice across the periodic boundaries. They are
fixed when the Context is created (`getDensityHaloExchange()`, `getVelocityHaloExchange()`) and serialized with the
force.

## Fluid properties

### `setFluidDensity(density)`, `getFluidDensity()`

Mass density of the fluid at rest, in Da/nm³. The default 602.214 Da/nm³ is water (1 g/cm³). The density must be
positive. It sets the mass of a lattice cell, $`m_c = \text{density}\times\Delta x^3`$, and scales the densities
returned by `getFluidFields()`. A density in g/cm³ must be multiplied by `unit.AVOGADRO_CONSTANT_NA` (see
[units](getting_started.md#units)); without it the method raises `TypeError`.

### `setKinematicViscosity(viscosity)`, `getKinematicViscosity()`

Kinematic viscosity $`\nu`$ of the fluid, in nm²/ps. The default 1.0035 nm²/ps is water at 20 C. It must be
positive. With the lattice spacing $`\Delta x`$ and the time step $`\Delta t`$ it fixes the relaxation time
$`\tau = 3\nu\Delta t/\Delta x^2 + \tfrac12`$; a warning is printed if $`\tau`$ is outside [0.505, 2] (see
[relaxation time](lattice.md#relaxation-time)).

## Driving and initial state

### `setBodyAcceleration(acceleration)`, `getBodyAcceleration()`

Uniform acceleration $`\mathbf g`$ applied to the fluid, as a `Vec3` in nm/ps². Every fluid node receives the
force density $`\rho\mathbf g`$, where $`\rho`$ is the local density; this is the usual way to drive a pressure-driven
flow in a periodic channel. The default is zero. To keep the momentum that it gives to the fluid,
disable the [removal of the fluid momentum](#removal-of-the-fluid-momentum).

```python
force.setBodyAcceleration(mm.Vec3(0.02, 0, 0))     # nm/ps^2 along x
```

### `setInitialFluidVelocity(velocity)`, `getInitialFluidVelocity()`

Uniform velocity of the fluid when the Context is created, as a `Vec3` in nm/ps. The fluid starts at
equilibrium with this velocity and the density set by `setFluidDensity()`. The default is zero. It has
no effect on an existing Context.

## Removal of the fluid momentum

### `setFluidMomentumRemovalFrequency(frequency)`, `getFluidMomentumRemovalFrequency()`

Every `frequency` lattice steps, the centre-of-mass velocity of the fluid is subtracted from every
fluid node, so that the total momentum of the fluid becomes zero. The default 1 removes it at every
step; 0 never removes it. The value must not be negative. Steps are numbered by the step count of the
Context, which checkpoints restore; see [removal of the fluid momentum](lattice.md#removal-of-the-fluid-momentum).

Use 0 for flows driven by `setBodyAcceleration()`.

## Stability checks

### `setMachCheckFrequency(frequency)`, `getMachCheckFrequency()`

Every `frequency` lattice steps the plugin computes the largest Mach number of the fluid,
$`\mathrm{Ma} = \max\lvert\mathbf u\rvert/c_s`$ over the fluid nodes. If it exceeds the limit, the step raises an
exception that reports the value and the step. The default is 100; 0 disables the check. The value must not be
negative. Steps are numbered by the step count of the Context, as for the momentum removal.

### `setMachNumberLimit(limit)`, `getMachNumberLimit()`

Largest Mach number allowed by the check. The default 0.3 is the usual limit of the weakly
compressible regime. It must be positive.

See [Mach number and stability](lattice.md#mach-number-and-stability).

## Solid nodes

### `setSolidNodes(nodes)`, `getSolidNodes()`

Lattice nodes that are solid walls, as a sequence of node indices $`i + n_x(j + n_y k)`$: a Python list, a
range or a NumPy array of integers. The fluid does not occupy them, and does not slip on them (no-slip
walls). With the default wall scheme a population of the fluid that streams into a solid node is sent back
to the node it came from (halfway bounce-back): the wall lies halfway between a solid node and its fluid
neighbours. See [`setWallScheme()`](#setwallschemescheme-getwallscheme) for the other choice.

- The default is an empty list: the whole lattice is fluid.
- Indices must be in range and must not repeat, and at least one node must be fluid. The order does not
  matter.
- Solid nodes hold no fluid: `getFluidFields()` returns zero density and zero velocity there. They do
  not enter the removal of the fluid momentum or the Mach number.
- Solid nodes are supported on every platform, and coupled particles are reflected at the walls on every
  platform.

`getSolidNodes()` returns the list of indices.

### `setWallScheme(scheme)`, `getWallScheme()`

How the fluid meets the solid nodes: `LBMForce.BounceBack` (the default) or `LBMForce.Regularized`. Both
give no-slip walls, accurate to second order in the lattice spacing, and conserve the mass of the fluid.

| | `BounceBack` | `Regularized` |
|---|---|---|
| how | a population sent towards a solid node comes back to its node, reversed | the fluid node next to the wall rebuilds the populations that come from the solid nodes, as those of fluid at rest on the solid node |
| where the wall is | halfway between the solid node and the first fluid node | on the solid nodes |
| channel between the solid planes $`j = 0`$ and $`j = n_y - 1`$ | walls at $`y = \Delta x/2`$ and $`(n_y - 3/2)\Delta x`$ | walls at $`y = 0`$ and $`(n_y - 1)\Delta x`$ |
| exact Poiseuille flow at | $`\tau = 7/8`$ | $`\tau = 1`$ |
| with fluid fluctuations | exact thermal equilibrium next to the wall: the wall returns the fluctuations it receives (thermal accommodation zero, [theory.md](../theory.md#7-fluctuating-fluid-implemented-on-all-platforms), Walls) | equilibrium from the second node on; on the first node the momentum along the wall is 3 to 4 % low |
| platforms | all | all |

If you do not know which one to use, keep the default. The `Regularized` scheme is the thread-safe boundary
condition of M. Lauricella et al., Phys. Fluids 37, 072111 (2025), appendix. The theory and the exact solutions are
in [theory.md](../theory.md#solid-nodes-and-walls). It is fixed when the Context is created.

```python
force.setWallScheme(LBMForce.Regularized)
print(force.getWallScheme() == LBMForce.Regularized)
```

Output:

```
True
```

### `getWallForce(context)`

Returns the force exerted on the solid nodes during the last lattice step, as a `Vec3` in kJ/mol/nm.
It is the momentum given to the walls, divided by the time step, by:
- the fluid, through bounce-back (momentum exchange method of Ladd,
  [theory.md](../theory.md#solid-nodes-and-walls)) or, with `Regularized` walls, as the momentum of the
  populations that stream into the solid nodes minus the momentum that the walls put into their fluid nodes
  when they rebuild them;
- the coupled particles: the reaction of particles at solid nodes and their reflections. The fluid nodes next
  to the walls are fluid nodes like the others: the reaction there stays in the fluid.

It is zero before the first step and without solid nodes. With it, the total momentum of particles, fluid and walls
is conserved. In a steady flow driven by a body acceleration $`\mathbf g`$ it equals $`\mathbf g`$ times the mass of
the fluid. It is the total over all solid nodes: the force on each wall separately is not available yet. It does not
advance the fluid.

```python
import numpy as np

nx, ny, nz = force.getGridSize()
k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
index = i + nx*(j + ny*k)
force.setSolidNodes(index[(j == 0) | (j == ny - 1)])     # walls on the planes j = 0 and j = ny-1
```

## Open faces

By default the fluid fills a periodic box: what leaves through a face comes back through the opposite face.
An **open face** instead holds the fluid on it at a velocity or at a density that you choose. With open faces
you can build inlets and outlets, moving plates, flows driven by a pressure difference, on every platform.
The theory is in
[theory.md](../theory.md#open-faces), and two complete scripts are in the examples:
[Couette flow](examples.md#couette-flow-between-two-open-faces) and
[flow in a duct](examples.md#flow-in-a-duct-driven-by-a-pressure-difference).

**Step by step.**

1. **Choose the faces.** The six faces of the box are `LBMForce.XMin` (the face $`x = 0`$), `LBMForce.XMax` (the
   face at the end of the box along x), `YMin`, `YMax`, `ZMin` and `ZMax`. The two faces of the same axis
   must be both periodic or both open: you cannot open `YMin` alone.
2. **Choose the type of each open face** with `setFaceBoundary(face, type)`:
   - `LBMForce.Velocity`: the fluid beyond the face moves with the velocity of `setFaceVelocity(face, velocity)`
     (default zero). A velocity across the face is an inlet or an outlet with a given flow; a velocity
     along the face is a moving plate; zero is a wall at rest.
- `LBMForce.Density`: the fluid beyond the face has the density of `setFaceDensity(face, density)`. The density is
     the pressure: $`p = c_s^2\rho`$ with $`c_s^2 = \Delta x^2/(3\Delta t^2)`$. The default, 0, means the density of
     the fluid at rest (`setFluidDensity()`). The velocity across the face is filtered in time, half the value that
     the arriving fluid gives and half that of the face node at the start of the step (the velocity of
     `getFluidFields()`): this damps a spurious oscillation from one node to the next and from one step to the next,
     and does not change steady flows ([theory, Time filter of the Density faces](../theory.md#open-faces)).
3. **Switch off the removal of the fluid momentum**: `setFluidMomentumRemovalFrequency(0)`. With open faces
   the fluid exchanges momentum with the outside, and the plugin refuses to create the Context otherwise.
4. **Create the Context and run.** The velocities and densities of the faces can be changed during the run
   with `updateParametersInContext()`; the types cannot.

Each face has its own velocity (a vector) and its own density (a number): six faces, six vectors and six
numbers, independent.

```python
# A flow along y: inlet at y = 0 with 0.1 nm/ps, outlet at the end of the box at the density of the fluid.
force.setFaceBoundary(LBMForce.YMin, LBMForce.Velocity)
force.setFaceBoundary(LBMForce.YMax, LBMForce.Density)
force.setFaceVelocity(LBMForce.YMin, mm.Vec3(0, 0.1, 0)*unit.nanometer/unit.picosecond)
force.setFluidMomentumRemovalFrequency(0)
print(force.getFaceBoundary(LBMForce.YMin) == LBMForce.Velocity, force.getFaceVelocity(LBMForce.YMin))
```

Output:

```
True Vec3(x=0.0, y=0.1, z=0.0) nm/ps
```

**What to know.**

- **Where the face is.** The nodes of the face, for `ZMin` the nodes $`k = 0`$ and for `ZMax` the nodes
  $`k = n_z - 1`$, are ordinary fluid nodes. The populations that come from beyond the face are those of fluid with
  the velocity of a `Velocity` face, or the density of a `Density` face, placed one node outside the box: the
  velocity or the density of the face holds there, at $`k = -1`$ and $`k = n_z`$. A Couette flow between two
  `Velocity` faces is $`u(z) = U(z + 1)/(n_z + 1)`$. The grid needs at least 3 nodes along an open axis.
- **Edges and corners.** A node on several open faces takes the velocity of its first `Velocity` face in
  the order XMin, XMax, YMin, YMax, ZMin, ZMax; if all its faces are `Density` faces, it takes the density of
  the first one and the velocity zero. A face node next to a solid node with `Regularized` walls is a wall.
- **Walls and faces together** are fine: for example solid walls around a duct and open faces at its ends.
  Bounce-back is never applied across an open face.
- **The body acceleration** (`setBodyAcceleration()`) still acts on all the fluid.
- **Particles** still live in OpenMM's periodic box: a particle that crosses an open face reappears on the
  opposite side. Keep coupled particles away from the open faces.
- **Mass** is not conserved with open faces (fluid enters and leaves), and `getWallForce()` counts only the
  solid walls, not the open faces.
- **Keep the flow slow and the density differences small**: a few percent at most, so that the fluid stays
  nearly incompressible and the Mach number low.
- **Inlets.** Prefer a `Velocity` inlet with a `Density` outlet. A `Density` face through which the fluid enters can
  become unstable at small $`\tau`$: with a difference of density of 1 % it was stable at $`\tau \ge 0.6`$ and
  unstable at $`\tau \le 0.55`$ ([theory](../theory.md#open-faces)).
- **Next to the faces** the flow enters and leaves, so in a duct driven by two `Density` faces the pressure
  gradient in the middle is somewhat larger than the difference of the faces divided by the length.

### `setFaceBoundary(face, type)`, `getFaceBoundary(face)`

The type of a face: `LBMForce.Periodic` (default), `LBMForce.Velocity` or `LBMForce.Density`. Fixed when the
Context is created.

### `setFaceVelocity(face, velocity)`, `getFaceVelocity(face)`

The velocity of the fluid beyond a `Velocity` face, a `Vec3` in nm/ps; default zero. It can be changed in a
Context with `updateParametersInContext()`. Ignored on the other faces.

### `setFaceDensity(face, density)`, `getFaceDensity(face)`

The density of the fluid beyond a `Density` face, in Da/nm³; the default, 0, means the density of the fluid at
rest. It can be changed in a Context with `updateParametersInContext()`. Ignored on the other faces.

## Coupled particles

Each coupled particle feels a friction force $`-\gamma m(\mathbf v - \mathbf u)`$ relative to the fluid velocity
$`\mathbf u`$ at the nearest lattice node, plus a random force at the given temperature, and the fluid at that node
receives the opposite force (Euler-Maruyama random force, with the explicit or the centred drag of
[`setDragScheme()`](#setdragschemescheme-getdragscheme);
[theory.md](../theory.md#2-particle-fluid-coupling-implemented-on-all-platforms)), on every platform.

- **Forces between steps.** The fluid advances once per integration step. Between steps, a force
  evaluation such as `getState(getForces=True)` returns the coupling force of the next step, as OpenMM does
  for every force, without changing the fluid; the random numbers of a step are drawn once, so extra
  evaluations do not change the run. Before the first step the coupling forces are zero. The energy of the
  coupling is zero.
- **Walls.** A coupled particle whose nearest node is solid and that moves into the wall has every
  component of its velocity reversed at the start of the step, as for a no-slip wall. A particle that
  already moves out of the wall keeps its velocity. Uncoupled particles do not see the walls.
- **Stability.** With the explicit drag (the default), in one step the drag multiplies the velocity of a particle
  relative to the fluid by $`1 - \gamma\Delta t`$. A warning is printed when $`\gamma\Delta t > 1`$, and the motion
  is unstable for $`\gamma\Delta t \ge 2`$. The centred drag is stable for any friction and prints no warning.
- **Temperature.** The temperature that OpenMM reports for the System (`StateDataReporter`, the kinetic
  energy of a State) is that of the full step. With the explicit drag it is the temperature of the coupled
  particles ([temperature example](examples.md#temperature-of-coupled-particles)); with the centred drag
  the right one is that of the half step, which
  [`LBMTemperatureReporter`](#openmmlbmlbmtemperaturereporterfile-reportinterval-force) reports.

### `addParticle(particle)`

Couples the particle with index `particle` in the System to the fluid and returns its index among the
coupled particles. Its mass is taken from the System and must be positive. A particle can be coupled
only once. Particles that are not added (for example the atoms of a rigid wall) do not interact with
the fluid.

```python
for i in range(system.getNumParticles()):
    force.addParticle(i)
```

### `getNumParticles()`, `getParticle(index)`, `setParticle(index, particle)`

Number of coupled particles; System index of the coupled particle `index`; change it.

### `setCouplingScheme(scheme)`, `getCouplingScheme()`

Scheme of the coupling:
- `LBMForce.EulerMaruyama`, the default: friction and random force at the temperature of
  `setTemperature()`;
- `LBMForce.NVE`: friction only, without random force whatever the temperature, so particles and fluid
  exchange momentum through the drag with no thermostat.

The total momentum is conserved with both schemes; the kinetic energy is dissipated by the drag and by
the viscosity of the fluid ([theory.md](../theory.md#2-particle-fluid-coupling-implemented-on-all-platforms)).
The scheme is saved with the force by `XmlSerializer`. It can be changed in a Context with
`updateParametersInContext()`.

```python
force.setCouplingScheme(LBMForce.NVE)
```

### `setDragScheme(scheme)`, `getDragScheme()`

Time discretization of the drag
([theory.md](../theory.md#2-particle-fluid-coupling-implemented-on-all-platforms), Drag schemes):
- `LBMForce.Explicit`, the default: the drag compares the velocity of the particle half a step before the force with
  that of the fluid before the force. The temperature that `StateDataReporter` reports is right. The velocity of a
  particle relative to the fluid changes sign at every step for $`\gamma\Delta t > 1`$ and grows without bound for
  $`\gamma\Delta t \ge 2`$.
- `LBMForce.Centered`: the drag compares the velocities of particle and fluid at the time of the force, and
  is solved exactly for all the particles of a node. It is stable for any friction. The velocities of the
  State have the right temperature, while `StateDataReporter` reports a lower one; use
  [`LBMTemperatureReporter`](#openmmlbmlbmtemperaturereporterfile-reportinterval-force). `LBMForce` must be the
  last force of the System, and the System must not contain virtual sites. It runs on every platform, and
  costs a few percent more than the explicit drag on a GPU.

The drag scheme is fixed when the Context is created and saved with the force by `XmlSerializer`; a
checkpoint can only be loaded in a Context with the same drag scheme.

```python
force.setDragScheme(LBMForce.Centered)      # then system.addForce(force), after all the other forces
```

### `openmmlbm.LBMTemperatureReporter(file, reportInterval, force)`

A reporter for `openmm.app.Simulation` that writes, every `reportInterval` steps, the step, the time (ps) and the
temperature (K) of the particles coupled to `force`, with three degrees of freedom per particle. It uses the
velocity that has the right temperature for the drag scheme of the force: the full-step velocity
$`\mathbf v + \Delta t\,\mathbf F/(2m)`$ with the explicit drag (the same temperature as `StateDataReporter`), the
velocity of the State with the centred drag. `file` is a path or an open file such as `sys.stdout`.

```python
from openmmlbm import LBMTemperatureReporter
reporter = LBMTemperatureReporter('temperature.txt', 1000, force)
# simulation.reporters.append(reporter)
```

### `openmmlbm.LBMVTKReporter(prefix, reportInterval, force)`

A reporter for `openmm.app.Simulation` that writes, every `reportInterval` steps, the fluid of `force` and the
particles of the System in VTK files, which ParaView and VisIt read. Everything is in OpenMM units, and the files
of one report have the step in their name, written with 10 digits (for example `run_fluid_0000001000.vti`):

| File | Content |
|---|---|
| `<prefix>_fluid_<step>.vti` | the lattice (VTK XML ImageData): point $`(i, j, k)`$ is the node at $`(i\Delta x, j\Delta x, k\Delta x)`$, in nm; `density` in Da/nm³ and `velocity` in nm/ps, as [`getFluidFields()`](#getfluidfieldscontext-gatherfalse-halofalse) returns them; with solid nodes also `solid`, 1 for a solid node and 0 for a fluid one |
| `<prefix>_particles_<step>.vtp` | the particles (VTK XML PolyData): positions in nm, `velocity` in nm/ps (the velocities of the State, at the half step), `mass` in Da, `index` in the System, and `coupled`, 1 for the particles coupled to the fluid and 0 for the others |
| `<prefix>.pvd` | the list of the files written, with their times in ps: open it in ParaView to load the whole series |

Each file records its units: a comment at the top of the file and, in the field data, the string array `units`,
one line per array (for example `density: Da/nm^3`), which ParaView shows in its Information panel.

Optional arguments: `fluid=False` or `particles=False` to leave one of the two out; `double=True` to write in
double precision (the default is single precision, which halves the size); `wrap=False` to keep the positions as
they are, instead of wrapping each molecule into the periodic box (as `getState(enforcePeriodicBox=True)` does) so
that the particles overlay the lattice; `append=True`, for a run continued from a checkpoint, to keep the files
already listed in `<prefix>.pvd`. The numbers are binary (raw appended data, little endian). Writing the files
neither advances the fluid nor draws random numbers, so it does not change the run. A file of the fluid holds
16 bytes per node in single precision (4 MB for $`64^3`$ nodes). With more than one domain
([decomposition](#setdomaindecompositionpx-py-pz-getdomaindecomposition)) the fluid cannot be written yet: the
reporter raises an error. To write the particles, attach a reporter with `fluid=False` on rank 0 only
(`openmmlbm.mpiRank() == 0`): it asks the State for positions and velocities only, which needs no communication.

```python
from openmmlbm import LBMVTKReporter
reporter = LBMVTKReporter('run', 1000, force)
# simulation.reporters.append(reporter)
```

### `setFluidFluctuations(fluctuations)`, `getFluidFluctuations()`

Whether the fluid has thermal fluctuations of its own, at the temperature of
[`setTemperature()`](#settemperaturetemperature-gettemperature). The default is `False`: the fluid receives thermal
energy only from the reaction to the random forces on the coupled particles
([limitations](README.md#limitations-of-the-model)). With `True`, every collision adds to the populations of each
node a random part that conserves its mass and momentum and gives the stress and the higher moments their
equilibrium fluctuations (ghost-mode filtered fluctuating lattice Boltzmann,
[theory.md](../theory.md#7-fluctuating-fluid-implemented-on-all-platforms), section 7). It works with both coupling
schemes: with `NVE` the particles have no random force and are thermalized by the fluid only. The random numbers of
the fluid come from the seed of [`setRandomNumberSeed()`](#setrandomnumberseedseed-getrandomnumberseed). It is
fixed when the Context is created. On the CUDA, OpenCL and HIP platforms the random numbers come from OpenMM's
generator, which needs 64 bytes per lattice node and per boundary node (regularized walls and open faces); a step
costs about 30% to 40% more on an NVIDIA A100.

With the fluctuating fluid **use the centred drag**
([`setDragScheme(LBMForce.Centered)`](#setdragschemescheme-getdragscheme)) and measure the temperature with
[`LBMTemperatureReporter`](#openmmlbmlbmtemperaturereporterfile-reportinterval-force): the coupled particles then
have the set temperature, their diffusion coefficient contains the hydrodynamic contribution of the thermal flows,
and the Einstein relation holds. With the explicit drag the particles are too hot, by about
$`\gamma\Delta t\,m/(2m_c)`$, and a warning on stderr says so when the Context is created (measured: 14% for beads
of 100 Da with friction 10/ps and $`\Delta t = 0.02`$ ps, 56% for beads of 1000 Da with friction 10/ps and
$`\Delta t = 0.01`$ ps; [choosing the drag](lattice.md#choosing-the-drag)). Keep $`\tau`$ at 0.505 or above: closer
to 1/2 the fluctuating fluid becomes unstable (at $`\tau \le 0.501`$ with $`k_BT = 1/3000`$ in lattice units, at
$`\tau = 0.5001`$ for water with $`\Delta x = 0.5`$ nm and $`\Delta t = 0.01`$ ps;
[theory.md](../theory.md#7-fluctuating-fluid-implemented-on-all-platforms), stability near $`\tau = 1/2`$).

```python
force.setFluidFluctuations(True)
```

### `setFriction(friction)`, `getFriction()`

Friction coefficient $`\gamma`$ of the coupling, in 1/ps. The default is 1/ps. It must not be negative.
Typical values for coarse-grained beads are 1 to 10/ps; keep $`\gamma\Delta t`$ well below 1.

### `setTemperature(temperature)`, `getTemperature()`

Temperature of the random force on the coupled particles and, with
[fluid fluctuations](#setfluidfluctuationsfluctuations-getfluidfluctuations), of the fluid, in K: particles and
fluid share one heat bath. The default is 300 K. It must not be negative.

### `setRandomNumberSeed(seed)`, `getRandomNumberSeed()`

Seed of the random force, and of the fluctuations of the fluid when they are switched on. With 0, the
default, a different seed is chosen for every Context. Two
simulations with different seeds have different random forces. On the Reference platform the random
force has its own generator, so the same seed reproduces a simulation. On the CUDA, OpenCL and HIP
platforms it uses OpenMM's generator, which has a single seed per Context: a component that uses it with a
different seed, such as an `AndersenThermostat`, makes OpenMM stop with an error, and the two seeds must be
set equal. On a restart the seed does not matter: OpenMM checkpoints contain OpenMM's generator, and
`createCheckpoint()` the generator of the Reference platform ([checkpoints](#checkpoints)).

## Reading and writing the fluid of a Context

None of these methods advances the fluid.

### `getLocalDomain(context)`

Returns `((i0, j0, k0), (ni, nj, nk))`: the first node and the number of nodes along each axis of the domain of this
MPI rank ([domain decomposition](#setdomaindecompositionpx-py-pz-getdomaindecomposition)). With one domain it is
`((0, 0, 0), (nx, ny, nz))`.

```python
(i0, j0, k0), (ni, nj, nk) = force.getLocalDomain(context)
```

### `getFluidFields(context, gather=False, halo=False)`

Returns the tuple `(density, velocity)` with the fluid at every lattice node of the domain of this MPI rank, the whole
lattice with one domain, in node order: node $`(i_0 + i, j_0 + j, k_0 + k)`$ has index $`i + n_i(j + n_j k)`$.

- `density`: a list of densities in Da/nm³, as a `Quantity`;
- `velocity`: a list of `Vec3` velocities in nm/ps, as a `Quantity`. It is the velocity of the forced fluid,
  $`\mathbf u = \mathbf j/\rho + \mathbf g\Delta t/2`$ (see [velocity of the
  fluid](lattice.md#velocity-of-the-fluid)).

Solid nodes have zero density and velocity.

```python
density, velocity = force.getFluidFields(context)
rho = np.array(density.value_in_unit(unit.dalton/unit.nanometer**3))   # shape (numNodes,)
u = np.array(velocity.value_in_unit(unit.nanometer/unit.picosecond))   # shape (numNodes, 3)
```

- `gather=True`: rank 0 receives the fields of the whole lattice, node $`(i, j, k)`$ at index $`i + n_x(j + n_y k)`$,
  and the other ranks empty lists. Every rank must call it, and it moves the whole lattice to rank 0: 32 bytes per
  node, 67 MB for $`128^3`$ nodes and 4.3 GB for $`512^3`$. Use it for tests and small lattices.
- `halo=True`: the fields of the domain with a layer one node thick around it, as an array of
  $`(n_i + 2)(n_j + 2)(n_k + 2)`$ nodes: node $`(i_0 + i, j_0 + j, k_0 + k)`$, with $`i`$ from $`-1`$ to $`n_i`$ and so
  on, has index $`(i + 1) + (n_i + 2)\,[(j + 1) + (n_j + 2)(k + 1)]`$. The layer holds the values of the neighbouring
  nodes, across the periodic boundaries too, for the fields whose halo is exchanged (`setDensityHaloExchange()`,
  `setVelocityHaloExchange()`), and NaN for the other fields and beyond the open faces. It needs no communication.

`gather` and `halo` cannot be both `True`. With one domain, `gather=True` returns the same as the default.

```python
(i0, j0, k0), (ni, nj, nk) = force.getLocalDomain(context)
density, velocity = force.getFluidFields(context, halo=True)
rho = np.array(density.value_in_unit(unit.dalton/unit.nanometer**3)).reshape(nk + 2, nj + 2, ni + 2)
# Without setDensityHaloExchange(True) the layer rho[0, :, :], rho[:, 0, :], ... holds NaN.
```

### `getFluidState(context, gather=False)`, `setFluidState(context, state, scatter=False)`

`getFluidState()` returns the complete state of the fluid as a list of 19 × numNodes numbers.
`setFluidState()` sets it in a Context with the same grid size; it accepts a list or a NumPy array.
With the domain decomposition they work on the domain of the rank, 19 numbers per node of the domain in the order of
`getFluidFields()`, and every rank must call `setFluidState()`. With `gather=True` rank 0 receives the state of the
whole lattice and the other ranks an empty list; with `scatter=True` rank 0 passes the state of the whole lattice and
the state passed by the other ranks (for example `None`) is ignored. Both are collective, and with one domain they do
what the calls without them do. The whole state of $`128^3`$ nodes takes 320 MB, of $`512^3`$ nodes 20 GB.
Together they save and restore the fluid, which is not part of OpenMM checkpoints. To save and continue a
whole run use the [checkpoints](#checkpoints) instead: they also keep the random numbers. Unlike checkpoints,
the fluid state does not depend on the platform: it can move a fluid from one platform to another
([restart](restart.md#moving-a-run-to-another-platform)).

The state holds, for the 19 lattice populations of each node, their deviations from the rest equilibrium,
$`f_q - w_q`$, in lattice units, stored as `[q*numNodes + node]`. A population is the value plus the weight $`w_q`$:
1/3 for $`q = 0`$, 1/18 for $`q = 1`$ to 6, 1/36 for $`q = 7`$ to 18 (see
[theory.md](../theory.md#4-storage-and-ordering-implemented)). At rest the state is zero. Treat it as opaque unless
you know the model. Saving and restoring it is exact. For large lattices the list is long: convert it at once to a
NumPy array, for example `np.array(force.getFluidState(context))`; with $`64^3`$ nodes it takes 40 MB.

### `getFluidMachNumber(context)`

Returns the largest Mach number of the fluid, $`\max\lvert\mathbf j/\rho\rvert/c_s`$ over the fluid nodes, at the
current step.

### `getLatticeParametersInContext(context)`

Returns the tuple `(dx, dt, tau)`: the lattice spacing (a `Quantity` in nm), the lattice time step
(a `Quantity` in ps) and the relaxation time (a number).

```python
dx, dt, tau = force.getLatticeParametersInContext(context)
```

## Checkpoints

OpenMM checkpoints (`Context.createCheckpoint()`) do not contain the fluid. These methods and functions save
it, with the rest of what the force needs, so that a run continues exactly. The page
[saving and continuing a simulation](restart.md) explains how to use them, step by step.

### `createCheckpoint(context)`, `loadCheckpoint(context, data)`

`createCheckpoint()` returns, as `bytes`, the part of the state of the Context that belongs to the force and
that OpenMM checkpoints miss: the populations of the fluid, the random numbers already drawn for the next
step, the force on the walls of the last step and, on the Reference platform, the state of the random number
generator of the force. `loadCheckpoint()` restores it in a Context built from the same System, on the same
platform and with the same precision. Load the OpenMM checkpoint of the same step first. In C++ they take a
`std::ostream` and a `std::istream` opened in binary mode.

```python
data = force.createCheckpoint(context)
force.loadCheckpoint(context, data)
print(type(data).__name__, data[:8])
```

Output:

```
bytes b'LBMCKPT1'
```

Errors: a checkpoint written on another platform, with another precision, for a different grid size,
number of coupled particles, drag scheme, wall scheme, types of the faces (`setFaceBoundary()`) or switch of the
fluid fluctuations, by a newer version of the plugin, or data that are not a
checkpoint or are damaged or truncated, make `loadCheckpoint()` raise an exception
([troubleshooting](troubleshooting.md)). A checkpoint of version 0.1.0 has no drag scheme in its header,
and loads only in a Context with the explicit drag; checkpoints of versions 0.1 and 0.2 load only in a Context
without fluid fluctuations, with the `BounceBack` wall scheme and with periodic faces. The velocities and densities
of the faces are not checked: those of the new Context are used. The checkpoints work with every combination of
fluid fluctuations, wall scheme and open faces, on every platform: the restarted run is identical, bit for bit,
to the uninterrupted one, and the time filter of the `Density` faces needs nothing more, since it reads the
velocity of the previous step from the populations.

### `openmmlbm.saveCheckpoint(file, context, force)`, `openmmlbm.loadCheckpoint(file, context, force)`

Functions of the module `openmmlbm` that write and read one file with both checkpoints: OpenMM's
(`context.createCheckpoint()`) and the force's (`force.createCheckpoint(context)`). `saveCheckpoint()` writes
to `file + '.tmp'` and then renames it, so an interrupted write never damages an existing checkpoint.
`loadCheckpoint()` restores positions, velocities, box, time, step count and random numbers, and the fluid;
it raises `ValueError` if the file is not a checkpoint written by `saveCheckpoint()`, or is truncated.

### `openmmlbm.LBMCheckpointReporter(file, reportInterval, force)`

A reporter for `openmm.app.Simulation`: every `reportInterval` steps it calls
`saveCheckpoint(file, simulation.context, force)`, replacing the previous checkpoint. It is the counterpart of
OpenMM's `CheckpointReporter` for a System with an `LBMForce`.

## Changing parameters in a Context

### `updateParametersInContext(context)`

Copies the current values of the force into an existing Context. Change the values with the setters,
then call this method:

```python
force.setBodyAcceleration(mm.Vec3(0, 0, 0))
force.updateParametersInContext(context)
```

It updates:

- the body acceleration;
- the friction, the temperature (also of a fluctuating fluid) and the coupling scheme;
- the frequency of the removal of the fluid momentum;
- the frequency and the limit of the Mach number check;
- the velocities and the densities of the open faces.

The grid size, the fluid density and viscosity, the solid nodes, the wall scheme, the types of the faces,
the set of coupled particles, the drag scheme and the fluid fluctuations cannot be changed this way: the method raises an error if they differ from those of the Context. The
initial velocity and the random seed are used only when the Context is created. To change any of
these, create a new Context, and transfer the fluid with `getFluidState()` and `setFluidState()` if
the grid is the same. `Context.reinitialize()` also restarts the fluid from its initial state.

## Other methods

### `usesPeriodicBoundaryConditions()`

Always `True`: the fluid fills the periodic box.

### `LBMForce.isinstance(force)`, `LBMForce.cast(force)`

Static methods to recognize an `LBMForce` among the forces of a System and to obtain it with its own
methods, for example after deserialization:

```python
lbm = [LBMForce.cast(f) for f in system.getForces() if LBMForce.isinstance(f)][0]
```

### Methods inherited from `Force`

`setForceGroup()`, `getForceGroup()`, `setName()` and `getName()` work as for every OpenMM force. The
fluid advances during the force evaluation of each integration step, so the force must be in a group
that the integrator evaluates; by default the integrator evaluates all groups.

## Serialization

`openmm.XmlSerializer` saves and loads an `LBMForce`, alone or as part of a System, with all its
parameters: grid, fluid properties, friction, temperature, random number seed, body acceleration, initial
velocity, frequencies, Mach limit, coupling and drag schemes, fluid fluctuations, wall scheme, solid nodes, the
type, velocity and density of each face, coupled particles, domain decomposition and check of the copies of the
particles, force group and name. The fluid
of a Context is not part of it. The XML has version 8. Older XML still loads: versions 1 to 3 (written by version
0.1.0) with the explicit drag, versions 1 to 4 (versions 0.1 and 0.2 of the plugin) without fluid fluctuations,
versions 1 to 5 with the `BounceBack` wall scheme, versions 1 to 6 with periodic faces and versions 1 to 7 (version
0.3 of the plugin) with one domain and the check of the copies on. Older versions of the plugin cannot
read newer XML. Import `openmmlbm` before
deserializing. A force deserialized on its own is returned as a generic `openmm.Force`; obtain the
`LBMForce` with `LBMForce.cast()` (see the [serialization example](examples.md#serialization)).

## Errors

Invalid settings raise a Python `Exception` with the messages listed in
[troubleshooting](troubleshooting.md#error-messages), most of them when the Context is created (for example open
faces with the removal of the fluid momentum, or a single open face on an axis). Some conditions only print a
warning on stderr: a relaxation time outside [0.505, 2], and, with coupled particles and the explicit drag,
$`\tau > 1.7`$, $`\gamma\Delta t > 1`$, and fluid fluctuations (the particles are then too hot).
