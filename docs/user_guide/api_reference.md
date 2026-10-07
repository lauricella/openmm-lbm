# API reference

`openmmlbm.LBMForce` is an OpenMM `Force`. It is configured with setters before the Context is created,
added to the System with `System.addForce()`, and used with a `VerletIntegrator`. Methods that take a
`context` act on the fluid of that Context.

Values are in OpenMM units (nm, ps, Da, K). Setters accept plain numbers in these units or `Quantity`
objects (see [units](getting_started.md#units)); getters return `Quantity` objects where the quantity
has units.

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
| `setFluidDensity(density)` | Da/nm^3 | 602.214 (water) | no |
| `setKinematicViscosity(viscosity)` | nm^2/ps | 1.0035 (water) | no |
| `setBodyAcceleration(acceleration)` | nm/ps^2 | (0, 0, 0) | yes |
| `setInitialFluidVelocity(velocity)` | nm/ps | (0, 0, 0) | used only at creation |
| `setFluidMomentumRemovalFrequency(frequency)` | steps | 1 | yes |
| `setMachCheckFrequency(frequency)` | steps | 100 | yes |
| `setMachNumberLimit(limit)` | dimensionless | 0.3 | yes |
| `setSolidNodes(nodes)` | node indices | none | no |
| `setWallScheme(scheme)` | | `BounceBack` | no |
| `setFaceBoundary(face, type)` | | `Periodic` | no |
| `setFaceVelocity(face, velocity)` | nm/ps | (0, 0, 0) | yes |
| `setFaceDensity(face, density)` | Da/nm^3 | 0, meaning the fluid density | yes |
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

Number of lattice nodes along x, y and z. It must be set: the default 0 raises an error when the
Context is created. The lattice spans the periodic box of the System, which must be rectangular, with
cubic cells: dx = Lx/nx = Ly/ny = Lz/nz. `getGridSize()` returns the list `[nx, ny, nz]`.

```python
force.setGridSize(30, 30, 30)
nx, ny, nz = force.getGridSize()
```

See [the lattice](lattice.md#geometry) for the geometry and the numbering of the nodes.

## Fluid properties

### `setFluidDensity(density)`, `getFluidDensity()`

Mass density of the fluid at rest, in Da/nm^3. The default 602.214 Da/nm^3 is water (1 g/cm^3). The
density must be positive. It sets the mass of a lattice cell, m_c = density dx^3, and scales the
densities returned by `getFluidFields()`. A density in g/cm^3 must be multiplied by
`unit.AVOGADRO_CONSTANT_NA` (see [units](getting_started.md#units)).

### `setKinematicViscosity(viscosity)`, `getKinematicViscosity()`

Kinematic viscosity nu of the fluid, in nm^2/ps. The default 1.0035 nm^2/ps is water at 20 C. It must
be positive. With the lattice spacing dx and the time step dt it fixes the relaxation time
tau = 3 nu dt/dx^2 + 1/2; a warning is printed if tau is outside [0.505, 2] (see
[relaxation time](lattice.md#relaxation-time)).

## Driving and initial state

### `setBodyAcceleration(acceleration)`, `getBodyAcceleration()`

Uniform acceleration g applied to the fluid, as a `Vec3` in nm/ps^2. Every fluid node receives the
force density rho g, where rho is the local density; this is the usual way to drive a pressure-driven
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
Ma = max |u|/c_s over the fluid nodes. If it exceeds the limit, the step raises an exception that
reports the value and the step. The default is 100; 0 disables the check. The value must not be
negative. Steps are numbered by the step count of the Context, as for the momentum removal.

### `setMachNumberLimit(limit)`, `getMachNumberLimit()`

Largest Mach number allowed by the check. The default 0.3 is the usual limit of the weakly
compressible regime. It must be positive.

See [Mach number and stability](lattice.md#mach-number-and-stability).

## Solid nodes

### `setSolidNodes(nodes)`, `getSolidNodes()`

Lattice nodes that are solid walls, as a sequence of node indices i + nx*(j + ny*k): a Python list, a
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
| where the wall is | halfway between the solid node and the first fluid node | on the first fluid node next to the solid nodes |
| channel between the solid planes j = 0 and j = ny - 1 | walls at y = dx/2 and (ny - 3/2) dx | walls at y = dx and (ny - 2) dx |
| most accurate for | tau < 15/16 (water: tau about 0.6) | tau > 15/16 |
| with fluid fluctuations | exact thermal equilibrium next to the wall | fluctuations 3 to 9 % low on the first node next to the wall |
| platforms | all | all |

If you do not know which one to use, keep the default. The theory and the exact solutions are in
[theory.md](../theory.md#solid-nodes-and-walls). It is fixed when the Context is created.

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
- the coupled particles: the reaction of particles at solid nodes (and, with `Regularized` walls, at the fluid
  nodes on the walls) and their reflections.

It is zero before the first step and without solid nodes. With it, the total momentum of particles, fluid
and walls is conserved. In a steady flow driven by a body acceleration g it equals g times the mass of
the fluid. It is the total over all solid nodes: the force on each wall separately is not available yet.
It does not advance the fluid.

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

1. **Choose the faces.** The six faces of the box are `LBMForce.XMin` (the face x = 0), `LBMForce.XMax` (the
   face at the end of the box along x), `YMin`, `YMax`, `ZMin` and `ZMax`. The two faces of the same axis
   must be both periodic or both open: you cannot open `YMin` alone.
2. **Choose the type of each open face** with `setFaceBoundary(face, type)`:
   - `LBMForce.Velocity`: the fluid on the face moves with the velocity of `setFaceVelocity(face, velocity)`
     (default zero). A velocity across the face is an inlet or an outlet with a given flow; a velocity
     along the face is a moving plate; zero is a wall at rest.
   - `LBMForce.Density`: the fluid on the face has the density of `setFaceDensity(face, density)`, and moves
     only across the face. The density is the pressure: p = c_s^2 rho with c_s^2 = dx^2/(3 dt^2). The
     default, 0, means the density of the fluid at rest (`setFluidDensity()`). The velocity across the face
     is filtered in time, half the value that the arriving fluid gives and half that of the previous step:
     this removes a spurious oscillation from one node to the next and from one step to the next, and does
     not change steady flows ([theory, Time filter of the Density faces](../theory.md#open-faces)).
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

- **The face is on the nodes of the face**, not on the edge of the box: for `ZMin` the nodes k = 0, for
  `ZMax` the nodes k = nz - 1. The fluid on these nodes has exactly the velocity of a `Velocity` face, or the
  density of a `Density` face. The grid needs at least 3 nodes along an open axis.
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

### `setFaceBoundary(face, type)`, `getFaceBoundary(face)`

The type of a face: `LBMForce.Periodic` (default), `LBMForce.Velocity` or `LBMForce.Density`. Fixed when the
Context is created.

### `setFaceVelocity(face, velocity)`, `getFaceVelocity(face)`

The velocity of the fluid on a `Velocity` face, a `Vec3` in nm/ps; default zero. It can be changed in a
Context with `updateParametersInContext()`. Ignored on the other faces.

### `setFaceDensity(face, density)`, `getFaceDensity(face)`

The density of the fluid on a `Density` face, in Da/nm^3; the default, 0, means the density of the fluid at
rest. It can be changed in a Context with `updateParametersInContext()`. Ignored on the other faces.

## Coupled particles

Each coupled particle feels a friction force -gamma m (v - u) relative to the fluid velocity u at the
nearest lattice node, plus a random force at the given temperature, and the fluid at that node receives
the opposite force (Euler-Maruyama random force, with the explicit or the centred drag of
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
- **Stability.** With the explicit drag (the default), in one step the drag multiplies the velocity of a
  particle relative to the fluid by 1 - friction*dt. A warning is printed when friction*dt > 1, and the
  motion is unstable for friction*dt >= 2. The centred drag is stable for any friction and prints no
  warning.
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
- `LBMForce.Explicit`, the default: the drag compares the velocity of the particle half a step before the
  force with that of the fluid before the force. The temperature that `StateDataReporter` reports is right.
  The velocity of a particle relative to the fluid changes sign at every step for friction*dt > 1 and grows
  without bound for friction*dt >= 2.
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

A reporter for `openmm.app.Simulation` that writes, every `reportInterval` steps, the step, the time (ps) and
the temperature (K) of the particles coupled to `force`, with three degrees of freedom per particle. It uses
the velocity that has the right temperature for the drag scheme of the force: the full-step velocity
v + dt F/(2m) with the explicit drag (the same temperature as `StateDataReporter`), the velocity of the State
with the centred drag. `file` is a path or an open file such as `sys.stdout`.

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
| `<prefix>_fluid_<step>.vti` | the lattice (VTK XML ImageData): point (i, j, k) is the node at (i dx, j dx, k dx), in nm; `density` in Da/nm^3 and `velocity` in nm/ps, as [`getFluidFields()`](#getfluidfieldscontext) returns them; with solid nodes also `solid`, 1 for a solid node and 0 for a fluid one |
| `<prefix>_particles_<step>.vtp` | the particles (VTK XML PolyData): positions in nm, `velocity` in nm/ps (the velocities of the State, at the half step), `mass` in Da, `index` in the System, and `coupled`, 1 for the particles coupled to the fluid and 0 for the others |
| `<prefix>.pvd` | the list of the files written, with their times in ps: open it in ParaView to load the whole series |

Optional arguments: `fluid=False` or `particles=False` to leave one of the two out; `double=True` to write in
double precision (the default is single precision, which halves the size); `wrap=False` to keep the positions as
they are, instead of wrapping each molecule into the periodic box (as `getState(enforcePeriodicBox=True)` does) so
that the particles overlay the lattice; `append=True`, for a run continued from a checkpoint, to keep the files
already listed in `<prefix>.pvd`. The numbers are binary (raw appended data, little endian). Writing the files
neither advances the fluid nor draws random numbers, so it does not change the run. A file of the fluid holds
16 bytes per node in single precision (4 MB for 64^3 nodes).

```python
from openmmlbm import LBMVTKReporter
reporter = LBMVTKReporter('run', 1000, force)
# simulation.reporters.append(reporter)
```

### `setFluidFluctuations(fluctuations)`, `getFluidFluctuations()`

Whether the fluid has thermal fluctuations of its own, at the temperature of
[`setTemperature()`](#settemperaturetemperature-gettemperature). The default is `False`: the fluid receives
thermal energy only from the reaction to the random forces on the coupled particles
([limitations](README.md#limitations-of-the-model)). With `True`, every collision adds to the populations of
each node a random part that conserves its mass and momentum and gives the stress and the higher moments their
equilibrium fluctuations (ghost-mode filtered fluctuating lattice Boltzmann,
[theory.md](../theory.md#7-fluctuating-fluid-implemented-on-all-platforms),
section 7). It works with both coupling schemes: with `NVE` the particles have no random force and are
thermalized by the fluid only. The random numbers of the fluid come from the seed of
[`setRandomNumberSeed()`](#setrandomnumberseedseed-getrandomnumberseed). It is fixed when the Context is
created. On the CUDA, OpenCL and HIP platforms the random numbers come from OpenMM's generator, which needs
64 bytes per lattice node; a step costs about 30% to 40% more on an NVIDIA A100.

With the fluctuating fluid **use the centred drag**
([`setDragScheme(LBMForce.Centered)`](#setdragschemescheme-getdragscheme)) and measure the temperature with
[`LBMTemperatureReporter`](#openmmlbmlbmtemperaturereporterfile-reportinterval-force): the coupled particles then
have the set temperature, their diffusion coefficient contains the hydrodynamic contribution of the thermal
flows, and the Einstein relation holds. With the explicit drag the particles are too hot, by about
friction x dt x m/(2 m_c), and a warning on stderr says so when the Context is created (measured: 14% for beads of 100 Da with friction 10/ps and dt = 0.02 ps, 56% for beads of
1000 Da with friction 10/ps and dt = 0.01 ps; [choosing the drag](lattice.md#choosing-the-drag)). Keep tau at
0.505 or above: closer to 1/2 the fluctuating fluid becomes unstable (at tau <= 0.501 with kT = 1/3000 in
lattice units, at tau = 0.5001 for water with dx = 0.5 nm and dt = 0.01 ps;
[theory.md](../theory.md#7-fluctuating-fluid-implemented-on-all-platforms), stability near tau = 1/2).

```python
force.setFluidFluctuations(True)
```

### `setFriction(friction)`, `getFriction()`

Friction coefficient gamma of the coupling, in 1/ps. The default is 1/ps. It must not be negative.
Typical values for coarse-grained beads are 1 to 10/ps; keep friction*dt well below 1.

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

### `getFluidFields(context)`

Returns the tuple `(density, velocity)` with the fluid at every lattice node, in node order:

- `density`: a list of densities in Da/nm^3, as a `Quantity`;
- `velocity`: a list of `Vec3` velocities in nm/ps, as a `Quantity`. It is the velocity of the forced
  fluid, u = j/rho + g dt/2 (see [velocity of the fluid](lattice.md#velocity-of-the-fluid)).

Solid nodes have zero density and velocity.

```python
density, velocity = force.getFluidFields(context)
rho = np.array(density.value_in_unit(unit.dalton/unit.nanometer**3))   # shape (numNodes,)
u = np.array(velocity.value_in_unit(unit.nanometer/unit.picosecond))   # shape (numNodes, 3)
```

### `getFluidState(context)`, `setFluidState(context, state)`

`getFluidState()` returns the complete state of the fluid as a list of 19 x numNodes numbers.
`setFluidState()` sets it in a Context with the same grid size; it accepts a list or a NumPy array.
Together they save and restore the fluid, which is not part of OpenMM checkpoints. To save and continue a
whole run use the [checkpoints](#checkpoints) instead: they also keep the random numbers. Unlike checkpoints,
the fluid state does not depend on the platform: it can move a fluid from one platform to another
([restart](restart.md#moving-a-run-to-another-platform)).

The state holds, for the 19 lattice populations of each node, their deviations from the rest
equilibrium, f_q - w_q, in lattice units, stored as [q*numNodes + node]. A population is the value plus
the weight w_q: 1/3 for q = 0, 1/18 for q = 1 to 6, 1/36 for q = 7 to 18 (see
[theory.md](../theory.md#4-storage-and-ordering-implemented)). At rest the state is zero. Treat it as
opaque unless you know the model. Saving and restoring it is exact. For large lattices the list is long: convert it at once to a NumPy array, for example
`np.array(force.getFluidState(context))`; with 64^3 nodes it takes 40 MB.

### `getFluidMachNumber(context)`

Returns the largest Mach number of the fluid, max |j/rho|/c_s over the fluid nodes, at the current
step.

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
type, velocity and density of each face, coupled particles, force group and name. The fluid of a Context is not
part of it. The XML has version 7. Older XML still loads: versions 1 to 3 (written by version 0.1.0) with the
explicit drag, versions 1 to 4 (versions 0.1 and 0.2 of the plugin) without fluid fluctuations, versions 1 to 5
with the `BounceBack` wall scheme and versions 1 to 6 with periodic faces. Older versions of the plugin cannot
read newer XML. Import `openmmlbm` before
deserializing. A force deserialized on its own is returned as a generic `openmm.Force`; obtain the
`LBMForce` with `LBMForce.cast()` (see the [serialization example](examples.md#serialization)).

## Errors

Invalid settings raise a Python `Exception` with the messages listed in
[troubleshooting](troubleshooting.md#error-messages), most of them when the Context is created (for
example open faces with the removal of the fluid momentum, or a single open face on an axis). Some
conditions only print a warning on stderr: a relaxation time outside [0.505, 2], and, with coupled
particles and the explicit drag, tau > 1.7, friction*dt > 1, and fluid fluctuations (the particles are then too
hot).
