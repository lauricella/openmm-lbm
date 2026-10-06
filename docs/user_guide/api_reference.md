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
- [Coupled particles](#coupled-particles)
- [Reading and writing the fluid of a Context](#reading-and-writing-the-fluid-of-a-context)
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
| `addParticle(particle)`, `setParticle(index, particle)` | particle indices | none | no |
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
range or a NumPy array of integers. The fluid does not occupy them. A population of the fluid that
streams into a solid node is sent back to the node it came from (halfway bounce-back). The wall
therefore lies halfway between a solid node and its fluid neighbours, and has no-slip conditions.

- The default is an empty list: the whole lattice is fluid.
- Indices must be in range and must not repeat, and at least one node must be fluid. The order does not
  matter.
- Solid nodes hold no fluid: `getFluidFields()` returns zero density and zero velocity there. They do
  not enter the removal of the fluid momentum or the Mach number.
- Solid nodes are supported on the Reference platform only in this version. On the other platforms,
  creating a Context raises an error.

`getSolidNodes()` returns the list of indices.

### `getWallForce(context)`

Returns the force exerted on the solid nodes during the last lattice step, as a `Vec3` in kJ/mol/nm.
It is the momentum given to the walls, divided by the time step, by:
- the fluid, through bounce-back (momentum exchange method of Ladd,
  [theory.md](../theory.md#solid-nodes-implemented-on-the-reference-platform));
- the coupled particles: the reaction of particles at solid nodes and their reflections.

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

## Coupled particles

Each coupled particle feels a friction force -gamma m (v - u) relative to the fluid velocity u at the
nearest lattice node, plus a random force at the given temperature, and the fluid at that node receives
the opposite force (explicit Euler-Maruyama scheme,
[theory.md](../theory.md#2-particle-fluid-coupling-implemented-on-the-reference-platform)). This is
implemented on the Reference platform; on the other platforms no force acts on the particles yet.

- **Once per step.** The coupling forces are computed once per integration step. Every other force
  evaluation, for example `getState(getForces=True)`, returns the forces of the last step, without new
  random numbers. Before the first step they are zero. The energy of the coupling is zero.
- **Walls.** A coupled particle whose nearest node is solid and that moves into the wall has every
  component of its velocity reversed at the start of the step, as for a no-slip wall. A particle that
  already moves out of the wall keeps its velocity. Uncoupled particles do not see the walls.
- **Stability.** In one step the drag multiplies the velocity of a particle relative to the fluid by
  1 - friction*dt. A warning is printed when friction*dt > 1, and the motion is unstable for
  friction*dt >= 2.
- **Temperature.** The temperature that OpenMM reports for the System (`StateDataReporter`, the kinetic
  energy of a State) is not valid for coupled particles. Use full-step velocities, as in the
  [temperature example](examples.md#temperature-of-coupled-particles).

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
the viscosity of the fluid ([theory.md](../theory.md#2-particle-fluid-coupling-implemented-on-the-reference-platform)).
The scheme is saved with the force by `XmlSerializer`. It can be changed in a Context with
`updateParametersInContext()`.

```python
force.setCouplingScheme(LBMForce.NVE)
```

### `setFriction(friction)`, `getFriction()`

Friction coefficient gamma of the coupling, in 1/ps. The default is 1/ps. It must not be negative.
Typical values for coarse-grained beads are 1 to 10/ps; keep friction*dt well below 1.

### `setTemperature(temperature)`, `getTemperature()`

Temperature of the random force on the coupled particles, in K. The default is 300 K. It must not be
negative.

### `setRandomNumberSeed(seed)`, `getRandomNumberSeed()`

Seed of the random force. With 0, the default, a different seed is chosen for every Context. Two
simulations with different seeds have different random forces. On the Reference platform the random
force has its own generator, so the same seed reproduces a simulation; as for other OpenMM forces, this
is not guaranteed on the other platforms. The state of the generator is not part of OpenMM checkpoints.

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
Together they save and restore the fluid, which is not part of OpenMM checkpoints (see the
[restart example](examples.md#saving-and-restoring-the-fluid)).

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
- the friction, the temperature and the coupling scheme;
- the frequency of the removal of the fluid momentum;
- the frequency and the limit of the Mach number check.

The grid size, the fluid density and viscosity, the solid nodes and the set of coupled particles
cannot be changed this way: the method raises an error if they differ from those of the Context. The
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
parameters: grid, fluid properties, body acceleration, initial velocity, frequencies, Mach limit, solid
nodes and coupled particles. The fluid of a Context is not part of it. Import `openmmlbm` before
deserializing. A force deserialized on its own is returned as a generic `openmm.Force`; obtain the
`LBMForce` with `LBMForce.cast()` (see the [serialization example](examples.md#serialization)).

## Errors

Invalid settings raise a Python `Exception` with the messages listed in
[troubleshooting](troubleshooting.md#error-messages), most of them when the Context is created. A
relaxation time outside [0.505, 2] only prints a warning on stderr.
