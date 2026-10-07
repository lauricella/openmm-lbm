# The lattice

This page explains how the lattice is built from the System and the integrator, how to read the fluid
fields node by node, how to choose the parameters, and how the plugin keeps the fluid in the regime
where the model is accurate. The model itself is described in [theory.md](../theory.md).

## Quick recipe

For coarse-grained proteins in water, with one bead per residue, these choices work and are those of the
examples. The rest of this page explains each of them.

| Parameter | Typical choice | Why, and the limits |
|---|---|---|
| Lattice spacing dx | 0.5 nm | about the size of a residue bead. Each bead is coupled to its nearest node |
| Time step dt | 0.01 ps (10 fs) | the time step of the coarse-grained model; it is also the lattice time step |
| Box | a whole number of dx along each side; at least 3 times the size of the protein and 2 times the cutoff of the nonbonded forces | the box is periodic: a protein feels its images through the fluid |
| Kinematic viscosity | 1.0035 nm^2/ps (water, tau = 0.62 with the values above), or larger | tau = 3 nu dt/dx^2 + 1/2 must stay between about 0.505 and 1.7 ([relaxation time](#relaxation-time)) |
| Friction | 5 to 10 /ps, so that friction x dt = 0.05 to 0.1 | with the explicit drag (the default) friction x dt must be below 1 (warning) and below 2 (stability), at 0.1 or below it is accurate; for larger values use the centred drag ([choosing the drag](#choosing-the-drag)) |
| Temperature | that of the simulation, e.g. 298 K | the fluid is the thermostat of the coupled particles: no other thermostat |
| Integrator | `VerletIntegrator(dt)` | friction and random force are part of `LBMForce` |
| Removal of the fluid momentum | every step (the default) | keeps the system at rest; set 0 for flows driven by a body force |

**Checking a new setup.** Create the Context, then:

1. print `force.getLatticeParametersInContext(context)`: tau must be in the range above (the plugin also
   prints a warning when it is not);
2. run a few hundred steps and print `force.getFluidMachNumber(context)`: it should stay below 0.1
   ([Mach number](#mach-number-and-stability));
3. check the temperature in the log of `StateDataReporter`: a few percent below the set temperature is
   normal ([limitations](README.md#limitations-of-the-model)), much more is not.

## Geometry

The fluid fills the periodic box of the System, which must be rectangular. `setGridSize(nx, ny, nz)`
divides it into cubic cells:

- the lattice spacing is dx = Lx/nx, and Ly/ny and Lz/nz must be equal to it (to a relative 1e-6);
- node (i, j, k) sits at (i dx, j dx, k dx), with 0 <= i < nx, 0 <= j < ny and 0 <= k < nz;
- the lattice is periodic in all three directions, like the box.

The box is read from the default periodic box vectors of the System when the Context is created.
For a box of 15 x 15 x 7.5 nm and dx = 0.5 nm:

```python
import openmm as mm
from openmmlbm import LBMForce

system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(15, 0, 0), mm.Vec3(0, 15, 0), mm.Vec3(0, 0, 7.5))
force = LBMForce()
force.setGridSize(30, 30, 15)
```

## Node indexing and NumPy arrays

Node (i, j, k) has index i + nx*(j + ny*k): i varies fastest. All per-node lists use this order:

- the fields returned by `getFluidFields()`;
- the solid nodes passed to `setSolidNodes()`;
- the state returned by `getFluidState()`, which holds 19 values per node.

Reshaped to (nz, ny, nx), a list becomes an array indexed as [k, j, i]:

```python
import numpy as np
import openmm as mm
import openmm.unit as unit
from openmmlbm import LBMForce

nx, ny, nz = 8, 6, 4
system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 2))
system.addParticle(1.0)
force = LBMForce()
force.setGridSize(nx, ny, nz)
force.setInitialFluidVelocity(mm.Vec3(0.1, 0, 0))
system.addForce(force)
context = mm.Context(system, mm.VerletIntegrator(0.01), mm.Platform.getPlatformByName('Reference'))
context.setPositions([mm.Vec3(0, 0, 0)])


def fluid_arrays(force, context):
    """Density (nz, ny, nx) in Da/nm^3 and velocity (nz, ny, nx, 3) in nm/ps, indexed [k, j, i]."""
    nx, ny, nz = force.getGridSize()
    density, velocity = force.getFluidFields(context)
    rho = np.array(density.value_in_unit(unit.dalton/unit.nanometer**3)).reshape(nz, ny, nx)
    u = np.array(velocity.value_in_unit(unit.nanometer/unit.picosecond)).reshape(nz, ny, nx, 3)
    return rho, u


rho, u = fluid_arrays(force, context)
print(rho.shape, u.shape, u[2, 3, 4])           # velocity of node (i, j, k) = (4, 3, 2)

# Index and position of every node, as arrays indexed [k, j, i].
k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
index = i + nx*(j + ny*k)
dx = 0.5
x, y, z = i*dx, j*dx, k*dx
print(index[2, 3, 4], (x[2, 3, 4], y[2, 3, 4], z[2, 3, 4]))
```

Boolean masks on these arrays select sets of nodes. For example, `index[y < 1.0]` lists the nodes
with y < 1 nm, which can be passed to `setSolidNodes()` (see the [channel example](examples.md#channel-flow-between-two-walls)).

## Time step

The lattice time step dt is the step size of the `VerletIntegrator`, read when the Context is created.
The fluid advances by one lattice step at each integration step. The step size must not change
afterwards: if it does, the next step raises an exception. To use a different step size, create a new
Context.

## Units on the lattice

Internally the plugin works in lattice units. The conversions follow from dx, dt and the fluid density
rho0 (`setFluidDensity()`). The most useful ones are listed below; [theory.md](../theory.md#3-units-and-conversions-implemented)
has the complete table.

| Quantity | Lattice unit | With dx = 0.5 nm, dt = 0.01 ps, water |
|---|---|---|
| velocity | dx/dt | 50 nm/ps |
| lattice sound speed | c_s = dx/(dt sqrt(3)) | 28.9 nm/ps |
| acceleration | dx/dt^2 | 5000 nm/ps^2 |
| mass of a cell | m_c = rho0 dx^3 | 75.3 Da |
| kinematic viscosity | dx^2/dt | 25 nm^2/ps |

## Relaxation time

The kinematic viscosity nu enters the model through the relaxation time

  tau = 3 nu dt/dx^2 + 1/2,

which must be larger than 1/2. The model is accurate for tau between about 0.505 and 2; outside this
range the plugin prints a warning on stderr when the Context is created. Near 1/2 the fluid is close
to the stability limit, and at large tau the error on the position of walls grows (see
[theory.md](../theory.md#solid-nodes-implemented-on-all-platforms)).

Since dt is the time step of the molecular dynamics, the viscosity and the resolution are coupled:
tau - 1/2 = 3 nu dt/dx^2. With dt = 0.01 ps:

| dx | water, nu = 1.0035 nm^2/ps | 5 x water |
|---|---|---|
| 0.5 nm | tau = 0.62 | tau = 1.10 |
| 0.25 nm | tau = 0.98 | tau = 2.91 |

To check a choice before creating a Context:

```python
def relaxation_time(viscosity, dx, dt):
    """tau for a kinematic viscosity in nm^2/ps, a spacing in nm and a time step in ps."""
    return 3*viscosity*dt/dx**2 + 0.5


for dx in (1.0, 0.5, 0.25):
    for dt in (0.002, 0.01, 0.02):
        print('dx = %.2f nm  dt = %.3f ps  tau = %.3f' % (dx, dt, relaxation_time(1.0035, dx, dt)))
```

If tau is too close to 1/2, increase dt or the viscosity, or decrease dx. If tau is too large, do the
opposite. Coarse-grained models often use a viscosity larger than that of water, which also moves tau
up. Inside a Context, `getLatticeParametersInContext(context)` returns dx, dt and tau.

**With the explicit drag, coupled particles need tau below about 1.7.** With the explicit drag at the
nearest node, the hydrodynamic part of the mobility of a particle decreases as tau grows. At tau = 1.7 it
is about 15% of its value at tau = 1.1, and above tau = 1.79 it is negative: particles then move less than
a Langevin particle with the same friction. A warning is printed when a Context with coupled particles and
the explicit drag has tau > 1.7. With the centred drag this part stays positive at every tau, but it grows
with tau ([theory.md](../theory.md#2-particle-fluid-coupling-implemented-on-all-platforms), self-mobility).

## Choosing the drag

`setDragScheme()` chooses between two time discretizations of the drag, with the same friction and the
same random force ([API](api_reference.md#setdragschemescheme-getdragscheme)). The choice is made when the
Context is created.

| | `LBMForce.Explicit` (default) | `LBMForce.Centered` |
|---|---|---|
| Velocities compared | particle half a step before the force, fluid before the force | particle and fluid at the time of the force |
| Stability | friction x dt < 2 for one particle; with M the mass of the particles at a node and m_c that of the fluid in a cell, friction x dt (1 + M/m_c)/2 < 1 | any friction |
| Velocity with the right temperature | full step: the temperature of `StateDataReporter` | half step: the velocities of the State; `StateDataReporter` reports less |
| Requirements | none | `LBMForce` is the last force of the System; no virtual sites |
| Cost on a GPU | | 0 to 6% more |
| Same results as | version 0.1.0 and the DragOpenMM library | |

**Use the explicit drag** (the default) in most cases: it is accurate for friction x dt up to about 0.1, it
reproduces the previous results, the temperature that OpenMM reports is right, and with the fluid of this
version (without thermal fluctuations) its particles are the closest to T (next paragraph).

**Use the centred drag** when the explicit drag is unstable: large friction, or several heavy beads at a
node. With beads of 130 Da, dx = 0.5 nm and dt = 0.01 ps, three or four beads at a node make the explicit
drag unstable already for friction above 26 to 32/ps. The centred drag is stable for any friction.

With the centred drag measure the temperature with `openmmlbm.LBMTemperatureReporter`, which uses the
velocity that has the right temperature for each drag
([API](api_reference.md#openmmlbmlbmtemperaturereporterfile-reportinterval-force)):

```python
import openmmlbm
# simulation.reporters.append(openmmlbm.LBMTemperatureReporter('temperature.txt', 1000, force))
```

**Both drags give particles colder than T** at the density of water, because the fluid has no thermal
fluctuations of its own ([limitations](README.md#limitations-of-the-model)). The centred drag couples a
particle to its own cell within the step, and is colder, the more so the larger friction x dt and the
mass of the bead in units of the mass of fluid in a cell, m/m_c (m_c = 75 Da with dx = 0.5 nm), while the
diffusion coefficient is the same with both drags. Measured at 298 to 300 K, each drag with its right
velocity ([validation](../validation.md)):

| System | friction x dt | m/m_c | explicit | centred |
|---|---|---|---|---|
| free beads of 100 Da | 0.1 | 1.3 | 2 to 5% below T | 8 to 10% below T |
| SOD1, COCOMO2 | 0.1 | 1.3 (mean) | 1.3% below T | 7% below T |
| SOD1, COCOMO2 | 0.3 | 1.3 (mean) | 1% below T | 16% below T |
| free beads of 1000 Da | 0.1 | 13 | 15% below T | 46% below T |

With a fluid much heavier than the particles both give T
([theory.md](../theory.md#2-particle-fluid-coupling-implemented-on-all-platforms), temperature with a fluid
without fluctuations). A fluid with thermal fluctuations (not implemented yet) would give T with both.

## Velocity of the fluid

The fluid responds to the body force F = rho g with the forced velocity

  u = j/rho + g dt/2,

where j is the momentum density on the lattice. `getFluidFields()` returns this velocity, which is the
physical velocity of the fluid: it is the one that matches the analytical flow profiles. Without a
body force it equals j/rho. The Mach number uses j/rho.

## Mach number and stability

The Mach number Ma = |u|/c_s compares the speed of the fluid with the lattice sound speed. The model
is a weakly compressible fluid: density fluctuations grow as Ma^2 and the viscous stress has an error
of order Ma^3. For accurate results Ma should stay below about 0.1. In the target applications it is
between 1e-3 and 1e-2: a 100 Da bead at 298 K with dx = 0.5 nm and dt = 0.01 ps gives Ma = 5e-3.

The plugin checks the largest Mach number of the fluid every 100 lattice steps. If it exceeds 0.3,
the step raises an exception that reports the step and the value. Both numbers can be changed:

- `setMachCheckFrequency(n)` checks every n steps, and `setMachCheckFrequency(0)` disables the check;
- `setMachNumberLimit(limit)` changes the limit.

`getFluidMachNumber(context)` returns the current value at any time, for monitoring. A
[monitoring example](examples.md#monitoring-a-run-and-the-mach-number-check) shows the check at work.

## Removal of the fluid momentum

By default the plugin removes the momentum of the fluid at every step. It subtracts the
centre-of-mass velocity of the fluid, u_cm = sum(j)/sum(rho), from every fluid node, before the
collision of every lattice step whose number is a multiple of the frequency. Steps are numbered by
the step count of the Context (`context.getStepCount()`): a new Context starts at 0, so the first
removal happens before the first step, and a checkpoint restores the count, so a restarted run removes
the momentum at the same steps as an uninterrupted one. The Mach number check counts steps in the same
way.

- `setFluidMomentumRemovalFrequency(n)` removes it every n steps.
- `setFluidMomentumRemovalFrequency(0)` never removes it. Use 0 for flows driven by a body force,
  otherwise the removal cancels the momentum that the force gives to the fluid.

## Initial state

A new Context starts the fluid at equilibrium:

- the density of every fluid node is the density set by `setFluidDensity()`;
- the velocity of every fluid node is the velocity set by `setInitialFluidVelocity()` (zero by
  default);
- solid nodes hold no fluid.

The fluid is not part of the State or of the checkpoints of OpenMM. `Context.reinitialize()` also
restarts it from this initial state. To save and continue a run, fluid included, use
`openmmlbm.saveCheckpoint()` and `openmmlbm.loadCheckpoint()` ([restart](restart.md)); to keep only the
fluid, `getFluidState()` and `setFluidState()` ([example](examples.md#saving-and-restoring-the-fluid)).
