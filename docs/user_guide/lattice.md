# The lattice

This page explains how the lattice is built from the System and the integrator, how to read the fluid
fields node by node, how to choose the parameters, and how the plugin keeps the fluid in the regime
where the model is accurate. The model itself is described in [theory.md](../theory.md).

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
[theory.md](../theory.md#solid-nodes-implemented-on-the-reference-platform)).

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

**Coupled particles need tau below about 1.7.** With the explicit drag at the nearest node, the
hydrodynamic part of the mobility of a particle decreases as tau grows. At tau = 1.7 it is about 15% of
its value at tau = 1.1, and above tau = 1.79 it is negative: particles then move less than a Langevin
particle with the same friction. A warning is printed when a Context with coupled particles has
tau > 1.7 ([theory.md](../theory.md#2-particle-fluid-coupling-implemented-on-the-reference-platform)).

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
restarts it from this initial state. To keep it, save it with `getFluidState()` and restore it with
`setFluidState()` (see the [restart example](examples.md#saving-and-restoring-the-fluid)).
