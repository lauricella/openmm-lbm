# Getting started

This page checks the installation, runs a first simulation with a lattice Boltzmann fluid and explains
each step. The [lattice](lattice.md) page explains how to choose the parameters, and the
[API reference](api_reference.md) describes every method.

## Checking the installation

Build and install the plugin as described in the [installation guide](installation.md) (step by step)
or in the [README](../../README.md#building) (short version). The module
`openmmlbm` must then import in the same Python environment as OpenMM:

```python
import openmm as mm
from openmmlbm import LBMForce

system = mm.System()
system.addForce(LBMForce())
print('openmm-lbm is installed, OpenMM', mm.Platform.getOpenMMVersion())
print('platforms:', [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())])
```

If `addForce()` fails with a `TypeError`, the Python wrapper of the plugin was generated with a
different SWIG version from the OpenMM Python module (see [troubleshooting](troubleshooting.md)).

## A first simulation

The script below fills a periodic box with fluid, pushes the fluid with a uniform acceleration for
1 ps and reads back its density and velocity.

```python
import numpy as np
import openmm as mm
import openmm.unit as unit
from openmmlbm import LBMForce

# A periodic box of 4 x 4 x 4 nm with one particle (OpenMM needs at least one particle).
system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
system.addParticle(100.0)

# The fluid: 8 x 8 x 8 lattice nodes span the box, so the lattice spacing is 0.5 nm.
force = LBMForce()
force.setGridSize(8, 8, 8)
force.setFluidDensity(602.214*unit.dalton/unit.nanometer**3)          # water
force.setKinematicViscosity(1.0035*unit.nanometer**2/unit.picosecond)  # water at 20 C
force.setBodyAcceleration(mm.Vec3(0.05, 0, 0)*unit.nanometer/unit.picosecond**2)
force.setFluidMomentumRemovalFrequency(0)  # keep the momentum given by the acceleration
system.addForce(force)

# The step size of the integrator is the time step of the lattice.
integrator = mm.VerletIntegrator(0.01*unit.picosecond)
context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
context.setPositions([mm.Vec3(1, 1, 1)])

dx, dt, tau = force.getLatticeParametersInContext(context)
print('dx =', dx, ' dt =', dt, ' tau = %.4f' % tau)

integrator.step(100)                       # 100 lattice steps: 1 ps

density, velocity = force.getFluidFields(context)
rho = np.array(density.value_in_unit(unit.dalton/unit.nanometer**3))     # shape (512,)
u = np.array(velocity.value_in_unit(unit.nanometer/unit.picosecond))     # shape (512, 3)
print('mean density  %.3f Da/nm^3' % rho.mean())
print('mean velocity', np.round(u.mean(axis=0), 6) + 0.0, 'nm/ps')   # rounded: y and z are 0 to 1e-18
print('Mach number   %.5f' % force.getFluidMachNumber(context))
```

Output:

```
dx = 0.5 nm  dt = 0.01 ps  tau = 0.6204
mean density  602.214 Da/nm^3
mean velocity [0.05025 0.      0.     ] nm/ps
Mach number   0.00173
```

### What each part does

**The box defines the lattice.** The fluid fills the periodic box of the System. `setGridSize(nx, ny, nz)` sets the
number of lattice nodes along each box vector. The box must be rectangular and the lattice cells must be cubic: the
lattice spacing $`\Delta x = L_x/n_x`$ must equal $`L_y/n_y`$ and $`L_z/n_z`$. Here
$`\Delta x = 4\ \mathrm{nm}/8 = 0.5\ \mathrm{nm}`$.

**The fluid has a density and a kinematic viscosity.** The defaults are those of water:
602.214 Da/nm³ (1 g/cm³) and 1.0035 nm²/ps (1.0035e-6 m²/s, water at 20 C).

**The integrator sets the lattice time step.** `LBMForce` requires a `VerletIntegrator`; any other integrator is
rejected when the Context is created. The step size of the integrator is the lattice time step $`\Delta t`$, and it
cannot change after the Context has been created. Together, $`\Delta x`$, $`\Delta t`$ and the viscosity fix the
relaxation time $`\tau = 3\nu\Delta t/\Delta x^2 + \tfrac12`$ of the model, returned by
`getLatticeParametersInContext()`. The [lattice](lattice.md) page explains how to choose them.

**The fluid starts at rest, at equilibrium.** A new Context starts the fluid at uniform density and at
the velocity set by `setInitialFluidVelocity()`, zero by default.

**The fluid advances once per integration step.** `integrator.step(100)` advances the fluid by 100
lattice steps. Reading the fluid, computing forces or energies with `context.getState()`, or
minimizing the energy does not advance it.

**The body acceleration pushes the fluid.** A uniform acceleration $`\mathbf g`$ acts on every fluid node as the
force density $`\rho\mathbf g`$. The default removal of the fluid momentum (every step) would cancel its effect, so
this example disables it with `setFluidMomentumRemovalFrequency(0)`.

**Reading the fluid.** `getFluidFields(context)` returns the density and the velocity at every lattice node, as two
lists with units. The velocity is that of the forced fluid, $`\mathbf u = \mathbf j/\rho + \mathbf g\Delta t/2`$,
where $`\mathbf j`$ is the momentum density on the lattice. After 1 ps it is
$`g(t + \Delta t/2) = 0.05\times1.005 = 0.05025`$ nm/ps. The [lattice](lattice.md#node-indexing-and-numpy-arrays)
page shows how to arrange the lists as three-dimensional NumPy arrays.

**The Mach number measures the fluid speed against the lattice sound speed.** The model is accurate
only for small Mach numbers. The plugin checks the Mach number every 100 steps and stops the
simulation if it exceeds 0.3 (see [stability](lattice.md#mach-number-and-stability)).

## Units

All methods use OpenMM units: nm, ps, Da (g/mol), K and kJ/mol. A setter accepts either a plain number
in these units or a `Quantity` in any compatible unit, which is converted:

```python
import openmm as mm
import openmm.unit as unit
from openmmlbm import LBMForce

force = LBMForce()
force.setKinematicViscosity(1.0035e-6*unit.meter**2/unit.second)       # stored as 1.0035 nm^2/ps
force.setBodyAcceleration(mm.Vec3(1e15, 0, 0)*unit.meter/unit.second**2)  # stored as 1 nm/ps^2
print(force.getKinematicViscosity(), force.getBodyAcceleration())
```

Getters return `Quantity` objects, as the methods of OpenMM forces do.

A `Quantity` in a unit that cannot be converted, for example a friction in nm, raises `TypeError` with the
method and the two units, instead of being turned into a wrong number.

**Densities are per mole.** In OpenMM a mass is a molar mass: the dalton is 1 g/mol. A density in
g/cm³ must therefore be multiplied by the Avogadro constant; without it `setFluidDensity()` and
`setFaceDensity()` raise `TypeError` (OpenMM itself would read 1 g/cm³ as 1e-21 Da/nm³):

```python
import openmm.unit as unit
from openmmlbm import LBMForce

force = LBMForce()
force.setFluidDensity(1.0*unit.gram/unit.centimeter**3*unit.AVOGADRO_CONSTANT_NA)
print(force.getFluidDensity())    # 602.214... Da/nm^3
```

## Platforms

The fluid, the solid nodes with both wall schemes, the open faces and the coupling of the particles run on
every platform (see the [status table](README.md#what-works-in-this-version)). The Reference platform is for tests and small
systems; CUDA, OpenCL and HIP are for production. On those platforms the fluid is stored in the "mixed"
type of the platform:

- single precision with `Precision` = `single`;
- double precision with `mixed` and `double`.

In `mixed` and `double` precision the fluid agrees with the Reference platform to rounding (relative
differences of order 1e-15 after 1000 steps); in `single` precision the differences grow to about 1e-5.
The forces on the particles are handled by OpenMM in single precision unless the platform runs in `double`
precision. `mixed` is recommended for production.

## Next steps

- [Tutorial](tutorial.md): the examples of `examples/`, one by one, with exercises.
- [The lattice](lattice.md): a quick recipe for the parameters, then geometry, units, stability.
- [Saving and continuing a simulation](restart.md): checkpoints and long runs.
- [Examples](examples.md): a channel between two walls, monitoring, restarts, serialization,
  `openmm.app.Simulation`, flows between open faces.
- [API reference](api_reference.md): every method of `LBMForce`.
- [Glossary](glossary.md): the words used in this guide.
