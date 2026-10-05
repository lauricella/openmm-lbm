# Troubleshooting

## Error messages

Errors are raised as a Python `Exception`. Most of them are raised when the Context is created, when
the plugin converts the parameters to lattice units.

| Message | Cause and fix |
|---|---|
| `the grid size must be set to positive values with setGridSize()` | Call `setGridSize(nx, ny, nz)` before creating the Context. |
| `the periodic box must be rectangular` | The default box vectors of the System are triclinic. The fluid needs a rectangular box. |
| `the lattice cells must be cubic, but the box and grid sizes give spacings ...` | Lx/nx, Ly/ny and Lz/nz differ. Choose the grid so that the three spacings are equal, or adjust the box. |
| `the integrator step size must be positive` | Set a positive step size on the integrator. |
| `the fluid density must be positive`, `the kinematic viscosity must be positive` | Check `setFluidDensity()` and `setKinematicViscosity()`. |
| `the friction must not be negative`, `the temperature must not be negative` | Check `setFriction()` and `setTemperature()`. |
| `the fluid momentum removal frequency must not be negative`, `the Mach number check frequency must not be negative` | Use 0 to disable, or a positive number of steps. |
| `the Mach number limit must be positive` | Check `setMachNumberLimit()`. |
| `a solid node index is out of range` | Solid node indices must be between 0 and nx ny nz - 1; see [node indexing](lattice.md#node-indexing-and-numpy-arrays). |
| `a solid node is listed more than once` | Remove the duplicates, for example with `np.unique()`. |
| `all lattice nodes are solid` | At least one node must be fluid. |
| `a coupled particle index is out of range`, `a particle is coupled more than once`, `coupled particles must have a positive mass` | Check the indices passed to `addParticle()` and the masses in the System. Massless particles (virtual sites) cannot be coupled. |
| `LBMForce requires a VerletIntegrator: drag and random forces are part of the force` | Use `VerletIntegrator`. Langevin and other thermostatted integrators would add a second friction. |
| `solid nodes are supported only on the Reference platform in this version` | Run walls on the Reference platform, or remove the solid nodes. |
| `LBMForce does not support running on multiple devices` | Use a single GPU (`DeviceIndex` with one value). |
| `the integrator step size changed after the Context was created; reinitialize the Context` | The step size is the lattice time step and cannot change. Create a new Context, and transfer the fluid with `getFluidState()` and `setFluidState()`. |
| `the Mach number of the fluid is ... after ... lattice steps, above the limit ...` | The fluid is too fast for the model. Reduce the body acceleration or the forces on the fluid, or the time step; see [Mach number and stability](lattice.md#mach-number-and-stability). |
| `setFluidState() was called with a state of the wrong size` | The state comes from a different grid. It must have 19 nx ny nz values. |
| `updateParametersInContext: the grid size cannot be changed` (and the similar messages for the density and viscosity, the coupled particles and the solid nodes) | These parameters are fixed when the Context is created. Create a new Context. |

The warning `the relaxation time tau = ... is outside the range [0.505, 2]` is printed on stderr and
does not stop the simulation. See [relaxation time](lattice.md#relaxation-time) for how to bring tau
into the range.

## Common pitfalls

**The fluid does not move under a body force.** The momentum of the fluid is removed at every step by
default. Call `setFluidMomentumRemovalFrequency(0)` for flows driven by `setBodyAcceleration()`.

**The density is wrong by a factor of 6e23.** OpenMM masses are molar masses (1 Da = 1 g/mol), so a
density in g/cm^3 must be multiplied by `unit.AVOGADRO_CONSTANT_NA`; see [units](getting_started.md#units).

**The fluid restarts from rest after a restart.** OpenMM checkpoints and `Context.reinitialize()` do
not keep the fluid. Save it with `getFluidState()` and restore it with `setFluidState()`; see
[saving and restoring the fluid](examples.md#saving-and-restoring-the-fluid).

**The fluid does not change on the CUDA, OpenCL or HIP platform.** The fluid update runs only on the
Reference platform in this version; see the [status table](README.md#what-works-in-this-version).

**The particles do not feel the fluid.** The particle-fluid coupling is not implemented yet:
`LBMForce` applies no force to the particles.

**`System.addForce(LBMForce())` raises a `TypeError`.** The Python wrapper of the plugin was generated
with a different SWIG version from the OpenMM Python module, so OpenMM does not recognize the
`LBMForce` as a `Force`. Rebuild the plugin with the SWIG version that CMake names in its error
message; see the [README](../../README.md#building).

**Minimizing the energy or calling `getState()` does not advance the fluid.** This is intended: the
fluid advances once per integration step, and only `integrator.step()` (or `Simulation.step()`)
advances it.
