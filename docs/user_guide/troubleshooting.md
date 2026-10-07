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
| `LBMForce does not support running on multiple devices` | Use a single GPU (`DeviceIndex` with one value). |
| `the integrator step size changed after the Context was created; reinitialize the Context` | The step size is the lattice time step and cannot change. Create a new Context, and transfer the fluid with `getFluidState()` and `setFluidState()`. |
| `the Mach number of the fluid is ... after ... lattice steps, above the limit ...` | The fluid is too fast for the model. Reduce the body acceleration or the forces on the fluid, or the time step; see [Mach number and stability](lattice.md#mach-number-and-stability). |
| `setFluidState() was called with a state of the wrong size` | The state comes from a different grid. It must have 19 nx ny nz values. |
| `the checkpoint was written on the platform X, not on Y` | A checkpoint can only be loaded on the platform where it was written. | Use the same platform, or move the run with `saveState()` and `getFluidState()` ([restart](restart.md#moving-a-run-to-another-platform)). |
| `the checkpoint was written with a different precision` | The precision (single, mixed, double) differs from that of the checkpoint. | Create the Context with the same `Precision` property. |
| `the checkpoint was written for a different grid size or number of coupled particles` | The System built for the restart is not the same as the one of the checkpoint. | Build the System exactly as in the first run. |
| `the data are not a checkpoint written by LBMForce::createCheckpoint()`, or `... is not a checkpoint written by openmmlbm.saveCheckpoint()` | The file or the bytes are not a checkpoint of openmm-lbm (for example an OpenMM checkpoint alone). | Save with `openmmlbm.saveCheckpoint()` and load with `openmmlbm.loadCheckpoint()`. |
| `with the Centered drag scheme LBMForce must be the last force of the System; add it after all the other forces` | The centred drag reads the other forces on the particles, which are complete only when `LBMForce` comes last. Call `system.addForce(force)` after adding every other force. |
| `the Centered drag scheme does not support virtual sites` | OpenMM moves the forces of virtual sites to their particles after the forces are computed, so the centred drag would miss them. Use the explicit drag. |
| `the checkpoint was written with a different drag scheme` | The drag scheme of the restarted run differs from that of the checkpoint. | Use the same `setDragScheme()` as in the first run. |
| `updateParametersInContext: the grid size cannot be changed` (and the similar messages for the density and viscosity, the coupled particles, the solid nodes and the drag scheme) | These parameters are fixed when the Context is created. Create a new Context. |
| `IntegrationUtilities::initRandomNumberGenerator(): Requested two different values for the random number seed` | On the CUDA, OpenCL and HIP platforms the random force uses OpenMM's generator, which has one seed per Context. Another component of the System (an `AndersenThermostat`, for example) uses it with a different seed: give both the same seed. |

The warning `tau = ... > 1.7: with the explicit drag at the nearest node the hydrodynamic self-mobility of
a coupled particle is small` is printed when particles are coupled and the relaxation time is large:
above tau = 1.79 the hydrodynamic mobility of a particle is negative. Reduce the viscosity or the time
step, or use a coarser lattice; see [relaxation time](lattice.md#relaxation-time).

The warning `friction*dt = ... > 1` is printed on stderr when the explicit drag overshoots: the velocity
of a particle relative to the fluid changes sign at every step, and grows without bound for
friction*dt >= 2. Reduce the friction or the time step, or use the centred drag
(`setDragScheme(LBMForce.Centered)`), which is stable for any friction.

The warning `the relaxation time tau = ... is outside the range [0.505, 2]` is printed on stderr and
does not stop the simulation. See [relaxation time](lattice.md#relaxation-time) for how to bring tau
into the range.

## Common pitfalls

**The fluid does not move under a body force.** The momentum of the fluid is removed at every step by
default. Call `setFluidMomentumRemovalFrequency(0)` for flows driven by `setBodyAcceleration()`.

**The density is wrong by a factor of 6e23.** OpenMM masses are molar masses (1 Da = 1 g/mol), so a
density in g/cm^3 must be multiplied by `unit.AVOGADRO_CONSTANT_NA`; see [units](getting_started.md#units).

**The fluid restarts from rest after a restart.** OpenMM checkpoints and `Context.reinitialize()` do
not keep the fluid. Save the run with `openmmlbm.saveCheckpoint()` or `openmmlbm.LBMCheckpointReporter` and
continue it with `openmmlbm.loadCheckpoint()` ([restart](restart.md)), or save only the fluid with
`getFluidState()` and restore it with `setFluidState()`; see
[saving and restoring the fluid](examples.md#saving-and-restoring-the-fluid).

**The OpenCL platform crashes on an NVIDIA GPU while the CUDA platform works.** If the CUDA
forward-compatibility libraries (`cuda-compat`, needed when the driver is older than the CUDA version of
OpenMM) are on `LD_LIBRARY_PATH`, the OpenCL driver can crash with a segmentation fault while it
compiles kernels, also without this plugin. Run OpenCL simulations without those libraries on the
library path.

**A run on a GPU differs from the same run on the Reference platform.** With T > 0 the random forces
come from different generators, so the trajectories differ while their statistics agree. At T = 0, or with
the NVE scheme, the platforms agree to rounding in double precision.

**The temperature in the log is a little below the set temperature.** The temperature that OpenMM
reports for coupled particles is the full-step one. It is below T because the fluid has no thermal
fluctuations of its own and takes part of the momentum of the particles: about 1-2% with friction*dt =
0.1, more with a large friction (13% for the disordered protein of `examples/cocomo/diffusion.py
--preset rlp`, friction 100/ps). See [validation.md](../validation.md). The velocities that OpenMM stores
are those of the half step, whose temperature is higher, T/(1 - friction*dt/2) for a free particle.

**Particles that are not coupled cross the walls.** Only coupled particles are reflected at solid
nodes.

**`System.addForce(LBMForce())` raises a `TypeError`.** The Python wrapper of the plugin was generated
with a different SWIG version from the OpenMM Python module, so OpenMM does not recognize the
`LBMForce` as a `Force`. Rebuild the plugin with the SWIG version that CMake names in its error
message; see the [README](../../README.md#building).

**Minimizing the energy or calling `getState()` does not advance the fluid.** This is intended: the
fluid advances once per integration step, and only `integrator.step()` (or `Simulation.step()`)
advances it.
