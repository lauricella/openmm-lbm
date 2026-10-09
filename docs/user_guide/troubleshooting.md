# Troubleshooting

## Error messages

Errors are raised as a Python `Exception`. Most of them are raised when the Context is created, when
the plugin converts the parameters to lattice units.

| Message | Cause and fix |
|---|---|
| `the grid size must be set to positive values with setGridSize()` | Call `setGridSize(nx, ny, nz)` before creating the Context. |
| `the lattice has ... nodes, more than the 2147483647 that the plugin can index` | The nodes are indexed with 32-bit integers; use a coarser lattice ([`setGridSize()`](api_reference.md#setgridsizenx-ny-nz-getgridsize)). |
| `a domain of the lattice holds up to ... nodes (with its halo), more than the 113025455 whose 19 populations the plugin can index` | Divide the lattice into more domains (`setDomainDecomposition()`, with MPI); with one domain the limit is about $`480^3`$ nodes. |
| `the periodic box must be rectangular` | The default box vectors of the System are triclinic. The fluid needs a rectangular box. |
| `the lattice cells must be cubic, but the box and grid sizes give spacings ...` | $`L_x/n_x`$, $`L_y/n_y`$ and $`L_z/n_z`$ differ. Choose the grid so that the three spacings are equal, or adjust the box. |
| `the integrator step size must be positive` | Set a positive step size on the integrator. |
| `the fluid density must be positive`, `the kinematic viscosity must be positive` | Check `setFluidDensity()` and `setKinematicViscosity()`. |
| `the friction must not be negative`, `the temperature must not be negative` | Check `setFriction()` and `setTemperature()`. |
| `the fluid momentum removal frequency must not be negative`, `the Mach number check frequency must not be negative` | Use 0 to disable, or a positive number of steps. |
| `the Mach number limit must be positive` | Check `setMachNumberLimit()`. |
| `a solid node index is out of range` | Solid node indices must be between 0 and $`n_xn_yn_z - 1`$; see [node indexing](lattice.md#node-indexing-and-numpy-arrays). |
| `a solid node is listed more than once` | Remove the duplicates, for example with `np.unique()`. |
| `all lattice nodes are solid` | At least one node must be fluid. |
| `expected a sequence of integers` (a `TypeError` from `setSolidNodes()`) | Pass the node indices as a list, a range or a NumPy array of integers. |
| `a coupled particle index is out of range`, `a particle is coupled more than once`, `coupled particles must have a positive mass` | Check the indices passed to `addParticle()` and the masses in the System. Massless particles (virtual sites) cannot be coupled. |
| `LBMForce requires a VerletIntegrator: drag and random forces are part of the force` | Use `VerletIntegrator`. Langevin and other thermostatted integrators would add a second friction. |
| `LBMForce does not support running on multiple devices` | Use a single GPU per Context (`DeviceIndex` with one value). To use several GPUs, divide the lattice into domains, one MPI rank per GPU (`setDomainDecomposition()`, [running on several GPUs](parallel.md)). |
| `the domain decomposition (setDomainDecomposition()) cannot be negative` | Use positive numbers of domains, or 0 to let MPI choose. |
| `setDomainDecomposition() asks for more than one domain, but the plugin was built without MPI` | Build the plugin with `-DOPENMM_LBM_MPI=ON` ([installation](installation.md)), or use one domain. A 0 also needs MPI. |
| `the domain decomposition X x Y x Z (setDomainDecomposition()) does not match the N MPI ranks` | The product of the domains must be the number of processes of `mpirun` or `srun`; check `-n` and the decomposition, or use zeros. |
| `the domain decomposition has P domains along axis A, more than the N nodes of the lattice` | Each domain needs at least one node along every axis: use fewer domains along that axis. |
| `with the domain decomposition every MPI rank must use the same platform and precision: rank R has ...` | Every rank must create its Context on the same platform with the same `Precision`. |
| `with the domain decomposition and coupled particles the platform must compute the forces deterministically ...` | On the CUDA and HIP platforms set the property `DeterministicForces` to `"true"`, so that every rank computes the same forces on its copies of the particles. |
| `with the domain decomposition (setDomainDecomposition()) the System cannot contain an AndersenThermostat or a Monte Carlo barostat` | Their random numbers would differ between the ranks, which hold copies of the same particles. Remove them: the temperature comes from the coupling to the fluid. |
| `the positions or velocities of the particles differ between the MPI ranks at lattice step N` | Every rank must set the same positions and velocities: read them from the same file, and draw velocities with a fixed seed (`setVelocitiesToTemperature(T, seed)`). The check can be switched off with `setParticleCopiesCheck(False)` only when the copies are known to be identical. |
| `getFluidFields() cannot gather the fields and add the halo in the same call` | Use `gather=True` or `halo=True`, not both. |
| `updateParametersInContext: the domain decomposition cannot be changed`, `... the exchange of the halo cannot be switched on or off` | Create a new Context for another decomposition or another exchange of the halo. |
| `the integrator step size changed after the Context was created; reinitialize the Context` | The step size is the lattice time step and cannot change. Create a new Context, and transfer the fluid with `getFluidState()` and `setFluidState()`. |
| `the Mach number of the fluid is ... after ... lattice steps, above the limit ...` | The fluid is too fast for the model. Reduce the body acceleration, the forces on the fluid or the velocities of the open faces, or the time step; see [Mach number and stability](lattice.md#mach-number-and-stability). With fluid fluctuations and $`\tau`$ very close to 1/2 (below about 0.502) the fluid can become unstable by itself: keep $`\tau`$ at 0.505 or above ([`setFluidFluctuations()`](api_reference.md#setfluidfluctuationsfluctuations-getfluidfluctuations)). With fluid fluctuations and two `Density` faces at the same density, without walls along the flow, the mean flow across the faces has nothing that stops it and grows until this error: add walls or use a `Velocity` face ([open faces](api_reference.md#open-faces)). A `Density` face through which the fluid enters can become unstable at small $`\tau`$ (with a difference of density of 1 %, at $`\tau \le 0.55`$): use a `Velocity` inlet. |
| `setFluidState() was called with a state of the wrong size` | The state comes from a different grid or domain. It must have 19 values per node: of the whole lattice, $`19\,n_xn_yn_z`$, with one domain or with `scatter=True` (on rank 0); of the domain of the rank otherwise ([`getLocalDomain()`](api_reference.md#getlocaldomaincontext)). |
| `the checkpoint was written on the platform X, not on Y`, or OpenMM's `loadCheckpoint: Checkpoint was created with a different Platform: ...` | A checkpoint can only be loaded on the platform where it was written. Use the same platform, or move the run with `saveState()` and `getFluidState()` ([restart](restart.md#moving-a-run-to-another-platform)). |
| `the checkpoint was written with a different precision`, or OpenMM's `Checkpoint was created with a different numeric precision` | The precision (single, mixed, double) differs from that of the checkpoint. Create the Context with the same `Precision` property. |
| `Checkpoint was created with a different version of OpenMM` (from OpenMM) | The OpenMM part of the checkpoint was written by another version of OpenMM. Continue the run with the version of OpenMM that wrote it. |
| `the checkpoint was written for a different grid size or number of coupled particles` | The System built for the restart is not the same as the one of the checkpoint. Build the System exactly as in the first run. |
| `the data are not a checkpoint written by LBMForce::createCheckpoint()`, or `... is not a checkpoint written by LBMForce.saveCheckpointFile() or openmmlbm.saveCheckpoint()` | The file or the bytes are not a checkpoint of openmm-lbm (for example an OpenMM checkpoint alone). Save with `openmmlbm.saveCheckpoint()` and load with `openmmlbm.loadCheckpoint()`. |
| `... was written in X precision, and this Context uses Y`, `... was written for a System with a different number of particles` | A checkpoint file is loaded into a Context with another precision, or a System with another number of particles. Create the Context with the `Precision` of the first run and build the same System. |
| `... was written with one domain by openmmlbm.saveCheckpoint(), and only a Context with one domain can load it` | A file written with one domain can be loaded only with one domain. To continue a run with another decomposition, write its checkpoints with `saveCheckpointFile()` ([restart](restart.md)). |
| `with the domain decomposition the checkpoints are written to a file, with saveCheckpointFile()` (or `read from a file, with loadCheckpointFile()`) | `createCheckpoint()` and `loadCheckpoint()` hold the state of one process; with more than one domain use `saveCheckpointFile()` and `loadCheckpointFile()`, or `openmmlbm.saveCheckpoint()` and `openmmlbm.loadCheckpoint()`. |
| `cannot open ... to write it`, `cannot open ... to read it`, `cannot rename ... to ...`, `error writing ...`, `... is truncated or cannot be read` | A checkpoint or VTK file could not be written or read: check the path, the permissions and the free space (the checkpoint is written to `<file>.tmp` and then renamed). |
| `unsupported checkpoint version`, `the checkpoint is damaged`, `the checkpoint is truncated`, `... is truncated or damaged` | The checkpoint was written by a newer version of the plugin, or the file was cut or changed (for example by a job that stopped while writing it). Use the version that wrote it, or an earlier checkpoint. |
| `error writing the checkpoint` | In C++, the stream passed to `createCheckpoint()` could not be written. Check that it is open, in binary mode, and that the disk is not full. |
| `the checkpoint must be bytes` | `LBMForce.loadCheckpoint(context, data)` takes the bytes returned by `createCheckpoint()`; to read a file, use `openmmlbm.loadCheckpoint(file, context, force)`. |
| `Unsupported version number` (from `XmlSerializer.deserialize()`) | The XML was written by a newer version of the plugin: for example, version 0.3 cannot read the XML of this version (XML version 8). Use the newer version. |
| `the serialized force must have 6 faces` (from `XmlSerializer.deserialize()`) | The XML of the force was changed by hand or is damaged: its `Faces` element must hold one `Face` for each of the six faces. Serialize the force again. |
| `unknown coupling scheme`, `unknown drag scheme` | `setCouplingScheme()` or `setDragScheme()` received a number that is not a scheme. Use `LBMForce.EulerMaruyama` or `LBMForce.NVE`, and `LBMForce.Explicit` or `LBMForce.Centered`. |
| `with the Centered drag scheme LBMForce must be the last force of the System; add it after all the other forces` | The centred drag reads the other forces on the particles, which are complete only when `LBMForce` comes last. Call `system.addForce(force)` after adding every other force. |
| `the Centered drag scheme does not support virtual sites` | OpenMM moves the forces of virtual sites to their particles after the forces are computed, so the centred drag would miss them. Use the explicit drag. |
| `the checkpoint was written with a different drag scheme` | The drag scheme of the restarted run differs from that of the checkpoint (a checkpoint of version 0.1.0 has the explicit drag). Use the same `setDragScheme()` as in the first run. |
| `unknown wall scheme`, `unknown boundary type of a face`, `unknown face of the box` | `setWallScheme()`, `setFaceBoundary()` or a face method received a number that is not a scheme, a type or a face. Use `LBMForce.BounceBack` or `LBMForce.Regularized`; `LBMForce.Periodic`, `LBMForce.Velocity` or `LBMForce.Density`; `LBMForce.XMin` ... `LBMForce.ZMax`. |
| `the two faces perpendicular to x must be both periodic or both open (Velocity or Density)` | An axis has one open face and one periodic face. Open both faces of the axis (`setFaceBoundary()` on `XMin` and `XMax`, for example), with any combination of `Velocity` and `Density`. See [open faces](api_reference.md#open-faces). |
| `with open faces perpendicular to y the grid needs at least 3 nodes along y` | The nodes of the two faces are the first and the last node of the axis; there must be fluid between them. Use more nodes. |
| `with open faces the fluid exchanges momentum with the outside, and its momentum cannot be removed: call setFluidMomentumRemovalFrequency(0)` | The default removes the momentum of the fluid at every step, which makes no sense with inlets and outlets. Call `force.setFluidMomentumRemovalFrequency(0)`. |
| `TypeError: LBMForce.setFluidDensity(): the unit gram/(centimeter**3) cannot be converted to dalton/(nanometer**3)` (or another method and unit) | A setter received a `Quantity` whose unit does not convert to the unit of the method. A density in g/cm³ needs the Avogadro constant: multiply it by `unit.AVOGADRO_CONSTANT_NA`. Otherwise pass the value in the unit of the method ([summary](api_reference.md#summary)) or a `Quantity` in a compatible unit. |
| `the density of a face must not be negative` | `setFaceDensity()` takes a density in Da/nm³; 0 means the density of the fluid at rest. |
| `the checkpoint was written with a different wall scheme` | Use the same `setWallScheme()` as in the first run (checkpoints of versions 0.1 and 0.2 have bounce-back walls). |
| `the checkpoint was written with different boundary types of the faces (setFaceBoundary())` | Use the same `setFaceBoundary()` for the six faces as in the first run (checkpoints of versions 0.1 and 0.2 have periodic faces). |
| `updateParametersInContext: the wall scheme cannot be changed`, `updateParametersInContext: the boundary types of the faces cannot be changed` | These are fixed when the Context is created. The velocities and densities of the faces can be changed. |
| `the checkpoint was written with fluid fluctuations, and this Context has them off` (or the opposite) | The fluid fluctuations of the restarted run differ from those of the checkpoint (checkpoints of versions 0.1 and 0.2 have none). Use the same `setFluidFluctuations()` as in the first run. |
| `updateParametersInContext: the grid size cannot be changed` (and the similar messages for the density and viscosity, the coupled particles, the solid nodes, the drag scheme and the fluid fluctuations) | These parameters are fixed when the Context is created. Create a new Context. |
| `IntegrationUtilities::initRandomNumberGenerator(): Requested two different values for the random number seed` | On the CUDA, OpenCL and HIP platforms the random force uses OpenMM's generator, which has one seed per Context. Another component of the System (an `AndersenThermostat`, for example) uses it with a different seed: give both the same seed. |

The warning `tau = ... > 1.7: with the explicit drag at the nearest node the hydrodynamic self-mobility of
a coupled particle is small` is printed when particles are coupled with the explicit drag (the default)
and the relaxation time is large: above $`\tau = 1.79`$ the hydrodynamic mobility of a particle is negative.
With the centred drag it is not printed. Reduce the viscosity or the time
step, or use a coarser lattice; see [relaxation time](lattice.md#relaxation-time).

The warning `friction*dt = ... > 1` is printed on stderr when the explicit drag overshoots: the velocity of a
particle relative to the fluid changes sign at every step, and grows without bound for
$`\text{friction}\times\Delta t \ge 2`$. Reduce the friction or the time step, or use the centred drag
(`setDragScheme(LBMForce.Centered)`), which is stable for any friction.

The warning `with fluid fluctuations the explicit drag makes the coupled particles hotter than the set
temperature, by about friction*dt*m/(2 m_c) = ...%` is printed on stderr when fluid fluctuations are switched on
with the explicit drag, coupled particles, the `EulerMaruyama` scheme, $`T > 0`$ and a friction above zero. Use the
centred drag (`setDragScheme(LBMForce.Centered)`) with the fluctuating fluid; see
[choosing the drag](lattice.md#choosing-the-drag).

The warning `the relaxation time tau = ... is outside the range [0.505, 2]` is printed on stderr and
does not stop the simulation. See [relaxation time](lattice.md#relaxation-time) for how to bring $`\tau`$
into the range.

## Common pitfalls

**A run on several nodes stops at the start with `no remote ep address for lane` (UCX) in `MPI_Comm_split_type`.**
UCX failed to connect the ranks over several InfiniBand ports. Choose one port, for example
`mpirun -x UCX_NET_DEVICES=mlx5_0:1 ...` (the names are those of `ucx_info -d`), or keep all the ports with
`-x UCX_PROTO_ENABLE=n` ([running on several GPUs](parallel.md#launching)).

**A run with several MPI ranks hangs.** A collective call made on some ranks only waits forever for the others. With
the domain decomposition every rank must create the Context, evaluate the forces when there are coupled particles
(`getState()` with forces or with the energy: the `VerletIntegrator` needs the forces for the kinetic energy) and
call `getWallForce()`, `getFluidMachNumber()`, `setFluidState()`, the calls with `gather=True` and the checkpoints
and VTK files of the fluid, in the same order. A reporter that makes such a call, such as `StateDataReporter` with
the temperature, must be on every rank, with its file only on rank 0 (`os.devnull` on the others); see
[running on several GPUs](parallel.md).

**The fluid does not move under a body force.** The momentum of the fluid is removed at every step by
default. Call `setFluidMomentumRemovalFrequency(0)` for flows driven by `setBodyAcceleration()`.

**The density is wrong by a factor of 6e23.** OpenMM masses are molar masses (1 Da = 1 g/mol), so a
density in g/cm³ must be multiplied by `unit.AVOGADRO_CONSTANT_NA`; see [units](getting_started.md#units).

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

**A run on a GPU differs from the same run on the Reference platform.** With $`T > 0`$ the random forces come from
different generators, so the trajectories differ while their statistics agree. At $`T = 0`$, or with the NVE scheme
and without fluid fluctuations, the platforms agree to rounding in double precision.

**The temperature in the log is a little below the set temperature.** The temperature that OpenMM reports for
coupled particles is the full-step one. With the explicit drag (the default) it is the right one, and it is below
$`T`$ because the fluid has no thermal fluctuations of its own and takes part of the momentum of the particles:
about 1-2% with $`\text{friction}\times\Delta t = 0.1`$, more with a large friction (13% for the disordered protein
of `examples/cocomo/diffusion.py --preset rlp`, friction 100/ps). See [validation.md](../validation.md). The
velocities that OpenMM stores are those of the half step, whose temperature is higher,
$`T/(1 - \text{friction}\times\Delta t/2)`$ for a free particle. With the centred drag the right temperature is that
of the half step, which `openmmlbm.LBMTemperatureReporter` reports, and the log of `StateDataReporter` is lower
still; the centred drag is colder than the explicit one ([choosing the drag](lattice.md#choosing-the-drag)).

**With fluid fluctuations the particles are hotter than the set temperature.** With the explicit drag the thermal
motion of the fluid heats the coupled particles by about $`\text{friction}\times\Delta t\times m/(2m_c)`$, where
$`m_c`$ is the mass of fluid in a cell (56% for beads of 1000 Da with friction 10/ps, $`\Delta t = 0.01`$ ps and
$`\Delta x = 0.5`$ nm); a warning says so when the Context is created. Use the centred drag with the fluctuating
fluid, and measure the temperature with `openmmlbm.LBMTemperatureReporter` ([choosing the
drag](lattice.md#choosing-the-drag)).

**Particles that are not coupled cross the walls.** Only coupled particles are reflected at solid
nodes.

**`System.addForce(LBMForce())` raises a `TypeError`.** The Python wrapper of the plugin was generated
with a different SWIG version from the OpenMM Python module, so OpenMM does not recognize the
`LBMForce` as a `Force`. Rebuild the plugin with the SWIG version that CMake names in its error
message; see the [README](../../README.md#building).

**Minimizing the energy or calling `getState()` does not advance the fluid.** This is intended: the
fluid advances once per integration step, and only `integrator.step()` (or `Simulation.step()`)
advances it.
