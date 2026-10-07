# Architecture

openmm-lbm follows the structure of the OpenMM example plugin
(https://github.com/openmm/openmmexampleplugin). The HIP platform follows openmm-torch.

## Files

| Path | Content |
|---|---|
| `openmmapi/include/LBMForce.h`, `openmmapi/src/LBMForce.cpp` | public API: parameters, coupled particles, access to the fluid |
| `openmmapi/include/LBMKernels.h` | `CalcLBMForceKernel`, the interface every platform implements, and `LBMLatticeParameters` |
| `openmmapi/include/internal/LBMForceImpl.h`, `openmmapi/src/LBMForceImpl.cpp` | checks the setup, computes the lattice parameters for all platforms (Invariants, below), prints the warnings, writes and checks the header of the checkpoints |
| `openmmapi/include/internal/D3Q19.h` | velocity set, weights, opposite velocities, ordering of the populations, equilibrium and its deviation from the rest equilibrium, Hermite polynomial H2, regularized non-equilibrium part, Guo forcing (host code) |
| `platforms/reference/` | `ReferenceCalcLBMForceKernel`: plain C++ in double precision, the correctness reference |
| `platforms/common/` | `CommonCalcLBMForceKernel` and the device kernels (`src/kernels/*.cc`), written once in the OpenMM common compute dialect |
| `platforms/cuda/`, `platforms/opencl/`, `platforms/hip/` | only the kernel factories, which create `CommonCalcLBMForceKernel` with the context of the platform (on OpenCL its subclass `OpenCLCalcLBMForceKernel`, step 3 below), and the tests |
| `serialization/` | XML proxy of `LBMForce` (parameters only; version 5, which reads versions 1 to 4) |
| `python/` | SWIG wrapper `openmmlbm`, with the Python helpers `LBMTemperatureReporter`, `LBMVTKReporter`, `saveCheckpoint()`, `loadCheckpoint()` and `LBMCheckpointReporter`, and its tests (`TestExamples.py` runs every script of `examples/` for a few steps) |
| `examples/` | example scripts, ports of the examples of the DragOpenMM plugin (`examples/README.md`) |
| `tests/TestLBMForce.h` | tests shared by all platforms; each platform has a `Test<Platform>LBMForce.cpp` |
| `tests/TestLBMFluid.h` | tests of the fluid on its own, `runFluidTests()`, and of the solid nodes, `runWallTests()`, on every platform (`docs/validation.md`) |
| `tests/TestLBMCoupling.h` | tests of the particle-fluid coupling, `runCouplingTests()`, on every platform |
| `tests/TestLBMCentered.h` | tests of the centred drag, `runCenteredTests()`, which also runs the coupling tests that do not depend on the drag with the centred one, on every platform |

## Libraries

| Library | Installed in | Content |
|---|---|---|
| `libOpenMMLBM` | `lib/` | API, implementation of the force, serialization |
| `libOpenMMLBMReference` | `lib/plugins/` | Reference platform |
| `libOpenMMLBMCUDA`, `libOpenMMLBMOpenCL`, `libOpenMMLBMHIP` | `lib/plugins/` | GPU platforms (factory + common implementation) |

OpenMM loads the libraries in `lib/plugins/` at startup. A GPU platform library is only usable when
the corresponding OpenMM platform is available. For example, `libOpenMMLBMCUDA` needs the CUDA driver.

## Data flow

1. **Context creation.**
   - `LBMForceImpl::initialize()` checks the integrator (Verlet), the box (rectangular, cubic cells),
     the viscosity (tau > 1/2) and the coupled particles.
   - It computes `LBMLatticeParameters` and passes them to the kernel of the platform.
   - The kernel allocates the fluid and sets it to equilibrium.
2. **Integration step.** `VerletIntegrator::step()` calls `ContextImpl::updateContextState()`, which
   calls `LBMForceImpl::updateContextState()`, which calls the kernel's `beginStep()`; the kernel
   records the step count of the Context, which times the momentum removal and the Mach number check.
   Then
   `LBMForceImpl::calcForcesAndEnergy()` checks that the step size has not changed and calls the
   kernel's `execute()`. The first `execute()` after `beginStep()` advances the fluid by one lattice
   step (moments, momentum removal, coupling of the particles, collision, with the random part of a
   fluctuating fluid, and streaming, bounce-back at solid nodes; see `docs/theory.md` sections 1, 2 and 7) and adds the coupling forces of the step to the
   particles. The other force evaluations (`getState()`, for example) do not advance the fluid: they add the
   coupling forces of the next step, computed on the current fluid without its reaction, as OpenMM does for
   every force. The random numbers of a step are drawn once, by the first evaluation that needs them, and
   reused by the step. Before the first step of the Context the coupling forces are zero. A repeated
   evaluation of a step (OpenMM repeats all the evaluations of a step when it enlarges the neighbor list of a
   nonbonded force) is recognized because the step count of the Context has not been incremented yet, and
   it adds the forces of the step again. `beginStep()` also reverses the velocity of a coupled particle whose
   nearest node is solid and that moves into the wall (v.n > 0). Every platform does all of this.
   With the centred drag (`LBMForce::Centered`) the coupling needs the other forces on the particles:
   `LBMForceImpl::initialize()` checks that `LBMForce` is the last force of the System and that there are
   no virtual sites. On the Reference platform, when its `execute()` runs, OpenMM's force array already
   holds the sum of the other forces; `ReferenceCalcLBMForceKernel::coupleParticlesCentered()` reads them,
   sorts the keys node*N_p + i, solves the drag of each node in closed form (`docs/theory.md` section 2,
   Solution of the centred drag) and gives the node the reaction -S. On the CUDA, OpenCL and HIP platforms
   `execute()` does nothing: `CommonCalcLBMForceKernel::initialize()` registers a `ForcePostComputation`
   (`CenteredDragPostComputation`), which OpenMM calls at the end of every force evaluation, when its force
   buffers hold all the other forces, and which does the work of `execute()` (step 3).
3. **Lattice step on the CUDA, OpenCL and HIP platforms** (`CommonCalcLBMForceKernel::advanceFluid()`,
   kernels in `platforms/common/src/kernels/lbmFluid.cc` and `lbmCoupling.cc`):

   | Kernel | Threads | Reads | Writes |
   |---|---|---|---|
   | `reflectParticles` (in `beginStep()`, with coupled particles and solid nodes) | one per atom | positions, velocities, solid mask | velocities of the reflected particles, their momentum given to the wall |
   | `computeFluidMoments` | one per node | populations | rho - 1, j, Pi^neq of the node |
   | `sumFluidMomentum` (when the removal is due) | work groups of 64 | rho - 1, j | one partial sum per group |
   | `computeFluidCenterVelocity` | one work group | partial sums | u_cm |
   | `removeFluidMomentum` | one per node | rho - 1, u_cm | j |
   | `coupleParticles` (explicit drag, with coupled particles) | one per atom | positions, velocities, moments of the nearest node, OpenMM's random numbers | force of the particle, its key node*N_p + i, the random numbers of the step (`noise`) when it draws them, the reaction at a solid node (wall momentum) |
   | `prepareCenteredDrag` (centred drag, with coupled particles) | one per atom | positions, velocities, OpenMM's force buffers (the other forces), OpenMM's random numbers | v~ = v + dt Fc/(2m) and random force of the particle, its key node*N_p + i, `noise` when it draws the random numbers |
   | OpenMM's `ComputeSort` (explicit drag: in a lattice step; centred drag: in every evaluation) | | keys | sorted keys |
   | `sumCellReactions` (explicit drag, in a lattice step) | one per key | sorted keys, forces | the reaction of each node, written by its first key |
   | `solveCenteredDrag` (centred drag) | one per key | sorted keys, v~, random forces, moments of the node, body acceleration | the forces of the particles of each node, written by its first key; in a lattice step also the reaction -S of the node, and at a solid node -S as wall momentum |
   | `collideAndStream` | one per node (solid nodes do nothing) | moments and reaction of the node; with a fluctuating fluid, four float4 of OpenMM's random numbers per node, drawn after those of the particles, and the coefficients of the basis | the 19 populations it sends to the neighbours |
   | `clearCellReactions` | one per key | sorted keys | zero reaction at the nodes of the step |
   | `bounceBack` (with solid nodes) | one per solid node | populations of the solid node | the populations it returns to the fluid neighbours, and its momentum exchange |
   | `computeMaxFluidSpeed` (when the Mach check is due) | work groups of 64 | populations | one maximum per group, reduced on the host |
   | `applyCouplingForces` (with coupled particles, in every force evaluation that includes forces) | one per atom | coupling forces | OpenMM's fixed point force buffer |

   The moments are stored component by component, [k numNodes + node], so that consecutive threads read
   consecutive addresses; the Reference platform stores them node by node. With solid nodes, the kernels
   are compiled with `HAS_SOLID_NODES` and read a mask of the fluid nodes; without them they do not read it.
   `getWallForce()` sums the momentum exchange of the solid nodes of the last step on the host, so a step
   costs no transfer. The atoms are addressed through OpenMM's atom index array, since OpenMM may reorder
   them: every per-particle array is indexed by the position of the particle in the list of the force.
   Every force evaluation that includes forces runs `applyCouplingForces`, which adds the coupling forces to
   OpenMM's force buffer: in the evaluation of a step, those of the step; between steps, those of the next
   step, computed by `computeFluidMoments`, the momentum removal if due and `coupleParticles` (explicit drag)
   or `prepareCenteredDrag`, the sort and `solveCenteredDrag` (centred drag), without the reaction on the
   fluid; in a repeated evaluation of a step, those of the step again, without computing them. The random
   numbers of a step are copied from OpenMM's generator into the array `noise` once, by the first
   `coupleParticles` or `prepareCenteredDrag` that needs them (`drawNoise`), and reused by the step.
   With the centred drag all of this runs in the post-computation, which reads the other forces from
   OpenMM's fixed point buffer and, on the OpenCL platform, also from its floating point buffers, which
   OpenCL adds to the fixed point one only after the post-computations. The OpenCL factory creates a
   subclass of the common kernel, `OpenCLCalcLBMForceKernel`, whose `hasFloatForceBuffers()` returns true;
   with the centred drag the coupling kernels are then compiled with `HAS_FLOAT_FORCE_BUFFERS`, and
   `prepareCenteredDrag` gets the buffers and their number on first use, since OpenMM creates them after the
   forces are initialized. The post-computation does nothing if `LBMForce`'s force
   group is not requested, or if OpenMM has marked the evaluation as invalid (neighbor list overflow): the
   repeated evaluation then does the step.
4. **Checkpoints.** `LBMForce::createCheckpoint()` and `loadCheckpoint()` go through `LBMForceImpl`, which
   writes and checks a header (tag, version, platform, grid size, number of coupled particles, from
   version 2 the drag scheme and from version 3 the switch of the fluid fluctuations; version 1 is read as the
   explicit drag, versions 1 and 2 as without fluctuations), to the kernel, which writes its state as
   it is. The common kernel writes the size of its floating point type (a checkpoint is refused in another
   precision), whether the random numbers of the next step are already drawn and whether the fluid has
   advanced, then the populations, those random numbers and the wall momentum of the particles and of the
   solid nodes. The Reference kernel writes the populations, the random numbers of the next step, the wall
   momentum and its SFMT generator, with the Gaussian number kept by its Box-Muller transform. The Python module adds
   `openmmlbm.saveCheckpoint()`, `loadCheckpoint()` and `LBMCheckpointReporter` (in `python/openmmlbm.i`),
   which store an OpenMM checkpoint and the checkpoint of the force in one file.
5. **Fluid access.** `getFluidFields()`, `getFluidState()` and `setFluidState()` go from `LBMForce`,
   through `LBMForceImpl`, to the kernel. The common implementation computes density and momentum on
   the device (`computeFluidMoments`), then converts them to OpenMM units on the host.

## Invariants

- **Unit conversion in two documented steps.** `LBMForceImpl::computeLatticeParameters()` validates the
  setup and computes tau, omega and the initial velocity and body acceleration in lattice units; it passes
  dx, dt, the density, the friction and kT in OpenMM units (`LBMLatticeParameters`). The kernels convert the
  per-particle quantities with the same factors on every platform: masses in units of the cell mass
  m_c = rho0 dx^3, gamma = friction dt, kT dt^2/(m_c dx^2), velocities with `getVelocityScale()` = dx/dt and
  forces with m_c dx/dt^2 (`docs/theory.md`, section 3).
- **Same conventions everywhere.** Every platform uses the population ordering and the node index of
  `D3Q19.h`, and stores the populations as deviations f_q - w_q from the rest equilibrium, so
  `getFluidState()` is portable across platforms.
- **Precision.** The fluid is stored in the "mixed" type of the platform (float in single precision,
  double otherwise); Reference uses double.
- **No atomic operations in the plugin kernels**, and the fluid is advanced once per step (see
  `docs/theory.md`).

## Adding a device kernel

1. Write it in `platforms/common/src/kernels/<file>.cc` with the common macros (`KERNEL`, `GLOBAL`,
   `GLOBAL_ID`, `GLOBAL_SIZE`, `real`, `mixed`, ...).
2. At build time CMake turns every `.cc` file into a string `CommonLBMKernelSources::<file>`.
3. Compile it at runtime with `cc.compileProgram(CommonLBMKernelSources::<file>, defines)` in
   `CommonCalcLBMForceKernel`.
4. Add the same algorithm to `ReferenceCalcLBMForceKernel`, and a test that compares the two: the shared
   C++ tests run on every platform, and the Python tests `test_fluid_agrees_with_reference`,
   `test_coupling_agrees_with_reference` (both drags) and `test_centered_drag_sees_all_forces` compare each
   GPU platform with the Reference platform, or with the closed form, directly.

## Adding a platform

1. Write a kernel factory in `platforms/<platform>/src/`, like `CudaLBMKernelFactory.cpp`, that creates
   `CommonCalcLBMForceKernel` with the `ComputeContext` of the platform. If the platform keeps part of the
   forces in floating point buffers that it adds to the fixed point buffer only after the
   post-computations (as OpenCL does), create a subclass whose `hasFloatForceBuffers()` returns true, as
   `OpenCLCalcLBMForceKernel` does.
2. Add the library in `platforms/<platform>/CMakeLists.txt` and, in the top `CMakeLists.txt`, the option
   `LBM_BUILD_<PLATFORM>_LIB` as for the other platforms: `LBM_OPENMM_PLATFORM()` checks that the OpenMM in
   `OPENMM_DIR` has the platform (its header and its library) and gives the full path of the library, which
   the platform links (the libraries of OpenMM are never linked with `-l`), a test checks the toolkit that
   compiles it, and `LBM_OPTION()` sets the default, says why the platform is not built, and stops `cmake` if
   the option is `ON` for a platform that cannot be built. Add the platform to the summary at the end of the
   file.
3. Add `platforms/<platform>/tests/Test<Platform>LBMForce.cpp`, which calls `runPlatformTests()`,
   `runFluidTests()`, `runWallTests()`, `runCouplingTests()` and `runCenteredTests()` and takes the
   precision as its argument, and register it in `platforms/<platform>/tests/CMakeLists.txt` in the three
   precision modes.
4. Add the platform to the Python tests that compare the GPU platforms with the Reference platform.
