# Architecture

openmm-lbm follows the structure of the OpenMM example plugin
(https://github.com/openmm/openmmexampleplugin). The HIP platform follows openmm-torch.

## Files

| Path | Content |
|---|---|
| `openmmapi/include/LBMForce.h`, `openmmapi/src/LBMForce.cpp` | public API: parameters, coupled particles, access to the fluid |
| `openmmapi/include/LBMKernels.h` | `CalcLBMForceKernel`, the interface every platform implements, and `LBMLatticeParameters` |
| `openmmapi/include/internal/LBMForceImpl.h`, `openmmapi/src/LBMForceImpl.cpp` | checks the setup and converts all parameters to lattice units once, for all platforms |
| `openmmapi/include/internal/D3Q19.h` | velocity set, weights, opposite velocities, ordering of the populations, equilibrium and its deviation from the rest equilibrium, Hermite polynomial H2, regularized non-equilibrium part, Guo forcing (host code) |
| `platforms/reference/` | `ReferenceCalcLBMForceKernel`: plain C++ in double precision, the correctness reference |
| `platforms/common/` | `CommonCalcLBMForceKernel` and the device kernels (`src/kernels/*.cc`), written once in the OpenMM common compute dialect |
| `platforms/cuda/`, `platforms/opencl/`, `platforms/hip/` | only the kernel factories, which create `CommonCalcLBMForceKernel` with the context of the platform, and the tests |
| `serialization/` | XML proxy of `LBMForce` (parameters only) |
| `python/` | SWIG wrapper `openmmlbm` and its tests (`TestExamples.py` runs every script of `examples/` for a few steps) |
| `examples/` | example scripts, ports of the examples of the DragOpenMM plugin (`examples/README.md`) |
| `tests/TestLBMForce.h` | tests shared by all platforms; each platform has a `Test<Platform>LBMForce.cpp` |
| `tests/TestLBMFluid.h` | tests of the fluid on its own, `runFluidTests()`, and of the solid nodes, `runWallTests()`, on every platform (`docs/validation.md`) |

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
   step (moments, momentum removal, coupling of the particles, collision and streaming, bounce-back at
   solid nodes; see `docs/theory.md` sections 1 and 2) and computes the coupling forces; every
   `execute()` adds those forces to the particles, so other force evaluations neither advance the fluid
   nor draw new random numbers. `beginStep()` also reverses the velocity of a coupled particle that has
   entered a solid node. Every platform does all of this.
3. **Lattice step on the CUDA, OpenCL and HIP platforms** (`CommonCalcLBMForceKernel::advanceFluid()`,
   kernels in `platforms/common/src/kernels/lbmFluid.cc` and `lbmCoupling.cc`):

   | Kernel | Threads | Reads | Writes |
   |---|---|---|---|
   | `reflectParticles` (in `beginStep()`, with coupled particles and solid nodes) | one per atom | positions, velocities, solid mask | velocities of the reflected particles, their momentum given to the wall |
   | `computeFluidMoments` | one per node | populations | rho - 1, j, Pi^neq of the node |
   | `sumFluidMomentum` (when the removal is due) | work groups of 64 | rho - 1, j | one partial sum per group |
   | `computeFluidCenterVelocity` | one work group | partial sums | u_cm |
   | `removeFluidMomentum` | one per node | rho - 1, u_cm | j |
   | `coupleParticles` (with coupled particles) | one per atom | positions, velocities, moments of the nearest node, OpenMM's random numbers | force of the particle, its key node*N_p + i |
   | OpenMM's `ComputeSort` | | keys | sorted keys |
   | `sumCellReactions` | one per key | sorted keys, forces | the reaction of each node, written by its first key |
   | `collideAndStream` | one per fluid node | moments and reaction of the node | the 19 populations it sends to the neighbours |
   | `clearCellReactions` | one per key | sorted keys | zero reaction at the nodes of the step |
   | `bounceBack` (with solid nodes) | one per solid node | populations of the solid node | the populations it returns to the fluid neighbours, and its momentum exchange |
   | `computeMaxFluidSpeed` (when the Mach check is due) | work groups of 64 | populations | one maximum per group, reduced on the host |

   The moments are stored component by component, [k numNodes + node], so that consecutive threads read
   consecutive addresses; the Reference platform stores them node by node. With solid nodes, the kernels
   are compiled with `HAS_SOLID_NODES` and read a mask of the fluid nodes; without them they do not read it.
   `getWallForce()` sums the momentum exchange of the solid nodes of the last step on the host, so a step
   costs no transfer. The atoms are addressed through OpenMM's atom index array, since OpenMM may reorder
   them: every per-particle array is indexed by the position of the particle in the list of the force.
   Every force evaluation runs `applyCouplingForces`, which adds the coupling forces to OpenMM's force
   buffer: in the evaluation of a step, those of the step; between steps, those of the next step, computed
   by `computeFluidMoments`, the momentum removal if due and `coupleParticles` without the reaction on the
   fluid. The random numbers of a step are copied from OpenMM's generator into the array `noise` once, by the
   first `coupleParticles` that needs them (`drawNoise`), and reused by the step.
4. **Checkpoints.** `LBMForce::createCheckpoint()` and `loadCheckpoint()` go through `LBMForceImpl`, which
   writes and checks a header (tag, version, platform, grid size, number of coupled particles), to the
   kernel, which writes its arrays as they are: populations, random numbers drawn for the next step, wall
   momentum and, on the Reference platform, its SFMT generator. The Python module adds
   `openmmlbm.saveCheckpoint()`, `loadCheckpoint()` and `LBMCheckpointReporter` (in `python/openmmlbm.i`),
   which store an OpenMM checkpoint and the checkpoint of the force in one file.
5. **Fluid access.** `getFluidFields()`, `getFluidState()` and `setFluidState()` go from `LBMForce`,
   through `LBMForceImpl`, to the kernel. The common implementation computes density and momentum on
   the device (`computeFluidMoments`), then converts them to OpenMM units on the host.

## Invariants

- **Unit conversion in one place.** It happens only in `LBMForceImpl::computeLatticeParameters()`;
  kernels receive lattice units.
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
   C++ tests run on every platform, and the Python test `test_fluid_agrees_with_reference` compares the
   fluid of each GPU platform with the Reference platform directly.
