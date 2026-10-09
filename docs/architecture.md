# Architecture

openmm-lbm follows the structure of the OpenMM example plugin
(https://github.com/openmm/openmmexampleplugin). The HIP platform follows openmm-torch.

## Files

| Path | Content |
|---|---|
| `openmmapi/include/LBMForce.h`, `openmmapi/src/LBMForce.cpp` | public API: parameters, coupled particles, access to the fluid |
| `openmmapi/include/LBMKernels.h` | `CalcLBMForceKernel`, the interface every platform implements, and `LBMLatticeParameters` |
| `openmmapi/include/internal/LBMForceImpl.h`, `openmmapi/src/LBMForceImpl.cpp` | checks the setup, computes the lattice parameters for all platforms (Invariants, below), prints the warnings, writes and checks the header of the checkpoints; with the domain decomposition it gathers and scatters the fields and the state of the fluid, and writes and reads the checkpoint files and the VTK files of the fluid with `LBMParallelFile` |
| `openmmapi/include/internal/LBMDecomposition.h`, `openmmapi/src/LBMDecomposition.cpp` | `LBMDecomposition`: the blocks of the domain decomposition and every MPI call of the plugin (collective errors, sums in rank order, the exchanges, gathering and scattering); `LBMParallelFile`: the files written and read by all the ranks together with MPI-IO, each rank the nodes of its domain at their place in an array of the lattice (checkpoint files, VTK files of the fluid); the initialization of MPI on first use and its finalization (`finalizeMPI()`, which the Python module calls when the script ends). The only file that includes `mpi.h`; its MPI code is compiled only with `OPENMM_LBM_MPI`, and without it there is one domain and the files use the standard library |
| `openmmapi/include/internal/D3Q19.h` | velocity set, weights, opposite velocities, ordering of the populations, equilibrium and its deviation from the rest equilibrium, Hermite polynomial $`H^{(2)}`$, regularized non-equilibrium part, Guo forcing, orthogonal basis of Lulli et al. and random part of the fluctuating fluid (host code) |
| `openmmapi/include/internal/LBMBoundaries.h` | `LBMBoundaries`: finds the boundary nodes of regularized walls and open faces, with the bits of their unknown and solid directions, what each one imposes and the face that gives it, and the links of the bounce-back walls from the fluid nodes to the solid nodes (host code, used by every platform) |
| `platforms/reference/` | `ReferenceCalcLBMForceKernel`: plain C++ in double precision, the correctness reference |
| `platforms/common/` | `CommonCalcLBMForceKernel` and the device kernels (`src/kernels/*.cc`), written once in the OpenMM common compute dialect |
| `platforms/cuda/`, `platforms/opencl/`, `platforms/hip/` | only the kernel factories, which create `CommonCalcLBMForceKernel` with the context of the platform (on CUDA its subclass `CudaCalcLBMForceKernel`, which gives the MPI library the address of the device buffers of the exchange, `getDeviceAddress()`; on OpenCL its subclass `OpenCLCalcLBMForceKernel`, step 3 below; the CUDA and HIP factories also pass the platform property `DeterministicForces`), and the tests |
| `serialization/` | XML proxy of `LBMForce` (parameters only; version 8, which reads versions 1 to 7) |
| `python/` | SWIG wrapper `openmmlbm`, with the Python helpers `LBMTemperatureReporter`, `LBMVTKReporter`, `saveCheckpoint()`, `loadCheckpoint()` and `LBMCheckpointReporter`, and its tests (`TestExamples.py` runs every script of `examples/` for a few steps) |
| `examples/` | example scripts, ports of the examples of the DragOpenMM plugin (`examples/README.md`) |
| `tests/TestLBMForce.h` | tests shared by all platforms; each platform has a `Test<Platform>LBMForce.cpp` |
| `tests/TestLBMFluid.h` | tests of the fluid on its own, `runFluidTests()`, and of the walls, with both wall schemes, and the open faces, `runWallTests()`, on every platform (`docs/validation.md`) |
| `tests/TestLBMCoupling.h` | tests of the particle-fluid coupling, `runCouplingTests()`, on every platform |
| `tests/TestLBMCentered.h` | tests of the centred drag, `runCenteredTests()`, which also runs the coupling tests that do not depend on the drag with the centred one, on every platform |
| `tests/TestLBMFluctuations.h` | tests of the fluctuating fluid, `runFluctuationTests()`, on every platform |
| `devtools/`, `.github/workflows/CI.yml` | continuous integration and its conda environments; `devtools/check_user_guide.py` runs the Python blocks of `docs/user_guide/` and checks their output |

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
     the viscosity ($`\tau > 1/2`$), the solid nodes, the faces and the coupled particles.
   - It computes `LBMLatticeParameters` and passes them to the kernel of the platform.
   - The kernel allocates the fluid and sets it to equilibrium.
   - With the domain decomposition (`docs/theory.md` section 8), `computeLatticeParameters()` resolves the
     decomposition (`LBMDecomposition::resolve()`, the zeros with `MPI_Dims_create`) and checks the lattice and its
     largest domain against the 32-bit indices of the plugin; `initialize()` refuses an `AndersenThermostat` or a
     Monte Carlo barostat. The kernel's `initialize()` checks over the ranks that all use the same platform and
     precision and, on CUDA and HIP with coupled particles, that `DeterministicForces` is on; it lists the frame and
     the interior of the block and the slots exchanged with each rank, and exchanges the halo of the fields that are
     exchanged.
2. **Integration step.** `VerletIntegrator::step()` calls `ContextImpl::updateContextState()`, which calls
   `LBMForceImpl::updateContextState()`, which calls the kernel's `beginStep()`; the kernel records the step count of
   the Context, which times the momentum removal and the Mach number check. Then `LBMForceImpl::calcForcesAndEnergy()`
   checks that the step size has not changed and calls the kernel's `execute()`. The first `execute()` after
   `beginStep()` advances the fluid by one lattice step (moments, momentum removal, coupling of the particles,
   collision, with the random part of a fluctuating fluid, and streaming, the bounce-back at the fluid nodes next to
   the solid nodes with the `BounceBack` wall scheme, then the rebuild of the boundary nodes of regularized walls and
   open faces; see `docs/theory.md` sections 1, 2 and 7) and adds the coupling forces of the step to the particles.
   The other force evaluations (`getState()`, for example) do not advance the fluid: they add the coupling forces of
   the next step, computed on the current fluid without its reaction, as OpenMM does for every force. The random
   numbers of a step are drawn once, by the first evaluation that needs them, and reused by the step. Before the first
   step of the Context the coupling forces are zero. A repeated evaluation of a step (OpenMM repeats all the
   evaluations of a step when it enlarges the neighbor list of a nonbonded force) is recognized because the step count
   of the Context has not been incremented yet, and it adds the forces of the step again. `beginStep()` also reverses
   the velocity of a coupled particle whose nearest node is solid and that moves into the wall
   ($`\mathbf v\cdot\mathbf n > 0`$). Every platform does all of this. With the domain decomposition `beginStep()` also
   compares the copies of the particles over the ranks when due (the first lattice step, then with the period of the
   Mach number check), the lattice step collides the frame of the block, starts the exchange of its populations,
   collides the interior and then waits for the exchange, and the coupling forces are summed over the ranks after the
   coupling (download, `MPI_Allreduce`, upload). The kernel's `initialize()` lists the boundary
   nodes once with `LBMBoundaries` (`internal/LBMBoundaries.h`, the same code on every platform): the fluid nodes next
   to solid nodes with regularized walls, and the nodes of the open faces, with the bits of their unknown directions
   and what each one imposes. `ReferenceCalcLBMForceKernel::applyBoundaries()`, and the kernel `applyBoundaries` on
   the other platforms (step 3), rebuild them after the streaming, each node from its own populations and the solid
   slots it wrote itself. With the centred drag (`LBMForce::Centered`) the coupling needs the other forces on the
   particles: `LBMForceImpl::initialize()` checks that `LBMForce` is the last force of the System and that there are
   no virtual sites. On the Reference platform, when its `execute()` runs, OpenMM's force array already holds the sum
   of the other forces; `ReferenceCalcLBMForceKernel::coupleParticlesCentered()` reads them, sorts the keys
   $`\mathrm{node}\cdot N_p + i`$, solves the drag of each node in closed form (`docs/theory.md` section 2, Solution
   of the centred drag) and gives the node the reaction $`-\mathbf S`$. On the CUDA, OpenCL and HIP platforms
   `execute()` does nothing: `CommonCalcLBMForceKernel::initialize()` registers a `ForcePostComputation`
   (`CenteredDragPostComputation`), which OpenMM calls at the end of every force evaluation, when its force buffers
   hold all the other forces, and which does the work of `execute()` (step 3).
3. **Lattice step on the CUDA, OpenCL and HIP platforms** (`CommonCalcLBMForceKernel::advanceFluid()`,
   kernels in `platforms/common/src/kernels/lbmFluid.cc` and `lbmCoupling.cc`):

   | Kernel | Threads | Reads | Writes |
   |---|---|---|---|
   | `reflectParticles` (in `beginStep()`, with coupled particles and solid nodes) | one per atom | positions, velocities, solid mask | velocities of the reflected particles, their momentum given to the wall |
   | `computeFluidMoments` | one per node | populations | $`\rho - 1`$, $`\mathbf j`$, $`\Pi^{\mathrm{neq}}`$ of the node |
   | `sumFluidMomentum` (when the removal is due) | work groups of 64 | $`\rho - 1`$, $`\mathbf j`$ | one partial sum per group |
   | `computeFluidCenterVelocity` | one work group | partial sums | $`\mathbf u_{\mathrm{cm}}`$ |
   | `removeFluidMomentum` | one per node | $`\rho - 1`$, $`\mathbf u_{\mathrm{cm}}`$ | $`\mathbf j`$ |
   | `coupleParticles` (explicit drag, with coupled particles) | one per atom | positions, velocities, moments of the nearest node, OpenMM's random numbers | force of the particle, its key $`\mathrm{node}\cdot N_p + i`$, the random numbers of the step (`noise`) when it draws them, the reaction at a solid node (wall momentum) |
   | `prepareCenteredDrag` (centred drag, with coupled particles) | one per atom | positions, velocities, OpenMM's force buffers (the other forces), OpenMM's random numbers | $`\tilde{\mathbf v} = \mathbf v + \Delta t\,\mathbf F_c/(2m)`$ and random force of the particle, its key $`\mathrm{node}\cdot N_p + i`$, `noise` when it draws the random numbers |
   | OpenMM's `ComputeSort` (explicit drag: in a lattice step; centred drag: in every evaluation) | | keys | sorted keys |
   | `sumCellReactions` (explicit drag, in a lattice step) | one per key | sorted keys, forces | the reaction of each node, written by its first key |
   | `solveCenteredDrag` (centred drag) | one per key | sorted keys, $`\tilde{\mathbf v}`$, random forces, moments of the node, body acceleration | the forces of the particles of each node, written by its first key; in a lattice step also the reaction $`-\mathbf S`$ of the node, and at a solid node $`-\mathbf S`$ as wall momentum |
   | `collideAndStream` | one per node (solid nodes do nothing); with the domain decomposition one per node of a list, run for the frame of the block and then for its interior | moments and reaction of the node; with a fluctuating fluid, four float4 of OpenMM's random numbers per node, drawn after those of the particles, and the coefficients of the basis | the 19 populations it sends to the neighbours |
   | `packPopulations`, `unpackPopulations` (with the domain decomposition: packing after the `collideAndStream` of the frame, unpacking after that of the interior and the end of the exchange) | one per slot | the populations pushed into the halo; the buffer received from the other ranks | the buffer sent to the other ranks (from the device with CUDA-aware MPI on CUDA, through the host otherwise); the populations of the nodes of the block that come from other ranks |
   | `clearCellReactions` | one per key | sorted keys | zero reaction at the nodes of the step |
   | `bounceBack` (with solid nodes and the `BounceBack` wall scheme) | one per fluid node next to a solid node | the populations that the node sent into its solid neighbours, except across an open face (`wallNodes`, `wallLinks`) | the same populations, as its own populations in the opposite directions |
   | `computeWallExchange` (with solid nodes and the `BounceBack` wall scheme) | one per solid node | populations that the fluid neighbours sent into the solid node, except across an open face (the links of `solidLinks`; with the domain decomposition only those to fluid nodes of the block) | its momentum exchange |
   | `applyBoundaries` (with regularized walls or open faces) | one per boundary node | populations of the node and the solid slots it wrote in the streaming, its moments of the step, velocities and densities of the faces, body acceleration, relaxation rate; with a fluctuating fluid four float4 of OpenMM's random numbers per boundary node, after those of the nodes | its rebuilt unknown populations and, next to solid nodes, its momentum exchange |
   | `computeMaxFluidSpeed` (when the Mach check is due) | work groups of 64 | populations | one maximum per group, reduced on the host |
   | `applyCouplingForces` (with coupled particles, in every force evaluation that includes forces) | one per atom | coupling forces | OpenMM's fixed point force buffer |

   With the exchange of the halo of the density or the velocity, `packFields` (one thread per node sent) copies
   $`\rho - 1`$ and $`\mathbf j`$ of the nodes of the block in the halo of other ranks into a buffer, at the end of the
   lattice step, when the Context is created and when the state is set. When the populations are uploaded in parts
   (more than 2 GB, `docs/theory.md` section 8), `copyPart` copies each part from a smaller array into its place.
   With the domain decomposition `computeFluidCenterVelocity` writes the four sums of the block, which the host adds
   in rank order and uploads as $`\mathbf u_{\mathrm{cm}}`$.

   The moments are stored component by component, $`[k\cdot\mathrm{numNodes} + \mathrm{node}]`$, so that consecutive
   threads read consecutive addresses; the Reference platform stores them node by node. With the domain decomposition
   (`DOMAIN_DECOMPOSITION`, `docs/theory.md` section 8) the arrays hold the block of the rank and a layer of halo nodes
   along the divided axes (`PAD_X`, `PAD_Y`, `PAD_Z`; the extents `SX`, `SY` and `NUM_STORED` entries per component),
   and `STORAGE_INDEX` and `neighborIndex()` give the position of a node; the coupling kernels also get the size of the
   lattice (`GNX`, `GNY`, `GNZ`) and the first node of the block (`OX`, `OY`, `OZ`). There the node of a sort key is the
   position of the node in the arrays of the block; a particle whose nearest node belongs to another rank gets no
   coupling force and the key $`(N_{\mathrm{stored}} + \mathrm{node})\,N_p + i`$, past every node of the block
   (`otherRankKey()`); `reflectParticles` reads the mask of the whole lattice, and only rank 0 counts the momentum given
   to the walls (the other ranks compile with `SKIP_REFLECTION_MOMENTUM`); without it they reduce to the node index and
   to the periodic wrap of the lattice. With solid nodes, the kernels are compiled with `HAS_SOLID_NODES` and read a
   mask of the fluid nodes; without them they do not read it. With boundary nodes they are also compiled with
   `HAS_BOUNDARY_NODES` (the boundary nodes are fluid nodes in the mask); `OPEN_X`, `OPEN_Y` and `OPEN_Z` mark the axes
   with open faces, across which the bounce-back returns nothing; with the `BounceBack` wall scheme they are compiled
   with `BOUNCE_BACK_WALLS`. `getWallForce()` sums the momentum exchange of the last step on the host, of the solid
   nodes with bounce-back or of the nodes of regularized walls, and adds the static part of the weights $`w_q`$,
   computed once in `initialize()`, so a step costs no transfer. The atoms are addressed through OpenMM's atom index
   array, since OpenMM may reorder them: every per-particle array is indexed by the position of the particle in the list
   of the force. Every force evaluation that includes forces runs `applyCouplingForces`, which adds the coupling forces
   to OpenMM's force buffer: in the evaluation of a step, those of the step; between steps, those of the next step,
   computed by `computeFluidMoments`, the momentum removal if due and `coupleParticles` (explicit drag) or
   `prepareCenteredDrag`, the sort and `solveCenteredDrag` (centred drag), without the reaction on the fluid; in a
   repeated evaluation of a step, those of the step again, without computing them. The random numbers of a step are
   copied from OpenMM's generator into the array `noise` once, by the first `coupleParticles` or `prepareCenteredDrag`
   that needs them (`drawNoise`), and reused by the step. With the centred drag all of this runs in the
   post-computation, which reads the other forces from OpenMM's fixed point buffer and, on the OpenCL platform, also
   from its floating point buffers, which OpenCL adds to the fixed point one only after the post-computations. The
   OpenCL factory creates a subclass of the common kernel, `OpenCLCalcLBMForceKernel`, whose `hasFloatForceBuffers()`
   returns true; with the centred drag the coupling kernels are then compiled with `HAS_FLOAT_FORCE_BUFFERS`, and
   `prepareCenteredDrag` gets the buffers and their number on first use, since OpenMM creates them after the forces are
   initialized. The post-computation does nothing if `LBMForce`'s force group is not requested, or if OpenMM has marked
   the evaluation as invalid (neighbor list overflow): the repeated evaluation then does the step.
4. **Checkpoints.** `LBMForce::createCheckpoint()` and `loadCheckpoint()` go through `LBMForceImpl`, which writes and
   checks a header (tag, version, platform, grid size, number of coupled particles, from version 2 the drag scheme, from
   version 3 the switch of the fluid fluctuations, from version 4 the wall scheme and from version 5 the types of the
   six faces; version 1 is read as the explicit drag, versions 1 and 2 as without fluctuations, versions 1 to 3 as with
   bounce-back walls, versions 1 to 4 as with periodic faces), to the kernel, which writes its state as it is. The
   common kernel writes the size of its floating point type (a checkpoint is refused in another precision), whether the
   random numbers of the next step are already drawn and whether the fluid has advanced, then the populations, those
   random numbers and the wall momentum of the particles, of the solid nodes and of the nodes of regularized walls. The
   Reference kernel writes the populations, the random numbers of the next step, the wall momentum and its SFMT
   generator, with the Gaussian number kept by its Box-Muller transform. The Python module adds
   `openmmlbm.saveCheckpoint()`, `loadCheckpoint()` and `LBMCheckpointReporter` (in `python/openmmlbm.i`), which store
   an OpenMM checkpoint and the checkpoint of the force in one file. The checkpoint files of `saveCheckpointFile()`,
   which any decomposition can load, are written by `LBMForceImpl::saveCheckpointFile()` through `LBMParallelFile`: a
   head from rank 0 (the header above, the precision, the blocks, the particles), the table of the data of the ranks,
   the populations of the lattice from every rank, and the data of each rank (its OpenMM checkpoint and
   `createRankCheckpoint()` of its kernel, which is what `createCheckpoint()` writes except the populations).
   `loadCheckpointFile()` loads the data of the rank with the same decomposition, and otherwise sets the particles from
   the head and calls `resetRankState()` of the kernel (`docs/theory.md`, section 8). With more than one domain the
   Python functions use these files. `loadCheckpointFile()` also loads the files that `openmmlbm.saveCheckpoint()`
   writes with one domain (tag `OPENMMLBM-CHECKPOINT-1`), but only into a Context with one domain; with the same
   decomposition it loads the OpenMM checkpoint of the rank, sets the fluid with `setFluidState()` and then the rest of
   the state of the rank with `loadRankCheckpoint()`. A file is written as `<path>.tmp` and renamed when it is complete.
5. **Fluid access.** `getFluidFields()`, `getFluidState()` and `setFluidState()` go from `LBMForce`,
   through `LBMForceImpl`, to the kernel. The common implementation computes density and momentum on
   the device (`computeFluidMoments`), then converts them to OpenMM units on the host. The kernels give the domain of
   the rank, with its halo on request; with more than one domain `LBMForceImpl` gathers the fields and the state on
   rank 0, or scatters the state from it, through `LBMDecomposition` (`MPI_Gatherv`, `MPI_Scatterv`).

## Invariants

- **Unit conversion in two documented steps.** `LBMForceImpl::computeLatticeParameters()` validates the setup and
  computes $`\tau`$, $`\omega`$, the initial velocity, the body acceleration and the velocities and densities of the
  faces in lattice units; it passes $`\Delta x`$, $`\Delta t`$, the density, the friction and $`k_BT`$ in OpenMM units
  (`LBMLatticeParameters`). The kernels convert the per-particle quantities with the same factors on every platform:
  masses in units of the cell mass $`m_c = \rho_0\Delta x^3`$, $`\gamma = \text{friction}\cdot\Delta t`$,
  $`k_BT\,\Delta t^2/(m_c\Delta x^2)`$, velocities with `getVelocityScale()` $`= \Delta x/\Delta t`$ and forces with
  $`m_c\Delta x/\Delta t^2`$ (`docs/theory.md`, section 3).
- **Same conventions everywhere.** Every platform uses the population ordering and the node index of
  `D3Q19.h`, and stores the populations as deviations $`f_q - w_q`$ from the rest equilibrium, so
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
   `OpenCLCalcLBMForceKernel` does. For the exchange from GPU to GPU with an MPI library that reads the memory of the
   device, override `getDeviceAddress()`, as `CudaCalcLBMForceKernel` does; if the platform has a property like
   `DeterministicForces`, pass it with `setDeterministicForces()`.
2. Add the library in `platforms/<platform>/CMakeLists.txt` and, in the top `CMakeLists.txt`, the option
   `LBM_BUILD_<PLATFORM>_LIB` as for the other platforms: `LBM_OPENMM_PLATFORM()` checks that the OpenMM in
   `OPENMM_DIR` has the platform (its header and its library) and gives the full path of the library, which
   the platform links (the libraries of OpenMM are never linked with `-l`), a test checks the toolkit that
   compiles it, and `LBM_OPTION()` sets the default, says why the platform is not built, and stops `cmake` if
   the option is `ON` for a platform that cannot be built. Add the platform to the summary at the end of the
   file.
3. Add `platforms/<platform>/tests/Test<Platform>LBMForce.cpp`, which calls `runPlatformTests()`,
   `runFluidTests()`, `runWallTests()`, `runCouplingTests()`, `runCenteredTests()` and
   `runFluctuationTests()` and takes the
   precision as its argument, and register it in `platforms/<platform>/tests/CMakeLists.txt` in the three
   precision modes.
4. Add the platform to the Python tests that compare the GPU platforms with the Reference platform.
