# Changelog

All notable changes to openmm-lbm are recorded here. Versions follow semantic versioning; in the 0.x
versions the API may still change.

## 0.4.0 (2026-10-09)

Decomposition of the lattice over MPI ranks, one GPU per rank, on every platform: the fluid is divided among the
ranks, the particles are replicated on every rank, and the files of checkpoints and of the fluid are written by all
the ranks together. Without the decomposition the algorithm is that of version 0.3.0; the fixes are listed below.

### Added
- Decomposition of the lattice over MPI ranks: CMake option `OPENMM_LBM_MPI` (off by default; only the MPI C API,
  in `openmmapi/src/LBMDecomposition.cpp`; the summary of `cmake` ends with `MPI: yes (MPI x.y)` or `MPI: no`),
  `setDomainDecomposition()`/`getDomainDecomposition()` (a 0 lets MPI choose, with the most domains along z, then y,
  since dividing x costs more on the GPUs), `isMPIAvailable()`, `getMPIRank()`/`getMPISize()`/`getMPILocalRank()`
  (`openmmlbm.mpiRank()` and the like in Python), serialization version 8 with the decomposition. Without MPI, or
  with one domain, nothing changes. The user guide has a new page, `docs/user_guide/parallel.md`.
- Reference platform: the fluid, the walls and the open faces with more than one domain (`docs/theory.md`, section 8):
  each rank advances the nodes of its block and exchanges the populations streamed between blocks, overlapping the
  exchange with the collision of the interior of the block; identical bit for
  bit to one domain without the removal of the fluid momentum (`python/tests/mpi_decomposition.py`).
- Reference platform: coupled particles with more than one domain. The particles are replicated on every rank; the rank
  of the nearest node computes the coupling force, and the forces are summed over the ranks (exact). The reflection at
  the walls is done by every rank, and its momentum counted once. Identical bit for bit to one domain at T = 0, with
  both drag schemes. Checks that stop every rank together: the same platform and precision on every rank, identical
  copies of the particles (a hash of positions and velocities, at the first step and then with the period of the Mach
  number check); an `AndersenThermostat` or a Monte Carlo barostat is refused with more than one domain. With more than
  one domain the warnings of the Context are printed by rank 0 only. `setParticleCopiesCheck()` turns the comparison of
  the copies off (on by default; serialized in version 8). With more than one MPI rank the Python module aborts every
  rank (`LBMForce.abortMPI()`, `MPI_Abort`) after printing an uncaught exception, so that an error on one rank does not
  leave the others waiting.
- Fields and state of the fluid with more than one domain: `getLocalDomain()`; `getFluidFields()`, `getFluidState()`
  and `setFluidState()` work on the domain of the rank, and gather the whole lattice on rank 0 or set it from rank 0
  only on request (`gather=True`, `scatter=True`); `getFluidFields(halo=True)` adds a layer one node thick around the
  domain, filled for the fields whose halo is exchanged at the end of every step (`setDensityHaloExchange()`,
  `setVelocityHaloExchange()`, off by default, serialized in version 8) and NaN otherwise. With one domain the new
  arguments change nothing else, and the halo of an exchanged field comes from the lattice across the periodic
  boundaries (NaN beyond the open faces).
- CUDA, OpenCL and HIP platforms: the fluid, the walls and the open faces with more than one domain, one GPU per rank.
  Each rank stores only its block and a layer of halo nodes along the divided axes; the populations pushed into the halo
  are packed on the device, sent with non-blocking MPI calls while the interior of the block collides, and unpacked into
  the blocks of their owners: on the CUDA platform with CUDA-aware MPI from device to device between the ranks of a node
  (over NVLink), through the host between nodes and on the other platforms. Identical bit for bit to one domain on the
  same platform and precision without the removal of the fluid momentum. Without the decomposition the arithmetic and
  the kernels do not change. Coupled particles as on the Reference platform: the rank of the nearest node computes the
  coupling force, the forces are summed over the ranks, and the CUDA and HIP platforms need the property
  `DeterministicForces`; the copies of the particles are compared over the ranks as on the Reference platform. The
  exchange of the halo of the density and the velocity: the device copies the moments of the nodes to send, and the host
  computes their fields as `getFluidFields()` does and sends them, so that the copies are identical bit for bit. The
  kernels of every platform give the fields and the state of the domain of the rank only (with its halo on request), so
  a rank holds those of the whole lattice only when they are gathered on it.
- With `OPENMM_LBM_MPI`, the test `TestMPIReferenceLBMForce`, which `ctest` runs under `mpiexec` with two ranks.
- The fluctuating fluid with more than one domain validated statistically (`docs/validation.md`): variances and spectra
  of the fluctuations as with one domain, no difference at the planes of the borders of the blocks (within 2e-4 of
  their value), independent random numbers on the ranks, and the temperature of coupled particles that cross the
  domains unchanged. `python/tests/mpi_decomposition.py` checks the independence of the ranks.
- Checkpoint files that any decomposition can load: `saveCheckpointFile()` and `loadCheckpointFile()`. One file for all
  the ranks, written and read with collective MPI-IO, each rank the populations of its domain at their place in the
  lattice, with the particles and, for every rank, its OpenMM checkpoint and its part of the state of the force.
  Loaded with the same decomposition the run continues bit for bit; with another one, another number of ranks or one
  domain, the fluid is restored exactly, the particles exactly in double precision and to float rounding in mixed and
  single precision (OpenMM keeps the positions in float), and the random numbers continue from the generators of the
  new Context. With one domain no MPI is needed, and the file can be continued with several domains.
  `openmmlbm.saveCheckpoint()`, `loadCheckpoint()` and `LBMCheckpointReporter` use it with more than one domain; with
  one domain they write and read the same files as before. `LBMForce.createCheckpoint()` (the state of one domain)
  gives an error with more than one domain.
- `LBMVTKReporter` with more than one domain: the files of the fluid are the same as with one domain, written together
  by all the ranks with MPI-IO (`writeFluidFile()`), and rank 0 writes the particles and the list of files.

- CUDA platform with more than one domain: with an MPI library that reads and writes device memory (CUDA-aware MPI)
  the populations exchanged between the ranks of the same node go from device to device, over NVLink, while the
  interior of the block collides; between nodes they go through the host, which was faster.
  `OPENMM_LBM_DEVICE_MPI=0` makes them go through the host also within a node. Only the first Context of a process
  uses it (the MPI library binds these transfers to the CUDA context of the first one). The populations exchanged
  through the host are in their own precision and without copies on the host. `devtools/benchmark_decomposition.py`
  measures the time per step (`docs/validation.md`), also with coupled particles (`--particles`, `--centered`).

- A job of the continuous integration builds with MPI and runs the decomposition on CPUs (Reference and OpenCL, two
  to eight ranks); the decomposition on GPUs is tested by hand before every release (`CONTRIBUTING.md`).
- With fluid fluctuations at a temperature above zero, the Context prints $`k_BT`$ in lattice units and the thermal
  Mach number $`\sqrt{3k_BT}`$ when it is created, and warns if $`k_BT`$ exceeds 1/3000, the largest value validated
  (`docs/theory.md`, section 7, Size of the fluctuations).

### Changed
- The preset `fabio-g30` of `examples/cocomo/diffusion.py` (SOD1 in a box of 30 nm, friction 30/ps) is now called
  `sod1-g30`.
- `LBMVTKReporter` writes the density and the velocity of the fluid in two files, `<prefix>_density_<step>.vti` and
  `<prefix>_velocity_<step>.vti` (each with the solid nodes, so that it opens alone), instead of one
  `<prefix>_fluid_<step>.vti`; `density=False` or `velocity=False` leaves one of them out (`fluid=False` both). The
  values are the same as before, byte for byte.

### Fixed
- All platforms: a lattice of more than 2147483647 nodes, or a domain whose 19 populations have $`2^{31}`$ entries or
  more (more than 113025455 nodes, for example $`512^3`$ on one GPU), overflowed the 32-bit indices of the plugin; on
  the GPUs the Context failed with `CUDA_ERROR_OUT_OF_MEMORY`. Such a lattice is now an error that names the limit
  when the Context is created (`test_lattice_too_large`). Affected since version 0.1.0.
- With the domain decomposition on several nodes, UCX printed hundreds of errors (`cudaHostUnregister() failed`,
  `failed to dereg from md[3]=cuda_cpy`) at the end of every run, after the work was done: MPI was finalized at the
  exit of the process, after OpenMM had destroyed the CUDA contexts in which UCX had registered the host buffers of
  the exchange. The Python module now finalizes MPI, if the plugin initialized it, as soon as the script ends.
- CUDA, OpenCL and HIP platforms: a lattice whose populations take more than 2 GB on one GPU (more than 14.1 million
  nodes in mixed and double precision, 28.3 million in single precision, for example $`256^3`$) failed when the
  Context was created, with `CUDA_ERROR_INVALID_VALUE` on CUDA: OpenMM computes the size of an upload as an `int`.
  The populations are now uploaded in parts. Affected since version 0.1.0.
- Reference platform: on the nodes of `Density` faces the velocity of the node used to rebuild the populations
  (along the face, and in the time filter across it) included the reaction of the coupled particles whose nearest
  node is a face node, while the CUDA, OpenCL and HIP platforms use $`(\mathbf j + \rho\mathbf g/2)/\rho`$, the
  velocity of `getFluidFields()`. The Reference platform now uses the same velocity: the rebuilt populations are
  those of a node beyond the face, on which the particles do not act. Runs without coupled particles on the nodes of
  `Density` faces are unchanged, bit for bit. `test_coupling_agrees_with_reference` has a case with open faces and a
  particle on a `Density` face.

### Documentation
- A new page of the user guide, `docs/user_guide/parallel.md`: when the decomposition pays, building with MPI, a
  complete script, job scripts for one and several nodes, the choice of the InfiniBand ports. With coupled particles
  and more than one domain `getState(getEnergy=True)` is a collective call, since the `VerletIntegrator` needs the
  forces for the kinetic energy: a `StateDataReporter` with the temperature must be on every rank.
- The documentation reviewed against the code: `docs/theory.md` section 8, `docs/architecture.md`, the API
  reference, troubleshooting (the errors of the decomposition and of the checkpoint files), `docs/validation.md`
  (every test file, the command lines of the MPI scripts, the cost of dividing x and of the coupled particles
  explained); the environment set up on the compute node after the session or the job has started; references to
  intermediate versions removed.
- The regularized walls and the open faces are the thread-safe boundary condition of M. Lauricella et al.,
  Phys. Fluids 37, 072111 (2025), appendix (introduced by A. Montessori et al., Phys. Fluids 36, 035171, 2024), a
  non-equilibrium extrapolation of Guo, Zheng and Shi written for the post-collision populations: cited in
  `docs/theory.md`, section 1, in the API reference, the glossary and `LBMForce.h`.
- Corrections after a review against the code: with regularized walls the wall lies on the solid nodes (comment of
  `setSolidNodes()`); the particles give momentum to the walls only through their reaction at solid nodes and
  their reflections, since the boundary nodes are fluid nodes (`getWallForce()`); the random numbers on the GPU
  platforms take 64 bytes per node and per boundary node; the random part of a rebuilt population has the
  amplitude of the density of the node; the mass balance of a face node next to a solid node; the direction of the
  velocity in the time filter of the `Density` faces; the definition of the errors of the two walls and the
  density in the middle of the duct in `docs/validation.md`; comments of the code that still described the
  boundary nodes of the first version.
- The formulas of the documentation (`README.md`, `CHANGELOG.md`, `docs/`, `examples/README.md`) are written in
  LaTeX math that GitHub renders (inline formulas, and display formulas in `math` blocks or, inside list items,
  where GitHub shows a `math` block as code, in display style on a line of their own). Notation only: the content
  is unchanged.

## 0.3.0 (2026-10-08)

Thermal fluctuations of the fluid, regularized walls and open faces, on all platforms; VTK output from Python. With
the fluctuations off, periodic faces and the default bounce-back walls, runs on the Reference platform are
identical, bit for bit, to those of version 0.2.1 (walls, coupled particles, both drags, EM and NVE).

### Added
- Thermal fluctuations of the fluid, `setFluidFluctuations(true)`, off by default, on all platforms: the ghost-mode
  filtered fluctuating lattice Boltzmann model (J. Chem. Phys. 164, 194905, 2026) on the orthogonal D3Q19 basis of
  Lulli et al. (Phys. Rev. E 109, 045304, 2024). Every collision adds to the populations of each node a random part
  that conserves its mass and momentum, with the equilibrium variance on the six stress modes, which relax with
  $`\omega`$, and on the nine ghost modes, which relax with rate 1. The fluid fluctuates at the temperature of
  `setTemperature()`, also with the `NVE` scheme. The random numbers come from the generator of the force on the
  Reference platform and from OpenMM's generator on the CUDA, OpenCL and HIP platforms (64 bytes per node and per
  boundary node). Without fluctuations, or at zero temperature, the run is identical, bit for bit, to that of
  version 0.2.1. Serialization version 5 and checkpoint version 3 record the switch; older files are read without
  fluctuations. `docs/theory.md`, section 7. Validated on an NVIDIA A100 (`docs/validation.md`): with the centred
  drag the coupled particles have the set temperature, their velocity autocorrelation equals the response to a kick
  and their diffusion coefficient follows the Einstein relation; the explicit drag makes them too hot, by about
  $`\text{friction}\times\Delta t\times m/(2m_c)`$, so the documentation recommends the centred drag with the
  fluctuating fluid, and a warning on stderr gives this estimate when a Context is created with fluid fluctuations,
  the explicit drag, a friction and coupled particles with the `EulerMaruyama` scheme at $`T > 0`$. Close to
  $`\tau = 1/2`$ the fluctuating fluid is unstable ($`\tau \le 0.501`$ at $`k_BT = 1/3000`$ in lattice units). The
  bounce-back walls are in exact thermal equilibrium with the fluctuating fluid: they return the fluctuations they
  receive (thermal accommodation coefficient zero), and there is no accommodation parameter (`docs/theory.md`,
  section 7, Walls).
- `examples/cocomo/diffusion.py --fluid-fluctuations`.
- A test of the spectrum of the velocity fluctuations on all platforms (`testVelocitySpectrum`): the velocity of
  `getFluidFields()`, minus the mean velocity of each sample, is Fourier transformed and its longitudinal and
  transverse parts are compared with equipartition, at rest, in a uniform flow and with a body force.
- `openmmlbm.LBMVTKReporter`, a reporter for `openmm.app.Simulation` that writes the fluid (density and
  velocity, solid nodes) and the particles (positions, velocities, masses, coupled or not) in VTK files for
  ParaView, in OpenMM units (nm, Da/nm³, nm/ps), with a `.pvd` file for the series; `--vtk N` in
  `examples/cocomo/diffusion.py`. Each file records its units, in a comment at the top and in the field data
  array `units` ("density: Da/nm^3", ...), which ParaView shows in its Information panel.
- Regularized walls, `setWallScheme(LBMForce.Regularized)`, on all platforms (the default stays the bounce-back of
  version 0.2). The fluid nodes next to the solid nodes rebuild the populations that come from the solid nodes as
  those of fluid at rest on the solid node,
  $`f^{\mathrm{eq}}(\rho_b, 0) + (1 - \omega) f^{\mathrm{neq}}(\Pi^{\mathrm{neq}}\ \text{of the node})`$ + the Guo
  term, plus a random part of their own with a fluctuating fluid, with $`\rho_b`$ from the mass balance of the
  rebuilt links, so the mass is conserved exactly; the other populations, and the collision of these nodes, are
  those of any fluid node. The wall lies on the solid nodes (halfway with bounce-back). Both walls are second order;
  bounce-back is exact for Poiseuille flow at $`\tau = 7/8`$, the regularized wall at $`\tau = 1`$, and the
  bounce-back is about four times more accurate at the $`\tau`$ of water. With fluid fluctuations the fluctuations
  are at equilibrium from the second node on. A first version with the local regularized boundary condition of Latt
  was replaced: imposing the velocity of the wall on the boundary node put the fluctuations next to it 3 to 9 %
  below equilibrium. Without the mass balance a fluctuating fluid lost half of its mass in 10000 steps
  (`docs/validation.md`). Serialization version 6 and checkpoint version 4 record the scheme; older files are read
  with bounce-back. `docs/theory.md`, section 1. Checkpoint version 5 also records the types of the faces, and
  refuses a Context with other faces; older checkpoints are read with periodic faces. Checkpoints and
  `setFluidState()` restart a run exactly with every combination of fluid fluctuations, walls and open faces.
- Open faces, `setFaceBoundary(face, LBMForce.Velocity | LBMForce.Density)` with `setFaceVelocity()` and
  `setFaceDensity()`, six independent faces, on all platforms: inlets, outlets, moving plates and
  flows driven by a pressure difference. As for the regularized walls, the nodes of a face rebuild the populations
  that come from beyond the face as those of fluid with the moments of the node and the velocity of a Velocity
  face ($`\rho_b`$ from the mass balance with the inflow, as Zou and He) or the density of a Density face; the
  velocity or the density of a face holds one node beyond it. The Density faces filter their velocity across the
  face in time, which damps the staggered mode that the lattice otherwise keeps; a Density face used as an inlet
  is unstable at $`\tau \le 0.55`$ with a difference of density of 1 %, so the documentation recommends a Velocity inlet
  at small $`\tau`$. The velocities and
  densities can be changed with `updateParametersInContext()`. Serialization version 7. With open faces the
  removal of the fluid momentum must be off. Examples: Couette flow and a duct driven by a pressure difference
  (`docs/user_guide/examples.md`).
- On the CUDA, OpenCL and HIP platforms the regularized walls and the open faces have the arithmetic of the
  Reference platform: the boundary nodes are found by the same code (`internal/LBMBoundaries.h`) and rebuilt by
  the kernel `applyBoundaries`, one thread per node, with four float4 of OpenMM's random numbers per boundary node
  for a fluctuating fluid. All the tests of the walls and of the faces run on every
  platform, and `test_fluid_agrees_with_reference` compares the GPU platforms with the Reference platform also
  with regularized walls and with open faces.

### Changed
- Build: the OpenMM library and the platform libraries are linked by their full path in `OPENMM_DIR`, for the
  plugin, the tests and the Python module, instead of `-lOpenMM` with the search path, in which the linker
  flags of an active conda environment (`LDFLAGS`) put the environment, which may hold another OpenMM, before
  `OPENMM_DIR`. CMake warns if the linker flags contain a folder with another OpenMM library, which the
  programs could load at run time. `make PythonInstall` builds the plugin library first. Section 12 of
  `docs/user_guide/installation.md` documents what CMake checks, the build options and their messages.
- Python: the setters with units (`setFluidDensity()`, `setKinematicViscosity()`, `setFriction()`,
  `setTemperature()`, `setBodyAcceleration()`, `setInitialFluidVelocity()`, `setFaceVelocity()`,
  `setFaceDensity()`) convert a `Quantity` to the unit of the method and raise `TypeError` if its unit does not
  convert. Before, OpenMM's typemaps stripped it in the MD unit system without a check, and a density in g/cm³
  became 1e-21 Da/nm³ without an error.
- The halfway bounce-back (`BounceBack` walls) is done by the fluid nodes next to the walls: after the streaming
  each of them takes back, in the opposite direction, the population that it built for a direction towards a
  solid node from its own moments, force and random part, instead of a pass over the solid nodes that wrote into
  their fluid neighbours. The results are identical bit for bit on all platforms and precisions
  (`docs/theory.md`, section 1; `docs/validation.md`, Walls).

## 0.2.1 (2026-10-07)

A fix of the build: CMake builds only the platforms that the OpenMM in `OPENMM_DIR` has. The physics and the
results are those of version 0.2.0.

### Fixed
- Build: a GPU platform is built only if the OpenMM in `OPENMM_DIR` has it, that is its header
  (`include/openmm/opencl/OpenCLContext.h`, `cuda/CudaContext.h`, `hip/HipContext.h`) and its library (in `lib` or
  `lib/plugins`), besides the toolkit that compiles it. Before, CMake looked only for the toolkit on the system, so with
  an OpenMM compiled from source without OpenCL, on a system with OpenCL, `cmake` succeeded and `make` stopped with
  `fatal error: openmm/opencl/OpenCLContext.h: No such file or directory`. Versions 0.1.0 and 0.2.0 are affected. The
  Python wrapper is built only if Python imports the OpenMM module, NumPy, setuptools and pip, SWIG is found, and OpenMM
  has its SWIG files and the headers of its plugins. CMake says why it leaves a part out and prints the list of what it
  builds; an option set to `ON` for a part that cannot be built stops `cmake` with an error, instead of the build. The
  tests of a platform that is not built are not built.

## 0.2.0 (2026-10-07)

The centred drag on all platforms, beside the explicit drag of version 0.1.0, which stays the default and is
unchanged bit for bit; and the fix of the repeated force evaluations on the GPU platforms, the known issue of
version 0.1.0.

### Added
- Centred drag, `setDragScheme(LBMForce.Centered)`, beside the explicit drag of version 0.1.0 (`LBMForce.Explicit`,
  the default, unchanged bit for bit). The drag compares the velocities of particle and fluid at the time of the
  force, the fluid velocity being the one that the collision puts in the equilibrium; the implicit system is solved
  in closed form for all the particles of a node, and conserves the total momentum exactly. It is stable for any
  friction, and its right temperature is that of the velocities of the State (half steps; `docs/theory.md`, section
  2). It requires `LBMForce` to be the last force of the System and no virtual sites. All platforms: on CUDA, OpenCL
  and HIP it runs in a `ForcePostComputation`, which reads the other forces at the end of the force evaluation; it
  costs 0 to 6% more than the explicit drag on an A100. With the fluid of this version, which has no thermal
  fluctuations, its particles are colder than with the explicit drag, the more so the larger
  $`\text{friction}\times\Delta t`$ and the bead mass in cell masses (SOD1 with COCOMO2: 7% below $`T`$ at 10/ps and
  16% at 30/ps, against 1 to 2%), while the diffusion coefficient is the same: the explicit drag stays the default,
  and the centred one is for the cases where the explicit drag is unstable (`docs/user_guide/lattice.md`, choosing
  the drag).
- `openmmlbm.LBMTemperatureReporter`: temperature of the coupled particles with the velocity that has the
  right temperature for the drag scheme.
- Option `--drag Explicit|Centered` in the examples with particles; section "Choosing the drag" of the user
  guide.
- Validation of the centred drag (`docs/validation.md`): stochastic tests T2, T6 and T7 (with the kick response
  of the same drag), SOD1 and the friction of 30/ps, 64 copies of SOD1 (7040 beads), energy budget, GPU
  platforms against the Reference platform.
- Serialization version 4 (the drag scheme; versions 1 to 3 are read with the explicit drag) and version 2
  of the checkpoint header of `LBMForce::createCheckpoint()` (the drag scheme; version 1 is read with the
  explicit drag, and a checkpoint is refused by a Context with the other drag scheme).

### Fixed
- CUDA, OpenCL and HIP: when OpenMM repeated the force evaluation of a step to enlarge the neighbor list of a
  nonbonded force (Systems of more than about 1250 atoms), the repeated evaluation applied the coupling force
  of the next step instead of that of the step, so particles and fluid received different momenta in that
  step (1.3e-2 of the momentum of the particles in the test that compresses 8000 particles). It now applies
  the force of the step again (`testRepeatedForceEvaluation`).

### Changed
- The warnings for $`\tau > 1.7`$ (small or negative self-mobility) and for $`\text{friction}\times\Delta t > 1`$
  are printed only with the explicit drag: with the centred drag the self-mobility is positive at every $`\tau`$ and
  the drag is stable for any friction (`docs/theory.md`, section 2; `testSelfMobilityWarning`,
  `testFrictionWarning`).
- README: the fluid is described as thread-safe; full name of Luis Enrique Coronas-Serna in README,
  `CITATION.cff` and the examples.
- `CITATION.cff`: the preferred citation is the article on openmm-lbm (in preparation), and the references
  are the ghost-mode filtered fluctuating lattice Boltzmann method (J. Chem. Phys. 164, 194905, 2026),
  LBsoft (Comput. Phys. Commun. 256, 107455, 2020) and the thread-safe lattice Boltzmann method (J. Comput.
  Sci. 74, 102165, 2023).
- `docs/theory.md` and `CONTRIBUTING.md` cite Kassen, Shankar and Fogelson (2022) for the sorting of keys
  and the segmented reduction of the per-cell reaction.
- `python/tests/TestEnergyBudget.py` takes the coupling force from the change of velocity of the bead, so
  that the budget holds for both drags.

## 0.1.0 (2026-10-07)

First release. A native OpenMM plugin (OpenMM 8.3 to 8.6) with a D3Q19 regularized lattice Boltzmann fluid
coupled to the particles by friction and random force (Euler-Maruyama, or NVE at zero temperature), solid
walls with momentum exchange, checkpoints of the fluid, on the Reference, CUDA, OpenCL and HIP platforms. It
reproduces the reference CUDA library of the DragOpenMM project: trajectories of the deterministic tests within
1e-12 and the quantities derived from them within 1e-5,
stochastic tests (equipartition, diffusion, velocity autocorrelation) and the examples of that project
within their statistical error (`docs/validation.md`). Tested on NVIDIA A100 with OpenMM 8.3.1 and 8.6.1,
and in continuous integration with every minor version from 8.3 to 8.6 and a HIP build.

### Added
- Plugin structure following the OpenMM example plugin: API, Reference platform, common implementation
  for CUDA, OpenCL and HIP, serialization, Python wrapper (`openmmlbm`), tests.
- `LBMForce` with an OpenMM-style API: grid size, fluid density and viscosity, friction, temperature,
  random seed, body acceleration, initial fluid velocity, fluid momentum removal, coupled particles;
  access to the fluid with `getFluidFields()`, `getFluidState()` and `setFluidState()`.
- Checks at Context creation: `VerletIntegrator`, rectangular box with cubic cells, $`\tau > 1/2`$, valid
  and distinct coupled particles with positive mass.
- Conversion between OpenMM and lattice units in one place (`LBMForceImpl`).
- Fluid storage in the "mixed" type of the platform; equilibrium initial state; device kernel for
  the density and momentum of the fluid.
- CMake check for OpenMM 8.3 or later; supported range 8.3 to 8.6. Tested on NVIDIA A100 against
  8.3.1 and 8.6.1 (Reference, CUDA, OpenCL, Python), and in CI against every minor version. With the
  CUDA forward-compatibility libraries on the library path, the NVIDIA OpenCL driver crashes
  intermittently while compiling kernels, also without the plugin: the OpenCL tests run without them.
- CMake check that SWIG has the version used for the OpenMM Python module.
- HIP platform compiled and linked against ROCm 6.3 (no AMD GPU available for tests yet).
- Fluid update on the Reference platform: moments, removal of the fluid momentum, regularized collision
  with Guo forcing (prefactor 1/2) and push streaming, with the conventions of the reference library.
  The fluid advances once per integration step, triggered by `updateContextState()`; other force
  evaluations do not advance it.
- Stability checks (`docs/theory.md`, section 6): Mach number of the fluid checked every N steps
  (`setMachCheckFrequency()`, default 100) against a limit (`setMachNumberLimit()`, default 0.3), with an
  exception above it; `getFluidMachNumber()`; warning at Context creation for $`\tau`$ outside [0.505, 2];
  `getLatticeParametersInContext()`; CMake option `LBM_DEBUG` for debug diagnostics. Serialization
  version 2 stores the new parameters and still reads version 1.
- Solid nodes set at run time (`setSolidNodes()`) with halfway bounce-back, on the Reference platform;
  stored in the serialization. The Poiseuille profile matches the exact solution of the scheme.
- Fluid tests (`tests/TestLBMFluid.h`): steady uniform flow, conservation of mass and momentum, body
  force at lattice densities 0.98, 1 and 1.02, timing of the momentum removal, viscosity from the decay
  of a shear wave, queries that must not advance the fluid. See `docs/validation.md`.
- User guide (`docs/user_guide/`): getting started, the lattice and the choice of the parameters, a
  reference of every method of `LBMForce`, complete Python examples and troubleshooting.
  `devtools/check_user_guide.py` runs every example on the Reference platform and checks its output.
- The removal of the fluid momentum and the Mach number check are timed by the step count of the
  Context, which checkpoints restore: a run restarted from a checkpoint, with the fluid restored by
  `setFluidState()`, is identical to an uninterrupted run at any restart step
  (`testRestartFromCheckpoint`).

- Particle-fluid coupling on the Reference platform: explicit Euler-Maruyama drag and random force at
  the nearest node, the opposite force on the fluid (summed per node in particle order), only for the
  particles added to the force; coupled particles reaching a solid node while moving into the wall have
  their velocity reversed;
  forces computed once per step and reused by other force evaluations; a random generator owned by the
  force; warning for $`\text{friction}\times\Delta t > 1`$. Tests in `tests/TestLBMCoupling.h`. Documentation:
  `docs/theory.md` section 2, the user guide (two new examples, including a reporter of the full-step
  temperature).

- Force on the walls, `getWallForce()`: the momentum given to the solid nodes in the last step by
  bounce-back (momentum exchange method of Ladd) and by the coupled particles (reaction at solid nodes,
  reflections), divided by $`\Delta t`$. Tests: momentum of particles, fluid and walls conserved; in steady
  Poiseuille flow the force on the walls equals the body force on the fluid.

- Warning at Context creation on the CUDA, OpenCL and HIP platforms when particles are coupled, while the
  coupling was implemented only on the Reference platform (removed when the coupling was ported, below).

- Equivalence of the coupling with the reference library (E0, 80 cases at $`T = 0`$): trajectories within
  5e-13, derived quantities within 2e-5. Measured self-mobility as a function of $`\tau`$
  (`docs/theory.md`, section 2), and a warning for coupled particles when $`\tau > 1.7`$. Equipartition test.

- The populations are stored as deviations from the rest equilibrium, $`f_q - w_q`$, on all platforms, so
  that small hydrodynamic signals keep the full precision of the type. The fluid state of
  `getFluidState()`/`setFluidState()` is made of these deviations (a population is the value plus $`w_q`$);
  saving and restoring it remains exact. In double precision the rounding drift of the total momentum on a
  $`64^3`$ lattice drops from 3e-5 to 1e-8 (E0 repeated: all 80 cases still agree with the reference library).

- Coupling schemes, `setCouplingScheme()`: `EulerMaruyama` (default) or `NVE`, friction only without random
  force. Saved by the serialization (version 3, which still reads versions 1 and 2).

- Fluid update on the CUDA, OpenCL and HIP platforms, with the arithmetic of the Reference platform:
  moments, removal of the fluid momentum (two-stage reduction without atomic operations), regularized
  collision with Guo forcing and push streaming, and the Mach number check. The fluid tests run on every
  platform and precision mode, with tolerances that follow the precision; in mixed and double precision
  the fluid agrees with the Reference platform to 1e-14 (`test_fluid_agrees_with_reference`).

- Solid nodes on the CUDA, OpenCL and HIP platforms: halfway bounce-back with one thread per solid node,
  and the force on the walls (`getWallForce()`) from the momentum exchange, with the static pressure of the
  weights computed once in double precision. The wall tests run on every platform; in mixed and double
  precision the fluid and the force on the walls agree with the Reference platform to 1e-14. On every
  platform the bounce-back now processes only the links from solid to fluid nodes, which leaves the fluid
  and the force on the walls unchanged and makes the populations stored at solid nodes independent of the
  order of the solid nodes.

- Particle-fluid coupling on the CUDA, OpenCL and HIP platforms, with the arithmetic of the Reference
  platform: drag and random force at the nearest node (random numbers from OpenMM's generator), the
  reactions summed per node in particle order without atomic operations (keys sorted with OpenMM's
  `ComputeSort`, one writer per node), reflection of the particles at walls and their contribution to
  `getWallForce()`, forces computed once per step and added at every force evaluation. The coupling tests
  run on every platform and precision mode; at $`T = 0`$ in double precision particles and fluid agree with the
  Reference platform to 2e-11 (`test_coupling_agrees_with_reference`). 61 us per step for 110 particles on
  a $`30^3`$ lattice on an NVIDIA A100, against 517 us for the reference library. The warning of the GPU
  platforms is removed.
- On the GPU platforms the generator of OpenMM is consumed as in the reference library (one float4 per
  padded atom per step), so that runs with the random force can be compared with it step by step.
- `examples/`: kick of a bead with and without the fluid, thermalization of beads, a bead in a uniform
  flow (NVE), a fluid started from a shear wave with saving and loading of the state, and a plotting
  script. They are ports of the examples of the DragOpenMM plugin with the Euler-Maruyama coupling, and
  reproduce its trajectories (`examples/README.md`). A Python test runs each of them for a few steps.
- `examples/cocomo/`: the COCOMO2 model of proteins written from its article (`cocomo2.py`; on SOD1 its
  energies and forces equal those of the COCOMO2 script of the DragOpenMM project), the diffusion of
  SOD1 and of a disordered protein with and without the fluid (`diffusion.py`, with the parameter sets
  of the DragOpenMM runs), kicks of a peptide and of ubiquitin (`kick.py`), and the diffusion
  coefficient of the centre of mass (`msd.py`).
- CMake reads the version of OpenMM from its library and stops with an error that names it below 8.3, or
  warns above 8.6, the newest tested version (before, only the presence of `ComputeSort.h` was checked).
- Checkpoints of the fluid and of the coupling: `LBMForce::createCheckpoint()` and `loadCheckpoint()` (bytes
  in Python) write what OpenMM checkpoints miss (populations, random numbers already drawn, wall force,
  the generator of the Reference platform), and `openmmlbm.saveCheckpoint()`, `loadCheckpoint()` and
  `LBMCheckpointReporter` keep them with an OpenMM checkpoint in one file. A restarted run is identical, bit
  for bit, to an uninterrupted one, also with the random force. `examples/cocomo/diffusion.py` has the
  options `--checkpoint` and `--restart`. New page of the user guide: `docs/user_guide/restart.md`.
- Between integration steps the coupling force is that of the next step, as OpenMM does for every force, instead of
  that of the last step. With `VerletIntegrator` the kinetic energy of a State, and so the temperature of
  `StateDataReporter`, is then that of the full step also for coupled particles (before, 363 K at 300 K for a free
  particle with $`\text{friction}\times\Delta t = 0.1`$). The random numbers of a step are drawn once and the
  trajectories are unchanged. Test `testFullStepKineticEnergy` on every platform.
- Approximate kinetic energy budget with the NVE scheme (kinetic energy plus viscous and drag dissipation, valid to
  $`O(\mathrm{Ma}^2, \mathrm{Kn}^2)`$), described in `docs/theory.md` and checked by
  `python/tests/TestEnergyBudget.py`.
- User guide: step-by-step installation of conda, OpenMM and the plugin for beginners
  (`docs/user_guide/installation.md`), and a tutorial that teaches OpenMM and the plugin through the
  examples (`docs/user_guide/tutorial.md`).

### Not yet implemented
- Thermal fluctuations of the fluid (fluctuating lattice Boltzmann): without them the diffusion coefficient
  of the coupled particles stays close to $`k_BT/(m\,\text{friction})`$ (user guide, limitations of the model).
- Time-centred drag.
- Open faces with imposed density or velocity (`docs/theory.md`, solid nodes).
- Tests of the HIP platform on AMD GPUs (it is built in continuous integration, not run).
