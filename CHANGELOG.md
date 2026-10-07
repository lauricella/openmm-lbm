# Changelog

All notable changes to openmm-lbm are recorded here. Versions follow semantic versioning; in the 0.x
versions the API may still change.

## Unreleased

### Added
- Centred drag, `setDragScheme(LBMForce.Centered)`, beside the explicit drag of version 0.1.0
  (`LBMForce.Explicit`, the default, unchanged bit for bit). The drag compares the velocities of particle
  and fluid at the time of the force, the fluid velocity being the one that the collision puts in the
  equilibrium; the implicit system is solved in closed form for all the particles of a node, and conserves
  the total momentum exactly. It is stable for any friction, and the velocities of the State (half steps)
  have the right temperature of the drag (`docs/theory.md`, section 2). It requires `LBMForce` to be the
  last force of the System and no virtual sites. Reference platform only in this version.
- `openmmlbm.LBMTemperatureReporter`: temperature of the coupled particles with the velocity that has the
  right temperature for the drag scheme.
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
- The warning for tau > 1.7 (small or negative self-mobility) is printed only with the explicit drag: with the
  centred drag the self-mobility is positive at every tau (`docs/theory.md`, section 2).
- README: the fluid is described as thread-safe; full name of Luis Enrique Coronas-Serna in README,
  `CITATION.cff` and the examples.
- `docs/theory.md` and `CONTRIBUTING.md` cite Kassen, Shankar and Fogelson (2022) for the sorting of keys
  and the segmented reduction of the per-cell reaction.

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
- Checks at Context creation: `VerletIntegrator`, rectangular box with cubic cells, tau > 1/2, valid
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
  exception above it; `getFluidMachNumber()`; warning at Context creation for tau outside [0.505, 2];
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
  force; warning for friction*dt > 1. Tests in `tests/TestLBMCoupling.h`. Documentation:
  `docs/theory.md` section 2, the user guide (two new examples, including a reporter of the full-step
  temperature).

- Force on the walls, `getWallForce()`: the momentum given to the solid nodes in the last step by
  bounce-back (momentum exchange method of Ladd) and by the coupled particles (reaction at solid nodes,
  reflections), divided by dt. Tests: momentum of particles, fluid and walls conserved; in steady
  Poiseuille flow the force on the walls equals the body force on the fluid.

- Warning at Context creation on the CUDA, OpenCL and HIP platforms when particles are coupled, while the
  coupling was implemented only on the Reference platform (removed when the coupling was ported, below).

- Equivalence of the coupling with the reference library (E0, 80 cases at T = 0): trajectories within
  5e-13, derived quantities within 2e-5. Measured self-mobility as a function of tau
  (`docs/theory.md`, section 2), and a warning for coupled particles when tau > 1.7. Equipartition test.

- The populations are stored as deviations from the rest equilibrium, f_q - w_q, on all platforms, so
  that small hydrodynamic signals keep the full precision of the type. The fluid state of
  `getFluidState()`/`setFluidState()` is made of these deviations (a population is the value plus w_q);
  saving and restoring it remains exact. In double precision the rounding drift of the total momentum on a
  64^3 lattice drops from 3e-5 to 1e-8 (E0 repeated: all 80 cases still agree with the reference library).

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
  run on every platform and precision mode; at T = 0 in double precision particles and fluid agree with the
  Reference platform to 2e-11 (`test_coupling_agrees_with_reference`). 61 us per step for 110 particles on
  a 30^3 lattice on an NVIDIA A100, against 517 us for the reference library. The warning of the GPU
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
- Between integration steps the coupling force is that of the next step, as OpenMM does for every force,
  instead of that of the last step. With `VerletIntegrator` the kinetic energy of a State, and so the
  temperature of `StateDataReporter`, is then that of the full step also for coupled particles (before,
  363 K at 300 K for a free particle with friction*dt = 0.1). The random numbers of a step are drawn once
  and the trajectories are unchanged. Test `testFullStepKineticEnergy` on every platform.
- Approximate kinetic energy budget with the NVE scheme (kinetic energy plus viscous and drag dissipation,
  valid to O(Ma^2, Kn^2)), described in `docs/theory.md` and checked by `python/tests/TestEnergyBudget.py`.
- User guide: step-by-step installation of conda, OpenMM and the plugin for beginners
  (`docs/user_guide/installation.md`), and a tutorial that teaches OpenMM and the plugin through the
  examples (`docs/user_guide/tutorial.md`).

### Not yet implemented
- Thermal fluctuations of the fluid (fluctuating lattice Boltzmann): without them the diffusion coefficient
  of the coupled particles stays close to kT/(m friction) (user guide, limitations of the model).
- Time-centred drag.
- Open faces with imposed density or velocity (`docs/theory.md`, solid nodes).
- Tests of the HIP platform on AMD GPUs (it is built in continuous integration, not run).
