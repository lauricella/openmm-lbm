# Changelog

All notable changes to openmm-lbm are recorded here. Versions follow semantic versioning; 0.x
versions precede equivalence with the reference CUDA lattice Boltzmann library.

## Unreleased (0.1.0)

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
  8.3.1 and 8.6.1 (Reference, CUDA, Python; OpenCL on 8.6.1, because the OpenCL platform of
  OpenMM 8.3.1 itself crashes intermittently on that machine), and in CI against every minor version.
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
  particles added to the force; coupled particles reaching a solid node have their velocity reversed;
  forces computed once per step and reused by other force evaluations; a random generator owned by the
  force; warning for friction*dt > 1. Tests in `tests/TestLBMCoupling.h`. Documentation:
  `docs/theory.md` section 2, the user guide (two new examples, including a reporter of the full-step
  temperature, since the temperature reported by OpenMM is not valid for coupled particles).

### Not yet implemented
- Fluid update, solid nodes and particle-fluid coupling on the CUDA, OpenCL and HIP platforms.
- Open faces with imposed density or velocity (`docs/theory.md`, solid nodes).
