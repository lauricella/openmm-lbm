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

### Not yet implemented
- Fluid update (regularized collision with Guo forcing, streaming) and particle-fluid coupling:
  `LBMForce` applies no force in this version.
