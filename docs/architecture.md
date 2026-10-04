# Architecture

openmm-lbm follows the structure of the OpenMM example plugin
(https://github.com/openmm/openmmexampleplugin). The HIP platform follows openmm-torch.

## Files

| Path | Content |
|---|---|
| `openmmapi/include/LBMForce.h`, `openmmapi/src/LBMForce.cpp` | public API: parameters, coupled particles, access to the fluid |
| `openmmapi/include/LBMKernels.h` | `CalcLBMForceKernel`, the interface every platform implements, and `LBMLatticeParameters` |
| `openmmapi/include/internal/LBMForceImpl.h`, `openmmapi/src/LBMForceImpl.cpp` | checks the setup and converts all parameters to lattice units once, for all platforms |
| `openmmapi/include/internal/D3Q19.h` | velocity set, weights, ordering of the populations, equilibrium (host code) |
| `platforms/reference/` | `ReferenceCalcLBMForceKernel`: plain C++ in double precision, the correctness reference |
| `platforms/common/` | `CommonCalcLBMForceKernel` and the device kernels (`src/kernels/*.cc`), written once in the OpenMM common compute dialect |
| `platforms/cuda/`, `platforms/opencl/`, `platforms/hip/` | only the kernel factories, which create `CommonCalcLBMForceKernel` with the context of the platform, and the tests |
| `serialization/` | XML proxy of `LBMForce` (parameters only) |
| `python/` | SWIG wrapper `openmmlbm` and its tests |
| `tests/TestLBMForce.h` | tests shared by all platforms; each platform has a `Test<Platform>LBMForce.cpp` |

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
2. **Force evaluation.** `LBMForceImpl::calcForcesAndEnergy()` checks that the step size has not
   changed and calls the kernel. The kernel will advance the fluid and add the coupling forces; in
   version 0.1.0 it adds nothing.
3. **Fluid access.** `getFluidFields()`, `getFluidState()` and `setFluidState()` go from `LBMForce`,
   through `LBMForceImpl`, to the kernel. The common implementation computes density and momentum on
   the device (`computeFluidMoments` in `lbmFluid.cc`), then converts them to OpenMM units on the host.

## Invariants

- **Unit conversion in one place.** It happens only in `LBMForceImpl::computeLatticeParameters()`;
  kernels receive lattice units.
- **Same conventions everywhere.** Every platform uses the population ordering and the node index of
  `D3Q19.h`, so `getFluidState()` is portable across platforms.
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
4. Add the same algorithm to `ReferenceCalcLBMForceKernel`, and a test in `tests/TestLBMForce.h`
   that compares the two.
