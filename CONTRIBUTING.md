# Contributing to openmm-lbm

These rules apply to everyone who works on this repository. They exist so that the code stays correct,
reproducible and understandable by someone who arrives without any prior context.

## Approvals and procedure

- The maintainer is Marco Lauricella. His explicit approval is required for:
  - pushes to `main`, tags and releases;
  - changes to the physics or the numerics;
  - changes to the public API;
  - changes to the repository visibility.
- API and release decisions are discussed with the other authors.
- Work in small steps: observe, verify, change one thing, verify again.

## Language and style

- **Language.** English for code, comments, documentation, commit messages and issues.
- **Code style.** Follow OpenMM and the official example plugin (`openmm/openmmexampleplugin`):
  - C++ as in OpenMM;
  - classes in `CamelCase`, methods in `camelCase`;
  - Doxygen comments in public headers;
  - 4-space indentation;
  - one public class per file (small helper classes may live in the file that uses them);
  - test programs named `Test*.cpp`, with a `main()` that returns 0 on success and 1 on failure.
- **Comments.** Match the density of the surrounding code and explain why, not what.

## Physics and numerics

- **Reference implementation.** The algorithm is the one of the validated CUDA lattice Boltzmann library (`libc-cuda`) of the DragOpenMM project:
  - D3Q19 with regularized, thread-safe collision (populations rebuilt from moments up to second order);
  - Guo forcing with prefactor 1/2;
  - exact weakly compressible density convention;
  - explicit Euler–Maruyama particle–fluid coupling at the half step.

  Until equivalence is reached, any difference from the reference values of that library in double precision is a defect, unless a written decision says otherwise. The centred drag (`LBMForce::Centered`, version 0.2.0) is an extension beyond that library, derived in `docs/theory.md` (section 2) and validated in `docs/validation.md`; the explicit drag stays identical to it.
- **Changes to the physics** (fluctuating LB, interpolation, boundaries; the centred drag followed this rule) require three things:
  - a written derivation in `docs/theory.md`;
  - a test that checks it;
  - a comparison with reference values whose provenance is recorded (run, date, version).
- **Units.** The public API uses OpenMM units only. Lattice units stay internal; the conversion is documented in `docs/theory.md` (section 3) and done with the same factors on every platform (`docs/architecture.md`, Invariants).

## Tests and acceptance

- **All platforms aligned.** Every feature, physical or not, is implemented and tested on every platform: Reference, CUDA, OpenCL and HIP (HIP at least built, where no AMD GPU is available). A feature available on one platform only may exist on `develop` as an intermediate step, with a clear error on the other platforms, but it does not reach `main` or a release. The same tests run on every platform, with tolerances tied to the precision mode.
- **Everything is documented**, including results that are negative or contradict an expectation, with their numbers and protocol (`docs/theory.md`, `docs/validation.md`).
- **Green tests before merging.** No commit reaches `main` with failing tests. Tests run on every available platform (Reference, CUDA, OpenCL, and HIP where possible), in the `single`, `mixed` and `double` precision modes.
- **Every new feature comes with a test**, and every fixed defect with a regression test.
- **Reference values** (smoke tests, single-particle drag and finite-size mobility, …) live in a versioned file, with tolerances and provenance.
- **Continuous integration.** GitHub CI builds every platform against each supported OpenMM version, runs the C++ tests of serialization, Reference and OpenCL on CPU (PoCL) and the Python tests, and builds the HIP platform. The CUDA platform is built but not run there. GPU tests are run on a GPU machine, and their outcome is recorded (machine, date, version).
- **OpenCL on NVIDIA GPUs with an older driver.** When the CUDA forward-compatibility libraries (`cuda-compat`) are on `LD_LIBRARY_PATH`, which the CUDA platform needs if the driver is older than the CUDA version of OpenMM, the NVIDIA OpenCL driver crashes intermittently while compiling kernels, also without this plugin. Run the CUDA tests with those libraries and the OpenCL tests (`ctest -R OpenCL`, `pytest -k OpenCL`) without them.

## Supported OpenMM versions

- **Supported range: OpenMM 8.3 to 8.6.** The range is declared in `README.md`. CMake reads the version of the OpenMM in `OPENMM_DIR` from its library (`Platform::getOpenMMVersion()`, with that library preloaded so that another OpenMM on the library path does not interfere), stops with an error below the minimum and warns above the newest tested version; it also always checks for `openmm/common/ComputeSort.h`. The minimum is 8.3 because the plugin uses `ComputeSort`, which entered the OpenMM common compute layer in 8.3.0.
- **What "supported" means.** A compiled plugin is tied to the OpenMM version it was built against. "Supported" therefore means that the plugin builds, and passes all tests, against every minor version in the range.
- **Test matrix.** Continuous integration builds and tests against every minor version in the range (Reference and OpenCL on CPU). GPU tests run on the minimum and on the maximum version.
- **New OpenMM releases** are added to the test matrix and, once the tests pass, to the supported range.
- **Raising the minimum** requires a written reason, recorded in `CHANGELOG.md`.

## Determinism and portability

- **No atomic operations in the plugin kernels.** Per-cell sums use sorting with unique keys followed by segmented reductions, with one writer per cell (as in A. Kassen, V. Shankar and A. L. Fogelson, Int. J. High Perform. Comput. Appl. 36, 443 (2022); `docs/theory.md`, section 2). Given the same input and seed, results are bitwise identical on the same device. Tests use fixed seeds.
- **Portable kernels.** They are written only in the OpenMM common compute dialect (`platforms/common/src/kernels/*.cc`). No CUDA-specific code is allowed outside the kernel factories.
- **Dependencies.** OpenMM in the supported range (see above), CMake, SWIG, and Python with NumPy (pytest for the tests) only. No thrust or CUB.

## Licensing and code provenance

- **License header.** Every C++, kernel and Python source file starts with the MIT header (CMake files excepted):

  ```
  /* -------------------------------------------------------------------------- *
   *                                openmm-lbm                                  *
   * -------------------------------------------------------------------------- *
   * Copyright (c) 2026 the Authors (see README.md).                            *
   * SPDX-License-Identifier: MIT                                               *
   * -------------------------------------------------------------------------- */
  ```

- **Files derived from `openmm/openmmexampleplugin`** keep its copyright notice (Stanford University and the Authors), as the MIT license requires.
- **Do not copy code from projects under incompatible licenses** (for example GPL or AGPL lattice Boltzmann codes such as Palabos). Ideas may be reimplemented from scratch and cited.
- **Third-party code** is accepted only under MIT-compatible licenses (MIT, BSD, Apache 2.0), with attribution.
- **Literature** is cited in `docs/` and in `CITATION.cff`.

## Git workflow

- **Branches.** `main` is stable and always green. Development happens on `develop` or on topic branches (`feature/...`), merged after the tests pass.
- **Commits.** Small and focused, authored by the person who signs them.
- **Do not commit** build artefacts, data, simulation output or system files.
- **Versions.** Semantic versioning (`0.x` until equivalence with the reference implementation), annotated tags, and a `CHANGELOG.md`.

## Documentation

The documentation is written for users and developers. It must be complete enough that a newcomer can build the plugin, run the tests and understand why the code is written the way it is, from these files alone:

| File | Content |
|---|---|
| `README.md` | what the plugin does, installation, minimal example, authors, license, how to cite |
| `docs/theory.md` | model (regularized D3Q19, weakly compressible, Guo forcing), Euler–Maruyama coupling with the explicit and the centred drag, time levels (leapfrog, half step), units and conversions, equations with references |
| `docs/architecture.md` | file map; data flow within one step; invariants (one fluid update per step, no atomics, kernel order); how to add a platform or a kernel |
| `docs/validation.md` | tests, reference values, tolerances, how to reproduce them |
| `CONTRIBUTING.md` | this file |
| `CHANGELOG.md`, `CITATION.cff` | version history; citation, with authors in the order of the README |

Rules for the documentation:
- Every non-obvious choice (physics, numerics, conventions, units, time levels) is written once, with its reason and its reference.
- It is updated in the same commit as the code that changes it.
- Only publishable content: no machine-specific paths, accounts or private e-mail addresses, and no unpublished results without the authors' agreement.
