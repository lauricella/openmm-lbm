# Installing OpenMM and openmm-lbm, step by step

This page installs everything from nothing: the conda package manager, OpenMM, the tools to build the
plugin, and openmm-lbm itself; then it checks the installation and runs a first example. It assumes
only that you can open a terminal and type a command. Nothing here needs administrator rights: all
the software goes into your home directory.

It takes about 30 minutes, most of it spent waiting for downloads and for the compiler.

When you have finished, continue with the [tutorial](tutorial.md), which teaches OpenMM and the plugin
through the examples.

## How to read this page

Commands are in grey boxes:

```bash
echo hello
```

Type each line in the terminal (or copy and paste it) and press Enter. Text after a `#` is a comment:
it explains the command and is ignored by the terminal, so you do not have to type it. Lines that
start with `...` in the expected outputs stand for lines that are not shown.

If a command prints an error, stop there and look at [When something goes wrong](#11-when-something-goes-wrong)
at the end of this page before going on: every later step depends on the previous ones.

## 1. What you need

- **A computer with Linux.** The plugin is developed and tested on Linux (x86_64).
  - On **Windows**, install the Windows Subsystem for Linux with Ubuntu (open PowerShell as
    administrator, type `wsl --install`, restart), open the "Ubuntu" application and follow this page
    inside it.
  - On **macOS** OpenMM works, but the plugin has not been tested there.
- **About 6 GB of free disk space** and an internet connection.
- **A graphics card (GPU) is optional.** Without one, everything runs on the processor with the
  `Reference` platform, which is correct but slow: it is enough to learn and to run the small examples.
  With an NVIDIA GPU the `CUDA` and `OpenCL` platforms are a hundred times faster or more. The GPU
  driver must already be installed (on a cluster it always is).
- **On a computing cluster**, read [section 10](#10-on-a-computing-cluster) first.

## 2. Look at your system

Open a terminal and type:

```bash
uname -m          # the processor type
nvidia-smi        # the NVIDIA GPU and its driver, if there is one
```

- `uname -m` prints `x86_64` on most computers.
- If `nvidia-smi` prints `command not found`, there is no NVIDIA GPU (or no driver): you will use
  the `Reference` platform, and you can skip everything about CUDA below.
- If it prints a table, look at its first line: it shows the driver version and `CUDA Version: X.Y`,
  the newest CUDA version that your driver supports. Write it down: you need it in step 4. For
  example, on a node with an older driver:

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.274.02             Driver Version: 535.274.02   CUDA Version: 12.2     |
...
```

## 3. Install conda (Miniforge)

OpenMM and its tools are installed with **conda**, a package manager that keeps programs in
*environments*: separate folders, each with its own Python and its own libraries. An environment can
be deleted without touching the rest of the computer. Miniforge is the small installer of conda that
uses the community package channel `conda-forge`, where OpenMM is published.

If the command `conda --version` already prints a version, conda is installed and you can go to
step 4. Otherwise download and run the installer:

```bash
cd ~
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

The installer asks a few questions:

1. It shows the license: press Enter, then the space bar to scroll to the end, type `yes` and press
   Enter.
2. It proposes a folder (`/home/<you>/miniforge3`): press Enter to accept it.
3. It asks whether to initialize conda in your shell: type `yes`.

Then **close the terminal and open a new one**, so that the shell reads the new settings. The prompt
now starts with `(base)`, the name of the active environment. Check that conda works:

```bash
conda --version
```

It prints a version number, for example `conda 26.1.1`; any recent version is fine.

## 4. Create an environment with OpenMM

The plugin is compiled against OpenMM, so the environment contains OpenMM and the tools to compile:
a C++ compiler, `cmake` and `make` to drive the build, `swig` to generate the Python module, NumPy
and pytest. openmm-lbm supports OpenMM 8.3 to 8.6; this page uses the latest, 8.6.1.

**SWIG must have the same version as the one used for OpenMM's own Python module**, otherwise the
plugin cannot be used from Python: OpenMM 8.6.1 needs `swig=4.5.1`, OpenMM 8.3.1 needs `swig=4.3.1`.
The build checks it and tells you if it is wrong.

Choose **one** of the two commands below and type it on a single line.

**Without an NVIDIA GPU:**

```bash
conda create -n lbm -c conda-forge python=3.12 openmm=8.6.1 swig=4.5.1 cmake make cxx-compiler numpy pytest git
```

**With an NVIDIA GPU:** OpenMM 8.6.1 exists for CUDA 12.9 and for CUDA 13. Choose with the
`CUDA Version` that `nvidia-smi` printed in step 2:

- 12.9 or a later 12.x: use the command below as it is (`cuda-version=12.9`);
- 13.0 or newer: replace `cuda-version=12.9` with `cuda-version=13`;
- older than 12.9: the driver is too old for OpenMM 8.6.1. Update it, or, on a cluster with
  data-centre GPUs, follow [section 10](#10-on-a-computing-cluster).

```bash
conda create -n lbm -c conda-forge python=3.12 openmm=8.6.1 swig=4.5.1 cmake make cxx-compiler numpy pytest git cuda-version=12.9 cuda-nvcc opencl-headers
```

conda lists the packages it will install and asks `Proceed ([y]/n)?`: type `y`. The download takes a
few minutes.

Then **activate** the environment. You must do this in every new terminal before using OpenMM:

```bash
conda activate lbm
```

The prompt now starts with `(lbm)`. One more setting keeps Python from mixing in packages that may be
installed in your home directory outside conda (a common source of strange errors); it is stored in
the environment, so you set it only once:

```bash
conda env config vars set PYTHONNOUSERSITE=1
conda activate lbm
```

## 5. Check OpenMM

OpenMM has a self-test that lists the platforms it can use and compares their forces:

```bash
python -m openmm.testInstallation
```

On a computer with an NVIDIA GPU the output looks like this:

```
OpenMM Version: 8.6.1
Git Revision: b399af4725573963b46d6c1083fdcf7a37615857

There are 4 Platforms available:

1 Reference - Successfully computed forces
2 CPU - Successfully computed forces
3 CUDA - Successfully computed forces
4 OpenCL - Successfully computed forces

Median difference in forces between platforms:

Reference vs. CPU: 6.302e-06
...
All differences are within tolerance.
```

- Without a GPU only `Reference` and `CPU` appear, sometimes `OpenCL` too: that is correct.
- `LBMForce` runs on `Reference`, `CUDA`, `OpenCL` and `HIP`. OpenMM's `CPU` platform is not
  supported by the plugin; on a computer without a GPU use `Reference`.
- If CUDA prints an error instead of "Successfully computed forces", see
  [CUDA errors](#11-when-something-goes-wrong).

## 6. Download openmm-lbm

```bash
mkdir -p ~/src
cd ~/src
git clone https://github.com/lauricella/openmm-lbm.git
```

This creates the folder `~/src/openmm-lbm` with the source code, the documentation and the examples.
While the repository is private, `git` asks for a GitHub user name and a token: ask the authors for
access.

## 7. Build and install the plugin

The plugin is compiled in a separate *build* folder, so the source folder stays clean. `$CONDA_PREFIX`
is a variable that holds the folder of the active environment: the plugin is installed there, next to
OpenMM, where OpenMM looks for plugins.

```bash
conda activate lbm
mkdir -p ~/src/openmm-lbm-build
cd ~/src/openmm-lbm-build
cmake ~/src/openmm-lbm -DOPENMM_DIR=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make -j4
make install
make PythonInstall
```

What each command does:

1. `cmake` looks for OpenMM, the compiler and SWIG, decides which platforms to build and writes the
   build instructions. Among its messages it prints the version of OpenMM it found, for example
   `-- OpenMM 8.6.1 in /home/<you>/miniforge3/envs/lbm`, says why it leaves out a platform, for example
   `-- HIP was not found on this system: the HIP plugin of openmm-lbm is not built.`, prints what it builds,
   `-- openmm-lbm 0.3.0: platforms to build: Reference, CUDA, OpenCL; Python wrapper: yes; MPI: no`, and it ends with
   `-- Build files have been written to: ...`. A GPU platform is built only if the OpenMM in `OPENMM_DIR` has
   it (an OpenMM compiled from source may lack OpenCL, CUDA or HIP) and the system can compile it. It stops with an error if OpenMM is older than 8.3 (the
   message gives the version found), and warns if it is newer than 8.6, the newest tested version. It
   also stops if the SWIG version is wrong: the message names the version of SWIG (major and minor) that
   made the OpenMM Python module; install it (for OpenMM 8.6.1 from conda-forge,
   `conda install -c conda-forge swig=4.5.1`) and run `cmake` again.
2. `make -j4` compiles, using 4 processor cores. It takes a few minutes and ends with
   `[100%] Built target ...`. Lines with `warning` are not errors; a line with `error` stops the build.
3. `make install` copies the libraries into the environment.
4. `make PythonInstall` builds and installs the Python module `openmmlbm`.

To see which platforms were built: the line `platforms to build` of `cmake`, or `cmake -LA . | grep LBM_BUILD`,
which prints `ON` or `OFF` for CUDA, OpenCL, HIP and the Python wrapper. The Reference platform is always
built. The tests of a platform that is not built are not built either. What `cmake` checks, its options and
its messages are in [section 12](#12-what-cmake-checks-and-the-build-options).

## 8. Run the tests

The tests check that the plugin computes what it should on every platform that was built:

```bash
cd ~/src/openmm-lbm-build
ctest --output-on-failure
```

On a computer with an NVIDIA GPU there are 8 tests (serialization, Reference, and OpenCL and CUDA in
single, mixed and double precision); they take a few minutes and end with:

```
100% tests passed out of 8
```

On a computer without a GPU the CUDA tests cannot run. Exclude them:

```bash
ctest --output-on-failure -E Cuda
```

Then the Python tests:

```bash
cd ~/src/openmm-lbm/python/tests
python -m pytest
```

The last line counts the tests, for example `50 passed, 31 skipped` on a computer without a GPU (the
numbers depend on the platforms and packages available). "Skipped" tests are those of
platforms that are not available on your computer: that is normal. "Failed" is not: see the next
section.

## 9. A first run

Check that Python finds the plugin:

```bash
python -c "import openmmlbm; print('openmm-lbm is installed')"
```

Then run a first example, which is fast even without a GPU. It puts a bead at rest in a fluid that
flows along x, and prints how the bead is dragged:

```bash
mkdir -p ~/lbm-runs
cd ~/lbm-runs
python ~/src/openmm-lbm/examples/particle/uniform_flow.py --platform Reference --steps 200
```

```
Platform Reference, box 6.00 nm, 20^3 nodes, dt 0.001 ps, tau 1, friction 10 1/ps (gamma dt 0.01)
Bead 16.1460 Da at rest, fluid 129167.7 Da at u0 = 0.09 nm/ps: final common velocity 0.089989 nm/ps
  step   time (ps)  v_bead/u0   p_bead/p0     p_total/p0 - 1   E_bead/E0     E_fluid/E0    E_total/E0
     0      0.000   0.000000    0.0000e+00     2.220e-16      0.0000e+00    1.000000      1.000000
    50      0.050   0.394205    4.9276e-05    -2.331e-15      1.9425e-05    0.999901      0.999921
...
```

The bead (`v_bead/u0`) speeds up towards the velocity of the fluid, while the total momentum of bead
and fluid stays constant (`p_total/p0 - 1` is zero to rounding). With a GPU, remove
`--platform Reference`: the script then takes the fastest platform.

**Next:** the [tutorial](tutorial.md) explains how an OpenMM simulation is built and goes through the
examples one by one.

### Every time you open a new terminal

```bash
conda activate lbm
```

### Updating the plugin

```bash
cd ~/src/openmm-lbm
git pull
cd ~/src/openmm-lbm-build
make -j4 && make install && make PythonInstall
```

## 10. On a computing cluster

The steps are the same, with a few differences.

- **Where to install.** The home directory of a cluster is often small. Put Miniforge and the
  environment on a larger filesystem (your project or work area): give the installer that folder in
  step 3, or create the environment with `conda create -p /path/to/envs/lbm ...` and activate it with
  `conda activate /path/to/envs/lbm`.
- **Modules.** Do not load other Python or OpenMM modules of the cluster together with this
  environment: they would mix two installations.
- **OpenMM compiled from source.** The plugin can only have the platforms that this OpenMM has: see
  [OpenMM compiled from source](#openmm-compiled-from-source) in section 12.
- **Login and compute nodes.** The machine you log in to usually has no GPU: build there, but run the
  GPU tests and the simulations on compute nodes, through the job scheduler (for example `sbatch` with
  Slurm). On the login node use `ctest -E Cuda`.
- **Old GPU drivers.** If the driver of the compute nodes supports an older CUDA than OpenMM needs
  (see [CUDA errors](#11-when-something-goes-wrong)) and the GPUs are data-centre models (A100, H100,
  ...), NVIDIA's forward-compatibility libraries solve it: install them in the environment with
  `conda install -c conda-forge cuda-compat=12.9` (the CUDA version of OpenMM) and, in the job script,
  before running Python:

  ```bash
  export LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat:$LD_LIBRARY_PATH
  ```

  Use this only for the CUDA platform: with these libraries in the path, NVIDIA's OpenCL driver can
  crash while it compiles kernels, so run OpenCL without them.

## 11. When something goes wrong

| What you see | Why | What to do |
|---|---|---|
| `conda: command not found` | the terminal was not reopened after installing Miniforge, or the initialization was refused | open a new terminal; if it persists, run `~/miniforge3/bin/conda init` and open a new terminal |
| `ModuleNotFoundError: No module named 'openmm'` or `'openmmlbm'` | the environment is not active, or `make PythonInstall` was not run | `conda activate lbm`; repeat step 7 |
| `TypeError` in `system.addForce(force)` | the plugin's Python module was made with a different SWIG version from OpenMM's | install the SWIG version named by `cmake` and repeat step 7 from `cmake` |
| `cmake` error `openmm-lbm requires OpenMM 8.3 or later, but the OpenMM in ... is version ...` | the environment has an OpenMM older than 8.3, or `OPENMM_DIR` points to another installation | create the environment again with `openmm=8.6.1`, and pass `-DOPENMM_DIR=$CONDA_PREFIX` with the environment active |
| `cmake` error `OpenMM was not found in ...` | `OPENMM_DIR` is wrong, or the environment is not active | `conda activate lbm`, then `cmake` with `-DOPENMM_DIR=$CONDA_PREFIX` |
| `cmake` error `LBM_BUILD_OPENCL_LIB is ON, but OpenMM in ... was built without the OpenCL platform` (or CUDA, HIP, the Python wrapper) | the option was set to `ON`, on the command line or by an earlier `cmake` in the same build folder, for a part that this OpenMM or system lacks | configure with `-DLBM_BUILD_OPENCL_LIB=OFF` (or the option named), or use a new build folder |
| errors that mention NumPy and a folder `~/.local` | Python mixes in packages installed outside conda | `conda env config vars set PYTHONNOUSERSITE=1` and reactivate the environment |
| CUDA: `Error loading CUDA module: CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)` | the driver supports an older CUDA than the OpenMM build | update the NVIDIA driver; on a cluster with data-centre GPUs use `cuda-compat` ([section 10](#10-on-a-computing-cluster)) |
| `Segmentation fault` with the OpenCL platform on an NVIDIA GPU | `cuda-compat` in `LD_LIBRARY_PATH`, or the `pocl` package exposing a second OpenCL device | remove `cuda-compat` from `LD_LIBRARY_PATH` for OpenCL runs; set `export OCL_ICD_VENDORS=/etc/OpenCL/vendors` to use only NVIDIA's driver |
| `cmake` warning `The linker flags (from LDFLAGS) contain ..., which holds another OpenMM library` | the active conda environment has another OpenMM than `OPENMM_DIR` | activate the environment of `OPENMM_DIR`, or configure in a new build folder without these flags ([section 12](#which-openmm-is-used)) |
| `make: *** No rule to make target 'PythonInstall'` | the Python module is not built: `cmake` printed why (`... the Python wrapper of openmm-lbm is not built.`) | install what is missing ([section 12](#what-is-built)) and run `cmake` again in a new build folder |
| `ctest` runs fewer tests than expected (5 instead of 8 on a computer with an NVIDIA GPU) | a platform was not built: `cmake` printed why | install what is missing ([section 12](#what-is-built)), or ignore it if that platform is not needed |
| a test marked `Failed` | a real problem | run that test alone with `ctest -R <name> --output-on-failure` and send the output to the developers |

Errors that appear while running simulations are explained in [troubleshooting](troubleshooting.md).

## 12. What CMake checks, and the build options

This section is a reference: steps 7 and 8 work without it with OpenMM from conda-forge. It is useful with
an OpenMM compiled from source, on a cluster, or when `cmake` leaves out a platform.

### What is built

The Reference platform is always built. Each other part is built only if everything it needs is installed:

| Part | Option | What it needs |
|---|---|---|
| CUDA platform | `LBM_BUILD_CUDA_LIB` | the CUDA platform of the OpenMM in `OPENMM_DIR`: `include/openmm/cuda/CudaContext.h` and the library `libOpenMMCUDA` in `lib` or `lib/plugins`; the CUDA toolkit (`nvcc`) with the CUDA driver library (`libcuda`, or the stub library of the toolkit on a computer without a GPU) |
| OpenCL platform | `LBM_BUILD_OPENCL_LIB` | the OpenCL platform of OpenMM: `include/openmm/opencl/OpenCLContext.h` and `libOpenMMOpenCL`; the OpenCL headers and library (`libOpenCL`; with conda, the packages `opencl-headers` and `ocl-icd`) |
| HIP platform | `LBM_BUILD_HIP_LIB` | the HIP platform of OpenMM: `include/openmm/hip/HipContext.h` and `libOpenMMHIP`; HIP (ROCm), found through `ROCM_PATH` or in `/opt/rocm` |
| Python module `openmmlbm` | `LBM_BUILD_PYTHON_WRAPPERS` | `python`, which must import `openmm`, `numpy`, `setuptools` and `pip`; `swig`, of the version that made the OpenMM Python module; the SWIG files of OpenMM (`include/swig`) and the headers of its plugins (`OpenMMAmoeba.h`, `OpenMMDrude.h`, `openmm/RPMDIntegrator.h`, `openmm/RPMDMonteCarloBarostat.h`), which OpenMM installs with its own Python module |

OpenMM from conda-forge has the CUDA and OpenCL platforms and its Python module, so with it only the system
side matters (the CUDA toolkit, the OpenCL library).

For each part that it leaves out, `cmake` says why, for example:

```
-- HIP was not found on this system: the HIP plugin of openmm-lbm is not built.
-- OpenMM in /work/openmm was built without the OpenCL platform (include/openmm/opencl/OpenCLContext.h and the OpenMMOpenCL library in lib or lib/plugins are required): the OpenCL plugin of openmm-lbm is not built.
-- /usr/bin/python3 cannot import the Python module openmm: the Python wrapper of openmm-lbm is not built.
```

and it ends with the list of what will be built:

```
-- openmm-lbm 0.3.0: platforms to build: Reference, CUDA, OpenCL; Python wrapper: yes; MPI: no
```

The tests of a platform that is not built are not built, so `ctest` runs fewer tests: 2 (serialization and
Reference), plus 3 for each GPU platform. Without the Python module, `make PythonInstall` stops with
`No rule to make target 'PythonInstall'`.

### Choosing the parts

- **Default.** Each option is `ON` when everything the part needs is found, `OFF` otherwise.
- **Leaving a part out.** `-DLBM_BUILD_OPENCL_LIB=OFF` (and likewise for the other options) does not build
  that part even if it could be built.
- **Asking for a part that cannot be built.** `-DLBM_BUILD_OPENCL_LIB=ON` when something is missing stops
  `cmake` with an error that says what is missing, instead of letting `make` fail later:

  ```
  CMake Error at CMakeLists.txt:... (MESSAGE):
    LBM_BUILD_OPENCL_LIB is ON, but OpenMM in /work/openmm was built without the OpenCL platform (...): the
    OpenCL plugin of openmm-lbm cannot be built.  Configure with -DLBM_BUILD_OPENCL_LIB=OFF, or install what
    is missing.
  ```

- **The build folder remembers the options** (in its `CMakeCache.txt`), also those chosen by an earlier
  `cmake`. After changing `OPENMM_DIR` or installing something that was missing, configure in a new, empty
  build folder: otherwise a part stays out, or `cmake` stops with the error above.

### Other options

| Option | Meaning |
|---|---|
| `OPENMM_DIR` | the OpenMM installation, with the headers in `include` and the libraries in `lib`; with conda, `$CONDA_PREFIX` |
| `CMAKE_INSTALL_PREFIX` | where `make install` copies the plugin; the platform libraries go to its `lib/plugins`, where OpenMM looks for plugins when it starts, so it is normally the same folder as `OPENMM_DIR` |
| `PYTHON_EXECUTABLE`, `SWIG_EXECUTABLE` | the full path of `python` and `swig`, if they are not the first ones in `PATH` |
| `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` | the compilers, if not the default ones; use those that compiled OpenMM |
| `LBM_DEBUG` | `ON` adds diagnostics of the fluid (`docs/theory.md`, section 6) |
| `OPENMM_LBM_MPI` | `ON` builds the decomposition of the lattice over MPI ranks (`setDomainDecomposition()`, in development for version 0.4.0: for now on the Reference platform only); it needs an MPI library (`mpicc` in `PATH`), the same one that runs the program. `OFF` (the default) needs no MPI |

### Checks on OpenMM

- **Version.** `cmake` compiles a small program linked to the OpenMM library of `OPENMM_DIR` and prints the
  version it reports (`-- OpenMM 8.6.1 in ...`). It stops with an error if the version is older than 8.3,
  and warns if it is newer than 8.6, the newest tested version. If the version cannot be read, it warns;
  in any case it requires `openmm/common/ComputeSort.h`, which OpenMM has since 8.3.
- **Library.** It stops if `OPENMM_DIR/lib` has no OpenMM library.
- **SWIG.** The Python module must be generated by the same SWIG version as the OpenMM Python module,
  otherwise OpenMM does not recognize an `LBMForce` as a `Force`: `cmake` compares the two versions and
  stops with an error that names the version to install.

### Which OpenMM is used

- **Headers.** They come from `OPENMM_DIR/include`, which the compiler searches before its own folders.
  A header that is missing there would be taken from the compiler's folders: the compilers of conda search
  `$CONDA_PREFIX/include`, where the environment may have another, complete OpenMM. This is why `cmake`
  checks each platform in `OPENMM_DIR` before building it.
- **Libraries.** The OpenMM library and the platform libraries are linked by their full path in
  `OPENMM_DIR`, for the plugin libraries, the tests and the Python module. With `-lOpenMM` the linker would
  take the first library of that name in its search path, where the linker flags of an active conda
  environment (`LDFLAGS`) put the environment's `lib` before `OPENMM_DIR`.
- **At run time.** Programs (the tests, Python) load the OpenMM they find through their run path and the
  library path, and the run path set by `LDFLAGS` comes first. If the linker flags contain a folder with
  another OpenMM library, `cmake` warns:

  ```
  CMake Warning at CMakeLists.txt:... (MESSAGE):
    The linker flags (from LDFLAGS) contain /home/me/miniforge3/envs/other/lib, which holds another OpenMM
    library than the one in /work/openmm.  openmm-lbm is linked to the OpenMM in OPENMM_DIR, but its tests
    and programs may load the other one when they run: build in the environment whose OpenMM is the one in
    OPENMM_DIR, or without these linker flags.
  ```

  The usual case has no warning: the environment is active and `OPENMM_DIR` is `$CONDA_PREFIX`. Otherwise
  activate the environment of `OPENMM_DIR`, or configure in a new build folder without the flags of the
  other environment (for example after `unset LDFLAGS`).

### OpenMM compiled from source

- **The platforms of the plugin are those of OpenMM.** If the `cmake` of OpenMM did not find OpenCL (or
  CUDA), OpenMM has no such platform, and the plugin cannot have it either: its `cmake` says so and builds
  the others. `python -m openmm.testInstallation` lists the platforms of OpenMM. To add one, compile OpenMM
  again with it, then configure the plugin in a new build folder.
- **The Python module of OpenMM** must be installed (`make PythonInstall` of OpenMM): the plugin's Python
  module needs its SWIG files and imports it.
- **The same compiler** for OpenMM and the plugin (`CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER`): a plugin
  compiled with an older compiler than OpenMM may fail to start with `version 'CXXABI_...' not found`.
- **No GPU on the build machine.** The CUDA driver library is needed to link: the CUDA toolkit has a stub
  of it (`lib64/stubs/libcuda.so`), which CMake finds by itself. If linking the tests then fails with
  `libcuda.so.1` not found, make a link named `libcuda.so.1` to the stub in a folder of its own and add that
  folder to `LD_LIBRARY_PATH` only for `make`. The programs must run with the real driver, on the GPU nodes,
  without that folder in `LD_LIBRARY_PATH`.

