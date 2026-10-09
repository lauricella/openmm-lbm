# Running on several GPUs

**In development for version 0.4.0.** A lattice too large or too slow for one GPU can be divided into domains, one
per MPI rank and GPU: every rank advances the fluid of its domain, and the ranks exchange the populations that cross
the borders at every step. The particles are not divided: every rank builds the same System and integrates all the
particles, so the decomposition speeds up the fluid only. It pays when the fluid takes most of the time of a step,
as with coarse-grained molecules in a large box. The method and its limits are in
[theory](../theory.md#8-domain-decomposition-in-development-for-version-040), the measured speed in
[validation](../validation.md#performance-of-the-domain-decomposition-cuda-nvidia-a100), and every method in the
[API reference](api_reference.md#setdomaindecompositionpx-py-pz-getdomaindecomposition).

## Building with MPI

The decomposition needs the plugin built with the CMake option `OPENMM_LBM_MPI` ([options](installation.md)) and an
MPI library: on a cluster the one of the cluster (a module), which knows its network; on a workstation, for example,
the `openmpi` package of conda-forge in the environment of OpenMM. Build and run with the same library. For the
exchange from GPU to GPU within a node the library must be able to read the memory of the GPUs (CUDA-aware MPI, such
as Open MPI with UCX built with CUDA); without it the populations go through the host, which works but is slower.

```bash
cmake .. -DOPENMM_DIR=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DOPENMM_LBM_MPI=ON
```

The summary at the end of `cmake` ends with `MPI: yes (MPI 3.1)` or similar. After `make install` and
`make PythonInstall`, `python -c "from openmmlbm import LBMForce; print(LBMForce.isMPIAvailable())"` prints `True`.
`ctest` then also runs `TestMPIReferenceLBMForce` with two ranks, and the script `python/tests/mpi_decomposition.py`
of the repository checks a build on CPUs and on GPUs: every decomposition it is given must give the same fluid as one
domain, bit for bit, and it prints a line with `FAILED` for any difference.

```bash
mpirun -n 4 python python/tests/mpi_decomposition.py 2 2 1                                  # Reference platform
mpirun -n 4 python python/tests/mpi_decomposition.py 1 2 2 --platform CUDA --devices 4     # on a node with 4 GPUs
```

On a computer with fewer cores than ranks Open MPI refuses to start them; `--oversubscribe` (or
`OMPI_MCA_rmaps_base_oversubscribe=1`) allows it.

## The script

The same script runs on every rank. This one couples 200 beads of 100 Da, with a soft repulsion between them, to the
fluid of a periodic box of $`8 \times 8 \times 16`$ nm. It runs as it is in one process on the Reference platform;
on GPUs set `PLATFORM` to `'CUDA'` (or `'HIP'`, `'OpenCL'`) and launch one rank per GPU (next section).

```python
import os
import sys

import numpy as np
import openmm as mm
import openmm.app as app
import openmmlbm
from openmmlbm import LBMForce, LBMVTKReporter, LBMCheckpointReporter

PLATFORM = 'Reference'      # 'CUDA' on NVIDIA GPUs
rank = openmmlbm.mpiRank()

# The same System on every rank.
system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(8, 0, 0), mm.Vec3(0, 8, 0), mm.Vec3(0, 0, 16))
repulsion = mm.CustomNonbondedForce('10*(1-r)^2')
repulsion.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
repulsion.setCutoffDistance(1.0)
force = LBMForce()
force.setGridSize(16, 16, 32)
force.setFriction(5.0)
force.setTemperature(300)
for i in range(200):
    system.addParticle(100.0)
    repulsion.addParticle()
    force.addParticle(i)
system.addForce(repulsion)
if LBMForce.isMPIAvailable():
    force.setDomainDecomposition(0, 0, 0)    # one domain per rank: along z first, then y
system.addForce(force)

properties = {}
if PLATFORM != 'Reference':
    properties['Precision'] = 'mixed'
    properties['DeviceIndex'] = str(openmmlbm.mpiLocalRank())     # one GPU per rank on each node
    if PLATFORM in ('CUDA', 'HIP'):
        properties['DeterministicForces'] = 'true'               # the same forces on every copy of the particles
simulation = app.Simulation(app.Topology(), system, mm.VerletIntegrator(0.01),
                            mm.Platform.getPlatformByName(PLATFORM), properties)

# The same positions and velocities on every rank: a fixed seed.
positions = np.random.default_rng(1).uniform(0, 1, (200, 3))*[8, 8, 16]
simulation.context.setPositions([mm.Vec3(*x) for x in positions])
simulation.context.setVelocitiesToTemperature(300, 1)

# Every rank has the reporters that compute forces or write the fluid (collective); the text goes out from rank 0.
out = sys.stdout if rank == 0 else open(os.devnull, 'w')
simulation.reporters.append(app.StateDataReporter(out, 100, step=True, temperature=True))
simulation.reporters.append(LBMVTKReporter('run', 100, force))
simulation.reporters.append(LBMCheckpointReporter('run.chk', 100, force))
simulation.step(200)

start, count = force.getLocalDomain(simulation.context)
print('rank', rank, 'of', openmmlbm.mpiSize(), 'advanced the nodes', start, 'to',
      tuple(s + c - 1 for s, c in zip(start, count)))
```

What matters for the decomposition:
- **The same System, positions and velocities on every rank.** Every rank holds a copy of all the particles, and the
  copies must stay identical: read the positions from the same file, and draw random velocities with a fixed seed.
  The plugin compares the copies at the first step and then every `setMachCheckFrequency()` steps, and stops every
  rank with an error if they differ. An `AndersenThermostat` or a Monte Carlo barostat, whose random numbers would
  differ between the ranks, is refused; the temperature comes from the coupling to the fluid.
- **One GPU per rank.** `DeviceIndex` is the rank among the processes of the node, `mpiLocalRank()`, when every
  process sees all the GPUs of its node. With `srun --gpus-per-task=1` every process sees only its own GPU, as
  device 0: leave `DeviceIndex` out.
- **`DeterministicForces`.** On CUDA and HIP some sums of OpenMM depend on the order of the threads unless this
  property is `'true'`, and the copies of coupled particles would drift apart; with coupled particles the plugin
  requires it.
- **The decomposition.** `setDomainDecomposition(0, 0, 0)` lets MPI choose, with the most domains along z, then y;
  the product of explicit numbers must be the number of ranks. Dividing x, along which the nodes are consecutive in
  memory, costs 30% to 90% more time per step on the GPUs: give x domains only when z and y are not enough.
- **Collective calls.** The creation of the Context and some calls must be made by every rank, in the same order,
  or the run waits forever: with coupled particles every evaluation of the forces, that is `getState()` with
  `getForces=True` or with `getEnergy=True` (the `VerletIntegrator` needs the forces for the kinetic energy);
  `getWallForce()`, `getFluidMachNumber()`, `setFluidState()`, the calls with `gather=True`, the checkpoints and the
  VTK files of the fluid. So must the reporters that make these calls: `StateDataReporter` with energies or the
  temperature, `LBMTemperatureReporter`, `LBMVTKReporter`, `LBMCheckpointReporter`, `CheckpointReporter`. Give a
  reporter that writes text a file only on rank 0 (`os.devnull` on the other ranks). A reporter that reads only the
  positions, such as `DCDReporter`, can stay on rank 0.
- **Errors.** An uncaught exception on one rank prints its message and stops every rank (`MPI_Abort`), instead of
  leaving the others waiting. Errors found by the plugin stop all the ranks together, with the same message.
- **The fluid of a rank.** `getFluidFields()` and `getFluidState()` return the domain of the rank
  (`getLocalDomain()`); `gather=True` collects the whole lattice on rank 0.
- **Restarts.** `LBMCheckpointReporter` writes one file for the whole run, which a run with another number of ranks
  can also load ([restart](restart.md)).

## Launching

Set up the environment where the ranks run: in the job script, or in the interactive session on the compute node
after it has started, not on the login node before submitting the job. A job with one rank per GPU on one node with
four GPUs, under Slurm:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # one rank per GPU
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

module load openmpi                  # the MPI library that the plugin was built with (the name depends on the cluster)
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lbm
# export LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat:$LD_LIBRARY_PATH    # only if needed (installation, section 10)

mpirun -n 4 python run.py            # or: srun python run.py
```

On several nodes, `mpirun` of Open MPI starts the ranks of the other nodes with only the environment variables named
with `-x`, taken from the job script; `srun` passes the whole environment of the job script by default. Place the
same number of ranks on every node, so that `mpiLocalRank()` matches the GPUs:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

module load openmpi
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lbm

mpirun -n 8 --map-by ppr:4:node -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x UCX_NET_DEVICES=mlx5_0:1 python run.py
```

`UCX_NET_DEVICES` chooses the InfiniBand ports that UCX, the transport of Open MPI, uses between nodes; the names are
those of `ucx_info -d` on a compute node. On the nodes where the plugin was measured (four ports per node, UCX 1.16),
all four ports, which UCX uses when the variable is not set, stopped every run when the ranks connected, unless UCX
used the protocols of its earlier versions (`-x UCX_PROTO_ENABLE=n`); one port always worked, and two ports, one port
per rank (the one next to its GPU) or all four with `UCX_PROTO_ENABLE=n` were up to 10% faster on four nodes
([validation](../validation.md#performance-of-the-domain-decomposition-cuda-nvidia-a100)). For OpenCL also pass the
variables of its loader, such as `OCL_ICD_VENDORS`. The variable `OPENMM_LBM_DEVICE_MPI=0` makes the populations go
through the host also within a node, for example if the MPI library fails with the memory of the GPUs.

## Measuring the speed

`devtools/benchmark_decomposition.py` times the fluid on any decomposition, with or without coupled particles, on
the GPUs of a machine:

```bash
mpirun -n 4 python devtools/benchmark_decomposition.py 256 256 256 1 2 2                   # 256^3 nodes on 4 GPUs
mpirun -n 4 python devtools/benchmark_decomposition.py 256 256 256 1 2 2 --particles 100000
python devtools/benchmark_decomposition.py 256 256 256 1 1 1                               # one GPU, to compare
```

On four NVIDIA A100 of one node, $`256^3`$ nodes took 1.84 ms per step in mixed precision, 4.0 times faster than one
GPU; $`128^3`$ nodes per GPU ran at 96% of the speed of one GPU. Between nodes the exchange goes through the host and
takes longer: $`512^3`$ nodes on eight nodes (32 GPUs) were 3.8 times faster than on one node. Coupled particles add
the forces of OpenMM, which every rank computes for all of them, and the sum of their coupling forces over the ranks
at every step: $`10^5`$ particles added about 1.5 ms per step on four GPUs.

## At the end of the run

MPI is finalized when the script ends. If MPI was initialized by the plugin, the `openmmlbm` module finalizes it before
the objects of the script are deleted; a script that initialized MPI itself, for example with mpi4py, should finalize it
while its Contexts still exist ([API
reference](api_reference.md#setdomaindecompositionpx-py-pz-getdomaindecomposition)).
