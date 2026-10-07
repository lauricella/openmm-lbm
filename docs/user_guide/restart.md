# Saving and continuing a simulation (restart)

A simulation often has to stop before it is finished: on a computing cluster a job has a time limit (for
example 24 hours), a node can fail, or you may want to look at the results halfway. This page shows how to
save the complete state of a run with a lattice Boltzmann fluid and continue it later, exactly as if it had
never stopped.

If you are in a hurry: add an `openmmlbm.LBMCheckpointReporter` to your `Simulation`
([first section](#the-simplest-way-a-reporter)), and to continue build the same `Simulation` again and call
`openmmlbm.loadCheckpoint()`. The script of the [third section](#one-script-for-the-first-run-and-every-restart)
does both and can be copied as it is.

## What has to be saved

OpenMM saves the state of a simulation in a *checkpoint* (`Context.createCheckpoint()`,
`Simulation.saveCheckpoint()`, `app.CheckpointReporter`). An OpenMM checkpoint does not contain the fluid:
OpenMM does not know about it, and a plugin cannot add data to OpenMM checkpoints. openmm-lbm therefore
writes a second checkpoint for its own data, and the functions of the module `openmmlbm` keep the two
together in one file.

| Saved by | What |
|---|---|
| OpenMM's checkpoint | positions, velocities, periodic box, time, step count, state of the integrator, OpenMM's random number generator (which `LBMForce` uses on the CUDA, OpenCL and HIP platforms) |
| `LBMForce.createCheckpoint()` | populations of the fluid, random numbers already drawn for the next step, force on the walls of the last step, and on the Reference platform the random number generator of the force |
| nothing | the System itself (particles, forces, parameters) and the files written by reporters |

So a restart needs **the same script**: it builds the same System and the same `Simulation`, then loads the
checkpoint instead of setting the initial positions and velocities. With the same platform and precision,
the continued run is identical, bit for bit, to an uninterrupted one.

## The simplest way: a reporter

`openmmlbm.LBMCheckpointReporter(file, interval, force)` works like OpenMM's `CheckpointReporter`: every
`interval` steps it writes the checkpoint of the whole run, fluid included, to `file`, replacing the
previous one. The example below runs 300 steps of 50 beads in a fluid at 300 K and saves a checkpoint every
100 steps.

```python
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import openmmlbm
from openmmlbm import LBMForce


def create_simulation():
    """Build the System and the Simulation: the same every time the script runs."""
    numBeads, L = 50, 6.0
    topology = app.Topology()
    chain = topology.addChain()
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0), mm.Vec3(0, L, 0), mm.Vec3(0, 0, L))
    force = LBMForce()
    force.setGridSize(12, 12, 12)
    force.setFriction(10.0)
    force.setTemperature(300.0)
    force.setRandomNumberSeed(7)
    for i in range(numBeads):
        topology.addAtom('B', None, topology.addResidue('BEA', chain))
        system.addParticle(100.0)
        force.addParticle(i)
    system.addForce(force)
    simulation = app.Simulation(topology, system, mm.VerletIntegrator(0.01*unit.picosecond),
                                mm.Platform.getPlatformByName('Reference'))
    return simulation, force


simulation, force = create_simulation()
simulation.context.setPositions(np.random.default_rng(1).uniform(0, 6.0, (50, 3)))
simulation.context.setVelocitiesToTemperature(300.0, 3)
simulation.reporters.append(openmmlbm.LBMCheckpointReporter('run.chk', 100, force))
simulation.step(300)
print('stopped at step', simulation.currentStep)
```

Output:

```
stopped at step 300
```

To continue, a new run of the script (or another script) builds the same `Simulation` and loads the file.
The positions, the velocities, the time and the step count come from the checkpoint, so it does not set
them:

```python
# (continued) A new run: build the same Simulation and load the checkpoint instead of the initial state.
simulation, force = create_simulation()
openmmlbm.loadCheckpoint('run.chk', simulation.context, force)
print('continuing from step', simulation.currentStep)
simulation.step(200)
print('stopped at step', simulation.currentStep)
```

Output:

```
stopped at step 300
continuing from step 300
stopped at step 500
```

(The first line comes from the first block, which runs before this one when the guide is checked.)

## The restart is exact

The run below goes 400 steps in one go; the same run stops after 250 steps, saves, and continues in a new
`Simulation` up to step 400. The final positions and fluid are identical, to the last bit:

```python
# (continued) Compare an uninterrupted run with a run saved and continued.
def run_from_start(steps):
    simulation, force = create_simulation()
    simulation.context.setPositions(np.random.default_rng(1).uniform(0, 6.0, (50, 3)))
    simulation.context.setVelocitiesToTemperature(300.0, 3)
    simulation.step(steps)
    return simulation, force

whole, wholeForce = run_from_start(400)
first, firstForce = run_from_start(250)
openmmlbm.saveCheckpoint('half.chk', first.context, firstForce)
second, secondForce = create_simulation()
openmmlbm.loadCheckpoint('half.chk', second.context, secondForce)
second.step(150)
x1 = whole.context.getState(getPositions=True).getPositions(asNumpy=True)._value
x2 = second.context.getState(getPositions=True).getPositions(asNumpy=True)._value
f1 = np.array(wholeForce.getFluidState(whole.context))
f2 = np.array(secondForce.getFluidState(second.context))
print('positions identical:', np.array_equal(x1, x2), ' fluid identical:', np.array_equal(f1, f2))
```

Output:

```
stopped at step 300
continuing from step 300
stopped at step 500
positions identical: True  fluid identical: True
```

## Moving a run to another platform

To continue a run on another platform or precision (for example from a GPU to the Reference platform for a
check), save the particles and the fluid in portable formats instead: the particles with
`Simulation.saveState()`, an XML file, and the fluid with `getFluidState()`, a list of numbers that does not
depend on the platform. The continued run is then statistically equivalent but not identical, because the
random numbers are drawn again.

```python
# (continued) Save portable files, then continue on another platform (here the Reference platform again).
simulation, force = create_simulation()
openmmlbm.loadCheckpoint('run.chk', simulation.context, force)
simulation.saveState('particles.xml')
np.save('fluid.npy', np.array(force.getFluidState(simulation.context)))

other, otherForce = create_simulation()      # in practice with another platform
other.loadState('particles.xml')
otherForce.setFluidState(other.context, np.load('fluid.npy').tolist())
print('continuing from step', other.currentStep)
```

Output:

```
stopped at step 300
continuing from step 300
stopped at step 500
positions identical: True  fluid identical: True
continuing from step 300
```

`loadState()` restores positions, velocities, periodic box, time and step count: the continued run starts
at step 300, the step of the checkpoint loaded above.

## One script for the first run and every restart

On a cluster the convenient way is a single script that starts the run if there is no checkpoint and
continues it if there is one. Submit the same job again and again until the run reaches its total number of
steps: each job continues where the previous one stopped.

```python
import os
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import openmmlbm
from openmmlbm import LBMForce

TOTAL_STEPS = 1000         # length of the whole run
REPORT = 100               # steps between reports
CHECKPOINT = 'run.chk'     # checkpoint file, written every 2 reports

# The System and the Simulation, exactly the same at every start.
numBeads, L = 50, 6.0
topology = app.Topology()
chain = topology.addChain()
system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0), mm.Vec3(0, L, 0), mm.Vec3(0, 0, L))
force = LBMForce()
force.setGridSize(12, 12, 12)
force.setFriction(10.0)
force.setTemperature(300.0)
for i in range(numBeads):
    topology.addAtom('B', None, topology.addResidue('BEA', chain))
    system.addParticle(100.0)
    force.addParticle(i)
system.addForce(force)
simulation = app.Simulation(topology, system, mm.VerletIntegrator(0.01*unit.picosecond),
                            mm.Platform.getPlatformByName('Reference'))

if os.path.exists(CHECKPOINT):
    # A restart: positions, velocities, time and fluid from the checkpoint.
    openmmlbm.loadCheckpoint(CHECKPOINT, simulation.context, force)
    restart = True
else:
    # The first run: initial state.
    simulation.context.setPositions(np.random.default_rng(1).uniform(0, L, (numBeads, 3)))
    simulation.context.setVelocitiesToTemperature(300.0)
    restart = False
print('starting at step', simulation.currentStep)

# append=True continues the files of the previous runs instead of overwriting them.
simulation.reporters.append(app.StateDataReporter('log.txt', REPORT, step=True, temperature=True,
                                                  append=restart))
simulation.reporters.append(app.DCDReporter('trajectory.dcd', REPORT, append=restart))
simulation.reporters.append(openmmlbm.LBMCheckpointReporter(CHECKPOINT, 2*REPORT, force))
simulation.step(TOTAL_STEPS - simulation.currentStep)
print('finished at step', simulation.currentStep)
```

Output:

```
starting at step 0
finished at step 1000
```

Run with `python run.py`. If it stops before the end, `python run.py` again continues it; when the run is
finished, a further `python run.py` does nothing (zero steps left).

A job script for Slurm that runs it (adapt the account, the partition and the environment to your cluster):

```bash
#!/bin/bash
#SBATCH --job-name=lbm-run
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
source ~/miniforge3/etc/profile.d/conda.sh     # makes conda available in the job
conda activate lbm
python run.py
```

Submit it once with `sbatch job.sh`. To chain several jobs, each starting when the previous one ends, use a
dependency: `sbatch --dependency=afterany:<number of the previous job> job.sh`.

## Things to know

- **Use the same platform, precision and OpenMM version.** Like OpenMM's checkpoints, the file is specific
  to them: loading it on another platform or with another precision gives an error. To move a run to a
  different computer or platform, see [moving a run to another platform](#moving-a-run-to-another-platform).
- **Build the same System.** Same particles in the same order, same forces, same `LBMForce` parameters and
  grid. `loadCheckpoint()` refuses a checkpoint written for a different grid, number of coupled particles,
  drag scheme, wall scheme, types of the faces (`setFaceBoundary()`) or switch of the fluid fluctuations, but
  it cannot check everything else: a different friction or viscosity, or different velocities and densities
  of the faces, would simply be used from then on. The random number seed of the script does not matter on a restart: the state of the
  generator comes from the checkpoint.
- **Reports written after the last checkpoint.** If a job stops between two checkpoints, its reporters have
  already written the steps after the last checkpoint, and the restart writes them again. Choose the
  checkpoint interval as a multiple of the report interval, and remove from the text files the lines after
  the step of the checkpoint before you analyse them. The example `examples/cocomo/diffusion.py` (options
  `--checkpoint` and `--restart`) does this for you: it cuts its text files at the checkpoint, and starts a
  new trajectory file `<prefix>_<step>.dcd`.
- **Cost.** A checkpoint holds the whole fluid: 19 numbers per node, 8 bytes each in mixed and double
  precision, so 4 MB for 30^3 nodes and 150 MB for 100^3. Save one every 10 to 60 minutes of computing time,
  not every few steps.
- **Safe writing.** The file is written under a temporary name and then renamed, so a job killed while it
  writes leaves the previous checkpoint intact.
- **Checkpoints in memory.** `context.createCheckpoint()` and `force.createCheckpoint(context)` return the two
  checkpoints as bytes, and `context.loadCheckpoint(data)` and `force.loadCheckpoint(context, data)` load
  them; `saveCheckpoint()` and `loadCheckpoint()` only add the file around them. In C++ the methods take
  binary streams.

## Finding the force of a System

`loadCheckpoint()` and the reporter need the `LBMForce` object of the System of the `Simulation`. If the
script did not keep it, for example because the System was read from an XML file, find it among the forces
of the System:

```python
import openmm as mm
from openmmlbm import LBMForce

system = mm.System()
system.addForce(mm.CMMotionRemover())
system.addForce(LBMForce())
force = [LBMForce.cast(f) for f in system.getForces() if LBMForce.isinstance(f)][0]
print(type(force).__name__)
```

Output:

```
LBMForce
```
