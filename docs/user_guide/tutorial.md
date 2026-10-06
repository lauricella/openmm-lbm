# Tutorial: learning OpenMM and openmm-lbm with the examples

This tutorial is for someone who has installed OpenMM and the plugin ([installation](installation.md))
and has never used OpenMM. It explains the pieces of an OpenMM simulation, what the plugin adds, and
then goes through the scripts of the [`examples/`](../../examples/README.md) folder one by one, with
what to look at and a few exercises: a kicked bead, momentum and energy, temperature, the fluid alone,
a protein, and a long run split into several pieces.

Each lesson takes from a few minutes to half an hour. You need a terminal with the environment active
(`conda activate lbm`) and a working folder:

```bash
mkdir -p ~/lbm-runs
cd ~/lbm-runs
export EX=~/src/openmm-lbm/examples      # a short name for the folder of the examples
```

The commands below use `--platform Reference`, which works on every computer. With a GPU, leave it
out: the scripts then take the fastest platform. For the figures, install matplotlib once:

```bash
conda install -c conda-forge matplotlib
```

## 1. The pieces of an OpenMM simulation

OpenMM is a library: a simulation is a Python script that builds a few objects and asks them to
advance in time. Open [`examples/particle/kick.py`](../../examples/particle/kick.py) in a text editor
and find them in the function `main()`:

- **System**: what is simulated. It holds the particles (only their masses), the periodic box, and a
  list of **forces**. Here: `system = mm.System()`, `system.setDefaultPeriodicBoxVectors(...)`,
  `system.addParticle(mass)`.
- **Force**: an object that computes forces on the particles. OpenMM has many (`HarmonicBondForce` for
  springs between particles, `NonbondedForce` for Lennard-Jones and Coulomb, `CustomNonbondedForce`
  for any pair potential, ...). The plugin adds **`LBMForce`**, the fluid.
- **Integrator**: the algorithm that advances positions and velocities by one time step. With
  `LBMForce` it must be `VerletIntegrator`: friction and random force are already part of the plugin,
  and a thermostat in the integrator would add them a second time.
- **Platform**: where the computation runs: `Reference` (simple C++ on one processor core, the
  slowest but the reference for correctness), `CUDA` and `OpenCL` (GPUs), `HIP` (AMD GPUs).
- **Context**: the running simulation. It joins a System, an Integrator and a Platform, and holds the
  current positions and velocities: `context.setPositions(...)`, `context.setVelocities(...)`.
- **State**: a snapshot of the Context, read with `context.getState(getPositions=True, ...)`.
- `integrator.step(n)` advances n steps.

**Units.** OpenMM uses nanometres (nm), picoseconds (ps), daltons (Da, the mass of a hydrogen atom is
about 1 Da), kelvin (K) and kJ/mol. A velocity is in nm/ps (1 nm/ps = 1000 m/s). The module
`openmm.unit` attaches units to numbers; a plain number is taken in these units.

The [OpenMM user guide](https://docs.openmm.org/latest/userguide/) and the
[OpenMM cookbook](https://openmm.github.io/openmm-cookbook/) explain all of this in detail, for
simulations of atoms and molecules.

## 2. What LBMForce adds

`LBMForce` fills the periodic box with a fluid described by the lattice Boltzmann method: the box is
divided into cubic cells, and at each node of the lattice the fluid has a density and a velocity
u. The parameters are those of a real fluid:

- `setGridSize(nx, ny, nz)`: the number of nodes along each side. The spacing dx = box length / nodes
  must be the same in the three directions.
- `setFluidDensity(rho)`: in Da/nm^3 (water: 602 Da/nm^3).
- `setKinematicViscosity(nu)`: in nm^2/ps (water at 300 K: about 0.9 nm^2/ps).

The particles added with `addParticle(i)` are **coupled** to the fluid. At every step each one feels,
from the node nearest to it,

- a friction force -gamma m (v - u): the particle is dragged towards the velocity of the fluid
  (`setFriction(gamma)`, in 1/ps);
- a random force whose size is set by the temperature (`setTemperature(T)`), which keeps the particles
  at the temperature T. This is the Euler-Maruyama scheme; the scheme `NVE` has no random force.

The opposite of these forces is given to the fluid, so the total momentum of particles and fluid is
conserved. A particle that moves pushes the fluid, and the fluid pushes the other particles: this is
the hydrodynamic interaction that a simple Langevin thermostat does not have. The
[lattice page](lattice.md) explains how to choose the parameters; [theory.md](../theory.md) gives the
equations.

## 3. Lesson 1: a kicked bead (`kick.py`)

A bead of 80.7 Da starts at 9 nm/ps in a fluid at rest, at zero temperature. Run it twice, with the
fluid and without (`--no-lb`: the same friction in OpenMM's Langevin integrator, without fluid):

```bash
python $EX/particle/kick.py --platform Reference --nodes 30
python $EX/particle/kick.py --platform Reference --nodes 30 --no-lb
python $EX/plot.py kick_bead_lb_on.txt kick_bead_lb_off.txt --x 1 --y 5 --logy --xlabel 'time (ps)' --ylabel 'speed (nm/ps)' --output kick.png
```

(`--nodes 30` makes the box smaller, 9 nm instead of 30 nm, so that the run takes about 15 seconds on
one processor core instead of 10 minutes; on a GPU leave it out.) Open `kick.png`.

- **Without the fluid** the speed falls as exp(-gamma t), a straight line on the logarithmic axis, and
  the bead stops after v0/gamma = 0.9 nm.
- **With the fluid** the first steps are the same, then the decay slows down: the fluid around the bead
  has been set in motion and drags it along. The bead travels further, and its speed decays as a power
  of time (about t^(-3/2), the "long-time tail" of hydrodynamics) until it reaches the velocity of the
  whole fluid, m v0 / (m + M), where M is the mass of the fluid.

Exercises:

1. Look at the end of the two output files: what is the distance travelled in each case?
2. Double the friction (`--friction 20`). How do the two curves change?
3. With `--nodes 20` the box is smaller and the fluid lighter: compare the final velocity with
   m v0 / (m + M). The script prints the mass of the bead; the fluid has mass density x box^3.

## 4. Lesson 2: momentum and energy (`uniform_flow.py`)

Now the fluid moves at 0.09 nm/ps and a bead of 16 Da is at rest:

```bash
python $EX/particle/uniform_flow.py --platform Reference
```

The table shows, every 50 steps, the velocity of the bead relative to the fluid, the momentum of the
bead, the change of the total momentum, and the kinetic energies of bead and fluid relative to the
initial one.

- The bead reaches the velocity of the fluid in about 1/gamma = 0.1 ps.
- `p_total/p0 - 1` stays at 1e-14 or less: momentum is conserved exactly, apart from rounding.
- `E_total/E0` decreases: kinetic energy is not conserved. The friction turns it into heat, and so
  does the viscosity of the fluid. The fluid of the lattice Boltzmann method has a constant
  temperature, so that heat leaves the model.

Exercises:

1. Change the viscosity through the relaxation time (`--tau 0.6`, `--tau 1.5`). Does the final energy
   change? (The relaxation time tau sets the viscosity: nu = (tau - 1/2) dx^2 / (3 dt).)
2. Make the bead heavier (`--cells 100`). The common final velocity is printed in the second line:
   check it at the end of a longer run (`--steps 5000 --interval 500`).

## 5. Lesson 3: temperature (`thermal.py`)

The fluid also works as a thermostat. One bead at rest is brought to 300 K by the friction and the
random force:

```bash
python $EX/particle/thermal.py --platform Reference --seed 1
python $EX/plot.py thermal.txt --x 1 --y 2 --xlabel 'time (ps)' --ylabel 'vx (nm/ps)' --output thermal.png
```

`thermal.png` shows the velocity of the bead fluctuating around zero. At the end the script prints the
mean temperature computed in two ways:

- from the velocities that OpenMM stores, which with `VerletIntegrator` are half a step behind the
  positions ("half step");
- from the mean of two consecutive velocities, the velocity at the same time as the positions ("full
  step"). This is the temperature of the coupled particles, and it is also the one that OpenMM itself
  reports, for example in `StateDataReporter`, because OpenMM shifts the stored velocities by half a step
  with the forces of the next step (see [examples](examples.md#temperature-of-coupled-particles)).

With one bead and a small friction the mean converges slowly. A gas of 100 beads with a larger
friction gives a precise value in a few seconds, on a GPU or on the Reference platform:

```bash
python $EX/particle/thermal.py --beads 100 --friction 10 --steps 20000 --equilibration 2000 --seed 2 --removal 1
```

```
Mean temperature over 18000 steps: full step 297.6 K, half step 299.1 K
Free bead in a fluid at rest: full step 300.0 K, half step T/(1 - gamma dt/2) = 301.5 K
```

(on an A100 GPU; another platform gives slightly different numbers, because the random numbers are
rounded differently.) The full-step temperature is about 1% below 300 K: the fluid itself has no
thermal fluctuations, and it takes part of the momentum of the beads.

Exercises:

1. Run the same command with `--seed 3`: how much does the mean temperature change? This is the
   statistical error.
2. Increase the box with `--nodes 20`: the fluid becomes 8 times heavier than before, compared with
   the beads. Does the temperature change by more than the statistical error of exercise 1?

## 6. Lesson 4: the fluid alone (`initial_state.py`)

The last example has no coupled particles. It starts the fluid from a wave, u_x = U sin(2 pi y / L),
and prints its amplitude while the viscosity damps it:

```bash
python $EX/fluid/initial_state.py --platform Reference --save state.npz
```

The two columns are the measured amplitude and the decay U exp(-nu k^2 t) of a fluid with the
viscosity nu that was set; they agree within 1%.

- The script builds the initial populations of the lattice from the density and the velocity with the
  function `equilibrium_deviation()`, and gives them to the plugin with `setFluidState()`. Read it:
  it is the way to start the fluid from any flow.
- `--save state.npz` writes the state of the fluid at the end. `--load state.npz` starts a new run from
  it, and the run continues exactly as if it had not stopped:

```bash
python $EX/fluid/initial_state.py --platform Reference --load state.npz
```

Exercises:

1. Measure the viscosity: take the amplitude at two times, A1 and A2, and compute
   nu = ln(A1/A2) / (k^2 (t2 - t1)) with k = 2 pi / L. Compare it with the value printed at the start.
2. Repeat with `--viscosity 3` and `--viscosity 0.3`. How does the decay rate change? The script
   prints the relaxation time tau: below about 0.505 the plugin warns that the fluid may become
   unstable.

## 7. Lesson 5: a protein in the fluid (`cocomo/diffusion.py`)

The last example is a real application: the protein SOD1, with one bead per amino acid (the COCOMO2
model), diffusing in water. Besides `LBMForce`, the System now contains the forces of the protein model:
springs along the chain, angles, an elastic network that keeps the protein folded, and nonbonded
interactions between the beads. Read `examples/cocomo/cocomo2.py` to see how a model is built from
standard OpenMM forces (`HarmonicBondForce`, `HarmonicAngleForce`, `CustomNonbondedForce`).

The script uses `openmm.app.Simulation`, the usual way to run long simulations: it minimizes the
energy, then runs with *reporters*, objects that write the trajectory and other data every few steps.

```bash
python $EX/cocomo/diffusion.py --preset smoke --platform Reference
```

On a GPU, run 1 ns and compute the diffusion coefficient of the protein from its centre of mass:

```bash
python $EX/cocomo/diffusion.py --preset sod1 --steps 100000 --report 1000 --output sod1_1ns
python $EX/cocomo/msd.py sod1_1ns_com.txt
```

Exercises:

1. Open `sod1_1ns_temperature.txt`: is the protein at 298 K?
2. Run the same with `--no-lb` (no fluid, Langevin integrator) and compare the diffusion coefficients.
   A run of 1 ns gives only a rough value; the presets run for 50 to 200 ns.

## 8. Lesson 6: a long run in several pieces

A run of 200 ns of SOD1 takes about 30 minutes on a GPU, but larger systems or longer runs take days, and
on a cluster a job stops at its time limit. The run must then be saved regularly and continued. Try it with
a short run that you stop by hand:

```bash
python $EX/cocomo/diffusion.py --preset sod1 --steps 200000 --report 1000 --checkpoint 10000 --output long
```

Stop it with Ctrl+C after at least one checkpoint has been written: the script writes one every 10000
steps, about every 1.5 minutes on a GPU, and the file `long.chk` appears. It holds the last checkpoint:
positions, velocities, fluid and random numbers. Now continue:

```bash
python $EX/cocomo/diffusion.py --preset sod1 --steps 200000 --report 1000 --checkpoint 10000 --output long --restart
```

The script prints the step it starts from and runs up to step 200000. The result is the same, bit for bit,
as a run that had never stopped. Look at the end of `long_com.txt` before and after the restart: the lines
written after the last checkpoint by the interrupted run have been removed and written again.

The page [saving and continuing a simulation](restart.md) explains how to do the same in your own scripts,
with `openmmlbm.LBMCheckpointReporter` and `openmmlbm.loadCheckpoint()`, and shows a script that you can
submit again and again on a cluster until the run is finished.

## 9. Where to go next

- The [examples page](examples.md) of this guide has shorter scripts on specific topics: a channel
  between two walls, monitoring the Mach number, saving and restoring the fluid, serialization, and a
  simulation with `openmm.app.Simulation` and reporters, the usual way to run long simulations with
  OpenMM.
- The [API reference](api_reference.md) describes every method of `LBMForce`.
- More examples with coarse-grained proteins are being prepared in `examples/`.
