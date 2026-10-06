# Examples

Complete scripts that use openmm-lbm from Python. Each one runs on its own, prints what it is doing and
explains its physics in the text at the top of the file (`python <script> --help` prints it).

If you are new to OpenMM, start from the [installation guide](../docs/user_guide/installation.md) and
the [tutorial](../docs/user_guide/tutorial.md), which goes through these examples in order.

| Script | What it shows | Default size | Time on an A100 GPU | Time on one CPU core (Reference) |
|---|---|---|---|---|
| [particle/kick.py](particle/kick.py) | a bead kicked in a fluid at rest, with and without the fluid | 100^3 nodes, 2000 steps | 2 s | 10 min |
| [particle/thermal.py](particle/thermal.py) | a bead brought to temperature by the fluid; full-step and half-step temperature | 10^3 nodes, 20000 steps | 4 s | 10 s |
| [particle/uniform_flow.py](particle/uniform_flow.py) | a bead dragged by a uniform flow; momentum and kinetic energy | 20^3 nodes, 1000 steps | 1 s | 3 s |
| [fluid/initial_state.py](fluid/initial_state.py) | a fluid started from a shear wave; saving and loading the fluid state | 32 x 32 x 4 nodes, 1000 steps | 1 s | 2 s |

The times include the start of Python and OpenMM. The thermal example reads the velocities at every
step, which costs more than the step itself on a GPU.

Examples with coarse-grained proteins (the COCOMO2 model, a peptide, ubiquitin with an elastic network,
an intrinsically disordered protein and SOD1) are being prepared.

## Running an example

Activate the environment where OpenMM and openmm-lbm are installed, go to a working directory and run
the script with Python:

```bash
conda activate lbm
mkdir -p ~/lbm-runs && cd ~/lbm-runs
python /path/to/openmm-lbm/examples/particle/kick.py
```

The output files are written in the current directory. Options common to all scripts:

- `--platform NAME`: `Reference`, `CUDA`, `OpenCL` or `HIP`. Without it the script takes CUDA, then
  OpenCL, then Reference, the first that is available.
- `--precision single|mixed|double`: precision on CUDA, OpenCL and HIP (default `mixed`).
- `--steps N`: number of steps. Every other parameter has an option too: see `--help`.

## The examples

### particle/kick.py

A bead starts with velocity v0 along x in a fluid at rest, at zero temperature (coupling scheme NVE:
drag only, no random force).

- With the fluid, the drag passes the momentum of the bead to the fluid at its node, and the fluid
  moves along with the bead. The velocity decays more slowly than an exponential and the bead travels
  further than v0/gamma: with the default parameters, 0.958 nm instead of 0.900 nm.
- With `--no-lb` the same bead is integrated by OpenMM's `LangevinMiddleIntegrator` at zero
  temperature: v(t) = v0 exp(-gamma t), with no hydrodynamics.
- `--preset alanine` uses a bead with the mass of an alanine residue in a lattice of 0.5 nm.

The output file `kick_bead_lb_on.txt` (or `_lb_off`) has one line per step: time, velocity (vx, vy, vz,
|v|) and position. Plot the speed against time on a logarithmic scale to see the difference between
the two cases.

### particle/thermal.py

One bead at rest in a box of 3 nm with 10^3 nodes; the friction and the random force of `LBMForce`
(the Euler-Maruyama scheme) bring it to 300 K. At the end the script prints the temperature computed
from the velocities that OpenMM stores, which are half a step behind the positions, and from the
full-step velocities (the mean of two consecutive steps). With `--beads N` there are N beads at random
positions.

- The default friction is small, 0.1/ps, so one bead needs about 10 ps to thermalize and its
  temperature fluctuates strongly. Use `--steps 200000` or more beads for a stable mean.
- `--removal N` removes the momentum of the fluid every N steps (default 0, never).
- `--seed S` fixes the random numbers: with the same seed, a run on the same platform is repeated
  exactly.

### particle/uniform_flow.py

The fluid starts with a uniform velocity of 0.09 nm/ps and a bead of one lattice cell at rest. The
table printed every 50 steps shows the bead reaching the velocity of the fluid, the total momentum of
bead and fluid (conserved to rounding, below 1e-13), and the kinetic energies. The total kinetic energy
decreases: the drag and the viscosity dissipate it (see [docs/theory.md](../docs/theory.md), section 2).

### fluid/initial_state.py

The fluid starts from a shear wave, u_x = U sin(2 pi y / L), built with `setFluidState()` from the
D3Q19 equilibrium; the script prints its amplitude next to the decay U exp(-nu k^2 t) of a Newtonian
fluid. `--save state.npz` writes the fluid state and the time at the end, and `--load state.npz` starts
a new run from them: two runs of 500 steps give exactly the run of 1000 steps. The function
`equilibrium_deviation()` can be reused to start the fluid from any density and velocity field.

## Comparison with the DragOpenMM plugin

These scripts are ports of the examples of the DragOpenMM plugin (the reference CUDA library of the
project), with the Euler-Maruyama coupling of openmm-lbm instead of OpenMM's Langevin integrator. With
the same parameters, the DragOpenMM plugin in double precision gives the same trajectories:

| Example | Platform of openmm-lbm | Largest difference from DragOpenMM |
|---|---|---|
| kick, bead, 2000 steps | CUDA double and mixed | velocity 1e-13 of v0, position 1e-11 of the distance |
| kick, bead, 2000 steps | CUDA single | velocity 5e-8 of v0 |
| kick, alanine, 100 steps | Reference, CUDA double and mixed | 7e-14 of v0 |
| thermal, 300 K, 20000 steps, same seed | CUDA double and mixed | 3e-11 of the largest velocity, including the random force |
| thermal, 300 K, 20000 steps, same seed | CUDA single, OpenCL | 2e-6 and 5e-6 |
| uniform flow, 400 steps | CUDA double | equal to the 6 digits printed |

The thermal run can be compared step by step because both plugins draw their random numbers from
OpenMM's generator in the same order. OpenCL generates them with slightly different rounding from
CUDA, hence the 5e-6.
