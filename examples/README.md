# Examples

Complete scripts that use openmm-lbm from Python. Each one runs on its own, prints what it is doing and
explains its physics in the text at the top of the file (`python <script> --help` prints it).

If you are new to OpenMM, start from the [installation guide](../docs/user_guide/installation.md) and
the [tutorial](../docs/user_guide/tutorial.md), which goes through these examples in order.

| Script | What it shows | Default size | Time on an A100 GPU | Time on one CPU core (Reference) |
|---|---|---|---|---|
| [particle/kick.py](particle/kick.py) | a bead kicked in a fluid at rest, with and without the fluid | $`100^3`$ nodes, 2000 steps | 2 s | 10 min |
| [particle/thermal.py](particle/thermal.py) | a bead brought to temperature by the fluid; full-step and half-step temperature | $`10^3`$ nodes, 20000 steps | 4 s | 10 s |
| [particle/uniform_flow.py](particle/uniform_flow.py) | a bead dragged by a uniform flow; momentum and kinetic energy | $`20^3`$ nodes, 1000 steps | 1 s | 3 s |
| [fluid/initial_state.py](fluid/initial_state.py) | a fluid started from a shear wave; saving and loading the fluid state | 32 x 32 x 4 nodes, 1000 steps | 1 s | 2 s |
| [cocomo/diffusion.py](cocomo/diffusion.py) | diffusion of a protein (COCOMO2): the folded SOD1 (110 beads) or a disordered one, with and without the fluid | $`30^3`$ nodes, 2000 steps (`--preset smoke`) to 2e7 steps (200 ns) | 2 s (smoke), 30 min (200 ns) | 25 s (smoke) |
| [cocomo/kick.py](cocomo/kick.py) | a peptide or ubiquitin kicked in the fluid | $`100^3`$ nodes, 2000 steps | 10 s | 20 s with `--nodes 30` |
| [cocomo/msd.py](cocomo/msd.py) | mean square displacement and diffusion coefficient of the centre of mass | - | - | seconds |

The times include the start of Python and OpenMM. The thermal example reads the velocities at every
step, which costs more than the step itself on a GPU.

## Running an example

Activate the environment where OpenMM and openmm-lbm are installed, go to a working directory and run
the script with Python:

```bash
conda activate lbm
mkdir -p ~/lbm-runs && cd ~/lbm-runs
python /path/to/openmm-lbm/examples/particle/kick.py
```

The output files are written in the current directory. Options common to the simulation scripts (not
`cocomo/msd.py` and `plot.py`, which analyse their output):

- `--platform NAME`: `Reference`, `CUDA`, `OpenCL` or `HIP`. Without it the script takes CUDA, then
  OpenCL, then Reference, the first that is available.
- `--precision single|mixed|double`: precision on CUDA, OpenCL and HIP (default `mixed`).
- `--drag Explicit|Centered`: drag scheme of the coupling (default `Explicit`; see
  [choosing the drag](../docs/user_guide/lattice.md#choosing-the-drag)); not in `fluid/initial_state.py`,
  which has no particles.
- `--steps N`: number of steps. Every other parameter has an option too: see `--help`.

## The examples

### particle/kick.py

A bead starts with velocity $`v_0`$ along $`x`$ in a fluid at rest, at zero temperature (coupling scheme NVE:
drag only, no random force).

- With the fluid, the drag passes the momentum of the bead to the fluid at its node, and the fluid
  moves along with the bead. The velocity decays more slowly than an exponential and the bead travels
  further than $`v_0/\gamma`$: with the default parameters, 0.958 nm instead of 0.900 nm.
- With `--no-lb` the same bead is integrated by OpenMM's `LangevinMiddleIntegrator` at zero
  temperature: $`v(t) = v_0\exp(-\gamma t)`$, with no hydrodynamics.
- `--preset alanine` uses a bead with the mass of an alanine residue in a lattice of 0.5 nm.

The output file `kick_bead_lb_on.txt` (or `_lb_off`) has one line per step: time, velocity (vx, vy, vz,
|v|) and position. Plot the speed against time on a logarithmic scale to see the difference between
the two cases.

### particle/thermal.py

One bead, started with the velocity $`\sqrt{k_BT/m}`$ along $`x`$, in a box of 3 nm with $`10^3`$ nodes; the
friction and the random force of `LBMForce` (the Euler-Maruyama scheme) bring it to 300 K. At the end the script
prints the temperature computed from the velocities that OpenMM stores, which are half a step behind the positions,
and from the full-step velocities (the mean of two consecutive steps). With `--beads N` there are N beads at random
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

The fluid starts from a shear wave, $`u_x = U\sin(2\pi y/L)`$, built with `setFluidState()` from the D3Q19
equilibrium; the script prints its amplitude next to the decay $`U\exp(-\nu k^2 t)`$ of a Newtonian fluid.
`--save state.npz` writes the fluid state and the time at the end, and `--load state.npz` starts a new run from
them: two runs of 500 steps give exactly the run of 1000 steps. The function `equilibrium_deviation()` can be reused
to start the fluid from any density and velocity field.

### cocomo/diffusion.py and cocomo/msd.py

A coarse-grained protein diffuses in a box of fluid. The model is COCOMO2
([cocomo/cocomo2.py](cocomo/cocomo2.py), see below). Every bead is coupled to the fluid, which is also
the thermostat; with `--no-lb` the beads are integrated by OpenMM's Langevin integrator at the same
friction, without hydrodynamics.

```bash
export EX=/path/to/openmm-lbm/examples                   # the folder of the examples
python $EX/cocomo/diffusion.py --preset smoke            # SOD1, 2000 steps: a quick check
python $EX/cocomo/diffusion.py --preset sod1 --seed 1    # SOD1, 200 ns, about 30 minutes on an A100
python $EX/cocomo/msd.py sod1_lb_on_com.txt              # diffusion coefficient of the protein
# with --drag Centered the prefix of the files is sod1_lb_on_centered
# with a fluctuating fluid use the centred drag: --drag Centered --fluid-fluctuations (prefix sod1_lb_on_centered_fluct)
# --vtk 10000 writes the fluid and the beads every 10000 steps for ParaView: open sod1_lb_on.pvd
```

- `--preset sod1` reproduces the runs of the DragOpenMM project with the folded protein SOD1 (box 15 nm, friction
  10/ps, 298 K, $`\Delta t`$ 10 fs, $`\nu = 5.0175`$ nm²/ps, removal of the fluid momentum at every step);
  `--preset sod1-g30` a larger box (30 nm) with friction 30/ps for 50 ns.
- `--preset rlp` is an intrinsically disordered protein of 166 residues, without elastic network (box 20 nm,
  friction 100/ps, $`\Delta t`$ 2 fs, $`\nu = 1.0035`$ nm²/ps, 10 ns), from the thermal-diffusion example of the
  DragOpenMM plugin. Its full-step temperature stays at about 260 K, 13% below 298 K: with this large friction the
  fluid, which has no thermal fluctuations of its own without `--fluid-fluctuations`, takes a large part of the
  momentum of the beads (docs/validation.md, T2 and T6). The DragOpenMM plugin gives the same value.
- The script writes the trajectory (DCD), the energies and temperature (OpenMM's log), the full-step and
  half-step temperature and the centre of mass of the protein, not wrapped into the box, which `msd.py`
  reads.
- **Long runs.** Every `--checkpoint` steps (default: every 100 reports) the script saves the whole run,
  fluid included, in `<prefix>.chk`. If the run stops, for example at the time limit of a job, the same
  command with `--restart` continues it from the last checkpoint up to `--steps`, exactly as an
  uninterrupted run: the text files are cut at the checkpoint and continued, and the trajectory continues in
  `<prefix>_<step>.dcd`. See [saving and continuing a simulation](../docs/user_guide/restart.md).
- For SOD1, `--domain` and `--enm-domain` choose the folded domain (residues counted from 1, both
  included) for the exposure scaling and for the elastic network. The defaults, 1-108 and 2-109, are
  what the original scripts did with their option `-d 1 109`, because of an offset of one residue in
  their loops; they are kept to reproduce those runs.

### cocomo/kick.py

The kick experiment of [particle/kick.py](particle/kick.py) with a protein: every bead starts at
9 nm/ps along x, at zero temperature. `--preset peptide` kicks the flexible peptide GRGDSPYS (8 beads),
`--preset ubiquitin` ubiquitin (76 beads) held folded by an elastic network over the whole chain. The
output has the velocity and position of the centre of mass and the radius of gyration at every step.
With the default box ($`100^3`$ nodes, 30 nm) a run takes about 10 seconds on a GPU; on the Reference
platform use a smaller box, for example `--nodes 30` (20 seconds).

Compared with the original scripts, the centre of mass is weighted by the masses (the original used
the plain mean of the positions), and the elastic network of ubiquitin covers the whole chain: the
original command line gave one-residue domains and therefore no elastic bonds. The exposure threshold
is the lambda = 0.7 of COCOMO2 (`--lam`; the original command used 0.9).

### cocomo/cocomo2.py

The COCOMO2 model (Jussupow, Bartley, Lapidus, Feig, J. Chem. Theory Comput. 21, 2095 (2025)) written
for these examples from the description in the article: bonds, angles, elastic network, a 10-5
short-range potential with cation-pi and pi-pi terms, Debye-Hueckel electrostatics, scaling of the
nonbonded terms by the solvent exposure. The text at the top of the file gives every term and
parameter. `read_beads()`, `exposure()`, `elastic_network()` and `create_system()` can be used for other
proteins. On SOD1 its energies and forces are equal, term by term, to those of the COCOMO2 script used
in the DragOpenMM project (energies identical to the last digit, forces within 1e-16 relative, also
away from the reference structure).

### Data

| File | Content | Source |
|---|---|---|
| `cocomo/data/sod1.pdb` | SOD1 coarse grained with one bead per residue (110 beads, positions in Angstrom) | prepared by the IBPC group (L. E. Coronas-Serna, F. Sterpone) for the thermal-diffusion runs of SOD1 with COCOMO2 |
| `cocomo/data/sod1.surface` | solvent accessible surface of each residue of `sod1.pdb`, in nm² | same |
| `cocomo/data/GRGDSPYS.pdb` | the peptide GRGDSPYS, one bead per residue | kick examples of the DragOpenMM plugin (IBPC group) |
| `cocomo/data/ubiquitin.pdb`, `cocomo/data/ubiquitin.surface` | ubiquitin, one bead per residue at the C-alpha positions (`ubione.min.pdb` of the original example), and the solvent accessible surface of each residue in nm² | same |
| `cocomo/data/rlp.pdb` | an intrinsically disordered protein of 166 residues, one bead per residue | thermal-diffusion example of the DragOpenMM plugin (IBPC group) |

## Comparison with the DragOpenMM plugin

These scripts are ports of the examples of the DragOpenMM plugin (the reference CUDA library of the
project), with the Euler-Maruyama coupling of openmm-lbm instead of OpenMM's Langevin integrator. With
the same parameters, the DragOpenMM plugin in double precision gives the same trajectories:

| Example | Platform of openmm-lbm | Largest difference from DragOpenMM |
|---|---|---|
| kick, bead, 2000 steps | CUDA double and mixed | velocity 1e-13 of $`v_0`$, position 1e-11 of the distance |
| kick, bead, 2000 steps | CUDA single | velocity 5e-8 of $`v_0`$ |
| kick, alanine, 100 steps | Reference | 7e-14 of $`v_0`$ |
| thermal, 300 K, 20000 steps, same seed | CUDA double and mixed | 3e-11 of the largest velocity, including the random force |
| thermal, 300 K, 20000 steps, same seed | CUDA single, OpenCL | 2e-6 and 5e-6 |
| uniform flow, 400 steps | CUDA double | equal to the 6 digits printed |
| protein kick, peptide (COCOMO2), 2000 steps | CUDA double | velocity of the centre of mass 1e-13 of $`v_0`$; positions equal to the printed digits |
| SOD1 diffusion, `--preset sod1`, three runs of 200 ns | CUDA mixed | apparent diffusion coefficient of the centre of mass 2.3-2.9 Å²/ns at lag times from 0.1 to 18 ns, against 2.3-2.6 in three runs of the DragOpenMM plugin (spread between runs 1.9-3.4 at 18 ns); $`k_BT/(M\gamma) = 2.26`$ Å²/ns |
| `--preset rlp`, 10 ns | CUDA mixed | full-step temperature 259.1 K and half-step 293.6 K at 298 K, against 259.9 K and 294.7 K for the DragOpenMM plugin (0.2 ns, double precision) |
| `--preset sod1-g30`, 50 ns | CUDA mixed | 0.89-1.0 Å²/ns between 0.1 and 1 ns, against 0.86 for a run of the DragOpenMM plugin (explicit drag); $`k_BT/(M\gamma) = 0.75`$ Å²/ns |

The thermal run can be compared step by step because both plugins draw their random numbers from
OpenMM's generator in the same order. OpenCL generates them with slightly different rounding from
CUDA, hence the 5e-6.
