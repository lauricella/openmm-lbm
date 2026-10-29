# Examples

Complete scripts that use openmm-lbm from Python. Each one runs on its own, prints what it is doing and
explains its physics in the text at the top of the file (`python <script> --help` prints it).

If you are new to OpenMM, start from the [installation guide](../docs/user_guide/installation.md) and
the [tutorial](../docs/user_guide/tutorial.md), which goes through these examples in order.

| Script | What it shows | Default size | Time on an A100 GPU | Time on one CPU core (Reference) |
|---|---|---|---|---|
| [particle/kick.py](particle/kick.py) | a bead kicked in a fluid at rest, with and without the fluid | $`100^3`$ nodes, 2000 steps | 2 s | 10 min |
| [particle/thermal.py](particle/thermal.py) | a bead brought to temperature by the fluid; its temperature from the reporter of the plugin | $`10^3`$ nodes, 20000 steps | 6 s | 12 s |
| [particle/uniform_flow.py](particle/uniform_flow.py) | a bead dragged by a uniform flow; momentum and kinetic energy | $`20^3`$ nodes, 1000 steps | 1 s | 3 s |
| [fluid/initial_state.py](fluid/initial_state.py) | a fluid started from a shear wave; saving and loading the fluid state | 32 x 32 x 4 nodes, 1000 steps | 1 s | 2 s |
| [cocomo/diffusion.py](cocomo/diffusion.py) | diffusion of a protein (COCOMO2): the folded SOD1 (110 beads) or a disordered one, with and without the fluid | $`30^3`$ nodes, 2000 steps (`--preset smoke`) to 2e7 steps (200 ns) | 2 s (smoke), 30 min (200 ns) | 25 s (smoke) |
| [cocomo/kick.py](cocomo/kick.py) | a peptide, ubiquitin or SOD1 kicked in the fluid | $`100^3`$ nodes, 2000 steps | 10 s | 20 s with `--nodes 30` |
| [cocomo/msd.py](cocomo/msd.py) | mean square displacement and diffusion coefficient of the centre of mass | - | - | seconds |
| [cocomo/surface.py](cocomo/surface.py) | solvent accessible surface of each residue of an all-atom structure, the input of the exposure scaling | - | - | 2 s (159 residues) |

The times include the start of Python and OpenMM. The thermal example reads the velocities and writes the
temperature at every step, which costs more than the step itself on a GPU.

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
- `--viscosity NU`: kinematic viscosity in nm²/ps, in physical units. The plugin converts it to the relaxation time
  $`\tau = 3\nu\Delta t/\Delta x^2 + \tfrac12`$ of the lattice with the lattice spacing and the time step, and the
  script prints $`\tau`$ ([relaxation time](../docs/user_guide/lattice.md#relaxation-time)).
- `--temperature T`: temperature in K. Every simulation script has it, also those that run without noise by
  default. At $`T = 0`$ there is nothing stochastic, no random force on the particles and no thermal fluctuations of
  the fluid, but the drag stays: particles and fluid exchange momentum without a thermostat (the same as the coupling
  scheme `NVE`). The default is 0 in the kick and uniform-flow examples and in `fluid/initial_state.py`, 300 K in
  `particle/thermal.py` and 298 K in `cocomo/diffusion.py`.
- `--steps N`: number of steps. Every other parameter has an option too: see `--help`.

## The examples

### particle/kick.py

A bead starts with velocity $`v_0`$ along $`x`$ in a fluid at rest, at zero temperature (`--temperature 0`, the
default: drag only, no random force).

- With the fluid, the drag passes the momentum of the bead to the fluid at its node, and the fluid
  moves along with the bead. The velocity decays more slowly than an exponential and the bead travels
  further than $`v_0/\gamma`$: with the default parameters, 0.958 nm instead of 0.900 nm.
- With `--no-lb` the same bead is integrated by OpenMM's `LangevinMiddleIntegrator` at the same friction and
  temperature: at zero temperature $`v(t) = v_0\exp(-\gamma t)`$, with no hydrodynamics.
- With `--temperature 300` the bead also receives the random force of the coupling: its velocity decays to thermal
  fluctuations around zero.
- `--preset alanine` uses a bead with the mass of an alanine residue in a lattice of 0.5 nm.

The output file `kick_bead_lb_on.txt` (or `_lb_off`) has one line per step: time, velocity (vx, vy, vz,
|v|) and position. Plot the speed against time on a logarithmic scale to see the difference between
the two cases.

### particle/thermal.py

One bead, started with the velocity $`\sqrt{k_BT/m}`$ along $`x`$, in a box of 3 nm with $`10^3`$ nodes; the
friction and the random force of `LBMForce` (the Euler-Maruyama scheme) bring it to 300 K. The temperature of the
beads is written at every step in `thermal_temperature.txt` by the reporter of the plugin,
`openmmlbm.LBMTemperatureReporter`, which gives the right temperature for the drag scheme
([theory.md](../docs/theory.md), section 2), and the script prints its mean at the end. With `--beads N` (or `--particles N`) there are N beads at random positions.

- The default friction is small, 0.1/ps, so one bead needs about 10 ps to thermalize and its
  temperature fluctuates strongly. Use `--steps 200000` or more beads for a stable mean.
- `--removal N` removes the momentum of the fluid every N steps (default 0, never).
- `--seed S` fixes the random numbers: with the same seed, a run on the same platform is repeated
  exactly.

### particle/uniform_flow.py

The fluid starts with a uniform velocity of 0.09 nm/ps and a bead of one lattice cell at rest, at zero temperature (no
random force). The table printed every 50 steps shows the bead reaching the velocity of the fluid, the total momentum of
bead and fluid (conserved to rounding, below 1e-13), and the kinetic energies. The total kinetic energy decreases: the
drag and the viscosity dissipate it (see [docs/theory.md](../docs/theory.md), section 2).

### fluid/initial_state.py

The fluid starts from a shear wave, $`u_x = U\sin(2\pi y/L)`$, built with `setFluidState()` from the D3Q19
equilibrium; the script prints its amplitude next to the decay $`U\exp(-\nu k^2 t)`$ of a Newtonian fluid.
`--save state.npz` writes the fluid state and the time at the end, and `--load state.npz` starts a new run from
them: two runs of 500 steps give exactly the run of 1000 steps. The function `equilibrium_deviation()` can be reused
to start the fluid from any density and velocity field. With `--temperature 298` the fluid has thermal fluctuations:
the amplitude of the wave then fluctuates around the same decay, by 0.002 to 0.004 nm/ps with the default grid
(the mean of eight runs follows $`U\exp(-\nu k^2 t)`$ within its error), and a run continued with `--load` draws new
random numbers.

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
python $EX/cocomo/diffusion.py --preset sod1-g30 --drag Centered --fluid-fluctuations --seed 1
                                                         # fluctuating fluid, 50 ns, about 20 minutes on an A100
# with --drag Centered the prefix of the files is sod1_lb_on_centered
# with a fluctuating fluid use the centred drag: --drag Centered --fluid-fluctuations (prefix sod1_lb_on_centered_fluct)
# for water instead of five times its viscosity add --viscosity 0.893 (298 K) or --viscosity 0.696 --temperature 310
# --vtk 10000 writes the fluid and the beads every 10000 steps for ParaView: open sod1_lb_on.pvd
```

- `--preset sod1` reproduces the runs of the DragOpenMM project with the folded protein SOD1 (box 15 nm, friction
  10/ps, 298 K, $`\Delta t`$ 10 fs, $`\nu = 5.0175`$ nm²/ps, five times the viscosity of water at 20 °C, removal of the
  fluid momentum at every step);
  `--preset sod1-g30` a larger box (30 nm) with friction 30/ps for 50 ns.
- `--preset rlp` is an intrinsically disordered protein of 166 residues, without elastic network (box 20 nm,
  friction 100/ps, $`\Delta t`$ 2 fs, $`\nu = 1.0035`$ nm²/ps, water at 20 °C, 10 ns), from the thermal-diffusion
  example of the DragOpenMM plugin. Its full-step temperature stays at about 260 K, 13% below 298 K: with this large
  friction the fluid, which has no thermal fluctuations of its own without `--fluid-fluctuations`, takes a large part
  of the momentum of the beads (docs/validation.md, T2 and T6). The DragOpenMM plugin gives the same value.
- The script writes the trajectory (DCD, positions in Å as in every DCD file, wrapped into the box), the energies
  (OpenMM's log), the temperature of the beads and the centre of mass of the protein (`<prefix>_com.txt`, in nm, not
  wrapped into the box), which `msd.py` reads.
- **Temperature.** `<prefix>_temperature.txt` (step, time in ps, temperature in K) is written by the reporter of the
  plugin, `openmmlbm.LBMTemperatureReporter`, which gives the right temperature for the drag scheme
  ([theory.md](../docs/theory.md), section 2, Which velocity has the right temperature); with `--no-lb` by OpenMM. The
  end of the run prints its mean. With `--fluid-fluctuations` the script also writes
  `<prefix>_fluid_temperature.txt`, the temperature of the fluid every 10 reports, from the velocities of the nodes:
  the fluid of a cell, of mass $`m_c = \rho\Delta x^3`$, has $`k_BT/2`$ per component of its momentum
  ([theory.md](../docs/theory.md), section 7). With the preset `sod1-g30` (friction 30/ps), the centred drag and the
  fluctuating fluid keep the beads and the fluid at the set temperature, at 298 K and with `--temperature 310`;
  without fluctuations the fluid is cold and the beads are 16% below $`T`$ with the centred drag, 0.7% with the
  explicit one; the explicit drag with a fluctuating fluid makes them 20% too hot (numbers in
  [validation.md](../docs/validation.md#temperature-of-sod1-with-a-fluctuating-fluid-examplescocomodiffusionpy-cuda-nvidia-a100)).
- **Viscosity.** The presets `sod1` and `sod1-g30` use five times the viscosity of water, as the runs of the
  DragOpenMM project. For a protein in water pass the viscosity of water at the temperature of the run, in nm²/ps:
  `--viscosity 0.893` at 298 K, `--viscosity 0.696 --temperature 310` at 310 K; the plugin converts it to the
  relaxation time. With the preset `sod1-g30`, the centred drag and the fluctuating fluid the temperatures are right
  at this viscosity as well, and the protein diffuses 3 to 6 times faster: 11 to 22 Å²/ns instead of 3 to 4 Å²/ns
  (runs of 20 ns, numbers in
  [validation.md](../docs/validation.md#temperature-of-sod1-with-a-fluctuating-fluid-examplescocomodiffusionpy-cuda-nvidia-a100)).
- **Long runs.** Every `--checkpoint` steps (default: every 100 reports) the script saves the whole run,
  fluid included, in `<prefix>.chk`. If the run stops, for example at the time limit of a job, the same
  command with `--restart` continues it from the last checkpoint up to `--steps`, exactly as an
  uninterrupted run: the text files are cut at the checkpoint and continued, and the trajectory continues in
  `<prefix>_<step>.dcd`. See [saving and continuing a simulation](../docs/user_guide/restart.md).
- `--domain FIRST LAST` chooses the folded domain (residues counted from 1, both included): the residues whose
  nonbonded terms are scaled by their solvent exposure and, unless `--enm-domain` is also given, the residues
  joined by the elastic network. A folded domain normally has both, so `--domain` alone is the safe choice for a
  new protein. The SOD1 presets keep two ranges, 1-108 for the exposure and 2-109 for the elastic network: that is
  what the original scripts did with their option `-d 1 109`, because of an offset of one residue in their loops,
  and the presets reproduce those runs. With `--domain` alone both ranges become the same.

### cocomo/kick.py

The kick experiment of [particle/kick.py](particle/kick.py) with a protein: every bead starts at
9 nm/ps along x, at zero temperature by default. `--preset peptide` kicks the flexible peptide GRGDSPYS (8 beads),
`--preset ubiquitin` ubiquitin (76 beads) held folded by an elastic network over the whole chain, `--preset sod1`
the SOD1 of `diffusion.py`, with the same structure, exposure and elastic network (`--domain` and `--enm-domain`
work as in `diffusion.py`). The output has the velocity and position of the centre of mass and the radius of
gyration at every step. With the default box (30 nm, $`100^3`$ nodes) a run takes about 10 seconds on a GPU; on
the Reference platform use a smaller box, for example `--box 9` or `--nodes 30` (20 seconds). The momentum of the
fluid is never removed, so the protein ends at the velocity of the whole system, $`M v_0/(M + M_\text{fluid})`$.

- **At a temperature.** With `--temperature T` the beads receive the random force, and with `--fluid-fluctuations`
  (and `--drag Centered`) the fluid fluctuates too. `--equilibration N` runs N steps at T, from velocities drawn at
  T, before the kick adds $`v_0`$ along x to every bead. A single kick is then lost in the thermal velocity of the
  centre of mass, $`\sqrt{k_BT/M}`$ (about 0.015 nm/ps for SOD1 at 300 K), after about a picosecond: `--repeats N`
  runs N independent kicks (seeds `--seed`, `--seed` + 1, ...) and writes their mean, with the standard error of vx in a
  last column. To follow the decay down to a velocity $`V`$ about $`(\sqrt{k_BT/M}/V)^2`$ kicks are needed. In
  linear response the mean decay is the one at zero temperature.

  ```bash
  python $EX/cocomo/kick.py --preset sod1 --temperature 300 --fluid-fluctuations --drag Centered \
      --equilibration 5000 --repeats 100 --output kick_sod1_300K.txt
  ```

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

The surface file given to `exposure()` holds, for each residue, its solvent accessible surface area in nm²,
computed from the all-atom structure of the protein with atomic radii, as in COCOMO2: the reference surfaces
of the residues in `cocomo2.RESIDUES` are of the same kind. `ubiquitin.surface` is such a file. `sod1.surface`,
instead, was computed from the coarse-grained structure `sod1.pdb`, with one sphere of radius 0.2 nm per bead
(the default radius of Biopython for an atom that is not a chemical element) and a probe of 0.14 nm; it is the
file of the original runs of the DragOpenMM project, and the SOD1 presets keep it to reproduce them.

`surface.py` writes such a file from an all-atom PDB file, by the method of Shrake and Rupley with the van der
Waals radii of Bondi and a probe of 0.14 nm, one area per residue in nm², in the order of the residues:

```bash
python $EX/cocomo/surface.py protein.pdb --output protein.surface
```

Hydrogen atoms, water and residues that are not amino acids are left out (`--hydrogens` keeps the hydrogens). An
atom whose element is not in the table of radii stops the script, so a coarse-grained file gives an error instead
of a surface. The coarse-grained model must have one bead for each residue of the file, in the same order. With
the default 960 points per atom the error on the area of an atom is 0.14 % of its sphere on average, at most
0.7 %; on dihydrofolate reductase (OpenMM's `5dfr_minimized.pdb`, 159 residues) the total is 89.8 nm².

### Data

| File | Content | Source |
|---|---|---|
| `cocomo/data/sod1.pdb` | SOD1 coarse grained with one bead per residue (110 beads, positions in Angstrom) | prepared by the IBPC group (L. E. Coronas-Serna, F. Sterpone) for the thermal-diffusion runs of SOD1 with COCOMO2 |
| `cocomo/data/sod1.surface` | solvent accessible surface of each residue of `sod1.pdb`, in nm², computed from the coarse-grained structure (see cocomo2.py above) | same |
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
