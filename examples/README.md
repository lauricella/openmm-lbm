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
| [cocomo/shear.py](cocomo/shear.py) | SOD1 in a double Couette flow between two open faces moving in opposite directions: drift, rotation, profile of the fluid | 60 x 60 x 100 nodes, 1e6 steps (10 ns) | 11 min on an NVIDIA A30 | 20 s with `--box 10 10 --height 12 --steps 1000` |
| [cocomo/poiseuille.py](cocomo/poiseuille.py) | SOD1 in a Poiseuille flow, a channel between two walls driven by a uniform acceleration: drift, rotation, profile of the fluid | 60 x 60 x 100 nodes, 1e6 steps (10 ns) | - | 22 s with `--box 10 10 --height 12 --steps 1000` |
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
- `--drag Explicit|Centered`: drag scheme of the coupling (default `Explicit`, `Centered` in `cocomo/shear.py` and
  `cocomo/poiseuille.py`; see
  [choosing the drag](../docs/user_guide/lattice.md#choosing-the-drag)); not in `fluid/initial_state.py`,
  which has no particles.
- `--viscosity NU`: kinematic viscosity in nm²/ps, in physical units. The plugin converts it to the relaxation time
  $`\tau = 3\nu\Delta t/\Delta x^2 + \tfrac12`$ of the lattice with the lattice spacing and the time step, and the
  script prints $`\tau`$ ([relaxation time](../docs/user_guide/lattice.md#relaxation-time)).
- `--temperature T`: temperature in K. Every simulation script has it, also those that run without noise by
  default. At $`T = 0`$ there is nothing stochastic, no random force on the particles and no thermal fluctuations of
  the fluid, but the drag stays: particles and fluid exchange momentum without a thermostat (the same as the coupling
  scheme `NVE`). The default is 0 in the kick and uniform-flow examples and in `fluid/initial_state.py`, 300 K in
  `particle/thermal.py` and 298 K in `cocomo/diffusion.py`, `cocomo/shear.py` and `cocomo/poiseuille.py`.
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

- `--preset sod1` runs the folded protein SOD1 (box 15 nm, friction
  10/ps, 298 K, $`\Delta t`$ 10 fs, $`\nu = 5.0175`$ nm²/ps, five times the viscosity of water at 20 °C, removal of the
  fluid momentum at every step);
  `--preset sod1-g30` a larger box (30 nm) with friction 30/ps for 50 ns.
- `--preset rlp` is an intrinsically disordered protein of 166 residues, without elastic network (box 20 nm,
  friction 100/ps, $`\Delta t`$ 2 fs, $`\nu = 1.0035`$ nm²/ps, water at 20 °C, 10 ns). Its full-step temperature
  stays at about 260 K, 13% below 298 K: with this large friction the fluid, which has no thermal fluctuations of its
  own without `--fluid-fluctuations`, takes a large part of the momentum of the beads (docs/validation.md, T2 and T6).
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
- **Viscosity.** The presets `sod1` and `sod1-g30` use five times the viscosity of water. For a protein in water pass the viscosity of water at the temperature of the run, in nm²/ps:
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
  new protein. The SOD1 presets keep two ranges, 1-108 for the exposure and 2-109 for the elastic network. With
  `--domain` alone both ranges become the same.
- `--pdb` and `--surface` run another structure with the parameters of the preset (its structure, surface and
  domains are then not used; a structure without domain options is a disordered protein), also one with several
  proteins, each with its own folded domains (`--domains FILE`): see
  [several proteins in one structure](#several-proteins-in-one-structure).

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

The centre of mass is weighted by the masses, and the elastic network of ubiquitin covers the whole chain. The
exposure threshold is the lambda = 0.7 of COCOMO2 (`--lam`).

### cocomo/shear.py

SOD1 in a flow out of equilibrium, a double Couette flow. The box is open along $`z`$, between two `DensityVelocity`
faces at the density of the fluid that move along $`x`$ in opposite directions, $`-v`$ at `ZMin` and $`+v`$ at `ZMax`
([open faces](../docs/user_guide/api_reference.md#open-faces)), and periodic along $`x`$ and $`y`$. The faces hold
their velocity one node beyond the box, so the shear rate is $`\dot\gamma = 2v/((n_z + 1)\Delta x)`$ and the flow is
at rest at $`z_0 = (n_z - 1)\Delta x/2`$, half a spacing below the middle of the box, where the protein starts.
Nothing holds the protein, neither a restraint nor a wall: it turns about $`y`$ with the vorticity of the flow and is
carried along $`x`$ when it diffuses away from $`z_0`$. The protein is the SOD1 of `diffusion.py`, with the same
structure, exposure and elastic network (`--domain` and `--enm-domain` work as there). The fluid starts with the
linear profile of the faces (`setFluidState()`, as in `fluid/initial_state.py`). The defaults are those of a protein
in water: 298 K, $`\nu`$ = 0.893 nm²/ps, friction 30/ps, centred drag and fluctuating fluid, a box of
30 x 30 x 50 nm (`--box 30 30 --height 50`), $`v`$ = 0.5 nm/ps ($`\dot\gamma`$ = 0.0198/ps) and 10 ns, which take
11 minutes on an NVIDIA A30 (1370 ns/day; 2400 ns/day without fluctuations of the fluid).

```bash
python $EX/cocomo/shear.py --velocity 0.5 --steps 1000000 --output shear
```

- **Speed of the faces.** The faces must move well below the speed of sound of the lattice,
  $`c_s = \Delta x/(\sqrt3\,\Delta t)`$, 28.9 nm/ps with $`\Delta x`$ = 0.5 nm and $`\Delta t`$ = 0.01 ps. The script
  refuses faces faster than Mach 0.1, 2.9 nm/ps. Keep them at 1 nm/ps or below (Mach 0.035): the fluctuating fluid
  adds to the flow a thermal velocity of standard deviation $`\sqrt{k_BT/(\rho\Delta x^3)}`$ per node and component,
  0.18 nm/ps at 298 K, and its largest values on the lattice reach about five times that. Faces moving at 1 nm/ps
  keep the fluid at the temperature of the run on every plane, the planes of the faces included
  ([validation.md](../docs/validation.md#fluctuating-fluid-teststestlbmfluctuationsh-all-platforms)).
- **Shear rate.** The shear rates within reach are those of molecular dynamics: $`v`$ = 0.5 nm/ps over 50 nm gives
  $`\dot\gamma = 2\cdot10^{10}`$ /s. With a rotation time of SOD1 in water of about 7 ns (Stokes-Einstein-Debye with
  a hydrodynamic radius of 2 nm) the Weissenberg number $`\dot\gamma\tau_r`$ is about 140, and about 30 at
  $`v`$ = 0.1 nm/ps. A Weissenberg number of 1 needs $`v`$ of about 0.004 nm/ps with this height, far below the
  thermal velocity of the fluid: the flow then shows only in averages over long runs.
- **Height and sides.** `--height` keeps the protein away from the faces: in water it diffuses by about 5 nm along
  $`z`$ in 100 ns. A bead that crossed a face would be coupled to the fluid next to the opposite face, which moves the
  other way: the coupling wraps the positions into the periodic box of OpenMM, and the faces are not a Lees-Edwards
  boundary. `--box LX LY` gives the sides along $`x`$ and $`y`$: the protein interacts through the fluid with its
  periodic images at that distance.
- **Wall potential.** The faces neither reflect the beads nor push them back. `--wall-potential wca` or `hertz` adds a
  repulsive potential of the two faces on every bead, an OpenMM `CustomExternalForce` (`wall_potential()` in the
  script), with $`d`$ the distance of the bead from the face, $`z + \Delta x/2`$ from `ZMin` and
  $`(n_z - 1/2)\Delta x - z`$ from `ZMax`:
  - `wca`: $`U(d) = 4\varepsilon[(\sigma/d)^{12} - (\sigma/d)^6] + \varepsilon`$ for $`d < 2^{1/6}\sigma`$, zero beyond:
    impenetrable;
  - `hertz`: $`U(d) = \varepsilon\,\max(0, 1 - d/\sigma)^{5/2}`$: soft, at most $`\varepsilon`$ at the face, so a
    bead pushed hard can still cross it.

  `--wall-epsilon` gives $`\varepsilon`$ in kJ/mol (default 2.5 for `wca`, about $`k_BT`$ at 300 K, and 125 for
  `hertz`, about $`50\,k_BT`$) and `--wall-sigma` gives $`\sigma`$ in nm (default 0.5, one lattice spacing). The
  force is along $`z`$ only: along the faces the protein slides freely, and only the fluid drags it. The default,
  `none`, gives the runs without the potential. A run continued with `--restart` needs the same options.
- **Output.** `<prefix>_motion.txt` has, every `--report` steps, the centre of mass (not wrapped into the box), the
  velocity of the imposed flow at its height, the rotation of the protein from its starting structure as a
  quaternion (best fit weighted by the masses) and the angle of rotation about $`y`$ accumulated since the start.
  `<prefix>_profile.txt` has the mean velocity $`u_x`$ of the fluid on each plane of nodes, every `--profile`
  steps (default 10000, 100 ps). The trajectory, the temperature of the beads, the checkpoints and `--restart` work as in `diffusion.py`;
  the accumulated angle is written with all its digits, so that a restarted run continues it exactly.
- **ParaView.** `--vtk N` writes every N steps the fluid, density and velocity on the nodes, and the beads, with
  their positions, velocities and masses, in VTK files in OpenMM units (`openmmlbm.LBMVTKReporter`, as in
  `diffusion.py`): open `<prefix>.pvd` in ParaView for the whole series. The velocity of the fluid is the whole
  velocity, the Couette flow plus the thermal fluctuations, which in a single snapshot are of the same size as the
  flow (0.18 nm/ps per node against 0.5 nm/ps at the faces): the profile shows in averages over planes or over
  time. With the default box the files of the fluid of one snapshot take about 6 MB, so `--vtk 10000` (every
  100 ps) writes about 0.6 GB in 10 ns. A restarted run rewrites the files after the checkpoint and keeps the
  series in order.
- **Rotation.** A sphere turns at $`\dot\gamma/2`$. A body of another shape turns faster or slower as its
  orientation changes (the Jeffery orbits of an ellipsoid), and its mean rate differs from $`\dot\gamma/2`$. In a
  run of 10 ns with the defaults (NVIDIA A30, mixed precision) SOD1 turned about $`y`$ at
  $`0.81 \pm 0.02\,\dot\gamma/2`$ (blocks of 1 ns between 0.73 and 0.89); over windows of 100 ps its rate went from
  0.35 to 1.3 $`\dot\gamma/2`$ and back twice per turn, as for an elongated body. Runs of 1 ns with a fluid
  without fluctuations and the explicit drag, at zero temperature and at 298 K, gave 0.74, within the spread of the
  blocks: the changes of
  the rate come from the shape, not from the noise. The beads were at 299.5 K: the temperature of
  `<prefix>_temperature.txt` includes the velocity of the flow at the beads.

### cocomo/poiseuille.py

SOD1 in a Poiseuille flow, the companion of `shear.py`: a channel between two walls, driven along $`x`$ by a
uniform acceleration $`g`$ of the fluid (`setBodyAcceleration()`), the force of a pressure gradient $`\rho g`$. The
plane of nodes $`k = 0`$ is solid, with bounce-back walls, and the lattice is periodic, so the plane bounds the channel
on both sides: the walls lie halfway between the solid plane and the first fluid planes, at $`z = \Delta x/2`$ and
$`z = (n_z - 1/2)\Delta x`$, the channel is $`H = (n_z - 1)\Delta x`$ wide and its centre line is the middle of the
box. The steady profile is the parabola
$`u_x(z) = g/(2\nu)\,(z - \Delta x/2)\,((n_z - 1/2)\Delta x - z)`$ plus the uniform slip of the bounce-back walls,
$`g\Delta x^2(16\Lambda - 3)/(24\nu)`$ with $`\Lambda = (\tau - 1/2)/2`$, the exact solution of the scheme
([theory.md](../docs/theory.md#solid-nodes-and-walls)), $`-7\cdot10^{-5}`$ of the centre-line speed with the
defaults. The script takes the speed at the centre line (`--velocity`, default 0.5 nm/ps) and computes
$`g = 8\nu v/H^2`$; the fluid starts with the steady profile and the beads with the flow at their height. The
protein starts on the centre line, or `--position` nm away from it (negative below), and nothing holds it.

```bash
python $EX/cocomo/poiseuille.py                       # SOD1 on the centre line, v = 0.5 nm/ps, 10 ns
python $EX/cocomo/poiseuille.py --position -10        # 10 nm below the centre line
```

- **What to expect.** On the centre line the shear rate is zero: the protein is carried along $`x`$ at about the
  centre-line speed and does not turn on average. The curvature of the profile makes it lag slightly behind the fluid
  at its centre of mass ($`a^2 g/(6\nu)`$ for a sphere of radius $`a`$, Faxen's law). Away from the centre line it
  turns about $`y`$ with the local vorticity, at about half the local shear rate, $`(g/2\nu)(z_c - z)`$, in opposite
  directions on the two sides of the centre line. The end of the run prints the mean velocity of the centre of mass
  along $`x`$ and the mean flow at its height, and the mean rate of rotation.
- **Speed.** As in `shear.py`: the script refuses a centre line faster than Mach 0.1 (2.9 nm/ps), and 1 nm/ps or
  less (Mach 0.035) is recommended. With $`v`$ = 0.5 nm/ps in a channel of 49.5 nm, $`g = 1.5\cdot10^{-3}`$ nm/ps²
  (a pressure gradient of $`1.5\cdot10^{15}`$ Pa/m), the shear rate is 0.04/ps at the walls and 0.004/ps at 2.5 nm
  from the centre line, and the Reynolds number of the channel, $`vH/\nu`$, is 28.
- **Walls.** A coupled bead whose nearest node is solid and that moves into the wall has its velocity reversed
  ([coupled particles](../docs/user_guide/api_reference.md#coupled-particles), Walls): unlike the open faces of
  `shear.py`, nothing crosses the walls. Bounce-back walls keep the fluctuating fluid at equilibrium up to the wall.
  The script refuses a `--position` that puts a bead within one spacing of a wall. `--wall-potential wca` or `hertz`,
  with `--wall-epsilon` and `--wall-sigma`, adds the repulsive potential of `shear.py` on the wall planes
  $`z = \Delta x/2`$ and $`(n_z - 1/2)\Delta x`$, which keeps the beads off the walls before they reach the solid
  nodes; the reflection stays.
- **Temperature.** `<prefix>_temperature.txt` has the temperature of the beads from their velocities relative to the
  imposed flow at the height of each bead: the translation with the flow, 0.5 nm/ps on the centre line, would add
  about 1000 K to the temperature of `openmmlbm.LBMTemperatureReporter`.
- **Output.** The other files are those of `shear.py`: motion (centre of mass, flow at its height, rotation),
  profile of the fluid every `--profile` steps (the solid plane has zero velocity), trajectory, checkpoints with
  `--restart`, and `--vtk N` for ParaView. `poiseuille.py` takes the reporters from `shear.py`, which must stay next
  to it.

### Several proteins in one structure

`cocomo/diffusion.py`, `cocomo/kick.py`, `cocomo/shear.py` and `cocomo/poiseuille.py` run any coarse-grained
structure (`--pdb`), also one with several proteins, or several copies of one, each folded protein held by its own
elastic network.

- **The PDB file.** One bead per residue, as `cocomo2.read_beads()` reads it: the residue name in columns 18-20 and
  the position in Å in columns 31-54. The chain of a bead is its segment identifier (columns 73-76) or, without it,
  its chain identifier (column 22). Consecutive beads with the same identifier form one chain, and bonds and angles
  join them. **Give every protein its own identifier**: two proteins that follow each other with the same identifier
  are one chain, bonded at the junction. The chains are numbered 1, 2, ... in the order of the file.
- **The folded domains of every chain** are given by `--domains FILE`, a text file with one line per chain, in the
  order of the chains:

  ```text
  # one line per chain, in the order of the PDB file
  1 108            # chain 1: one folded domain, residues 1 to 108 of the chain
  10 60, 70 150    # chain 2: two folded domains
  -                # chain 3: no folded domain (a disordered protein)
  ```

  - On each line, pairs `FIRST LAST` separated by commas: residue numbers counted from 1 inside the chain (the
    position of the bead in its chain, not the residue number written in the PDB file), both included, in increasing
    order and without overlaps.
  - `-` alone: a chain without folded domain.
  - `#` starts a comment; empty lines are skipped.
  - The file must have exactly one line per chain. The script stops with an error that names the file, the line and
    the chain for a wrong number of lines, a domain outside its chain, overlapping domains or a malformed pair.

  The script converts the domains to the numbering of the whole structure (`cocomo2.read_domains()`) and prints them
  at the start of the run. For chains of 100, 150 and 200 beads, the lines `10 12, 20 23`, `2 10` and
  `1 5, 13 34, 38 42` become the domains 10-12, 20-23, 102-110, 251-255, 263-284 and 288-292 of the structure.
- **What a domain does.** In a folded domain the nonbonded terms are scaled by the solvent exposure of each residue,
  and the elastic network joins the beads at least three residues apart and closer than 0.9 nm in the starting
  structure ([cocomo2.py](#cocomococomo2py)). The network joins only beads of the same domain: each folded structure is
  held on its own, and two chains are never joined, nor two domains of one chain (give one range to hold a whole
  chain). Springs between proteins, to hold a complex together, are not part of the model here: they would be an
  extra force of the System, for example a `HarmonicBondForce` between chosen beads.
- **Elastic network and exposure apart.** `--enm-domains FILE`, in the same format, gives the domains of the elastic
  network when they differ from those of the exposure (as the SOD1 presets, 1-108 and 2-109). Without it,
  `--domains` sets both. `--domain FIRST LAST` and `--enm-domain FIRST LAST` remain for one range in the numbering
  of the whole structure; ranges and files cannot be mixed.
- **Surfaces.** `--surface` takes one file with the surface of every residue of the structure, or one file per chain
  in the order of the chains, with `-` in place of the file of a chain without folded domain. Each file has one value
  per residue, in nm² (`surface.py` writes it from the all-atom structure of the protein). A folded domain on a chain
  given `-` is an error.
- **Output by chain.** With more than one chain, `diffusion.py` also writes `<prefix>_com_<k>.txt`, the centre of
  mass of chain $`k`$, which `msd.py` reads like `<prefix>_com.txt`; `shear.py` and `poiseuille.py` write
  `<prefix>_motion_<k>.txt`, the motion and rotation of chain $`k`$, instead of `<prefix>_motion.txt`. `kick.py`
  kicks every bead and writes the motion of all the beads together. The structure is centred as a whole: in the box
  (`diffusion.py`, `kick.py`), on the plane where the flow is at rest (`shear.py`) or on the centre line, moved by
  `--position` (`poiseuille.py`).
- **Box.** Make the box large enough for all the proteins, with the distances that the structure gives them: the
  scripts do not move the proteins apart (the energy minimization at the start removes small clashes).

For example, two copies of SOD1, the second with two domains, and a disordered peptide in one structure:

```bash
cat > three.domains <<'END'
1 108
2 50, 60 109
-
END
python $EX/cocomo/diffusion.py --preset sod1-g30 --pdb three.pdb --surface sod1.surface sod1.surface - \
    --domains three.domains --output three
python $EX/cocomo/msd.py three_com_1.txt
python $EX/cocomo/shear.py --pdb three.pdb --surface $EX/cocomo/data/sod1.surface $EX/cocomo/data/sod1.surface - \
    --domains three.domains --height 60 --wall-potential wca
```

`diffusion.py` looks for a `--pdb` or `--surface` file that is not in the working directory in `cocomo/data/`; the
other scripts take the paths as given.

### cocomo/cocomo2.py

The COCOMO2 model (Jussupow, Bartley, Lapidus, Feig, J. Chem. Theory Comput. 21, 2095 (2025)) written
for these examples from the description in the article: bonds, angles, elastic network, a 10-5
short-range potential with cation-pi and pi-pi terms, Debye-Hueckel electrostatics, scaling of the
nonbonded terms by the solvent exposure. The text at the top of the file gives every term and
parameter. `read_beads()`, `exposure()`, `elastic_network()` and `create_system()` can be used for other
proteins, and `chain_ranges()`, `read_domains()`, `read_surfaces()` and `folded_domains()` for structures with
several chains ([several proteins in one structure](#several-proteins-in-one-structure)).

The surface file given to `exposure()` holds, for each residue, its solvent accessible surface area in nm²,
computed from the all-atom structure of the protein with atomic radii, as in COCOMO2: the reference surfaces
of the residues in `cocomo2.RESIDUES` are of the same kind. `ubiquitin.surface` is such a file. `sod1.surface`,
instead, was computed from the coarse-grained structure `sod1.pdb`, with one sphere of radius 0.2 nm per bead
(the default radius of Biopython for an atom that is not a chemical element) and a probe of 0.14 nm; the SOD1
presets use it.

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
| `cocomo/data/GRGDSPYS.pdb` | the peptide GRGDSPYS, one bead per residue | IBPC group |
| `cocomo/data/ubiquitin.pdb`, `cocomo/data/ubiquitin.surface` | ubiquitin, one bead per residue at the C-alpha positions, and the solvent accessible surface of each residue in nm² | same |
| `cocomo/data/rlp.pdb` | an intrinsically disordered protein of 166 residues, one bead per residue | IBPC group |

