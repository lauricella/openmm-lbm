# Glossary

The words used in this guide, explained briefly, with a link to where they are explained in full.

**Body acceleration.** A uniform acceleration g applied to the whole fluid, as gravity or a pressure
gradient would: it drives a flow, for example in a channel. Set with `setBodyAcceleration()`
([lattice](lattice.md#velocity-of-the-fluid)).

**Bounce-back.** The rule used at solid nodes: what the fluid sends into a wall comes back the way it came,
so that the fluid does not slip along the wall (no-slip) ([theory.md](../theory.md)).

**Checkpoint.** A file with everything needed to continue a simulation exactly where it stopped. OpenMM's
checkpoints do not contain the fluid; `openmmlbm.saveCheckpoint()` saves both ([restart](restart.md)).

**Context.** In OpenMM, the running simulation: a System, an Integrator and a Platform together, with the
current positions and velocities ([tutorial](tutorial.md#1-the-pieces-of-an-openmm-simulation)).

**Coupled particle.** A particle added to the force with `addParticle()`: it feels the friction and the
random force of the fluid, and the fluid feels it. The other particles do not see the fluid.

**Coupling scheme.** How the coupled particles and the fluid exchange forces. `EulerMaruyama` (the default):
friction and random force; `NVE`: friction only, as at zero temperature ([API](api_reference.md)).

**Euler-Maruyama.** The simple explicit rule used to add friction and random force over one time step.

**Friction (gamma).** How strongly a coupled particle is dragged towards the velocity of the fluid, in 1/ps.
With the explicit drag (the default) the product friction x dt must be below 1, and is best at 0.1 or below;
the centred drag is stable for any friction ([recipe](lattice.md#quick-recipe),
[choosing the drag](lattice.md#choosing-the-drag)).

**Full step, half step.** `VerletIntegrator` (a "leapfrog" integrator) stores the velocities half a step
behind the positions. The velocity at the same time as the positions, the full step, is the mean of two
consecutive stored velocities. Temperatures computed from the two differ
([examples](examples.md#temperature-of-coupled-particles)).

**Kinematic viscosity (nu).** The viscosity divided by the density, in nm^2/ps; for water about 1 nm^2/ps.
It sets how fast velocity differences in the fluid are smoothed out.

**Lattice, node, spacing (dx).** The fluid is described on a regular cubic grid that fills the periodic box:
the points of the grid are the nodes, and dx is the distance between neighbouring nodes
([lattice](lattice.md#geometry)).

**Lattice Boltzmann.** The method used for the fluid: at every node, 19 numbers (the populations) describe
how much fluid moves in each of 19 directions; at every step they collide and move to the neighbouring nodes
([theory.md](../theory.md)).

**Lattice units.** The internal units of the fluid, in which dx, dt and the mass of a cell are 1. You never
need them: the API uses OpenMM units ([lattice](lattice.md#units-on-the-lattice)).

**Mach number (Ma).** The speed of the fluid divided by the speed of sound of the lattice. The model is
accurate only when it is small: below 0.1; above 0.3 the plugin stops
([lattice](lattice.md#mach-number-and-stability)).

**Platform.** Where OpenMM computes: `Reference` (one processor core, slow, the reference for correctness),
`CUDA` and `OpenCL` (GPUs), `HIP` (AMD GPUs).

**Precision.** The number format used on a GPU platform: `single`, `mixed` (recommended) or `double`.

**Relaxation time (tau).** The parameter of the lattice Boltzmann method that corresponds to the viscosity:
tau = 3 nu dt/dx^2 + 1/2. It must stay between about 0.505 and 2, and below about 1.7 with coupled particles
and the explicit drag ([lattice](lattice.md#relaxation-time)).

**Removal of the fluid momentum.** At regular steps the plugin subtracts the mean velocity of the fluid,
so that the fluid as a whole stays at rest ([lattice](lattice.md#removal-of-the-fluid-momentum)).

**Reporter.** In `openmm.app.Simulation`, an object that writes data every few steps: a trajectory
(`DCDReporter`), energies and temperature (`StateDataReporter`), checkpoints (`LBMCheckpointReporter`).

**Solid nodes.** Nodes that hold no fluid and act as walls, set with `setSolidNodes()`
([API](api_reference.md#solid-nodes)).

**State.** In OpenMM, a snapshot of a Context: positions, velocities, forces, energies at one time.

**System.** In OpenMM, the description of what is simulated: particles with their masses, the periodic
box, and the forces.

**Temperature of the fluid.** By default the fluid of this plugin has no thermal fluctuations of its own: only
the coupled particles receive a random force ([limitations](README.md#limitations-of-the-model)). With
`setFluidFluctuations(True)` the fluid fluctuates at the temperature of
the force.

**Time step (dt).** The step of the integrator, which is also the step of the lattice: the fluid advances
once per integration step.
