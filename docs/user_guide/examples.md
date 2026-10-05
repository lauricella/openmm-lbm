# Examples

Complete scripts that run on the Reference platform in a few seconds. Each one can be copied into a
file and run as it is. The outputs shown were obtained with OpenMM 8.6.1.

1. [Channel flow between two walls](#channel-flow-between-two-walls): solid nodes, body force,
   comparison with the analytical profile, changing a parameter during the run.
2. [Monitoring a run and the Mach number check](#monitoring-a-run-and-the-mach-number-check).
3. [Saving and restoring the fluid](#saving-and-restoring-the-fluid).
4. [Serialization](#serialization).
5. [Using openmm.app.Simulation](#using-openmmappsimulation): a reporter for the fluid, checkpoints.

The page on [the lattice](lattice.md#node-indexing-and-numpy-arrays) shows how to turn the fluid
fields into NumPy arrays indexed by node.

## Channel flow between two walls

A fluid between two parallel walls, driven by a uniform acceleration g along x, reaches the parabolic
Poiseuille profile u(y) = g/(2 nu) (y - y0)(y1 - y), where y0 and y1 are the positions of the walls.
The script below builds the channel with two planes of solid nodes and runs until the flow is steady.
It then compares the profile with the parabola, switches the acceleration off with
`updateParametersInContext()`, and measures how fast the flow decays.

```python
import numpy as np
import openmm as mm
import openmm.unit as unit
from openmmlbm import LBMForce

nx, ny, nz = 4, 22, 4                 # 20 fluid planes between two solid planes
dx = 0.5                              # nm
dt = 0.01                             # ps
nu = 1.0035                           # nm^2/ps
g = 0.02                              # nm/ps^2, along x

system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(nx*dx, 0, 0), mm.Vec3(0, ny*dx, 0), mm.Vec3(0, 0, nz*dx))
system.addParticle(1.0)               # OpenMM needs at least one particle; it is not coupled

force = LBMForce()
force.setGridSize(nx, ny, nz)
force.setKinematicViscosity(nu)
force.setBodyAcceleration(mm.Vec3(g, 0, 0))
force.setFluidMomentumRemovalFrequency(0)      # keep the momentum given by the body force

# Solid planes j = 0 and j = ny-1; node (i, j, k) has index i + nx*(j + ny*k).
k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
index = i + nx*(j + ny*k)
force.setSolidNodes(index[(j == 0) | (j == ny - 1)])
system.addForce(force)

integrator = mm.VerletIntegrator(dt)
context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
context.setPositions([mm.Vec3(0, 0, 0)])

def ux_profile():
    """Velocity along x averaged over each plane j, in nm/ps."""
    density, velocity = force.getFluidFields(context)
    u = np.array(velocity.value_in_unit(unit.nanometer/unit.picosecond)).reshape(nz, ny, nx, 3)
    return u[..., 0].mean(axis=(0, 2))

integrator.step(20000)                # 200 ps, about 20 viscous times
u = ux_profile()[1:-1]                # fluid planes j = 1 .. ny-2

# The walls lie halfway between the solid and the first fluid plane.
y = np.arange(1, ny - 1)*dx
y0, y1 = 0.5*dx, (ny - 1.5)*dx
parabola = g/(2*nu)*(y - y0)*(y1 - y)
dx_, dt_, tau = force.getLatticeParametersInContext(context)
Lambda = (tau - 0.5)/2
exact = parabola + g*dx**2*(16*Lambda - 3)/(24*nu)   # slip of halfway bounce-back, docs/theory.md
print('tau = %.4f, Mach = %.4f' % (tau, force.getFluidMachNumber(context)))
print('centre: %.6f nm/ps, parabola %.6f, exact solution of the scheme %.6f' % (u.max(), parabola.max(), exact.max()))
print('max relative deviation: from the parabola %.1e, from the exact solution %.1e'
      % (np.abs(u - parabola).max()/parabola.max(), np.abs(u - exact).max()/exact.max()))

# Switch the driving off and follow the decay of the centre-line velocity.
force.setBodyAcceleration(mm.Vec3(0, 0, 0))
force.updateParametersInContext(context)
integrator.step(1000)
u1 = ux_profile().max()
integrator.step(1000)                 # 10 ps later
u2 = ux_profile().max()
H = (ny - 2)*dx
print('decay rate %.4f /ps, slowest viscous mode pi^2 nu/H^2 = %.4f /ps' % (np.log(u1/u2)/(1000*dt), np.pi**2*nu/H**2))
```

Output:

```
tau = 0.6204, Mach = 0.0086
centre: 0.248082 nm/ps, parabola 0.248505, exact solution of the scheme 0.248082
max relative deviation: from the parabola 1.7e-03, from the exact solution 2.5e-09
decay rate 0.0992 /ps, slowest viscous mode pi^2 nu/H^2 = 0.0990 /ps
```

Notes:

- **Where the walls are.** With halfway bounce-back the walls lie halfway between the solid planes
  and the first fluid planes, at y0 = dx/2 and y1 = (ny - 3/2) dx. The channel is H = (ny - 2) dx =
  10 nm wide.
- **Agreement with the parabola.** The profile has exactly the curvature of the parabola. The residual
  difference of 0.17% is a uniform slip of the bounce-back scheme, which depends on tau and vanishes at
  tau = 7/8. With this slip included, the profile matches the exact solution of the scheme to 1e-9
  (see [theory.md](../theory.md#solid-nodes-implemented-on-the-reference-platform)).
- **Periodic directions.** The walls are planes of solid nodes. In x and z the lattice stays periodic,
  so the channel is infinite along the flow.
- **Momentum removal.** It must be off (`setFluidMomentumRemovalFrequency(0)`): otherwise the plugin
  would subtract the momentum given by the acceleration.
- **Changing a parameter during the run.** `updateParametersInContext()` applies the new body
  acceleration from the next step, without touching the fluid. After the driving stops, the flow decays
  at the rate of the slowest viscous mode of the channel.
- **Other geometries.** Any set of nodes can be solid: a cylinder, a pore, a rough wall. Build the
  mask with NumPy, as here, and pass the indices to `setSolidNodes()`.

## Monitoring a run and the Mach number check

The model is accurate only while the fluid is slow compared with the lattice sound speed (see
[Mach number and stability](lattice.md#mach-number-and-stability)). This script pushes the fluid with
an acceleration that is far too strong, prints the Mach number while it grows, and catches the
exception raised by the check.

```python
import openmm as mm
import openmm.unit as unit
from openmmlbm import LBMForce

system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
system.addParticle(100.0)
force = LBMForce()
force.setGridSize(8, 8, 8)
force.setFluidMomentumRemovalFrequency(0)
force.setBodyAcceleration(mm.Vec3(5.0, 0, 0))   # far too strong: the fluid speeds up without limit
force.setMachCheckFrequency(50)                 # check every 50 steps instead of 100
system.addForce(force)
integrator = mm.VerletIntegrator(0.01)
context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
context.setPositions([mm.Vec3(1, 1, 1)])

try:
    for block in range(20):
        integrator.step(25)
        print('t = %4.2f ps  Mach = %.3f' % (context.getTime().value_in_unit(unit.picosecond),
                                            force.getFluidMachNumber(context)))
except Exception as error:
    print('stopped after %d steps: %s' % (context.getStepCount(), error))
```

Output:

```
t = 0.25 ps  Mach = 0.043
t = 0.50 ps  Mach = 0.087
t = 0.75 ps  Mach = 0.130
t = 1.00 ps  Mach = 0.173
t = 1.25 ps  Mach = 0.217
t = 1.50 ps  Mach = 0.260
t = 1.75 ps  Mach = 0.303
stopped after 199 steps: LBMForce: the Mach number of the fluid is 0.34641 after 200 lattice steps, above the limit 0.3 (docs/theory.md, section 6). Reduce the forces on the fluid, the time step or the friction.
```

Notes:

- `getFluidMachNumber()` can be called at any time and does not advance the fluid. Calling it every
  few hundred steps is a cheap way to watch a run.
- The check runs only after the lattice steps whose number is a multiple of the check frequency. At
  1.75 ps (step 175) the Mach number is already above 0.3, but the run stops at the next check, at step
  200.
- The exception is raised inside the 200th step, which OpenMM does not count: the step count is 199.
  After the exception the run should not continue; restart it from a saved state with different
  parameters.

## Saving and restoring the fluid

OpenMM checkpoints and `State` objects do not contain the fluid. To restart a run, save the
checkpoint and the fluid state together, and restore both in the new Context. This script checks that
the restarted run is identical to an uninterrupted one.

```python
import numpy as np
import openmm as mm
from openmmlbm import LBMForce


def create_context():
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    system.addParticle(100.0)
    force = LBMForce()
    force.setGridSize(8, 8, 8)
    force.setBodyAcceleration(mm.Vec3(0.05, 0.02, 0))
    force.setFluidMomentumRemovalFrequency(10)
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.01)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions([mm.Vec3(1, 1, 1)])
    return force, integrator, context


# First run: 100 steps, then save the OpenMM checkpoint and the fluid.
force, integrator, context = create_context()
integrator.step(100)
with open('run.chk', 'wb') as f:
    f.write(context.createCheckpoint())
np.save('run_fluid.npy', np.array(force.getFluidState(context)))
integrator.step(100)                     # the first run continues, for comparison
final = np.array(force.getFluidState(context))

# Second run: a new Context restarts from the saved files.
force, integrator, context = create_context()
with open('run.chk', 'rb') as f:
    context.loadCheckpoint(f.read())
force.setFluidState(context, np.load('run_fluid.npy'))
integrator.step(100)
print('steps:', context.getStepCount())
print('largest difference from the uninterrupted run:', np.abs(np.array(force.getFluidState(context)) - final).max())
```

Output:

```
steps: 200
largest difference from the uninterrupted run: 0.0
```

Notes:

- **Same grid.** The fluid state can be restored only in a Context with the same grid size; it holds
  19 numbers per node.
- **Counters restart at zero.** The counters of the momentum removal and of the Mach number check
  start from zero in every new Context. To reproduce an uninterrupted run exactly, save at a step that
  is a multiple of both frequencies, as here: the removal every 10 steps and the check every 100.
  Otherwise the restarted run removes the momentum at shifted steps; it remains a valid simulation, but
  not an identical one.
- **`Context.reinitialize()`** also resets the fluid to its initial state. Use the same two calls
  around it: `getFluidState()` before and `setFluidState()` after.
- **Positions and velocities.** `context.loadCheckpoint()` restores the particles, the time and the
  step count; `setFluidState()` restores the fluid. Both are needed.

## Serialization

An `LBMForce` is saved by `XmlSerializer` with all its parameters, as part of a System or alone.

```python
import numpy as np
import openmm as mm
from openmmlbm import LBMForce

system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
system.addParticle(100.0)
force = LBMForce()
force.setGridSize(8, 8, 8)
force.setKinematicViscosity(5.0)
force.setSolidNodes(range(64))                  # the plane k = 0
force.addParticle(0)
system.addForce(force)

with open('system.xml', 'w') as f:
    f.write(mm.XmlSerializer.serialize(system))

with open('system.xml') as f:
    system2 = mm.XmlSerializer.deserialize(f.read())
force2 = [LBMForce.cast(f) for f in system2.getForces() if LBMForce.isinstance(f)][0]
print(force2.getGridSize(), force2.getKinematicViscosity(), len(force2.getSolidNodes()), force2.getNumParticles())

# A force on its own is deserialized as a generic Force.
xml = mm.XmlSerializer.serialize(force)
force3 = LBMForce.cast(mm.XmlSerializer.deserialize(xml))
print(force3.getKinematicViscosity())
```

Output:

```
[8, 8, 8] 5.0 nm**2/ps 64 1
5.0 nm**2/ps
```

`openmmlbm` must be imported before deserializing, so that OpenMM knows how to read an `LBMForce`. The
fluid of a Context is not part of the XML; save it as in the [previous example](#saving-and-restoring-the-fluid).

## Using openmm.app.Simulation

`LBMForce` works with `openmm.app.Simulation` like any other force. This script builds a chain of
coarse-grained beads and couples them to the fluid. It records the fluid with a custom reporter next
to a standard `StateDataReporter`, and saves the run. In this version the coupling is not implemented,
so the beads move as if the fluid were absent; the setup does not change when it is.

```python
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmlbm import LBMForce


class FluidReporter:
    """Writes the time, the mean velocity of the fluid and its Mach number every reportInterval steps."""

    def __init__(self, file, reportInterval, force):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self._force = force
        print('# time (ps), mean fluid velocity ux uy uz (nm/ps), Mach number', file=self._out)

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return {'steps': steps, 'periodic': None, 'include': []}

    def report(self, simulation, state):
        density, velocity = self._force.getFluidFields(simulation.context)
        u = np.array(velocity.value_in_unit(unit.nanometer/unit.picosecond)).mean(axis=0)
        mach = self._force.getFluidMachNumber(simulation.context)
        time = state.getTime().value_in_unit(unit.picosecond)
        print('%g %g %g %g %g' % (time, u[0], u[1], u[2], mach), file=self._out, flush=True)


# A chain of 10 beads of 100 Da joined by harmonic bonds, in a box of 5 nm.
numBeads, box = 10, 5.0
topology = app.Topology()
chain = topology.addChain()
for i in range(numBeads):
    topology.addAtom('B', None, topology.addResidue('BEA', chain))
topology.setPeriodicBoxVectors([mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box)])

system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
bonds = mm.HarmonicBondForce()
for i in range(numBeads):
    system.addParticle(100.0)
    if i > 0:
        bonds.addBond(i - 1, i, 0.38, 1000.0)
system.addForce(bonds)

force = LBMForce()
force.setGridSize(10, 10, 10)                       # dx = 0.5 nm
force.setFriction(5.0)
force.setTemperature(300.0)
force.setBodyAcceleration(mm.Vec3(0.1, 0, 0))
force.setFluidMomentumRemovalFrequency(0)
for i in range(numBeads):
    force.addParticle(i)                            # coupling not implemented yet: no force on the beads
system.addForce(force)

simulation = app.Simulation(topology, system, mm.VerletIntegrator(0.01*unit.picosecond),
                            mm.Platform.getPlatformByName('Reference'))
simulation.context.setPositions([mm.Vec3(1.0 + 0.38*i, 2.5, 2.5) for i in range(numBeads)])
simulation.context.setVelocitiesToTemperature(300.0)

simulation.reporters.append(FluidReporter('fluid.txt', 100, force))
simulation.reporters.append(app.StateDataReporter('state.txt', 100, step=True, time=True, potentialEnergy=True))
simulation.step(500)

# Save the run: the OpenMM checkpoint and, separately, the fluid.
simulation.saveCheckpoint('run.chk')
np.save('run_fluid.npy', np.array(force.getFluidState(simulation.context)))

print(open('fluid.txt').read(), end='')
```

Output:

```
# time (ps), mean fluid velocity ux uy uz (nm/ps), Mach number
1 0.1005 0 0 0.0034641
2 0.2005 0 0 0.0069282
3 0.3005 0 0 0.0103923
4 0.4005 0 0 0.0138564
5 0.5005 0 0 0.0173205
```

To continue the run later, create the Simulation in the same way and restore both files:

```python
simulation.loadCheckpoint('run.chk')
force.setFluidState(simulation.context, np.load('run_fluid.npy'))
simulation.step(500)
```

Notes:

- **Integrator.** Only `VerletIntegrator` is accepted: when the coupling is implemented, friction and
  noise will be part of `LBMForce`, which will act as the thermostat of the coupled particles.
- **Reporters.** A reporter can read the fluid with any of the methods of `LBMForce` that take the
  Context; none of them advances the fluid.
- **Which particles to couple.** Only the particles passed to `addParticle()` will interact with the
  fluid.
- **GPU platforms.** To run on CUDA, replace the platform with `mm.Platform.getPlatformByName('CUDA')`
  and pass `{'Precision': 'mixed'}`. In this version the fluid does not advance on the GPU platforms
  yet, and solid nodes are not accepted there (see the
  [status table](README.md#what-works-in-this-version)).
