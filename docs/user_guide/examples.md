# Examples

Complete scripts that run on the Reference platform in a few seconds. Each one can be copied into a
file and run as it is. The outputs shown were obtained with OpenMM 8.6.1.

1. [Channel flow between two walls](#channel-flow-between-two-walls): solid nodes, body force,
   comparison with the analytical profile, changing a parameter during the run.
2. [A particle kicked in the fluid](#a-particle-kicked-in-the-fluid): drag, momentum passed to the fluid.
3. [Temperature of coupled particles](#temperature-of-coupled-particles): a reporter of the full-step
   temperature.
4. [Monitoring a run and the Mach number check](#monitoring-a-run-and-the-mach-number-check).
5. [Saving and restoring the fluid](#saving-and-restoring-the-fluid).
6. [Serialization](#serialization).
7. [Using openmm.app.Simulation](#using-openmmappsimulation): a reporter for the fluid, checkpoints.

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

# In the steady state the walls carry the whole body force on the fluid, g times its mass.
density, velocity = force.getFluidFields(context)
mass = np.sum(density.value_in_unit(unit.dalton/unit.nanometer**3))*dx**3
wall = force.getWallForce(context)[0].value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
print('force on the walls %.4f kJ/mol/nm, body force g M %.4f kJ/mol/nm' % (wall, g*mass))

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
force on the walls 481.7712 kJ/mol/nm, body force g M 481.7712 kJ/mol/nm
decay rate 0.0992 /ps, slowest viscous mode pi^2 nu/H^2 = 0.0990 /ps
```

Notes:

- **Where the walls are.** With halfway bounce-back the walls lie halfway between the solid planes
  and the first fluid planes, at y0 = dx/2 and y1 = (ny - 3/2) dx. The channel is H = (ny - 2) dx =
  10 nm wide.
- **Agreement with the parabola.** The profile has exactly the curvature of the parabola. The residual
  difference of 0.17% is a uniform slip of the bounce-back scheme, which depends on tau and vanishes at
  tau = 7/8. With this slip included, the profile matches the exact solution of the scheme to 1e-9
  (see [theory.md](../theory.md#solid-nodes-implemented-on-all-platforms)).
- **Periodic directions.** The walls are planes of solid nodes. In x and z the lattice stays periodic,
  so the channel is infinite along the flow.
- **Momentum removal.** It must be off (`setFluidMomentumRemovalFrequency(0)`): otherwise the plugin
  would subtract the momentum given by the acceleration.
- **Force on the walls.** `getWallForce()` measures the momentum that the fluid gives to the solid nodes
  (momentum exchange method). In the steady state it balances the body force on the fluid.
- **Changing a parameter during the run.** `updateParametersInContext()` applies the new body
  acceleration from the next step, without touching the fluid. After the driving stops, the flow decays
  at the rate of the slowest viscous mode of the channel.
- **Other geometries.** Any set of nodes can be solid: a cylinder, a pore, a rough wall. Build the
  mask with NumPy, as here, and pass the indices to `setSolidNodes()`.

## A particle kicked in the fluid

A particle of 100 Da starts with velocity v0 = 1 nm/ps in a fluid at rest. The temperature is zero, so
there is no random force. The drag slows the particle down and passes its momentum to the fluid, and the
total momentum of particle and fluid is conserved.

```python
import numpy as np
import openmm as mm
import openmm.unit as unit
from openmmlbm import LBMForce

L, n, dt = 8.0, 16, 0.01           # box (nm), nodes per side, time step (ps): dx = 0.5 nm
mass, friction, v0 = 100.0, 10.0, 1.0

system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0), mm.Vec3(0, L, 0), mm.Vec3(0, 0, L))
system.addParticle(mass)
force = LBMForce()
force.setGridSize(n, n, n)
force.setKinematicViscosity(5.0175)            # tau = 1.10
force.setFriction(friction)
force.setTemperature(0.0)                      # no random force: a deterministic kick
force.setFluidMomentumRemovalFrequency(0)      # keep the momentum given to the fluid
force.addParticle(0)
system.addForce(force)

integrator = mm.VerletIntegrator(dt)
context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
context.setPositions([mm.Vec3(4.0, 4.0, 4.0)])
context.setVelocities([mm.Vec3(v0, 0, 0)])

cellMass = force.getFluidDensity().value_in_unit(unit.dalton/unit.nanometer**3)*(L/n)**3
def fluid_momentum():
    """Total momentum of the fluid (Da nm/ps) from its populations."""
    f = np.array(force.getFluidState(context)).reshape(19, -1)
    cx = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1])
    return (cx[:, None]*f).sum()*cellMass*(L/n)/dt

print(' t (ps)   v/v0     (1-gamma dt)^n   p_fluid/p0   p_total/p0')
for block in range(6):
    t = context.getTime().value_in_unit(unit.picosecond)
    v = context.getState(getVelocities=True).getVelocities()[0][0].value_in_unit(unit.nanometer/unit.picosecond)
    pf = fluid_momentum()
    print('%5.2f   %.4f   %.4f          %.4f       %.9f' % (t, v/v0, (1 - friction*dt)**round(t/dt), pf/(mass*v0), (mass*v + pf)/(mass*v0)))
    integrator.step(10)
```

Output:

```
 t (ps)   v/v0     (1-gamma dt)^n   p_fluid/p0   p_total/p0
 0.00   1.0000   1.0000          0.0000       1.000000000
 0.10   0.3560   0.3487          0.6440       1.000000000
 0.20   0.1297   0.1216          0.8703       1.000000000
 0.30   0.0483   0.0424          0.9517       1.000000000
 0.40   0.0185   0.0148          0.9815       1.000000000
 0.50   0.0074   0.0052          0.9926       1.000000000
```

Notes:

- **Explicit drag.** With the fluid at rest, one step multiplies the velocity by 1 - gamma dt = 0.9.
  The particle slows down more slowly than (1 - gamma dt)^n, because the fluid at its node starts to move
  with it. This hydrodynamic response is what the lattice Boltzmann fluid adds to a Langevin thermostat.
- **Momentum.** The momentum lost by the particle is in the fluid: p_total/p0 = 1 to the rounding of the
  sum over the populations. The removal of the fluid momentum must be off, otherwise the plugin would
  subtract it.
- **Long times.** Particle and fluid end up moving together at P/(m + M_fluid), 3e-4 v0 here.

## Temperature of coupled particles

The random force at temperature T and the friction keep the coupled particles at about T: `LBMForce`
is their thermostat. OpenMM's leapfrog stores the velocities at half steps. The temperature that OpenMM
reports (`StateDataReporter`) is that of the full step, because OpenMM shifts the velocities by half a
step with the coupling force of the next step
([theory.md](../theory.md#2-particle-fluid-coupling-implemented-on-all-platforms)). The
reporter below computes both temperatures explicitly: the full-step one from the mean of the velocities
of two consecutive steps, and the half-step one from the stored velocities.

```python
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmlbm import LBMForce


class FullStepTemperatureReporter:
    """Writes the temperature of the coupled particles computed from full-step velocities.

    With the leapfrog of VerletIntegrator the stored velocities are at half steps.  The reporter takes the
    velocities of two consecutive steps, n - 1 and n, and uses their mean, the velocity at the full step.
    """

    def __init__(self, file, reportInterval, particles):
        self._out = open(file, 'w') if isinstance(file, str) else file
        self._reportInterval = reportInterval
        self._particles = np.array(particles)
        self._previous = None
        print('# step, full-step temperature (K), half-step temperature (K)', file=self._out)

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        if steps > 1:
            steps -= 1                    # first stop one step before the report
        return {'steps': steps, 'periodic': None, 'include': ['velocities']}

    def report(self, simulation, state):
        v = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)[self._particles]
        if simulation.currentStep%self._reportInterval != 0:
            self._previous = v
            return
        system = simulation.system
        m = np.array([system.getParticleMass(int(i)).value_in_unit(unit.dalton) for i in self._particles])
        kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
        dof = 3*len(m)
        half = (m[:, None]*v**2).sum()/(dof*kB)
        if self._previous is None:
            print('%d nan %.1f' % (simulation.currentStep, half), file=self._out, flush=True)
            return
        full = (m[:, None]*((v + self._previous)/2)**2).sum()/(dof*kB)
        print('%d %.1f %.1f' % (simulation.currentStep, full, half), file=self._out, flush=True)


# 200 free beads of 100 Da coupled to the fluid at 300 K, gamma dt = 0.1.
numBeads, L = 200, 8.0
topology = app.Topology()
chain = topology.addChain()
system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0), mm.Vec3(0, L, 0), mm.Vec3(0, 0, L))
force = LBMForce()
force.setGridSize(16, 16, 16)
force.setKinematicViscosity(5.0175)
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
simulation.context.setPositions(np.random.default_rng(1).uniform(0, L, (numBeads, 3)))
simulation.context.setVelocitiesToTemperature(300.0, 3)
simulation.reporters.append(FullStepTemperatureReporter('temperature.txt', 50, range(numBeads)))
simulation.step(5000)

data = np.loadtxt('temperature.txt')[20:]          # skip the first 1000 steps
print('full step %.0f K, half step %.0f K' % (data[:, 1].mean(), data[:, 2].mean()))
```

Output:

```
full step 293 K, half step 308 K
```

Notes:

- **Full step.** Slightly below T: the fluid has no thermal fluctuations of its own and takes part of the
  momentum of the particles. Over 20000 steps the same system gives 295.8 +- 0.4 K, 1.4% below T
  ([validation.md](../validation.md)).
- **Half step.** For a free particle it is T/(1 - gamma dt/2) = 315.8 K, lowered by the same factor:
  311.7 +- 0.4 K over 20000 steps.
- **OpenMM's own temperature.** `StateDataReporter(..., temperature=True)` gives the full-step
  temperature at the report step; the reporter above computes it one step earlier, from steps n - 1 and n.
  On this system the means over the same reports are 292 K and 293 K, equal within the statistical error.
- **Using the reporter.** It works with any Simulation: pass the indices of the coupled particles. It
  stops one step before each report to record the velocities.

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


# First run: 105 steps, then save the OpenMM checkpoint and the fluid.
force, integrator, context = create_context()
integrator.step(105)
with open('run.chk', 'wb') as f:
    f.write(context.createCheckpoint())
np.save('run_fluid.npy', np.array(force.getFluidState(context)))
integrator.step(95)                      # the first run continues, for comparison
final = np.array(force.getFluidState(context))

# Second run: a new Context restarts from the saved files.
force, integrator, context = create_context()
with open('run.chk', 'rb') as f:
    context.loadCheckpoint(f.read())
force.setFluidState(context, np.load('run_fluid.npy'))
integrator.step(95)
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
- **Any step.** The removal of the momentum and the Mach number check follow the step count of the
  Context, which the checkpoint restores. The restart can therefore happen at any step: here at step
  105, while the removal runs every 10 steps.
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
to a standard `StateDataReporter`, and saves the run. The beads and the fluid exchange momentum, so the
mean velocity of the fluid also changes in y and z. Fixed seeds, for the random force and for the initial
velocities, make the run reproducible.

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
force.setRandomNumberSeed(1)                       # a fixed seed makes the run reproducible
force.setBodyAcceleration(mm.Vec3(0.1, 0, 0))
force.setFluidMomentumRemovalFrequency(0)
for i in range(numBeads):
    force.addParticle(i)
system.addForce(force)

simulation = app.Simulation(topology, system, mm.VerletIntegrator(0.01*unit.picosecond),
                            mm.Platform.getPlatformByName('Reference'))
simulation.context.setPositions([mm.Vec3(1.0 + 0.38*i, 2.5, 2.5) for i in range(numBeads)])
kT = (unit.MOLAR_GAS_CONSTANT_R*300.0*unit.kelvin).value_in_unit(unit.kilojoule_per_mole)
simulation.context.setVelocities(np.random.default_rng(1).normal(0, np.sqrt(kT/100.0), (numBeads, 3)))

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
1 0.0983797 0.00133459 0.0012744 0.00451306
2 0.195794 0.000860838 0.000198581 0.00759161
3 0.296034 0.000747286 0.00100197 0.0116595
4 0.394395 0.000109059 0.00082861 0.0172864
5 0.492879 0.00146615 0.000630083 0.0186065
```

To continue the run later, create the Simulation in the same way and restore both files:

```python
simulation.loadCheckpoint('run.chk')
force.setFluidState(simulation.context, np.load('run_fluid.npy'))
simulation.step(500)
```

Notes:

- **Integrator.** Only `VerletIntegrator` is accepted: friction and random force are part of
  `LBMForce`, which is the thermostat of the coupled particles.
- **Initial velocities.** They come from NumPy here because `setVelocitiesToTemperature()` gives
  different velocities in different OpenMM versions; either is fine in practice.
- **Temperature.** The temperature column of `StateDataReporter` is that of the full step, as in the
  [temperature example](#temperature-of-coupled-particles).
- **Reporters.** A reporter can read the fluid with any of the methods of `LBMForce` that take the
  Context; none of them advances the fluid.
- **Which particles to couple.** Only the particles passed to `addParticle()` interact with the fluid.
- **GPU platforms.** To run on CUDA, replace the platform with `mm.Platform.getPlatformByName('CUDA')`
  and pass `{'Precision': 'mixed'}`. Every example runs unchanged; the random forces, and therefore
  the outputs with T > 0, differ from those of the Reference platform (see the
  [status table](README.md#what-works-in-this-version)).
