# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Thermalization of free beads by the fluid with the Euler-Maruyama coupling.

The beads start at the centre of the box with velocity v0 along x, sqrt(kT/m) by default, and the
friction and the random force of LBMForce bring them to the temperature T.  The default parameters are
those of the old plugin's single-particle test: one bead of 5 lattice cells (80.7 Da) in a box of 3 nm
with 10^3 nodes, time step 1 fs, friction 0.1/ps, tau = 1, 300 K, 20000 steps.  With --beads N the beads
are placed at random in the box.

The script writes, for every step, the time (ps), the velocity vx vy vz (nm/ps) and the position x y z (nm) of
the first bead; OpenMM's leapfrog stores the velocities half a step behind the positions.  The temperature of the
beads is written at every step by the plugin's reporter, openmmlbm.LBMTemperatureReporter, in <output>_temperature.txt
(step, time in ps, temperature in K): it gives the right temperature for the drag scheme (docs/theory.md, section 2).
At the end the script prints its mean after --equilibration steps.  The kinetic temperature of one bead fluctuates
strongly: its mean converges slowly, with a correlation time of about 1/gamma.
"""

import argparse
import math
import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import openmmlbm
from openmmlbm import LBMForce


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--beads', type=int, default=1, help='number of beads (default 1)')
    parser.add_argument('--nodes', type=int, default=10, help='lattice nodes per side (default 10)')
    parser.add_argument('--spacing', type=float, default=0.3, help='lattice spacing in nm (default 0.3)')
    parser.add_argument('--dt', type=float, default=0.001, help='time step in ps (default 0.001)')
    parser.add_argument('--tau', type=float, default=1.0, help='relaxation time, sets the viscosity (default 1)')
    parser.add_argument('--density', type=float, default=993.0, help='fluid density in kg/m^3 (default 993)')
    parser.add_argument('--cells', type=float, default=5.0, help='bead mass in lattice cells (default 5)')
    parser.add_argument('--friction', type=float, default=0.1, help='friction in 1/ps (default 0.1)')
    parser.add_argument('--temperature', type=float, default=300.0, help='temperature in K (default 300)')
    parser.add_argument('--velocity', type=float, help='initial velocity along x in nm/ps (default sqrt(kT/m))')
    parser.add_argument('--steps', type=int, default=20000, help='number of steps (default 20000)')
    parser.add_argument('--equilibration', type=int, default=0,
                        help='steps excluded from the temperature averages (default 0)')
    parser.add_argument('--removal', type=int, default=0,
                        help='steps between removals of the fluid momentum, 0 = never (default, as the old example)')
    parser.add_argument('--seed', type=int, default=0, help='random number seed, 0 = chosen at random (default)')
    parser.add_argument('--drag', choices=['Explicit', 'Centered'], default='Explicit',
                        help='drag scheme of LBMForce (default Explicit)')
    parser.add_argument('--platform', help='OpenMM platform (default: CUDA, then OpenCL, then Reference)')
    parser.add_argument('--precision', default='mixed', help='precision on CUDA, OpenCL and HIP (default mixed)')
    parser.add_argument('--output', default='thermal.txt', help='output file (default thermal.txt)')
    return parser.parse_args()


def select_platform(name, precision):
    """The named platform, or the fastest available one; the precision property on CUDA, OpenCL and HIP."""
    names = [name] if name else ['CUDA', 'OpenCL', 'Reference']
    for candidate in names:
        try:
            platform = mm.Platform.getPlatformByName(candidate)
        except mm.OpenMMException:
            if name:
                raise
            continue
        properties = {'Precision': precision} if candidate in ('CUDA', 'OpenCL', 'HIP') else {}
        return platform, properties


class FirstBeadReporter:
    """Writes at every step the time (ps), the velocity (nm/ps, half step) and the position (nm, not wrapped into the
    box) of the first bead."""

    def __init__(self, out, dt):
        self._out = out
        self._dt = dt

    def describeNextReport(self, simulation):
        return {'steps': 1, 'periodic': False, 'include': ['positions', 'velocities']}

    def report(self, simulation, state):
        write_first_bead(self._out, simulation.currentStep*self._dt, state)


def write_first_bead(out, time, state):
    v = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
    x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    print('%.5f %.10e %.10e %.10e %.10e %.10e %.10e' % (time, *v[0], *x[0]), file=out)


def main():
    args = parse_arguments()
    kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
    density = (args.density*unit.kilogram/unit.meter**3*unit.AVOGADRO_CONSTANT_NA).value_in_unit(
        unit.dalton/unit.nanometer**3)
    mass = args.cells*density*args.spacing**3
    box = args.nodes*args.spacing
    viscosity = (args.tau - 0.5)/3*args.spacing**2/args.dt
    v0 = args.velocity if args.velocity is not None else math.sqrt(kB*args.temperature/mass)

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
    force = LBMForce()
    force.setGridSize(args.nodes, args.nodes, args.nodes)
    force.setFluidDensity(density)
    force.setKinematicViscosity(viscosity)
    force.setFriction(args.friction)
    force.setTemperature(args.temperature)
    force.setRandomNumberSeed(args.seed)
    force.setFluidMomentumRemovalFrequency(args.removal)
    force.setDragScheme(getattr(LBMForce, args.drag))
    for i in range(args.beads):
        system.addParticle(mass)
        force.addParticle(i)
    system.addForce(force)
    integrator = mm.VerletIntegrator(args.dt)
    platform, properties = select_platform(args.platform, args.precision)
    simulation = app.Simulation(app.Topology(), system, integrator, platform, properties)
    context = simulation.context
    if args.beads == 1:
        positions = np.full((1, 3), box/2)
    else:
        positions = np.random.default_rng(args.seed).uniform(0, box, (args.beads, 3))
    context.setPositions(positions)
    context.setVelocities(np.tile([v0, 0.0, 0.0], (args.beads, 1)))

    print('Platform %s, box %.2f nm, %d^3 nodes, dt %g ps, %d steps' % (platform.getName(), box, args.nodes,
                                                                       args.dt, args.steps))
    print('%d bead(s) of %.4f Da, v0 %.4f nm/ps, friction %g 1/ps (gamma dt %g), T %g K' % (
        args.beads, mass, v0, args.friction, args.friction*args.dt, args.temperature))
    print('Fluid: density %.4f Da/nm^3, viscosity %.4f nm^2/ps, tau %g' % (density, viscosity, args.tau))

    temperatureFile = os.path.splitext(args.output)[0] + '_temperature.txt'
    with open(args.output, 'w') as out:
        print('# time (ps), vx vy vz (nm/ps, half step), x y z (nm) of the first bead', file=out)
        write_first_bead(out, 0.0, context.getState(getPositions=True, getVelocities=True))
        simulation.reporters.append(FirstBeadReporter(out, args.dt))
        simulation.reporters.append(openmmlbm.LBMTemperatureReporter(temperatureFile, 1, force))
        simulation.step(args.steps)
        del simulation.reporters[:]

    temperature = np.loadtxt(temperatureFile, ndmin=2)
    temperature = temperature[temperature[:, 0] > args.equilibration]
    print('Mean temperature over %d steps: %.1f K (%s drag, set temperature %g K)' % (
        len(temperature), temperature[:, 2].mean(), args.drag.lower(), args.temperature))
    print('Trajectory written to %s, temperature to %s' % (args.output, temperatureFile))


if __name__ == '__main__':
    main()
