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

OpenMM's leapfrog stores the velocities half a step behind the positions.  The script writes, for every
step, the time (ps), the velocity vx vy vz (nm/ps) and the position x y z (nm) of the first bead, and at
the end it prints the temperature of the beads computed from the half-step velocities and from the
full-step velocities, the mean of two consecutive steps (docs/theory.md, section 2): with the explicit
drag (the default) the full-step one is right, with the centred drag (--drag Centered) the half-step one.
The kinetic temperature of one bead fluctuates strongly: its mean converges slowly, with a correlation time of
about 1/gamma.
"""

import argparse
import math

import numpy as np
import openmm as mm
import openmm.unit as unit
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
    parser.add_argument('--precision', default='mixed', help='precision on CUDA and OpenCL (default mixed)')
    parser.add_argument('--output', default='thermal.txt', help='output file (default thermal.txt)')
    return parser.parse_args()


def select_platform(name, precision):
    """The named platform, or the fastest available one; the precision property on CUDA and OpenCL."""
    names = [name] if name else ['CUDA', 'OpenCL', 'Reference']
    for candidate in names:
        try:
            platform = mm.Platform.getPlatformByName(candidate)
        except mm.OpenMMException:
            if name:
                raise
            continue
        properties = {'Precision': precision} if candidate in ('CUDA', 'OpenCL') else {}
        return platform, properties


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
    context = mm.Context(system, integrator, platform, properties)
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

    # Sums of m v^2 over the beads and the steps after equilibration, at half and full steps.
    halfSum, fullSum, count = 0.0, 0.0, 0
    previous = None
    with open(args.output, 'w') as out:
        print('# time (ps), vx vy vz (nm/ps, half step), x y z (nm) of the first bead', file=out)
        for step in range(args.steps + 1):
            if step > 0:
                integrator.step(1)
            state = context.getState(getPositions=True, getVelocities=True)
            v = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
            x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            print('%.5f %.10e %.10e %.10e %.10e %.10e %.10e' % (step*args.dt, *v[0], *x[0]), file=out)
            if step > args.equilibration and previous is not None:
                halfSum += mass*(v**2).sum()
                fullSum += mass*(((v + previous)/2)**2).sum()
                count += 1
            previous = v

    dof = 3*args.beads*count
    a = args.friction*args.dt
    print('Mean temperature over %d steps: full step %.1f K, half step %.1f K' % (count, fullSum/(dof*kB),
                                                                                halfSum/(dof*kB)))
    if args.drag == 'Explicit':
        print('Free bead in a fluid at rest, explicit drag: full step %.1f K, half step T/(1 - gamma dt/2) = %.1f K' % (
            args.temperature, args.temperature/(1 - a/2)))
    else:
        print('Free bead in a fluid at rest, centred drag: full step T/(1 + gamma dt/2) = %.1f K, half step %.1f K' % (
            args.temperature/(1 + a/2), args.temperature))
    print('Trajectory written to', args.output)


if __name__ == '__main__':
    main()
