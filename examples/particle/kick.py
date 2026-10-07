# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Kick experiment: one bead starts with a velocity v0 in a fluid at rest, at zero temperature.

With the fluid (the default) the bead gives its momentum to the fluid through the drag, and the fluid at
its node starts to move with it.  The velocity decays more slowly than the exponential of a Langevin
thermostat, and the bead travels further.  With --no-lb the same bead is integrated with OpenMM's
LangevinMiddleIntegrator at the same friction and zero temperature, which gives v(t) = v0 exp(-gamma t).

Two parameter sets of the old plugin's examples are available:
  --preset bead     a bead of 5 lattice cells, 100^3 nodes of 0.3 nm, tau = 1, v0 = 9 nm/ps, 2000 steps
  --preset alanine  a bead with the mass of alanine, 30^3 nodes of 0.5 nm, nu = 1.0035 nm^2/ps, v0 = 15 nm/ps
Every parameter can be changed from the command line.

The output file has one line per step: time (ps), velocity vx vy vz and its norm (nm/ps), position x y z
(nm).  With VerletIntegrator the velocities are those that OpenMM stores, half a step behind the
positions.
"""

import argparse
import math

import openmm as mm
import openmm.unit as unit
from openmmlbm import LBMForce

PRESETS = {
    'bead': dict(nodes=100, spacing=0.3, dt=0.01, friction=10.0, viscosity=1.5, density=993.0, mass=None,
                 cells=5.0, velocity=9.0, steps=2000),
    'alanine': dict(nodes=30, spacing=0.5, dt=0.01, friction=10.0, viscosity=1.0035, density=1000.0,
                    mass=71.079, cells=None, velocity=15.0, steps=100),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', choices=sorted(PRESETS), default='bead', help='parameter set (default bead)')
    parser.add_argument('--nodes', type=int, help='lattice nodes per side of the cubic box')
    parser.add_argument('--spacing', type=float, help='lattice spacing (nm)')
    parser.add_argument('--dt', type=float, help='time step (ps)')
    parser.add_argument('--friction', type=float, help='friction gamma (1/ps)')
    parser.add_argument('--viscosity', type=float, help='kinematic viscosity (nm^2/ps)')
    parser.add_argument('--density', type=float, help='fluid density (kg/m^3)')
    parser.add_argument('--mass', type=float, help='bead mass (Da); default from the preset')
    parser.add_argument('--velocity', type=float, help='initial velocity along x (nm/ps)')
    parser.add_argument('--steps', type=int, help='number of steps')
    parser.add_argument('--no-lb', action='store_true', help='no fluid: LangevinMiddleIntegrator at zero temperature')
    parser.add_argument('--drag', choices=['Explicit', 'Centered'], default='Explicit',
                        help='drag scheme of LBMForce (default Explicit)')
    parser.add_argument('--platform', help='OpenMM platform (default: CUDA, then OpenCL, then Reference)')
    parser.add_argument('--precision', default='mixed', help='precision on CUDA and OpenCL (default mixed)')
    parser.add_argument('--output', help='output file (default kick_<preset>_lb_on.txt or _lb_off.txt)')
    args = parser.parse_args()
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if args.output is None:
        args.output = 'kick_%s_lb_%s.txt' % (args.preset, 'off' if args.no_lb else 'on')
    return args


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
    density = (args.density*unit.kilogram/unit.meter**3*unit.AVOGADRO_CONSTANT_NA).value_in_unit(
        unit.dalton/unit.nanometer**3)
    mass = args.mass if args.mass is not None else args.cells*density*args.spacing**3
    box = args.nodes*args.spacing

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
    system.addParticle(mass)
    if args.no_lb:
        integrator = mm.LangevinMiddleIntegrator(0.0, args.friction, args.dt)
    else:
        force = LBMForce()
        force.setGridSize(args.nodes, args.nodes, args.nodes)
        force.setFluidDensity(density)
        force.setKinematicViscosity(args.viscosity)
        force.setFriction(args.friction)
        force.setCouplingScheme(LBMForce.NVE)        # zero temperature: drag only, no random force
        force.setFluidMomentumRemovalFrequency(0)    # keep the momentum given to the fluid
        force.setDragScheme(getattr(LBMForce, args.drag))
        force.addParticle(0)
        system.addForce(force)
        integrator = mm.VerletIntegrator(args.dt)

    platform, properties = select_platform(args.platform, args.precision)
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions([mm.Vec3(0, 0, 0)])
    context.setVelocities([mm.Vec3(args.velocity, 0, 0)])

    print('Platform %s, box %.2f nm, %d^3 nodes, dt %g ps' % (platform.getName(), box, args.nodes, args.dt))
    print('Bead mass %.4f Da, v0 %g nm/ps, friction %g 1/ps' % (mass, args.velocity, args.friction))
    if args.no_lb:
        print('No fluid: LangevinMiddleIntegrator at 0 K')
    else:
        dx, dt, tau = force.getLatticeParametersInContext(context)
        print('Fluid: density %.4f Da/nm^3, viscosity %g nm^2/ps, tau %.4f' % (density, args.viscosity, tau))
        print('In lattice units: mass %.4f cells, v0 %.4f, gamma dt %.4f' % (
            mass/(density*args.spacing**3), args.velocity*args.dt/args.spacing, args.friction*args.dt))

    with open(args.output, 'w') as out:
        print('# time (ps), vx vy vz |v| (nm/ps), x y z (nm)', file=out)
        for step in range(args.steps + 1):
            if step > 0:
                integrator.step(1)
            state = context.getState(getPositions=True, getVelocities=True)
            v = state.getVelocities()[0].value_in_unit(unit.nanometer/unit.picosecond)
            x = state.getPositions()[0].value_in_unit(unit.nanometer)
            speed = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            print('%.4f %.10e %.10e %.10e %.10e %.10e %.10e %.10e' % (step*args.dt, v[0], v[1], v[2], speed,
                                                                         x[0], x[1], x[2]), file=out)

    t = args.steps*args.dt
    print('After %g ps: v/v0 = %.4e, distance travelled %.4f nm (exp(-gamma t) = %.4e, v0/gamma = %.4f nm)' % (
        t, v[0]/args.velocity, x[0], math.exp(-args.friction*t), args.velocity/args.friction))
    print('Trajectory written to', args.output)


if __name__ == '__main__':
    main()
