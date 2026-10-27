# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Kick experiment with a coarse-grained protein: every bead starts with the velocity v0 along x, in a
fluid at rest, at the temperature T (--temperature), zero by default.

As in ../particle/kick.py, but the kicked object is a protein of the COCOMO2 model (cocomo2.py):
  --preset peptide    the peptide GRGDSPYS (8 beads), flexible
  --preset ubiquitin  ubiquitin (76 beads), held folded by an elastic network over the whole chain,
                      with the nonbonded terms scaled by the solvent exposure of each residue
  --preset sod1       the folded protein SOD1 (110 beads) of ./diffusion.py, with the same structure, exposure
                      and elastic network (--domain and --enm-domain as in diffusion.py)
At zero temperature there is no random force on the beads, but the drag stays: beads and fluid exchange
momentum without a thermostat.  With --no-lb the beads are integrated by LangevinMiddleIntegrator at the same
temperature and friction, without fluid.  The viscosity is given in physical units (--viscosity, nm^2/ps); the
plugin converts it to the relaxation time tau of the lattice, which the script prints.  The parameters of the
fluid are those of the single-bead kick: a box of 30 nm with 100^3 nodes of 0.3 nm (--box gives the side in nm,
--nodes the number of nodes), nu = 1.5 nm^2/ps (tau = 1), density 993 kg/m^3, friction 10/ps, time step 10 fs,
v0 = 9 nm/ps, 2000 steps.  The momentum of the fluid is never removed: the momentum that the protein gives to the
fluid stays there, and the protein ends at the velocity of the whole system, M v0/(M + M_fluid).

Above zero temperature the beads receive the random force of the coupling, and with --fluid-fluctuations the fluid
fluctuates at the same temperature (use it with --drag Centered, which gives the right temperature).  The kick
should then start from equilibrium: --equilibration N first runs N steps with the velocities drawn at T, then adds
v0 along x to every bead.  A single kick is lost in the thermal motion of the centre of mass, of size
sqrt(kT/M), after about a picosecond: --repeats N runs N independent kicks in one run of the script, with the
seeds seed, seed + 1, ..., and writes their mean.  In linear response the mean decay at temperature T is the decay at zero temperature.

The output file has one line per step from the kick: time (ps), velocity of the centre of mass vx vy vz and the
norm of that velocity (nm/ps), position of the centre of mass x y z (nm), radius of gyration (nm).  With --repeats
these are the means over the kicks, and a last column gives the standard error of the mean of vx.
"""

import argparse
import math
import os

import numpy as np
import openmm as mm
import openmm.unit as unit
from openmmlbm import LBMForce

import cocomo2

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

PRESETS = {
    'peptide': dict(pdb='GRGDSPYS.pdb', surface=None, domain=None, enm_domain=None),
    'ubiquitin': dict(pdb='ubiquitin.pdb', surface='ubiquitin.surface', domain=(1, 76), enm_domain=(1, 76)),
    'sod1': dict(pdb='sod1.pdb', surface='sod1.surface', domain=(1, 108), enm_domain=(2, 109)),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', choices=sorted(PRESETS), default='peptide', help='protein (default peptide)')
    parser.add_argument('--box', type=float, help='side of the cubic box in nm (default 30: 100 nodes of 0.3 nm)')
    parser.add_argument('--nodes', type=int, help='lattice nodes per side, instead of --box (default 100)')
    parser.add_argument('--spacing', type=float, default=0.3, help='lattice spacing in nm (default 0.3)')
    parser.add_argument('--dt', type=float, default=0.01, help='time step in ps (default 0.01)')
    parser.add_argument('--viscosity', type=float, default=1.5,
                        help='kinematic viscosity in nm^2/ps (default 1.5: tau = 1 with the default spacing and dt)')
    parser.add_argument('--density', type=float, default=993.0, help='fluid density in kg/m^3 (default 993)')
    parser.add_argument('--friction', type=float, default=10.0, help='friction in 1/ps (default 10)')
    parser.add_argument('--velocity', type=float, default=9.0, help='initial velocity along x in nm/ps (default 9)')
    parser.add_argument('--steps', type=int, default=2000, help='number of steps after the kick (default 2000)')
    parser.add_argument('--lam', type=float, default=cocomo2.LAMBDA, help='exposure threshold lambda (default 0.7)')
    parser.add_argument('--domain', type=int, nargs=2, metavar=('FIRST', 'LAST'),
                        help='folded domain, residues counted from 1: the exposure scaling and, without --enm-domain, '
                        'the elastic network (default from the preset)')
    parser.add_argument('--enm-domain', type=int, nargs=2, metavar=('FIRST', 'LAST'),
                        help='residues joined by the elastic network, when they differ from --domain')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='temperature in K (default 0: no random force, drag only)')
    parser.add_argument('--fluid-fluctuations', action='store_true',
                        help='the fluid fluctuates at --temperature (above zero; use with --drag Centered)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random number seed (default 1234); kick i of --repeats uses seed + i')
    parser.add_argument('--equilibration', type=int, default=0,
                        help='steps at --temperature before the kick, from velocities drawn at that temperature '
                        '(default 0)')
    parser.add_argument('--repeats', type=int, default=1, help='independent kicks, averaged (default 1)')
    parser.add_argument('--no-lb', action='store_true', help='no fluid: LangevinMiddleIntegrator at the same T')
    parser.add_argument('--drag', choices=['Explicit', 'Centered'], default='Explicit',
                        help='drag scheme of LBMForce (default Explicit)')
    parser.add_argument('--platform', help='OpenMM platform (default: CUDA, then OpenCL, then Reference)')
    parser.add_argument('--precision', default='mixed', help='precision on CUDA, OpenCL and HIP (default mixed)')
    parser.add_argument('--output', help='output file (default kick_<preset>_lb_on.txt or _lb_off.txt)')
    args = parser.parse_args()
    if args.box is not None and args.nodes is not None:
        parser.error('give either --box or --nodes')
    if args.box is not None:
        args.nodes = round(args.box/args.spacing)
        if abs(args.nodes*args.spacing - args.box) > 1e-9*args.box:
            parser.error('the box must be a whole number of lattice spacings')
    elif args.nodes is None:
        args.nodes = 100
    if args.repeats < 1 or args.equilibration < 0:
        parser.error('--repeats must be at least 1 and --equilibration not negative')
    if args.no_lb and args.fluid_fluctuations:
        parser.error('--fluid-fluctuations needs the fluid, so it cannot be used with --no-lb')
    preset = PRESETS[args.preset]
    if (args.domain or args.enm_domain) and preset['surface'] is None:
        parser.error('--domain and --enm-domain need a folded protein (presets ubiquitin, sod1)')
    if args.domain is not None and args.enm_domain is None:
        args.enm_domain = args.domain          # one folded domain for both, instead of the ranges of the preset
    for key in ('domain', 'enm_domain'):
        if getattr(args, key) is None:
            setattr(args, key, preset[key])
    if args.output is None:
        args.output = 'kick_%s_lb_%s.txt' % (args.preset, 'off' if args.no_lb else 'on')
    return args


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


def main():
    args = parse_arguments()
    preset = PRESETS[args.preset]
    box = args.nodes*args.spacing
    names, chains, positions = cocomo2.read_beads(os.path.join(DATA, preset['pdb']))
    positions = cocomo2.centred(positions, box)
    xi, pairs = None, []
    if args.domain:
        sasa = np.loadtxt(os.path.join(DATA, preset['surface']))
        xi = cocomo2.exposure(names, sasa, [tuple(args.domain)], args.lam)
        pairs = cocomo2.elastic_network(positions, [tuple(args.enm_domain)])
    system = cocomo2.create_system(names, chains, box, xi, pairs)
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())])
    weights = masses/masses.sum()

    density = (args.density*unit.kilogram/unit.meter**3*unit.AVOGADRO_CONSTANT_NA).value_in_unit(
        unit.dalton/unit.nanometer**3)
    fluctuations = args.fluid_fluctuations and args.temperature > 0
    force = None
    if not args.no_lb:
        force = LBMForce()
        force.setGridSize(args.nodes, args.nodes, args.nodes)
        force.setFluidDensity(density)
        force.setKinematicViscosity(args.viscosity)
        force.setFriction(args.friction)
        force.setTemperature(args.temperature)       # at 0 K no random force: the drag only
        force.setFluidFluctuations(fluctuations)
        force.setFluidMomentumRemovalFrequency(0)    # the momentum given to the fluid stays in the system
        force.setDragScheme(getattr(LBMForce, args.drag))
        for i in range(system.getNumParticles()):
            force.addParticle(i)
        system.addForce(force)          # the last force of the System, as the centred drag requires
    platform, properties = select_platform(args.platform, args.precision)

    # One row per step from the kick: vx vy vz of the centre of mass, its position x y z, radius of gyration.
    rows = np.empty((args.repeats, args.steps + 1, 7))
    for r in range(args.repeats):
        seed = args.seed + r
        if args.no_lb:
            integrator = mm.LangevinMiddleIntegrator(args.temperature, args.friction, args.dt)
            integrator.setRandomNumberSeed(seed)
        else:
            force.setRandomNumberSeed(seed)
            integrator = mm.VerletIntegrator(args.dt)
        context = mm.Context(system, integrator, platform, properties)
        context.setPositions(positions)
        if r == 0:
            print('Platform %s; %s: %d beads, %.1f Da, %d elastic bonds; box %.1f nm' % (
                platform.getName(), args.preset, len(names), masses.sum(), len(pairs), box))
            if args.no_lb:
                fluid = 'no fluid (Langevin)'
            else:
                dx, dt, tau = force.getLatticeParametersInContext(context)
                fluid = '%d^3 nodes, viscosity %g nm^2/ps (tau %.4f)' % (args.nodes, args.viscosity, tau)
            print('%s; friction %g 1/ps, v0 %g nm/ps, dt %g ps, %d steps' % (
                fluid, args.friction, args.velocity, args.dt, args.steps))
            print('Temperature %g K%s%s' % (
                args.temperature, ' (no random force)' if args.temperature == 0 else '',
                ', fluctuating fluid' if fluctuations else ''))
            if args.fluid_fluctuations and not fluctuations:
                print('--fluid-fluctuations has no effect at 0 K')
            if args.equilibration or args.repeats > 1:
                print('%d kicks, seeds %d to %d, each after %d steps of equilibration' % (
                    args.repeats, args.seed, args.seed + args.repeats - 1, args.equilibration))
        kick = np.array([args.velocity, 0.0, 0.0])
        if args.equilibration > 0:
            if args.temperature > 0:
                context.setVelocitiesToTemperature(args.temperature, seed)
            integrator.step(args.equilibration)
            v = context.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer/unit.picosecond)
            context.setVelocities(v + kick)
        else:
            context.setVelocities(np.tile(kick, (len(names), 1)))
        for step in range(args.steps + 1):
            if step > 0:
                integrator.step(1)
            state = context.getState(getPositions=True, getVelocities=True)
            x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            v = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
            com, vcom = weights @ x, weights @ v
            rows[r, step, 0:3] = vcom
            rows[r, step, 3:6] = com
            rows[r, step, 6] = math.sqrt(weights @ ((x - com)**2).sum(axis=1))
        del context, integrator

    mean = rows.mean(axis=0)
    with open(args.output, 'w') as out:
        header = '# time (ps), centre of mass: vx vy vz |v| (nm/ps), x y z (nm); radius of gyration (nm)'
        if args.repeats > 1:
            header += '; standard error of vx (nm/ps); means over %d kicks' % args.repeats
        print(header, file=out)
        for step in range(args.steps + 1):
            vcom, com, rg = mean[step, 0:3], mean[step, 3:6], mean[step, 6]
            line = '%.4f %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.6f' % (
                step*args.dt, *vcom, np.linalg.norm(vcom), *com, rg)
            if args.repeats > 1:
                line += ' %.4e' % (rows[:, step, 0].std(ddof=1)/math.sqrt(args.repeats))
            print(line, file=out)

    t = args.steps*args.dt
    travelled = mean[-1, 3] - mean[0, 3]
    print('After %g ps: v/v0 = %.4e, distance travelled %.4f nm (v0/gamma = %.4f nm), radius of gyration %.3f nm' % (
        t, mean[-1, 0]/args.velocity, travelled, args.velocity/args.friction, mean[-1, 6]))
    print('Trajectory written to', args.output)


if __name__ == '__main__':
    main()
