# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Kick experiment with a coarse-grained protein: every bead starts with the velocity v0 along x, in a
fluid at rest, at zero temperature.

As in ../particle/kick.py, but the kicked object is a protein of the COCOMO2 model (cocomo2.py):
  --preset peptide    the peptide GRGDSPYS (8 beads), flexible
  --preset ubiquitin  ubiquitin (76 beads), held folded by an elastic network over the whole chain,
                      with the nonbonded terms scaled by the solvent exposure of each residue
The coupling scheme is NVE (drag only, no random force).  With --no-lb the beads are integrated by
LangevinMiddleIntegrator at zero temperature and the same friction, without fluid.  The parameters of
the fluid are those of the single-bead kick: 100^3 nodes of 0.3 nm, tau = 1, friction 10/ps, time step
10 fs, v0 = 9 nm/ps, 2000 steps.

The output file has one line per step: time (ps), velocity of the centre of mass vx vy vz and its norm
(nm/ps), position of the centre of mass x y z (nm), radius of gyration (nm).
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
    'peptide': dict(pdb='GRGDSPYS.pdb', surface=None, domain=None),
    'ubiquitin': dict(pdb='ubiquitin.pdb', surface='ubiquitin.surface', domain=(1, 76)),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', choices=sorted(PRESETS), default='peptide', help='protein (default peptide)')
    parser.add_argument('--nodes', type=int, default=100, help='lattice nodes per side (default 100)')
    parser.add_argument('--spacing', type=float, default=0.3, help='lattice spacing in nm (default 0.3)')
    parser.add_argument('--dt', type=float, default=0.01, help='time step in ps (default 0.01)')
    parser.add_argument('--tau', type=float, default=1.0, help='relaxation time, sets the viscosity (default 1)')
    parser.add_argument('--density', type=float, default=993.0, help='fluid density in kg/m^3 (default 993)')
    parser.add_argument('--friction', type=float, default=10.0, help='friction in 1/ps (default 10)')
    parser.add_argument('--velocity', type=float, default=9.0, help='initial velocity along x in nm/ps (default 9)')
    parser.add_argument('--steps', type=int, default=2000, help='number of steps (default 2000)')
    parser.add_argument('--lam', type=float, default=cocomo2.LAMBDA, help='exposure threshold lambda (default 0.7)')
    parser.add_argument('--no-lb', action='store_true', help='no fluid: LangevinMiddleIntegrator at zero temperature')
    parser.add_argument('--platform', help='OpenMM platform (default: CUDA, then OpenCL, then Reference)')
    parser.add_argument('--precision', default='mixed', help='precision on CUDA and OpenCL (default mixed)')
    parser.add_argument('--output', help='output file (default kick_<preset>_lb_on.txt or _lb_off.txt)')
    args = parser.parse_args()
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
    preset = PRESETS[args.preset]
    box = args.nodes*args.spacing
    names, chains, positions = cocomo2.read_beads(os.path.join(DATA, preset['pdb']))
    positions = cocomo2.centred(positions, box)
    xi, pairs = None, []
    if preset['domain']:
        sasa = np.loadtxt(os.path.join(DATA, preset['surface']))
        xi = cocomo2.exposure(names, sasa, [preset['domain']], args.lam)
        pairs = cocomo2.elastic_network(positions, [preset['domain']])
    system = cocomo2.create_system(names, chains, box, xi, pairs)
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())])
    weights = masses/masses.sum()

    density = (args.density*unit.kilogram/unit.meter**3*unit.AVOGADRO_CONSTANT_NA).value_in_unit(
        unit.dalton/unit.nanometer**3)
    if args.no_lb:
        integrator = mm.LangevinMiddleIntegrator(0.0, args.friction, args.dt)
    else:
        force = LBMForce()
        force.setGridSize(args.nodes, args.nodes, args.nodes)
        force.setFluidDensity(density)
        force.setKinematicViscosity((args.tau - 0.5)/3*args.spacing**2/args.dt)
        force.setFriction(args.friction)
        force.setCouplingScheme(LBMForce.NVE)
        force.setFluidMomentumRemovalFrequency(0)
        for i in range(system.getNumParticles()):
            force.addParticle(i)
        system.addForce(force)
        integrator = mm.VerletIntegrator(args.dt)
    platform, properties = select_platform(args.platform, args.precision)
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(positions)
    context.setVelocities(np.tile([args.velocity, 0.0, 0.0], (len(names), 1)))

    print('Platform %s; %s: %d beads, %.1f Da, %d elastic bonds; box %.1f nm' % (
        platform.getName(), args.preset, len(names), masses.sum(), len(pairs), box))
    print('%s; friction %g 1/ps, v0 %g nm/ps, dt %g ps, %d steps' % (
        'no fluid (Langevin at 0 K)' if args.no_lb else '%d^3 nodes, tau %g, NVE coupling' % (args.nodes, args.tau),
        args.friction, args.velocity, args.dt, args.steps))

    with open(args.output, 'w') as out:
        print('# time (ps), centre of mass: vx vy vz |v| (nm/ps), x y z (nm); radius of gyration (nm)', file=out)
        for step in range(args.steps + 1):
            if step > 0:
                integrator.step(1)
            state = context.getState(getPositions=True, getVelocities=True)
            x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            v = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
            com, vcom = weights @ x, weights @ v
            rg = math.sqrt(weights @ ((x - com)**2).sum(axis=1))
            print('%.4f %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.6f' % (
                step*args.dt, *vcom, np.linalg.norm(vcom), *com, rg), file=out)

    t = args.steps*args.dt
    travelled = com[0] - (weights @ positions)[0]
    print('After %g ps: v/v0 = %.4e, distance travelled %.4f nm (v0/gamma = %.4f nm), radius of gyration %.3f nm' % (
        t, vcom[0]/args.velocity, travelled, args.velocity/args.friction, rg))
    print('Trajectory written to', args.output)


if __name__ == '__main__':
    main()
