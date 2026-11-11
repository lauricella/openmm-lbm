# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Diffusion of a protein, coarse grained with COCOMO2, in a lattice Boltzmann fluid.

Every bead is coupled to the fluid with the Euler-Maruyama scheme of LBMForce, which is also the
thermostat, with the explicit drag (the default) or the centred one (--drag Centered, docs/theory.md,
section 2).  With --fluid-fluctuations the fluid itself fluctuates at the same temperature (docs/theory.md,
section 7).  With --no-lb there is no fluid: OpenMM's LangevinMiddleIntegrator, at the same friction and
temperature, moves the beads without hydrodynamic interactions.

Parameter sets (--preset):
  sod1       the folded protein SOD1 (110 beads, data/sod1.pdb), held folded by the elastic network of
             COCOMO2, with the nonbonded terms scaled by the solvent exposure of each residue
             (data/sod1.surface); box 15 nm, friction 10/ps, time step 10 fs, nu = 5.0175 nm^2/ps
             (five times the viscosity of water at 20 C, tau = 1.10), 200 ns
  sod1-g30   SOD1 in a box of 30 nm, friction 30/ps, 50 ns
  smoke      SOD1 with friction 5/ps for 2000 steps: a quick check that everything runs
  rlp        an intrinsically disordered protein (166 beads, data/rlp.pdb), no elastic network; box
             20 nm, friction 100/ps, time step 2 fs, nu = 1.0035 nm^2/ps (water at 20 C, tau = 0.52), 10 ns
All use a lattice spacing of 0.5 nm, water density, 298 K and the removal of the fluid momentum at
every step.  For a protein in water pass the viscosity of water at the temperature of the run, in nm^2/ps:
--viscosity 0.893 at 298 K, --viscosity 0.696 --temperature 310 at 310 K.

Another structure: --pdb, a coarse-grained PDB file with one bead per residue, and --surface, the solvent accessible
surface of each residue in nm^2 (cocomo2.py, surface.py), with --domain FIRST LAST for its folded domain (residues
counted from 1: the exposure scaling and, without --enm-domain, the elastic network); a structure without --domain
is a disordered protein.  The preset then gives only the parameters of the run (box, friction, viscosity, time step,
steps), not its structure, surface or domains.  A --pdb or --surface file that is not in the working directory is
looked for in data/.

Several proteins.  The PDB file may hold several chains, for example several proteins or several copies of one, each
with its own identifier (segment identifier, columns 73-76, or chain identifier, column 22): consecutive beads with
the same identifier form a chain and are bonded, so two proteins that follow each other with the same identifier
would be joined.  Their folded domains are given by --domains FILE, a text file with one line per chain, in the
order of the chains in the PDB file: on each line the domains of that chain as pairs FIRST LAST separated by commas,
in residue numbers counted from 1 inside the chain (the position of the bead in its chain, not the residue number of
the PDB file), or "-" for a chain without folded domain; "#" starts a comment.  For three chains:

    1 108            # chain 1: one folded domain
    10 60, 70 150    # chain 2: two folded domains, each with its own elastic network
    -                # chain 3: disordered

The script converts them to the numbering of the whole structure (cocomo2.read_domains()) and checks that there is
one line per chain and that every domain lies inside its chain.  The elastic network joins only beads of the same
domain: it holds each folded structure on its own and never joins two chains.  --domains sets the exposure scaling
and the elastic network; --enm-domains FILE, in the same format, gives other domains to the elastic network.
--surface takes one file for the whole structure or one file per chain, in the order of the chains, with "-" in
place of the file of a chain without folded domain.  With more than
one chain the script also writes the centre of mass of every chain (<prefix>_com_<k>.txt below).

The protocol: velocities at T, energy minimization, new velocities at
T, production.  Files written, with the prefix given by --output (default <preset>_lb_on, <preset>_lb_on_centered with
--drag Centered, <preset>_lb_on_fluct with --fluid-fluctuations, or <preset>_lb_off):
  <prefix>.dcd               trajectory (DCD), every --report steps, written by OpenMM's DCDReporter: positions
                             in Angstrom, as in every DCD file, each molecule wrapped into the box as a whole
  <prefix>.log               step, time, potential energy, speed (OpenMM's StateDataReporter)
  <prefix>_temperature.txt   step, time (ps) and temperature of the beads, written by the plugin's
                             openmmlbm.LBMTemperatureReporter, which gives the right temperature for the drag scheme
                             (docs/theory.md, section 2); with --no-lb by OpenMM's StateDataReporter
  <prefix>_fluid_temperature.txt
                             with --fluid-fluctuations, the temperature of the fluid every 10 reports, from the
                             velocities of the nodes (FluidTemperatureReporter below): it shows that the fluid is
                             at the temperature of the run
  <prefix>_com.txt           time (ps) and centre of mass x y z (nm), not wrapped into the box: the
                             input of msd.py, which computes the diffusion coefficient
  <prefix>_com_<k>.txt       with more than one chain, the same for chain k = 1, 2, ... (the order of the PDB file)
  <prefix>_final.xml, <prefix>_fluid.npz   final state of the particles and of the fluid
  <prefix>.chk               checkpoint of the whole run, fluid included, every --checkpoint steps
  <prefix>_density_<step>.vti, <prefix>_velocity_<step>.vti, <prefix>_particles_<step>.vtp, <prefix>.pvd
                             with --vtk N, every N steps, the fluid and the beads in VTK files for ParaView,
                             in OpenMM units (openmmlbm.LBMVTKReporter); open <prefix>.pvd for the series

A long run can be split into several jobs: if it stops (the time limit of a job, for example), run the same
command with --restart.  The run continues from the last checkpoint and stops at --steps, exactly as an
uninterrupted run would (docs/user_guide/restart.md).  The text files are cut at the step of the checkpoint
and continued, so that the reports written after the checkpoint by the interrupted job are not repeated; the
trajectory continues in a new file, <prefix>_<step>.dcd, where <step> is the step of the checkpoint.
"""

import argparse
import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import openmmlbm
from openmmlbm import LBMForce

import cocomo2

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

SOD1 = dict(pdb='sod1.pdb', surface='sod1.surface', domain=[1, 108], enm_domain=[2, 109], dt=0.01, viscosity=5.0175)
PRESETS = {
    'sod1': dict(SOD1, box=15.0, friction=10.0, steps=20000000, report=10000),
    'sod1-g30': dict(SOD1, box=30.0, friction=30.0, steps=5000000, report=10000),
    'smoke': dict(SOD1, box=15.0, friction=5.0, steps=2000, report=100),
    'rlp': dict(pdb='rlp.pdb', surface=None, domain=None, enm_domain=None, dt=0.002, viscosity=1.0035,
                box=20.0, friction=100.0, steps=5000000, report=5000),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', choices=sorted(PRESETS), default='sod1', help='parameter set (default sod1)')
    parser.add_argument('--no-lb', action='store_true', help='no fluid: LangevinMiddleIntegrator')
    parser.add_argument('--drag', choices=['Explicit', 'Centered'], default='Explicit',
                        help='drag scheme of LBMForce (default Explicit)')
    parser.add_argument('--fluid-fluctuations', action='store_true',
                        help='fluctuating fluid at the temperature of the beads (default off)')
    parser.add_argument('--box', type=float, help='side of the cubic box (nm)')
    parser.add_argument('--spacing', type=float, default=0.5, help='lattice spacing (nm, default 0.5)')
    parser.add_argument('--friction', type=float, help='friction (1/ps)')
    parser.add_argument('--viscosity', type=float, help='kinematic viscosity (nm^2/ps)')
    parser.add_argument('--temperature', type=float, default=298.0, help='temperature (K, default 298)')
    parser.add_argument('--dt', type=float, help='time step (ps)')
    parser.add_argument('--steps', type=int, help='number of production steps')
    parser.add_argument('--report', type=int, help='steps between reports')
    parser.add_argument('--seed', type=int, default=1234, help='random number seed (default 1234)')
    parser.add_argument('--pdb', help='coarse-grained structure (default from the preset, in data/)')
    parser.add_argument('--surface', nargs='+', metavar='FILE',
                        help='solvent accessible surface of each residue in nm^2: one file, or one per chain in the '
                        'order of the chains, "-" for a chain without folded domain (default from the preset)')
    parser.add_argument('--domain', type=int, nargs=2, metavar=('FIRST', 'LAST'),
                        help='folded domain, residues counted from 1: the exposure scaling and, without --enm-domain, '
                        'the elastic network (sod1: 1 108)')
    parser.add_argument('--enm-domain', type=int, nargs=2, metavar=('FIRST', 'LAST'),
                        help='residues joined by the elastic network, when they differ from --domain (sod1: 2 109)')
    parser.add_argument('--domains', metavar='FILE',
                        help='folded domains of every chain, one line per chain (see above): the exposure scaling '
                        'and, without --enm-domains, the elastic network; instead of --domain')
    parser.add_argument('--enm-domains', metavar='FILE',
                        help='domains of the elastic network of every chain, in the format of --domains, when they '
                        'differ from --domains; instead of --enm-domain')
    parser.add_argument('--platform', help='OpenMM platform (default: CUDA, then OpenCL, then Reference)')
    parser.add_argument('--precision', default='mixed', help='precision on CUDA, OpenCL and HIP (default mixed)')
    parser.add_argument('--output', help='prefix of the output files (default <preset>_lb_on, <preset>_lb_on_centered '
                        'with --drag Centered, then _fluct with --fluid-fluctuations, or <preset>_lb_off)')
    parser.add_argument('--checkpoint', type=int,
                        help='steps between checkpoints in <prefix>.chk (default 100 reports, 0 = none)')
    parser.add_argument('--vtk', type=int, default=0,
                        help='steps between VTK files of the fluid and of the beads (default 0 = none)')
    parser.add_argument('--restart', action='store_true',
                        help='continue from <prefix>.chk, appending to the output files, up to --steps')
    args = parser.parse_args()
    if args.vtk > 0 and args.no_lb:
        parser.error('--vtk writes the fluid of LBMForce, so it cannot be used with --no-lb')
    if (args.domains or args.enm_domains) and (args.domain or args.enm_domain):
        parser.error('give the domains either as ranges (--domain, --enm-domain) or as files (--domains, '
                     '--enm-domains), not both')
    if args.domain is not None and args.enm_domain is None:
        args.enm_domain = args.domain          # one folded domain for both, instead of the ranges of the preset
    structure = args.pdb is not None           # another structure than that of the preset
    for key, value in PRESETS[args.preset].items():
        if key in ('domain', 'enm_domain', 'surface') and structure or key in ('domain', 'enm_domain') and (
                args.domains or args.enm_domains):
            continue                           # the surface and the ranges of the preset belong to its structure
        if getattr(args, key) is None:
            setattr(args, key, value)
    if isinstance(args.surface, str):
        args.surface = [args.surface]
    if args.pdb is not None and not os.path.exists(args.pdb):
        args.pdb = os.path.join(DATA, args.pdb)
    if args.surface:
        args.surface = [value if value == '-' or os.path.exists(value) else os.path.join(DATA, value)
                        for value in args.surface]
    if args.output is None:
        args.output = '%s_lb_%s' % (args.preset, 'off' if args.no_lb else 'on')
        if not args.no_lb and args.drag == 'Centered':
            args.output += '_centered'
        if not args.no_lb and args.fluid_fluctuations:
            args.output += '_fluct'
    if args.checkpoint is None:
        args.checkpoint = 100*args.report
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


class CentreOfMassReporter:
    """Time and centre of mass of the particles start ... end - 1 (default all), from positions that are not
    wrapped into the box."""

    def __init__(self, file, reportInterval, masses, append=False, start=0, end=None):
        self._out = open(file, 'a' if append else 'w')
        self._reportInterval = reportInterval
        self._slice = slice(start, end)
        masses = np.asarray(masses)[self._slice]
        self._weights = masses/np.sum(masses)
        if not append:
            print('# time (ps), centre of mass x y z (nm)', file=self._out)

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return {'steps': steps, 'periodic': False, 'include': ['positions']}

    def report(self, simulation, state):
        x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)[self._slice]
        com = self._weights @ x
        print('%.3f %.6f %.6f %.6f' % (state.getTime().value_in_unit(unit.picosecond), *com), file=self._out,
              flush=True)


class FluidTemperatureReporter:
    """Temperature of a fluctuating fluid, from the velocities of its nodes (LBMForce.getFluidFields).

    At equilibrium the fluid of each cell, of mass m = rho dx^3, moves with the velocity u of its node, and every
    component of its momentum carries kB T/2 (docs/theory.md, section 7): T = <m (u - <u>)^2>/kB, averaged over the
    fluid nodes and the three components, with the mean velocity of the fluid subtracted.  Without fluid
    fluctuations the fluid is not thermal and this temperature stays near zero.  Reading the fields of the whole lattice
    takes time (0.7 s for 60^3 nodes), so the script reports it every 10 reports.
    """

    def __init__(self, file, reportInterval, force, spacing, append=False):
        self._out = open(file, 'a' if append else 'w')
        self._reportInterval = reportInterval
        self._force = force
        self._cellVolume = spacing**3
        if not append:
            print('# step, temperature of the fluid (K)', file=self._out)

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return {'steps': steps, 'periodic': None, 'include': []}

    def report(self, simulation, state):
        density, velocity = self._force.getFluidFields(simulation.context)
        rho = np.array(density.value_in_unit(unit.dalton/unit.nanometer**3))
        u = np.array(velocity.value_in_unit(unit.nanometer/unit.picosecond))
        fluid = rho > 0                   # solid nodes have zero density
        du = u[fluid] - u[fluid].mean(axis=0)
        kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
        temperature = (rho[fluid, None]*self._cellVolume*du**2).mean()/kB     # Da nm^2/ps^2 = kJ/mol
        print('%d %.2f' % (simulation.currentStep, temperature), file=self._out, flush=True)


def describe(domains):
    """The domains as text, in the numbering of the whole structure."""
    return ', '.join('%d-%d' % domain for domain in domains) if domains else 'none'


def truncate(file, lastValue, column=0):
    """Remove from a text file of reports the lines whose first column is beyond lastValue."""
    if not os.path.exists(file):
        return
    with open(file) as lines:
        kept = [line for line in lines if line.startswith('#') or not line.strip() or
                float(line.split()[column]) <= lastValue + 1e-6]
    with open(file, 'w') as out:
        out.writelines(kept)


def main():
    args = parse_arguments()
    names, chains, positions = cocomo2.read_beads(args.pdb)
    ranges = cocomo2.chain_ranges(chains)
    positions = cocomo2.centred(positions, args.box)
    exposed, network = cocomo2.folded_domains(chains, args.domain, args.enm_domain, args.domains, args.enm_domains)
    if exposed and not args.surface:
        raise ValueError('folded domains need the surface of every residue (--surface)')
    xi = cocomo2.exposure(names, cocomo2.read_surfaces(args.surface, chains), exposed) if exposed else None
    pairs = cocomo2.elastic_network(positions, network)
    system = cocomo2.create_system(names, chains, args.box, xi, pairs)
    masses = [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())]

    if args.no_lb:
        integrator = mm.LangevinMiddleIntegrator(args.temperature, args.friction, args.dt)
        integrator.setRandomNumberSeed(args.seed)
    else:
        nodes = round(args.box/args.spacing)
        if abs(nodes*args.spacing - args.box) > 1e-9*args.box:
            raise ValueError('the box must be a whole number of lattice spacings')
        force = LBMForce()
        force.setGridSize(nodes, nodes, nodes)
        force.setFluidDensity((1000*unit.kilogram/unit.meter**3*unit.AVOGADRO_CONSTANT_NA).value_in_unit(
            unit.dalton/unit.nanometer**3))
        force.setKinematicViscosity(args.viscosity)
        force.setFriction(args.friction)
        force.setTemperature(args.temperature)
        force.setRandomNumberSeed(args.seed)
        force.setFluidMomentumRemovalFrequency(1)
        force.setDragScheme(getattr(LBMForce, args.drag))
        force.setFluidFluctuations(args.fluid_fluctuations)
        for i in range(system.getNumParticles()):
            force.addParticle(i)
        system.addForce(force)          # the last force of the System, as the centred drag requires
        integrator = mm.VerletIntegrator(args.dt)

    topology = app.Topology()
    topology.setPeriodicBoxVectors([mm.Vec3(args.box, 0, 0), mm.Vec3(0, args.box, 0), mm.Vec3(0, 0, args.box)])
    chain = None
    for i, name in enumerate(names):
        if i == 0 or chains[i] != chains[i - 1]:
            chain = topology.addChain(chains[i])
        topology.addAtom('CA', app.element.carbon, topology.addResidue(name, chain))
    platform, properties = select_platform(args.platform, args.precision)
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(positions)

    print('Structure %s: %d beads in %d chain%s; folded domains %s; elastic network %s' % (
        args.pdb, len(names), len(ranges), '' if len(ranges) == 1 else 's', describe(exposed), describe(network)))
    print('Platform %s; %d beads, %d elastic bonds, box %g nm, %s' % (
        platform.getName(), len(names), len(pairs), args.box,
        'no fluid (Langevin)' if args.no_lb else '%d^3 lattice nodes' % round(args.box/args.spacing)))
    print('Friction %g 1/ps, viscosity %g nm^2/ps, %g K, dt %g ps, %d steps (%g ns)%s' % (
        args.friction, args.viscosity, args.temperature, args.dt, args.steps, args.steps*args.dt/1000,
        '' if args.no_lb else ', %s drag%s' % (args.drag.lower(), ', fluctuating fluid' if args.fluid_fluctuations
                                               else '')))

    checkpointFile = args.output + '.chk'
    if args.restart:
        # Positions, velocities, time, step count and random numbers come from the checkpoint; for a System
        # without LBMForce an OpenMM checkpoint is enough.
        if args.no_lb:
            simulation.loadCheckpoint(checkpointFile)
        else:
            openmmlbm.loadCheckpoint(checkpointFile, simulation.context, force)
        print('Restarted from %s at step %d' % (checkpointFile, simulation.currentStep))
    else:
        simulation.context.setVelocitiesToTemperature(args.temperature, args.seed)
        simulation.minimizeEnergy(tolerance=100*unit.kilojoule_per_mole/unit.nanometer, maxIterations=500000)
        energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        print('Potential energy after minimization: %.1f kJ/mol' % energy.value_in_unit(unit.kilojoule_per_mole))
        simulation.context.setVelocitiesToTemperature(args.temperature, args.seed + 1)

    append = args.restart
    dcdFile = args.output + '.dcd'
    if args.restart:
        step = simulation.currentStep
        time = simulation.context.getTime().value_in_unit(unit.picosecond)
        truncate(args.output + '.log', step)
        truncate(args.output + '_temperature.txt', step)
        truncate(args.output + '_fluid_temperature.txt', step)
        truncate(args.output + '_com.txt', time)
        for k in range(len(ranges) if len(ranges) > 1 else 0):
            truncate('%s_com_%d.txt' % (args.output, k + 1), time)
        dcdFile = '%s_%d.dcd' % (args.output, step)
    simulation.reporters.append(app.DCDReporter(dcdFile, args.report))
    simulation.reporters.append(app.StateDataReporter(args.output + '.log', args.report, step=True, time=True,
                                                      potentialEnergy=True, speed=True,
                                                      separator='\t', append=append))
    if args.no_lb:
        simulation.reporters.append(app.StateDataReporter(args.output + '_temperature.txt', args.report, step=True,
                                                          time=True, temperature=True, separator=' ', append=append))
    else:
        temperatureFile = open(args.output + '_temperature.txt', 'a' if append else 'w')
        simulation.reporters.append(openmmlbm.LBMTemperatureReporter(temperatureFile, args.report, force))
    simulation.reporters.append(CentreOfMassReporter(args.output + '_com.txt', args.report, masses, append))
    if len(ranges) > 1:
        for k, (start, end) in enumerate(ranges):
            simulation.reporters.append(CentreOfMassReporter('%s_com_%d.txt' % (args.output, k + 1), args.report,
                                                             masses, append, start, end))
    if not args.no_lb and args.fluid_fluctuations:
        simulation.reporters.append(FluidTemperatureReporter(args.output + '_fluid_temperature.txt', 10*args.report,
                                                             force, args.spacing, append))
    if args.checkpoint > 0:
        if args.no_lb:
            simulation.reporters.append(app.CheckpointReporter(checkpointFile, args.checkpoint))
        else:
            simulation.reporters.append(openmmlbm.LBMCheckpointReporter(checkpointFile, args.checkpoint, force))
    if args.vtk > 0:
        simulation.reporters.append(openmmlbm.LBMVTKReporter(args.output, args.vtk, force, append=args.restart))
    simulation.step(max(0, args.steps - simulation.currentStep))

    simulation.saveState(args.output + '_final.xml')
    if not args.no_lb:
        np.savez(args.output + '_fluid.npz', state=np.array(force.getFluidState(simulation.context)),
                 time=simulation.context.getTime().value_in_unit(unit.picosecond))
    temperature = np.loadtxt(args.output + '_temperature.txt', ndmin=2)
    print('Mean temperature of the beads over %d reports: %.1f K' % (len(temperature), temperature[:, 2].mean()))
    if not args.no_lb and args.fluid_fluctuations:
        fluid = np.loadtxt(args.output + '_fluid_temperature.txt', ndmin=2)
        if len(fluid) > 0:
            print('Mean temperature of the fluid over %d values: %.1f K' % (len(fluid), fluid[:, 1].mean()))
    print('Files written with prefix', args.output)


if __name__ == '__main__':
    main()
