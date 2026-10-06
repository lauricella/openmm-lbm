# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Diffusion of a protein, coarse grained with COCOMO2, in a lattice Boltzmann fluid.

Every bead is coupled to the fluid with the Euler-Maruyama scheme of LBMForce, which is also the
thermostat.  With --no-lb there is no fluid: OpenMM's LangevinMiddleIntegrator, at the same friction and
temperature, moves the beads without hydrodynamic interactions.

Parameter sets (--preset):
  sod1       the folded protein SOD1 (110 beads, data/sod1.pdb), held folded by the elastic network of
             COCOMO2, with the nonbonded terms scaled by the solvent exposure of each residue
             (data/sod1.surface); box 15 nm, friction 10/ps, time step 10 fs, nu = 5.0175 nm^2/ps
             (tau = 1.10), 200 ns: the runs of the DragOpenMM project
  fabio-g30  SOD1 in a box of 30 nm, friction 30/ps, 50 ns
  smoke      SOD1 with friction 5/ps for 2000 steps: a quick check that everything runs
  rlp        an intrinsically disordered protein (166 beads, data/rlp.pdb), no elastic network; box
             20 nm, friction 100/ps, time step 2 fs, nu = 1.0035 nm^2/ps (tau = 0.52), 10 ns
All use a lattice spacing of 0.5 nm, water density, 298 K and the removal of the fluid momentum at
every step.

The protocol is that of the original scripts: velocities at T, energy minimization, new velocities at
T, production.  Files written, with the prefix given by --output:
  <prefix>.dcd               trajectory (DCD), every --report steps
  <prefix>.log               step, time, potential energy, temperature, speed (OpenMM's StateDataReporter;
                             its temperature is that of the full step, docs/theory.md, section 2)
  <prefix>_temperature.txt   temperature of the beads from full-step and from half-step velocities
  <prefix>_com.txt           time (ps) and centre of mass x y z (nm), not wrapped into the box: the
                             input of msd.py, which computes the diffusion coefficient
  <prefix>_final.xml, <prefix>_fluid.npz   final state of the particles and of the fluid
  <prefix>.chk               checkpoint of the whole run, fluid included, every --checkpoint steps

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
    'fabio-g30': dict(SOD1, box=30.0, friction=30.0, steps=5000000, report=10000),
    'smoke': dict(SOD1, box=15.0, friction=5.0, steps=2000, report=100),
    'rlp': dict(pdb='rlp.pdb', surface=None, domain=None, enm_domain=None, dt=0.002, viscosity=1.0035,
                box=20.0, friction=100.0, steps=5000000, report=5000),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', choices=sorted(PRESETS), default='sod1', help='parameter set (default sod1)')
    parser.add_argument('--no-lb', action='store_true', help='no fluid: LangevinMiddleIntegrator')
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
    parser.add_argument('--surface', help='solvent accessible surface of each residue in nm^2 (default from the preset)')
    parser.add_argument('--domain', type=int, nargs=2, metavar=('FIRST', 'LAST'),
                        help='folded domain for the exposure scaling, residues counted from 1 (sod1: 1 108)')
    parser.add_argument('--enm-domain', type=int, nargs=2, metavar=('FIRST', 'LAST'),
                        help='residues joined by the elastic network (sod1: 2 109)')
    parser.add_argument('--platform', help='OpenMM platform (default: CUDA, then OpenCL, then Reference)')
    parser.add_argument('--precision', default='mixed', help='precision on CUDA and OpenCL (default mixed)')
    parser.add_argument('--output', help='prefix of the output files (default <preset>_lb_on or _lb_off)')
    parser.add_argument('--checkpoint', type=int,
                        help='steps between checkpoints in <prefix>.chk (default 100 reports, 0 = none)')
    parser.add_argument('--restart', action='store_true',
                        help='continue from <prefix>.chk, appending to the output files, up to --steps')
    args = parser.parse_args()
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    for key in ('pdb', 'surface'):
        value = getattr(args, key)
        if value is not None and not os.path.exists(value):
            setattr(args, key, os.path.join(DATA, value))
    if args.output is None:
        args.output = '%s_lb_%s' % (args.preset, 'off' if args.no_lb else 'on')
    if args.checkpoint is None:
        args.checkpoint = 100*args.report
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


class FullStepTemperatureReporter:
    """Temperature of the beads from full-step velocities, the mean of two consecutive steps."""

    def __init__(self, file, reportInterval, masses, append=False):
        self._out = open(file, 'a' if append else 'w')
        self._reportInterval = reportInterval
        self._masses = np.asarray(masses)
        self._previous = None
        if not append:
            print('# step, full-step temperature (K), half-step temperature (K)', file=self._out)

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        if steps > 1:
            steps -= 1                    # first stop one step before the report
        return {'steps': steps, 'periodic': None, 'include': ['velocities']}

    def report(self, simulation, state):
        v = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
        if simulation.currentStep%self._reportInterval != 0:
            self._previous = v
            return
        kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
        dof = 3*len(self._masses)
        half = (self._masses[:, None]*v**2).sum()/(dof*kB)
        full = np.nan if self._previous is None else (self._masses[:, None]*((v + self._previous)/2)**2).sum()/(dof*kB)
        print('%d %.2f %.2f' % (simulation.currentStep, full, half), file=self._out, flush=True)


class CentreOfMassReporter:
    """Time and centre of mass of all the particles, from positions that are not wrapped into the box."""

    def __init__(self, file, reportInterval, masses, append=False):
        self._out = open(file, 'a' if append else 'w')
        self._reportInterval = reportInterval
        self._weights = np.asarray(masses)/np.sum(masses)
        if not append:
            print('# time (ps), centre of mass x y z (nm)', file=self._out)

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return {'steps': steps, 'periodic': False, 'include': ['positions']}

    def report(self, simulation, state):
        x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        com = self._weights @ x
        print('%.3f %.6f %.6f %.6f' % (state.getTime().value_in_unit(unit.picosecond), *com), file=self._out,
              flush=True)


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
    positions = cocomo2.centred(positions, args.box)
    xi = cocomo2.exposure(names, np.loadtxt(args.surface), [tuple(args.domain)]) if args.domain else None
    pairs = cocomo2.elastic_network(positions, [tuple(args.enm_domain)]) if args.enm_domain else []
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
        for i in range(system.getNumParticles()):
            force.addParticle(i)
        system.addForce(force)
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

    print('Platform %s; %d beads, %d elastic bonds, box %g nm, %s' % (
        platform.getName(), len(names), len(pairs), args.box,
        'no fluid (Langevin)' if args.no_lb else '%d^3 lattice nodes' % round(args.box/args.spacing)))
    print('Friction %g 1/ps, viscosity %g nm^2/ps, %g K, dt %g ps, %d steps (%g ns)' % (
        args.friction, args.viscosity, args.temperature, args.dt, args.steps, args.steps*args.dt/1000))

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
        truncate(args.output + '_com.txt', time)
        dcdFile = '%s_%d.dcd' % (args.output, step)
    simulation.reporters.append(app.DCDReporter(dcdFile, args.report))
    simulation.reporters.append(app.StateDataReporter(args.output + '.log', args.report, step=True, time=True,
                                                      potentialEnergy=True, temperature=True, speed=True,
                                                      separator='\t', append=append))
    simulation.reporters.append(FullStepTemperatureReporter(args.output + '_temperature.txt', args.report, masses,
                                                            append))
    simulation.reporters.append(CentreOfMassReporter(args.output + '_com.txt', args.report, masses, append))
    if args.checkpoint > 0:
        if args.no_lb:
            simulation.reporters.append(app.CheckpointReporter(checkpointFile, args.checkpoint))
        else:
            simulation.reporters.append(openmmlbm.LBMCheckpointReporter(checkpointFile, args.checkpoint, force))
    simulation.step(max(0, args.steps - simulation.currentStep))

    simulation.saveState(args.output + '_final.xml')
    if not args.no_lb:
        np.savez(args.output + '_fluid.npz', state=np.array(force.getFluidState(simulation.context)),
                 time=simulation.context.getTime().value_in_unit(unit.picosecond))
    temperature = np.loadtxt(args.output + '_temperature.txt', ndmin=2)
    print('Mean full-step temperature %.1f K over %d reports' % (np.nanmean(temperature[:, 1]), len(temperature)))
    print('Files written with prefix', args.output)


if __name__ == '__main__':
    main()
