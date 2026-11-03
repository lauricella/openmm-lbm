# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""A protein in a Poiseuille flow: a channel between two walls, driven by a uniform acceleration of the fluid.

The plane of nodes k = 0 is solid (setSolidNodes(), bounce-back walls) and the lattice is periodic, so the plane
bounds the channel on both sides: the walls lie halfway between the solid plane and the first fluid planes, at
z = dx/2 and z = (n_z - 1/2) dx, the channel is H = (n_z - 1) dx wide and its centre line is at z_c = n_z dx/2, the
middle of the box.  x and y are periodic.  The fluid is driven along x by the uniform acceleration g
(setBodyAcceleration()), the force of a pressure gradient rho g; the momentum that g gives to the fluid goes to the
walls.  The steady profile is the parabola

    u_x(z) = g/(2 nu) (z - dx/2) ((n_z - 1/2) dx - z) + g dx^2 (16 Lambda - 3)/(24 nu),   Lambda = (tau - 1/2)/2,

whose second term is the uniform slip of the bounce-back walls (docs/theory.md, Exact Poiseuille flow with
bounce-back; it vanishes at tau = 7/8 and is -7e-5 of the centre-line speed with the defaults).  The script takes the
speed at the centre line of the parabola (--velocity) and computes g = 8 nu v/H^2.  The fluid starts with this
profile (setFluidState()), so that the protein does not wait for the flow to build up.

The protein is the SOD1 of diffusion.py (COCOMO2, 110 beads, data/sod1.pdb, the same exposure and elastic network),
started on the centre line or --position nm away from it, and nothing holds it.  On the centre line the shear rate
is zero: the protein is carried along x at about the centre-line speed and does not turn on average; the curvature of
the profile makes it lag slightly behind the fluid at its centre of mass (by a^2 g/(6 nu) for a sphere of radius a,
Faxen's law).  Away from the centre line it turns about y with the local vorticity, at about half the local shear
rate, (g/(2 nu)) (z_c - z), and changes direction across the centre line.  A coupled bead that reaches a wall has its
velocity reversed (docs/user_guide/api_reference.md, Coupled particles, Walls): unlike the open faces of shear.py, nothing crosses the
walls.

By default the fluid fluctuates at the temperature of the run and the drag is centred, so that the protein diffuses
with its hydrodynamic mobility (docs/theory.md, sections 2 and 7); bounce-back walls keep the fluctuating fluid at
equilibrium up to the wall.  The viscosity is given in physical units (--viscosity, nm^2/ps, water at 298 K by
default); the plugin converts it to the relaxation time tau of the lattice, which the script prints.  The speed at
the centre line must stay well below the speed of sound of the lattice, c_s = dx/(sqrt(3) dt), 28.9 nm/ps with the
default lattice: the script refuses more than Mach 0.1, 2.9 nm/ps; keep it at 1 nm/ps or below (Mach 0.035), since
the thermal velocity of the fluid, 0.18 nm/ps per node and component at 298 K, adds to the flow.  These are the
flows of molecular dynamics: with v = 0.5 nm/ps in a channel of 49.5 nm, g = 1.5e-3 nm/ps^2 (a pressure gradient
of 1.5e15 Pa/m), the shear rate is 0.04/ps at the walls and 0.004/ps at 2.5 nm from the centre line, and the
Reynolds number of the channel, v H/nu, is 28.

How to run it.  Activate the environment where OpenMM and openmm-lbm are installed, go to a working directory, where
the output files are written, and give Python the path of the script in the repository: the script reads
shear.py (its reporters), cocomo2.py, data/ and ../fluid/initial_state.py next to itself, so keep it there instead of
copying it alone.  Below, poiseuille.py stands for /path/to/openmm-lbm/examples/cocomo/poiseuille.py.

  python poiseuille.py                                                 # the defaults: SOD1 on the centre line,
                                                                       # in water at 298 K, v = 0.5 nm/ps, 10 ns
  python poiseuille.py --position -10                                  # start 10 nm below the centre line
  python poiseuille.py --velocity 1 --steps 5000000 --output v1        # faster flow, 50 ns, files v1*
  python poiseuille.py --box 40 40 --height 80                         # a larger box and a wider channel
  python poiseuille.py --vtk 10000                                     # VTK files for ParaView every 100 ps
  python poiseuille.py --restart                                       # continue a stopped run
  python poiseuille.py --temperature 0 --no-fluid-fluctuations --drag Explicit --steps 100000
                                                                       # 1 ns without noise: the deterministic flow
  python poiseuille.py --platform Reference --box 10 10 --height 12 --steps 1000   # a quick try on a CPU

Options (python poiseuille.py --help lists them with their defaults):
  flow        --velocity V (nm/ps, default 0.5: speed along x at the centre line; at most 0.1 c_s, better at most
              1), --height LZ (nm, default 50: the period along z, the channel is LZ - dx wide), --box LX LY (nm,
              default 30 30); every side a whole number of lattice spacings
  protein     --position DZ (nm, default 0: start of the centre of the protein from the centre line, negative
              below), --domain FIRST LAST and --enm-domain FIRST LAST (as in diffusion.py; default 1 108 and 2 109)
  fluid       --viscosity (nm^2/ps, default 0.893, water at 298 K), --temperature (K, default 298; 0 = no noise),
              --no-fluid-fluctuations (a fluid without thermal fluctuations), --spacing (nm, default 0.5),
              --dt (ps, default 0.01)
  coupling    --friction (1/ps, default 30), --drag Centered|Explicit (default Centered)
  run         --steps (default 1000000, 10 ns), --seed, --platform CUDA|OpenCL|HIP|Reference (default the fastest
              available), --precision single|mixed|double (default mixed)
  output      --output PREFIX (default poiseuille), --report N (steps between reports, default 1000: 10 ps),
              --profile N (steps between profiles of the fluid, default 10000: 100 ps; 0 = none), --vtk N (steps
              between VTK files, default 0 = none), --checkpoint N (steps between checkpoints, a multiple of --report;
              default 100 reports, 0 = none), --restart

Files written, with the prefix given by --output (default poiseuille):
  <prefix>.dcd               trajectory, every --report steps (OpenMM's DCDReporter: positions in Angstrom)
  <prefix>.log               step, time, potential energy, speed
  <prefix>_temperature.txt   step, time (ps) and temperature of the beads (K), from their velocities relative to the
                             imposed flow at the height of each bead (the translation with the flow, 0.5 nm/ps on
                             the centre line, would add about 1000 K), with the velocity that has the right
                             temperature for the drag scheme, as openmmlbm.LBMTemperatureReporter
  <prefix>_motion.txt        every --report steps: time (ps); centre of mass x y z (nm, not wrapped into the box);
                             velocity of the imposed flow at the height of the centre of mass (nm/ps); rotation of
                             the protein from its starting structure as a quaternion w x y z (best fit weighted by
                             the masses); angle of rotation about y accumulated since the start (rad), as in shear.py
  <prefix>_profile.txt       every --profile steps: time (ps) and the mean velocity u_x of the fluid on each plane of
                             nodes k = 0 ... n_z - 1 (nm/ps), averaged over x and y; the solid plane k = 0 has zero
  <prefix>.chk               checkpoint of the whole run, fluid included, every --checkpoint steps; the same command
                             with --restart continues the run up to --steps (docs/user_guide/restart.md)
  <prefix>_density_<step>.vti, <prefix>_velocity_<step>.vti, <prefix>_particles_<step>.vtp, <prefix>.pvd
                             with --vtk N, every N steps, the fluid (density and velocity on the nodes) and the
                             beads in VTK files for ParaView, in OpenMM units (openmmlbm.LBMVTKReporter); open
                             <prefix>.pvd for the series
"""

import argparse
import math
import os
import sys

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import openmmlbm
from openmmlbm import LBMForce

import cocomo2
from shear import MotionReporter, ProfileReporter, select_platform, truncate, truncate_pvd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
sys.path.insert(0, os.path.join(HERE, '..', 'fluid'))
from initial_state import equilibrium_deviation          # noqa: E402

DENSITY = 1000.0                   # kg/m^3, water


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--velocity', type=float, default=0.5,
                        help='speed of the fluid along x at the centre line, in nm/ps (default 0.5)')
    parser.add_argument('--box', type=float, nargs=2, default=[30.0, 30.0], metavar=('LX', 'LY'),
                        help='sides of the box along x and y in nm (default 30 30)')
    parser.add_argument('--height', type=float, default=50.0,
                        help='side of the box along z in nm; the channel is one spacing narrower (default 50)')
    parser.add_argument('--position', type=float, default=0.0,
                        help='start of the centre of the protein from the centre line, in nm (default 0)')
    parser.add_argument('--spacing', type=float, default=0.5, help='lattice spacing in nm (default 0.5)')
    parser.add_argument('--dt', type=float, default=0.01, help='time step in ps (default 0.01)')
    parser.add_argument('--viscosity', type=float, default=0.893,
                        help='kinematic viscosity in nm^2/ps (default 0.893, water at 298 K)')
    parser.add_argument('--temperature', type=float, default=298.0, help='temperature in K (default 298)')
    parser.add_argument('--friction', type=float, default=30.0, help='friction in 1/ps (default 30)')
    parser.add_argument('--drag', choices=['Explicit', 'Centered'], default='Centered',
                        help='drag scheme of LBMForce (default Centered)')
    parser.add_argument('--no-fluid-fluctuations', action='store_true',
                        help='a fluid without thermal fluctuations (the default fluctuates at --temperature)')
    parser.add_argument('--steps', type=int, default=1000000, help='number of steps (default 1000000: 10 ns)')
    parser.add_argument('--report', type=int, default=1000, help='steps between reports (default 1000)')
    parser.add_argument('--profile', type=int, default=10000,
                        help='steps between profiles of the fluid (default 10000: 100 ps; 0 = none)')
    parser.add_argument('--seed', type=int, default=1234, help='random number seed (default 1234)')
    parser.add_argument('--domain', type=int, nargs=2, default=[1, 108], metavar=('FIRST', 'LAST'),
                        help='folded domain of SOD1 for the exposure scaling and, without --enm-domain, the elastic '
                        'network (default 1 108)')
    parser.add_argument('--enm-domain', type=int, nargs=2, metavar=('FIRST', 'LAST'),
                        help='residues joined by the elastic network (default 2 109 with the default --domain)')
    parser.add_argument('--platform', help='OpenMM platform (default: CUDA, then OpenCL, then Reference)')
    parser.add_argument('--precision', default='mixed', help='precision on CUDA, OpenCL and HIP (default mixed)')
    parser.add_argument('--output', default='poiseuille', help='prefix of the output files (default poiseuille)')
    parser.add_argument('--checkpoint', type=int, help='steps between checkpoints (default 100 reports, 0 = none)')
    parser.add_argument('--vtk', type=int, default=0,
                        help='steps between VTK files of the fluid and of the beads, for ParaView (default 0 = none)')
    parser.add_argument('--restart', action='store_true',
                        help='continue from <prefix>.chk, appending to the output files, up to --steps')
    args = parser.parse_args()
    if args.enm_domain is None:
        args.enm_domain = [2, 109] if args.domain == [1, 108] else args.domain
    cs = args.spacing/(math.sqrt(3)*args.dt)
    if abs(args.velocity) > 0.1*cs:
        parser.error('the centre line moves at Mach %.3f of the lattice: keep |--velocity| below 0.1 c_s = %.3f nm/ps'
                     % (abs(args.velocity)/cs, 0.1*cs))
    for side in args.box + [args.height]:
        if abs(round(side/args.spacing)*args.spacing - side) > 1e-9*side:
            parser.error('every side of the box must be a whole number of lattice spacings')
    if args.checkpoint is None:
        args.checkpoint = 100*args.report
    if args.checkpoint % args.report != 0:
        parser.error('--checkpoint must be a multiple of --report, so that a restart continues the reports exactly')
    return args


class FlowTemperatureReporter:
    """Temperature of the coupled beads from their velocities relative to the imposed flow at their height, with the
    velocity that has the right temperature for the drag scheme (as openmmlbm.LBMTemperatureReporter)."""

    def __init__(self, file, interval, force, flow, height, append=False):
        self._interval = interval
        self._force = force
        self._flow = flow
        self._height = height
        self._particles = None
        self._out = open(file, 'a' if append else 'w')
        if not append:
            print('#"Step","Time (ps)","Temperature relative to the flow (K)"', file=self._out)

    def describeNextReport(self, simulation):
        steps = self._interval - simulation.currentStep % self._interval
        include = ['positions', 'velocities']
        if self._force.getDragScheme() != LBMForce.Centered:
            include.append('forces')
        return {'steps': steps, 'periodic': False, 'include': include}

    def report(self, simulation, state):
        if self._particles is None:
            self._particles = [self._force.getParticle(i) for i in range(self._force.getNumParticles())]
            self._masses = np.array([simulation.system.getParticleMass(i).value_in_unit(unit.dalton)
                                     for i in self._particles])
        x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)[self._particles]
        v = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)[self._particles]
        if self._force.getDragScheme() != LBMForce.Centered:
            forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            dt = simulation.integrator.getStepSize().value_in_unit(unit.picosecond)
            v = v + forces[self._particles]*(0.5*dt/self._masses[:, None])
        v[:, 0] -= self._flow(np.mod(x[:, 2], self._height))
        kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
        temperature = np.sum(self._masses[:, None]*v**2)/(3*len(self._particles)*kB)
        print('%d %.4f %.3f' % (simulation.currentStep, state.getTime().value_in_unit(unit.picosecond), temperature),
              file=self._out, flush=True)


def main():
    args = parse_arguments()
    dx = args.spacing
    nodes = [round(args.box[0]/dx), round(args.box[1]/dx), round(args.height/dx)]
    box = np.array([nodes[0]*dx, nodes[1]*dx, nodes[2]*dx])
    walls = (0.5*dx, (nodes[2] - 0.5)*dx)              # halfway between the solid plane and the first fluid planes
    width = walls[1] - walls[0]
    centre = 0.5*(walls[0] + walls[1])
    names, chains, positions = cocomo2.read_beads(os.path.join(DATA, 'sod1.pdb'))
    positions = cocomo2.centred(positions, box)         # the middle of the box is the centre line
    positions[:, 2] += args.position
    if positions[:, 2].min() < walls[0] + dx or positions[:, 2].max() > walls[1] - dx:
        sys.exit('--position %g puts the protein against a wall: the channel is %g nm wide and SOD1 spans %.1f nm '
                 'along z' % (args.position, width, np.ptp(positions[:, 2])))
    xi = cocomo2.exposure(names, np.loadtxt(os.path.join(DATA, 'sod1.surface')), [tuple(args.domain)])
    pairs = cocomo2.elastic_network(positions, [tuple(args.enm_domain)])
    system = cocomo2.create_system(names, chains, float(box.min()), xi, pairs)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(box[0], 0, 0), mm.Vec3(0, box[1], 0), mm.Vec3(0, 0, box[2]))
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())])
    weights = masses/masses.sum()

    g = 8*args.viscosity*args.velocity/width**2
    fluctuations = not args.no_fluid_fluctuations and args.temperature > 0
    force = LBMForce()
    force.setGridSize(*nodes)
    force.setFluidDensity((DENSITY*unit.kilogram/unit.meter**3*unit.AVOGADRO_CONSTANT_NA).value_in_unit(
        unit.dalton/unit.nanometer**3))
    force.setKinematicViscosity(args.viscosity)
    force.setFriction(args.friction)
    force.setTemperature(args.temperature)
    force.setRandomNumberSeed(args.seed)
    force.setFluidFluctuations(fluctuations)
    force.setDragScheme(getattr(LBMForce, args.drag))
    force.setBodyAcceleration(mm.Vec3(g, 0, 0))
    force.setFluidMomentumRemovalFrequency(0)         # keep the momentum given by the body force
    force.setSolidNodes(list(range(nodes[0]*nodes[1])))    # the plane k = 0 (node i + n_x (j + n_y k))
    for i in range(system.getNumParticles()):
        force.addParticle(i)
    system.addForce(force)                            # the last force of the System, as the centred drag requires
    integrator = mm.VerletIntegrator(args.dt)

    topology = app.Topology()
    topology.setPeriodicBoxVectors([mm.Vec3(box[0], 0, 0), mm.Vec3(0, box[1], 0), mm.Vec3(0, 0, box[2])])
    chain = None
    for i, name in enumerate(names):
        if i == 0 or chains[i] != chains[i - 1]:
            chain = topology.addChain(chains[i])
        topology.addAtom('CA', app.element.carbon, topology.addResidue(name, chain))
    platform, properties = select_platform(args.platform, args.precision)
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(positions)

    dxl, dtl, tau = force.getLatticeParametersInContext(simulation.context)
    slip = g*dx**2*(16*(tau - 0.5)/2 - 3)/(24*args.viscosity)
    def flow(z):
        """The steady profile of the scheme between the walls (docs/theory.md, Exact Poiseuille flow with
        bounce-back)."""
        return g/(2*args.viscosity)*(z - walls[0])*(walls[1] - z) + slip

    start = centre + args.position
    print('Platform %s; SOD1, %d beads, %d elastic bonds; box %g x %g x %g nm, %d x %d x %d nodes' % (
        platform.getName(), len(names), len(pairs), *box, *nodes))
    print('Viscosity %g nm^2/ps (tau %.4f), %g K, friction %g 1/ps, %s drag%s' % (
        args.viscosity, tau, args.temperature, args.friction, args.drag.lower(),
        ', fluctuating fluid' if fluctuations else ''))
    mach = args.velocity/(dx/(math.sqrt(3)*args.dt))
    print('Channel %g nm wide, walls at z = %g and %g nm; acceleration %.4e nm/ps^2; centre line at z = %g nm, '
          '%g nm/ps (Mach %.4f); shear rate at the walls %.5f 1/ps' % (width, walls[0], walls[1], g, centre,
                                                                     flow(centre), mach, 4*args.velocity/width))
    print('The protein starts at z = %g nm: flow %.5f nm/ps, rotation of the flow %.5f rad/ps about y' % (
        start, flow(start), g/(2*args.viscosity)*(centre - start)))

    checkpointFile = args.output + '.chk'
    if args.restart:
        openmmlbm.loadCheckpoint(checkpointFile, simulation.context, force)
        print('Restarted from %s at step %d' % (checkpointFile, simulation.currentStep))
        reference = np.load(args.output + '_reference.npy')
    else:
        simulation.context.setVelocitiesToTemperature(args.temperature, args.seed)
        simulation.minimizeEnergy(tolerance=100*unit.kilojoule_per_mole/unit.nanometer, maxIterations=500000)
        simulation.context.setVelocitiesToTemperature(args.temperature, args.seed + 1)
        state = simulation.context.getState(getPositions=True, getVelocities=True)
        reference = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        np.save(args.output + '_reference.npy', reference)
        # The beads start with the flow at their height, added to their thermal velocities.
        velocities = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
        velocities[:, 0] += flow(reference[:, 2])
        simulation.context.setVelocities(velocities)
        # The fluid starts with the steady profile, in lattice units; the solid plane k = 0 holds no fluid.
        k = np.arange(nodes[0]*nodes[1]*nodes[2])//(nodes[0]*nodes[1])
        u = np.zeros((len(k), 3))
        u[k > 0, 0] = flow(k[k > 0]*dx)*args.dt/dx
        force.setFluidState(simulation.context, equilibrium_deviation(np.ones(len(k)), u).ravel().tolist())

    append = args.restart
    dcdFile = args.output + '.dcd'
    if args.restart:
        step = simulation.currentStep
        time = simulation.context.getTime().value_in_unit(unit.picosecond)
        truncate(args.output + '.log', step)
        truncate(args.output + '_temperature.txt', step)
        truncate(args.output + '_motion.txt', time)
        truncate(args.output + '_profile.txt', time)
        truncate_pvd(args.output + '.pvd', time)
        dcdFile = '%s_%d.dcd' % (args.output, step)
    simulation.reporters.append(app.DCDReporter(dcdFile, args.report))
    simulation.reporters.append(app.StateDataReporter(args.output + '.log', args.report, step=True, time=True,
                                                      potentialEnergy=True, speed=True, separator='\t',
                                                      append=append))
    simulation.reporters.append(FlowTemperatureReporter(args.output + '_temperature.txt', args.report, force, flow,
                                                        box[2], append))
    current = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    simulation.reporters.append(MotionReporter(args.output + '_motion.txt', args.report, weights, reference, flow,
                                               append, current))
    if args.profile > 0:
        simulation.reporters.append(ProfileReporter(args.output + '_profile.txt', args.profile, force, nodes, append))
    if args.checkpoint > 0:
        simulation.reporters.append(openmmlbm.LBMCheckpointReporter(checkpointFile, args.checkpoint, force))
    if args.vtk > 0:
        simulation.reporters.append(openmmlbm.LBMVTKReporter(args.output, args.vtk, force, append=append))
    simulation.step(max(0, args.steps - simulation.currentStep))

    motion = np.loadtxt(args.output + '_motion.txt', ndmin=2)
    if len(motion) > 1:
        t = motion[:, 0]
        drift = np.polyfit(t, motion[:, 1], 1)[0]
        print('Centre of mass at z = %.3f nm, %.3f nm from the centre line; along x it moved at %.5f nm/ps, the flow '
              'at its height at %.5f nm/ps on average' % (motion[-1, 3], motion[-1, 3] - centre, drift,
                                                          motion[:, 4].mean()))
        print('Angle about y %.3f rad in %.1f ps: mean rate %.5f rad/ps (the flow turns at %.5f at the start)' % (
            motion[-1, 9], t[-1], motion[-1, 9]/t[-1], g/(2*args.viscosity)*(centre - start)))
    temperature = np.loadtxt(args.output + '_temperature.txt', ndmin=2)
    print('Mean temperature of the beads relative to the flow over %d reports: %.1f K' % (
        len(temperature), temperature[:, 2].mean()))
    print('Files written with prefix', args.output)


if __name__ == '__main__':
    main()
