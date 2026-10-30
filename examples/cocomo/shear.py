# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""A protein in the middle of a double Couette flow: two open faces that move in opposite directions.

The box is open along z: the face z = 0 (ZMin) moves with -v along x and the face z = L_z (ZMax) with +v, both
DensityVelocity faces at the density of the fluid (docs/user_guide/api_reference.md, Open faces); x and y stay
periodic.  The velocity of a face holds one node beyond it, so the fluid is sheared at the rate
gamma_dot = 2 v/((n_z + 1) dx) and its velocity is zero at z0 = (n_z - 1) dx/2, half a spacing below the middle of
the box, where the protein starts.  There the protein turns about y with the vorticity of the flow, and it is carried
along x only when it diffuses away from the middle.  A sphere would turn at gamma_dot/2; a body of another shape
turns faster or slower as its orientation changes (the Jeffery orbits of an ellipsoid), so its rate differs from
gamma_dot/2 and changes along the run.  The protein is the SOD1 of diffusion.py (COCOMO2, 110 beads,
data/sod1.pdb, the same exposure and elastic network).  Nothing keeps it in the middle: make the box high enough that
it does not reach the faces during the run (with D of about 15 A^2/ns it moves by about 5 nm in 100 ns).  The beads
still live in the periodic box of OpenMM, and a bead that crossed a face would be coupled to the fluid next to the
opposite face, which moves the other way.

The fluid starts with the linear profile of the Couette flow (setFluidState()), so that the protein does not wait for
the flow to build up.  By default the fluid fluctuates at the temperature of the run and the drag is centred, so
that the protein diffuses with its hydrodynamic mobility (docs/theory.md, sections 2 and 7); --no-fluid-fluctuations
and --drag Explicit give the coupling of the original scripts.  The viscosity is given in physical units
(--viscosity, nm^2/ps, water at 298 K by default); the plugin converts it to the relaxation time tau of the lattice,
which the script prints.

The speed of the faces must stay well below the speed of sound of the lattice, c_s = dx/(sqrt(3) dt), 28.9 nm/ps
with the default lattice: the script refuses faces faster than Mach 0.1, 2.9 nm/ps.  Keep them at 1 nm/ps or below
(Mach 0.035): the thermal fluctuations of the fluid add to the flow a velocity of standard deviation
sqrt(k_B T/(rho dx^3)) per node and component, 0.18 nm/ps at 298 K, and the largest of them on the lattice reach
about five times that.  The shear rates of these runs are those of molecular dynamics: with v = 0.5 nm/ps and a
height of 50 nm, gamma_dot = 0.02/ps = 2e10/s, and with a rotation time of SOD1 in water of about 7 ns the
Weissenberg number gamma_dot*tau_r is about 140; a Weissenberg number of 1 needs v of about 0.004 nm/ps, well below
the thermal velocity of the fluid.

Files written, with the prefix given by --output (default shear):
  <prefix>.dcd               trajectory, every --report steps (OpenMM's DCDReporter: positions in Angstrom)
  <prefix>.log               step, time, potential energy, speed
  <prefix>_temperature.txt   temperature of the beads, from openmmlbm.LBMTemperatureReporter; it includes the
                             velocity of the flow at the beads, a few tenths of a percent at the default shear
  <prefix>_motion.txt        every --report steps: time (ps); centre of mass x y z (nm, not wrapped into the box);
                             velocity of the imposed flow at the height of the centre of mass (nm/ps); rotation of
                             the protein from its starting structure as a quaternion w x y z (best fit weighted by
                             the masses); angle of rotation about y accumulated since the start (rad, the sum of
                             the rotations about y between two reports, written with all its digits so that a
                             restarted run continues it exactly)
  <prefix>_profile.txt       every --profile reports: time (ps) and the mean velocity u_x of the fluid on each
                             plane of nodes k = 0 ... n_z - 1 (nm/ps), averaged over x and y
  <prefix>.chk               checkpoint of the whole run, fluid included, every --checkpoint steps; the same
                             command with --restart continues the run up to --steps (docs/user_guide/restart.md)
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

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
sys.path.insert(0, os.path.join(HERE, '..', 'fluid'))
from initial_state import equilibrium_deviation          # noqa: E402

DENSITY = 1000.0                   # kg/m^3, water


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--velocity', type=float, default=0.5,
                        help='speed of the faces along x in nm/ps: ZMin moves with -v, ZMax with +v (default 0.5)')
    parser.add_argument('--box', type=float, nargs=2, default=[30.0, 30.0], metavar=('LX', 'LY'),
                        help='sides of the box along x and y in nm (default 30 30)')
    parser.add_argument('--height', type=float, default=50.0, help='side of the box along z in nm (default 50)')
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
    parser.add_argument('--profile', type=int, default=100,
                        help='reports between profiles of the fluid (default 100; 0 = none)')
    parser.add_argument('--seed', type=int, default=1234, help='random number seed (default 1234)')
    parser.add_argument('--domain', type=int, nargs=2, default=[1, 108], metavar=('FIRST', 'LAST'),
                        help='folded domain of SOD1 for the exposure scaling and, without --enm-domain, the elastic '
                        'network (default 1 108)')
    parser.add_argument('--enm-domain', type=int, nargs=2, metavar=('FIRST', 'LAST'),
                        help='residues joined by the elastic network (default 2 109 with the default --domain)')
    parser.add_argument('--platform', help='OpenMM platform (default: CUDA, then OpenCL, then Reference)')
    parser.add_argument('--precision', default='mixed', help='precision on CUDA, OpenCL and HIP (default mixed)')
    parser.add_argument('--output', default='shear', help='prefix of the output files (default shear)')
    parser.add_argument('--checkpoint', type=int, help='steps between checkpoints (default 100 reports, 0 = none)')
    parser.add_argument('--restart', action='store_true',
                        help='continue from <prefix>.chk, appending to the output files, up to --steps')
    args = parser.parse_args()
    if args.enm_domain is None:
        args.enm_domain = [2, 109] if args.domain == [1, 108] else args.domain
    cs = args.spacing/(math.sqrt(3)*args.dt)
    if abs(args.velocity) > 0.1*cs:
        parser.error('the faces move at Mach %.3f of the lattice: keep |--velocity| below 0.1 c_s = %.3f nm/ps'
                     % (abs(args.velocity)/cs, 0.1*cs))
    for side in args.box + [args.height]:
        if abs(round(side/args.spacing)*args.spacing - side) > 1e-9*side:
            parser.error('every side of the box must be a whole number of lattice spacings')
    if args.checkpoint is None:
        args.checkpoint = 100*args.report
    if args.checkpoint % args.report != 0:
        parser.error('--checkpoint must be a multiple of --report, so that a restart continues the reports exactly')
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


def best_rotation(reference, current, weights):
    """The rotation matrix R that brings the centred reference positions onto the centred current ones, R r ~ x,
    weighted by the masses (Kabsch)."""
    a = (weights[:, None]*current).T @ reference
    u, s, vt = np.linalg.svd(a)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def quaternion(r):
    """Unit quaternion (w, x, y, z) of a rotation matrix, with w >= 0."""
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2]))/2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2]))/2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2]))/2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2]))/2, r[1, 0] - r[0, 1])
    q = np.array([w, x, y, z])
    return q/np.linalg.norm(q)


def rotation_vector(r):
    """Axis times angle of a rotation matrix (the angle in [0, pi])."""
    angle = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1)/2)))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])/(2*math.sin(angle))
    return angle*axis


class MotionReporter:
    """Centre of mass, rotation from the starting structure and angle of rotation about y accumulated."""

    def __init__(self, file, interval, weights, reference, flow, append=False, positions=None):
        """With append, the run continues from a checkpoint: the accumulated angle is read from the last line of the
        file and the orientation is computed from the positions of the checkpoint, as the uninterrupted run did at
        that report."""
        self._interval = interval
        self._weights = np.asarray(weights)
        self._reference = reference - self._weights @ reference
        self._flow = flow
        self._previous = np.eye(3)
        self._angle = 0.0
        if append and os.path.exists(file):
            last = [line for line in open(file) if line.strip() and not line.startswith('#')]
            if last:
                self._angle = float(last[-1].split()[9])
                self._previous = best_rotation(self._reference, positions - self._weights @ positions, self._weights)
        self._out = open(file, 'a' if append else 'w')
        if not append:
            print('# time (ps), centre of mass x y z (nm), flow velocity at its height (nm/ps), rotation from the '
                  'start as quaternion w x y z, angle of rotation about y accumulated (rad)', file=self._out)

    def describeNextReport(self, simulation):
        steps = self._interval - simulation.currentStep % self._interval
        return {'steps': steps, 'periodic': False, 'include': ['positions']}

    def report(self, simulation, state):
        x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        com = self._weights @ x
        r = best_rotation(self._reference, x - com, self._weights)
        self._angle += float(rotation_vector(r @ self._previous.T)[1])
        self._previous = r
        print('%.3f %.6f %.6f %.6f %.6f %.8f %.8f %.8f %.8f %r' % (
            state.getTime().value_in_unit(unit.picosecond), *com, self._flow(com[2]), *quaternion(r), self._angle),
            file=self._out, flush=True)


class ProfileReporter:
    """Mean velocity u_x of the fluid on each plane of nodes along z, averaged over x and y."""

    def __init__(self, file, interval, force, nodes, append=False):
        self._interval = interval
        self._force = force
        self._nodes = nodes
        self._out = open(file, 'a' if append else 'w')
        if not append:
            print('# time (ps), mean u_x (nm/ps) on the planes k = 0 ... %d' % (nodes[2] - 1), file=self._out)

    def describeNextReport(self, simulation):
        steps = self._interval - simulation.currentStep % self._interval
        return {'steps': steps, 'periodic': None, 'include': []}

    def report(self, simulation, state):
        nx, ny, nz = self._nodes
        density, velocity = self._force.getFluidFields(simulation.context)
        u = np.array(velocity.value_in_unit(unit.nanometer/unit.picosecond))[:, 0].reshape(nz, ny*nx)
        print('%.3f ' % state.getTime().value_in_unit(unit.picosecond) + ' '.join('%.5f' % m for m in u.mean(axis=1)),
              file=self._out, flush=True)


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
    dx = args.spacing
    nodes = [round(args.box[0]/dx), round(args.box[1]/dx), round(args.height/dx)]
    box = np.array([nodes[0]*dx, nodes[1]*dx, nodes[2]*dx])
    names, chains, positions = cocomo2.read_beads(os.path.join(DATA, 'sod1.pdb'))
    # The protein starts where the flow is at rest: z0 = (n_z - 1) dx/2, half a spacing below the middle of the box,
    # since the plates are at -dx and n_z dx.
    positions = cocomo2.centred(positions, box)
    positions[:, 2] += 0.5*(nodes[2] - 1)*dx - 0.5*box[2]
    xi = cocomo2.exposure(names, np.loadtxt(os.path.join(DATA, 'sod1.surface')), [tuple(args.domain)])
    pairs = cocomo2.elastic_network(positions, [tuple(args.enm_domain)])
    system = cocomo2.create_system(names, chains, float(box.min()), xi, pairs)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(box[0], 0, 0), mm.Vec3(0, box[1], 0), mm.Vec3(0, 0, box[2]))
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())])
    weights = masses/masses.sum()

    v = args.velocity
    def flow(z):
        """The imposed Couette profile: -v at z = -dx, +v at z = n_z dx."""
        return -v + 2*v*(z + dx)/((nodes[2] + 1)*dx)

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
    force.setFluidMomentumRemovalFrequency(0)         # required with open faces
    for face, sign in ((LBMForce.ZMin, -1.0), (LBMForce.ZMax, 1.0)):
        force.setFaceBoundary(face, LBMForce.DensityVelocity)
        force.setFaceVelocity(face, mm.Vec3(sign*v, 0, 0))
        force.setFaceDensity(face, force.getFluidDensity())
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
    rate = 2*v/((nodes[2] + 1)*dx)
    print('Platform %s; SOD1, %d beads, %d elastic bonds; box %g x %g x %g nm, %d x %d x %d nodes' % (
        platform.getName(), len(names), len(pairs), *box, *nodes))
    print('Viscosity %g nm^2/ps (tau %.4f), %g K, friction %g 1/ps, %s drag%s' % (
        args.viscosity, tau, args.temperature, args.friction, args.drag.lower(),
        ', fluctuating fluid' if fluctuations else ''))
    print('Faces at -/+ %g nm/ps along x (Mach %.4f): shear rate %.5f 1/ps, rotation of the flow %.5f rad/ps about y' % (
        v, v/(dx/(math.sqrt(3)*args.dt)), rate, rate/2))

    checkpointFile = args.output + '.chk'
    reference = positions
    if args.restart:
        openmmlbm.loadCheckpoint(checkpointFile, simulation.context, force)
        print('Restarted from %s at step %d' % (checkpointFile, simulation.currentStep))
        reference = np.load(args.output + '_reference.npy')
    else:
        simulation.context.setVelocitiesToTemperature(args.temperature, args.seed)
        simulation.minimizeEnergy(tolerance=100*unit.kilojoule_per_mole/unit.nanometer, maxIterations=500000)
        simulation.context.setVelocitiesToTemperature(args.temperature, args.seed + 1)
        reference = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(
            unit.nanometer)
        np.save(args.output + '_reference.npy', reference)
        # The fluid starts with the linear profile of the Couette flow, in lattice units.
        k = np.arange(nodes[0]*nodes[1]*nodes[2])//(nodes[0]*nodes[1])
        u = np.zeros((len(k), 3))
        u[:, 0] = flow(k*dx)*args.dt/dx
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
        dcdFile = '%s_%d.dcd' % (args.output, step)
    simulation.reporters.append(app.DCDReporter(dcdFile, args.report))
    simulation.reporters.append(app.StateDataReporter(args.output + '.log', args.report, step=True, time=True,
                                                      potentialEnergy=True, speed=True, separator='\t',
                                                      append=append))
    temperatureFile = open(args.output + '_temperature.txt', 'a' if append else 'w')
    simulation.reporters.append(openmmlbm.LBMTemperatureReporter(temperatureFile, args.report, force))
    start = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    simulation.reporters.append(MotionReporter(args.output + '_motion.txt', args.report, weights, reference, flow,
                                               append, start))
    if args.profile > 0:
        simulation.reporters.append(ProfileReporter(args.output + '_profile.txt', args.profile*args.report, force,
                                                    nodes, append))
    if args.checkpoint > 0:
        simulation.reporters.append(openmmlbm.LBMCheckpointReporter(checkpointFile, args.checkpoint, force))
    simulation.step(max(0, args.steps - simulation.currentStep))

    motion = np.loadtxt(args.output + '_motion.txt', ndmin=2)
    if len(motion) > 0:
        z0 = 0.5*(nodes[2] - 1)*dx
        print('Centre of mass at z = %.3f nm, %.3f nm from the plane at rest; angle about y %.3f rad in %.1f ps: mean '
              'rate %.5f rad/ps (the flow turns at %.5f)' % (motion[-1, 3], motion[-1, 3] - z0, motion[-1, 9],
                                                            motion[-1, 0], motion[-1, 9]/motion[-1, 0], rate/2))
    temperature = np.loadtxt(args.output + '_temperature.txt', ndmin=2)
    print('Mean temperature of the beads over %d reports: %.1f K' % (len(temperature), temperature[:, 2].mean()))
    print('Files written with prefix', args.output)


if __name__ == '__main__':
    main()
