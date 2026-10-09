# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""The domain decomposition against one domain, on every rank (docs/theory.md, section 8; docs/validation.md, Domain
decomposition).  It needs the plugin built with MPI (-DOPENMM_LBM_MPI=ON) and runs under MPI, so pytest does not
collect it:

    mpirun -n 4 python mpi_decomposition.py 2 2 1 [OpenCL] [--platform CUDA --precision double --devices 4]

By default it runs on the Reference platform.  With --platform the cases run on that platform and precision, against
one domain on the same platform (each rank on device local rank % devices with --devices; on CUDA and HIP with
DeterministicForces, which the decomposition with coupled particles needs).

Every rank runs each case twice in the same process, with one domain (no communication) and with px*py*pz domains,
and compares with the single domain: the state of the fluid nodes of its domain (getFluidState(), getLocalDomain()),
the state of the whole lattice gathered on rank 0 (gather=True), the density and velocity of its domain with the halo
(getFluidFields() with halo=True and the exchange of the halo on), those gathered on rank 0 and, with coupled
particles at T = 0, the positions and velocities of all the particles.  The initial state is set from rank 0
(scatter=True).  Without the removal of the fluid momentum they must be identical bit for bit; with it they agree to
rounding, because the sums of the ranks are added in another order.  Then the checks that must stop every rank
together: copies of the particles that differ between the ranks, an AndersenThermostat, LBMForce.createCheckpoint()
(the state of one domain), and (with the optional argument OpenCL) rank 0 on the Reference platform and the others on
OpenCL.  Last, the checkpoint files and the VTK files written by all the ranks with MPI-IO, in the folder
mpi_decomposition_files of the working directory: a run continued from a checkpoint with the same decomposition, with
another one and with one domain, and VTK files against those of one domain.  On the Reference platform the particle
cases run also with the interpolation stencils (explicit and centred drag, Trilinear, ThreePoint and Keys), and the
fluctuating case with Keys.  A line with FAILED marks a failure."""
import hashlib
import os
import sys
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import openmmlbm
from openmmlbm import LBMForce

args = sys.argv[1:]
options = {}
for name in ('--platform', '--precision', '--devices'):
    if name in args:
        i = args.index(name)
        options[name] = args[i+1]
        del args[i:i+2]
px, py, pz = (int(a) for a in args[0:3])
other_platform = args[3] if len(args) > 3 else None
PLATFORM = options.get('--platform', 'Reference')
rank, size = openmmlbm.mpiRank(), openmmlbm.mpiSize()
PROPERTIES = {}
if PLATFORM != 'Reference':
    PROPERTIES['Precision'] = options.get('--precision', 'double')
    if '--devices' in options:
        PROPERTIES['DeviceIndex'] = str(openmmlbm.mpiLocalRank() % int(options['--devices']))
    if PLATFORM in ('CUDA', 'HIP'):
        PROPERTIES['DeterministicForces'] = 'true'
N = (8, 6, 6)

DENSITY, VELOCITY = unit.dalton/unit.nanometer**3, unit.nanometer/unit.picosecond

def block(domain, pad=0):
    # The slices [k, j, i] of the domain of the rank in an array of the lattice, padded by pad nodes on every side.
    (i0, j0, k0), (ni, nj, nk) = domain
    return (slice(k0, k0 + nk + 2*pad), slice(j0, j0 + nj + 2*pad), slice(i0, i0 + ni + 2*pad))

def fluid_of_block(solid, domain):
    mask = np.ones(N[::-1], dtype=bool)
    mask.reshape(-1)[solid] = False
    return mask[block(domain)].reshape(-1)

def state_of_block(state, domain):
    return state.reshape((19,) + N[::-1])[(slice(None),) + block(domain)].reshape(19, -1)

def arrays(fields):
    return np.array(fields[0].value_in_unit(DENSITY)), np.array(fields[1].value_in_unit(VELOCITY)).reshape(-1, 3)

def run(case, decomposition, steps=60, density_halo=True, velocity_halo=True, platform='Reference', properties={},
        checkpoint=False):
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3))
    system.addParticle(1.0)
    force = LBMForce()
    force.setGridSize(*N)
    force.setKinematicViscosity((0.8 - 0.5)/3*0.25/0.01)
    force.setBodyAcceleration(mm.Vec3(0.5, -0.2, 0.1))
    force.setInitialFluidVelocity(mm.Vec3(0.5, 0.2, -0.3))
    force.setFluidMomentumRemovalFrequency(3 if case == 'removal' else 0)
    if case in ('fluctuating', 'noise'):
        force.setTemperature(300.0)
        force.setFluidFluctuations(True)
        force.setRandomNumberSeed(1234)
    solid = []
    if case in ('bounceback', 'regularized', 'faces', 'fluctuating'):
        nx, ny, nz = N
        solid = sorted(set([i + nx*ny*k for k in range(nz) for i in range(nx)] +
                           [i + nx*(j + ny*k) for k in (2, 3) for j in (2, 3) for i in (3, 4)]))
        force.setSolidNodes(solid)
    if case in ('regularized', 'faces'):
        force.setWallScheme(LBMForce.Regularized)
    if case == 'faces':
        force.setFaceBoundary(LBMForce.XMin, LBMForce.Velocity)
        force.setFaceBoundary(LBMForce.XMax, LBMForce.Density)
        force.setFaceVelocity(LBMForce.XMin, mm.Vec3(0.5, 0.1, 0.0))
        force.setFaceDensity(LBMForce.XMax, 605.0)
    force.setDomainDecomposition(*decomposition)
    force.setDensityHaloExchange(density_halo)
    force.setVelocityHaloExchange(velocity_halo)
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.01)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), properties)
    if checkpoint:
        force.createCheckpoint(context)     # the checkpoint of the fluid (OpenMM checkpoints do not hold it)
    context.setPositions([mm.Vec3(0.1, 0.7, 0.3)])
    # Populations perturbed by up to 1e-3 as a function of the global index: rank 0 gathers the state of the lattice,
    # changes it and scatters it (with one domain, gather and scatter change nothing).
    state = np.array(force.getFluidState(context, gather=True))
    if len(state) > 0 and case != 'noise':
        state += 1e-3*np.sin(1.3*np.arange(len(state)))
    force.setFluidState(context, state if len(state) > 0 else None, scatter=True)
    integrator.step(steps)
    return dict(wall=np.array(force.getWallForce(context).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)),
                solid=solid, domain=force.getLocalDomain(context),
                state=np.array(force.getFluidState(context)).reshape(19, -1),
                gathered=np.array(force.getFluidState(context, gather=True)),
                fields=arrays(force.getFluidFields(context, halo=True)),
                gathered_fields=arrays(force.getFluidFields(context, gather=True)))

def compare_halo(single, split, case, density_halo=True, velocity_halo=True):
    # The fields of the domain with the halo, against those of one domain padded periodically, with NaN beyond the open
    # faces and for the fields whose halo is not exchanged.
    nx, ny, nz = N
    density, velocity = single['gathered_fields']
    density = np.pad(density.reshape(nz, ny, nx), 1, mode='wrap')
    velocity = np.pad(velocity.reshape(nz, ny, nx, 3), ((1, 1), (1, 1), (1, 1), (0, 0)), mode='wrap')
    if case == 'faces':
        density[:, :, [0, -1]] = np.nan
        velocity[:, :, [0, -1]] = np.nan
    expected_density = density[block(split['domain'], 1)].copy()
    expected_velocity = velocity[block(split['domain'], 1)].copy()
    halo = np.ones(expected_density.shape, dtype=bool)
    halo[1:-1, 1:-1, 1:-1] = False
    if not density_halo:
        expected_density[halo] = np.nan
    if not velocity_halo:
        expected_velocity[halo] = np.nan
    got_density = split['fields'][0].reshape(expected_density.shape)
    got_velocity = split['fields'][1].reshape(expected_velocity.shape)
    if not (np.array_equal(np.isnan(got_density), np.isnan(expected_density)) and
            np.array_equal(np.isnan(got_velocity), np.isnan(expected_velocity))):
        return np.inf
    return max(np.nan_to_num(np.abs(got_density - expected_density)).max()/602.214,
               np.nan_to_num(np.abs(got_velocity - expected_velocity)).max())

# Coupled particles of 50 Da: near the borders of the blocks, sharing a node (0 and 6), and crossing the periodic
# boundaries into the solid plane y = 0, where it is reflected (2).  A constant field and a soft pair force act on them, so
# that the centred drag reads other forces.
POSITIONS = [(1.70, 0.40, 0.30), (1.70, 2.00, 1.30), (0.05, 2.90, 2.95), (3.20, 0.80, 1.30), (2.40, 2.10, 0.20),
             (1.78, 2.05, 1.22), (1.72, 1.95, 1.35)]
VELOCITIES = [(1.5, 0.3, -2.0), (-1.0, 0.5, 0.2), (-0.8, 1.0, 0.6), (0.4, -0.6, 0.9), (0.0, 0.0, -1.5),
              (0.3, 0.2, -0.4), (-0.2, -0.3, 0.5)]

def particle_context(case, decomposition, platform='Reference', thermostat=False, check=True, properties={}, halo=False,
                     simulation=False):
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3))
    field = mm.CustomExternalForce('-3*x+2*y-z')
    pair = mm.CustomNonbondedForce('50*(1-r)^2')
    pair.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    pair.setCutoffDistance(1.0)
    force = LBMForce()
    force.setGridSize(*N)
    force.setKinematicViscosity((0.8 - 0.5)/3*0.25/0.01)
    for i in range(len(POSITIONS)):
        system.addParticle(50.0)
        field.addParticle(i)
        pair.addParticle()
        force.addParticle(i)
    system.addForce(field)
    system.addForce(pair)
    if thermostat:
        system.addForce(mm.AndersenThermostat(300.0, 1.0))
    force.setFriction(10.0)
    force.setFluidMomentumRemovalFrequency(0)
    # A case 'drag_Stencil' couples the particles with an interpolation stencil (docs/theory.md, section 9).
    case, _, stencil = case.partition('_')
    if stencil:
        force.setInterpolationStencil(getattr(LBMForce, stencil))
    force.setDragScheme(LBMForce.Centered if case in ('centered', 'faces', 'thermal') else LBMForce.Explicit)
    if case == 'thermal':
        force.setTemperature(300.0)
        force.setFluidFluctuations(True)
        force.setMachCheckFrequency(10)
        force.setRandomNumberSeed(1234)
    else:
        force.setTemperature(0.0)
    nx, ny, nz = N
    solid = sorted(set([i + nx*ny*k for k in range(nz) for i in range(nx)] +
                       [i + nx*(j + ny*k) for k in (2, 3) for j in (2, 3) for i in (3, 4)]))
    force.setSolidNodes(solid)
    if case in ('centered', 'faces'):
        force.setWallScheme(LBMForce.Regularized)
    if case == 'faces':
        force.setFaceBoundary(LBMForce.XMin, LBMForce.Velocity)
        force.setFaceBoundary(LBMForce.XMax, LBMForce.Density)
        force.setFaceVelocity(LBMForce.XMin, mm.Vec3(0.5, 0.1, 0.0))
        force.setFaceDensity(LBMForce.XMax, 605.0)
    force.setDomainDecomposition(*decomposition)
    force.setParticleCopiesCheck(check)
    force.setVelocityHaloExchange(halo)
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.01)
    if simulation:
        sim = app.Simulation(app.Topology(), system, integrator, mm.Platform.getPlatformByName(platform), properties)
        context = sim.context
    else:
        sim = None
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), properties)
    return context, integrator, force, solid, sim

def run_particles(case, decomposition, steps=60, platform='Reference', thermostat=False, perturb=False, check=True,
                  properties={}, halo=False, checkpoint=False):
    context, integrator, force, solid, _ = particle_context(case, decomposition, platform, thermostat, check, properties,
                                                            halo)
    if checkpoint:
        context.createCheckpoint()
    context.setPositions([mm.Vec3(*x) for x in POSITIONS])
    # In single precision the velocities are stored as float: a perturbation of 1e-12 would vanish.
    epsilon = 1e-6 if properties.get('Precision') == 'single' else 1e-12
    scale = 1.0 + (epsilon if perturb and rank == size - 1 else 0.0)
    context.setVelocities([mm.Vec3(*v)*scale for v in VELOCITIES])
    integrator.step(steps)
    state = context.getState(getPositions=True, getVelocities=True)
    particles = np.concatenate([state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                                state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)])
    wall = np.array(force.getWallForce(context).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))
    return np.array(force.getFluidState(context)).reshape(19, -1), wall, solid, particles, force.getLocalDomain(context)

def expect_error(name, text, case='explicit', **options):
    # Every rank must stop with the same error, and none may hang in a communication.
    try:
        if case == 'fluid':
            run('periodic', (px, py, pz), steps=5, **options)
        else:
            run_particles(case, (px, py, pz), steps=5, **options)
        message = 'no error  FAILED'
    except Exception as e:
        message = ('stops with the expected error' if text in str(e) else 'unexpected error: %s  FAILED' % e)
    print('rank %d/%d %s %-12s %s' % (rank, size, (px, py, pz), name, message), flush=True)

# Without a body force the force on the walls is a small difference between the pressure forces on the two sides of
# the solid plane y = 0, each about rho c_s^2 times its area (6.0e6 kJ/mol/nm here); the ranks add the momenta in
# another order, so it agrees to the rounding of those.
PRESSURE_FORCE = 602.214*(0.5/0.01)**2/3*4*3

def verdict(diff):
    return 'bitwise identical' if diff == 0 else 'max difference %.1e' % diff

reference = (PLATFORM == 'Reference')
ON = dict(platform=PLATFORM, properties=PROPERTIES)
single_precision = (PROPERTIES.get('Precision') == 'single')
for case in ('periodic', 'bounceback', 'regularized', 'faces', 'removal', 'velocityhalo', 'densityhalo'):
    # velocityhalo and densityhalo: the periodic fluid with the exchange of the halo of one field only.
    halos = dict(density_halo=(case != 'velocityhalo'), velocity_halo=(case != 'densityhalo'))
    single = run('periodic' if case.endswith('halo') else case, (1, 1, 1), **ON)
    split = run('periodic' if case.endswith('halo') else case, (px, py, pz), **halos, **ON)
    # The slots of solid nodes only hold what the fluid nodes pushed into them in the last streaming, on the rank of
    # each fluid node: they are not part of the state of the fluid.
    fluid = fluid_of_block(single['solid'], split['domain'])
    diff = np.abs(split['state'][:, fluid] - state_of_block(single['state'], split['domain'])[:, fluid]).max()
    if rank == 0:
        everywhere = np.ones(np.prod(N), dtype=bool)
        everywhere[single['solid']] = False
        gathered = np.abs(split['gathered'].reshape(19, -1)[:, everywhere] - single['state'][:, everywhere]).max()
        gathered_fields = max(np.abs(split['gathered_fields'][0] - single['gathered_fields'][0]).max()/602.214,
                              np.abs(split['gathered_fields'][1] - single['gathered_fields'][1]).max())
    else:
        gathered = 0 if len(split['gathered']) == 0 else np.inf
        gathered_fields = 0 if len(split['gathered_fields'][0]) == 0 else np.inf
    halo = compare_halo(single, split, case, **halos)
    walldiff = np.abs(split['wall'] - single['wall']).max()/max(1.0, np.abs(single['wall']).max())
    tolerance = (1e-6 if single_precision else 1e-15) if case == 'removal' else 0
    # In single precision each rank sums in float the links of a solid node to its own fluid nodes, one domain all of
    # them: the force on the walls then agrees to float rounding.
    ok = max(diff, gathered, gathered_fields, halo) <= tolerance and walldiff < (1e-6 if single_precision else 1e-11)
    print('rank %d/%d %s %-12s state %s (%d nodes), gathered %s; fields with halo %s, gathered %s; wall force relative '
          'difference %.1e%s' % (rank, size, (px, py, pz), case, verdict(diff), fluid.sum(), verdict(gathered),
                                 verdict(halo), verdict(gathered_fields), walldiff, '' if ok else '  FAILED'), flush=True)

# With the fluctuating fluid each rank draws its own random numbers, so the run differs from that of one domain; the
# total mass is conserved in both.
single = run('fluctuating', (1, 1, 1), steps=40, density_halo=False, velocity_halo=False, **ON)
split = run('fluctuating', (px, py, pz), steps=40, density_halo=False, velocity_halo=False, **ON)
if rank == 0:
    fluid = np.ones(np.prod(N), dtype=bool)
    fluid[single['solid']] = False
    mass1, mass2 = (r['gathered'].reshape(19, -1)[:, fluid].sum() for r in (single, split))
    changed = np.abs(split['gathered'] - single['gathered']).max()
    ok = abs(mass2 - mass1) < (1e-3 if single_precision else 1e-10) and changed > 0
    print('rank %d/%d %s fluctuating  runs; mass deviation %.3e against %.3e with one domain, populations differ by '
          '%.1e%s' % (rank, size, (px, py, pz), mass2, mass1, changed, '' if ok else '  FAILED'), flush=True)

# The ranks draw independent random numbers.  A fluctuating fluid without walls and without the initial perturbation is
# invariant under translations, apart from the noise: the fluctuations of corresponding nodes of two equal blocks along
# x would be identical if two ranks drew the same numbers, and are uncorrelated otherwise.
if px in (2, 4) and N[0] % px == 0:
    noise = run('noise', (px, py, pz), steps=100, density_halo=False, velocity_halo=False, **ON)
    if rank == 0:
        f = noise['gathered'].reshape((19,) + N[::-1])
        f = f - f.reshape(19, -1).mean(1)[:, None, None, None]
        width = N[0]//px
        a, b = f[..., :width], f[..., width:2*width]
        corr = (a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum())
        print('rank %d/%d %s independence  correlation of the fluctuations of two blocks %.3f (identical noise: 1)%s'
              % (rank, size, (px, py, pz), corr, '' if abs(corr) < 0.3 else '  FAILED'), flush=True)

if PLATFORM in ('CUDA', 'HIP'):
    expect_error('determinism', 'DeterministicForces', platform=PLATFORM,
                 properties=dict(PROPERTIES, DeterministicForces='false'))
# LBMForce.createCheckpoint() writes the state of one domain: with the decomposition the checkpoints go to a file.
expect_error('checkpoint', 'checkpoints are written to a file', case='fluid', density_halo=False, velocity_halo=False,
             checkpoint=True, **ON)

# The interpolation stencils run on the Reference platform for now (in development for version 0.5.0).
STENCIL_CASES = [drag + '_' + stencil for drag in ('explicit', 'centered') for stencil in ('Trilinear', 'ThreePoint', 'Keys')]
for case in ['explicit', 'centered', 'faces'] + (STENCIL_CASES if reference else []):
    single, wall1, solid, particles1, _ = run_particles(case, (1, 1, 1), **ON)
    split, wall2, _, particles2, domain = run_particles(case, (px, py, pz), **ON)
    fluid = fluid_of_block(solid, domain)
    diff = np.abs(split[:, fluid] - state_of_block(single, domain)[:, fluid]).max()
    pdiff = np.abs(particles2 - particles1).max()
    absdiff = np.abs(wall2 - wall1).max()
    ok = diff == 0 and pdiff == 0 and absdiff < (1e-6 if single_precision else 1e-12)*PRESSURE_FORCE
    print('rank %d/%d %s particles_%-19s fluid %s, particles %s; wall force %.3e, difference %.1e = %.1e of the '
          'pressure force%s' % (rank, size, (px, py, pz), case,
                                'bitwise identical' if diff == 0 else 'max difference %.1e' % diff,
                                'bitwise identical' if pdiff == 0 else 'max difference %.1e' % pdiff,
                                np.abs(wall1).max(), absdiff, absdiff/PRESSURE_FORCE, '' if ok else '  FAILED'),
          flush=True)

# With the fluctuating fluid the ranks draw different random numbers, so the run differs from that of one domain; the
# copies of the particles must stay identical (the plugin compares them every 10 steps here, and the hash printed by
# every rank must be the same).
for case in ['thermal'] + (['thermal_Keys'] if reference else []):
    _, _, _, particles, _ = run_particles(case, (px, py, pz), steps=50, **ON)
    print('rank %d/%d %s %-12s runs; hash of the particles %s'
          % (rank, size, (px, py, pz), case, hashlib.md5(particles.tobytes()).hexdigest()[:16]), flush=True)

expect_error('copies', 'differ between the MPI ranks', perturb=True, **ON)
# With the check off (setParticleCopiesCheck(False)) the same copies run without an error.
try:
    run_particles('explicit', (px, py, pz), steps=5, perturb=True, check=False, **ON)
    message = 'not checked, runs'
except Exception as e:
    message = 'unexpected error: %s  FAILED' % e
print('rank %d/%d %s %-12s %s' % (rank, size, (px, py, pz), 'unchecked', message), flush=True)
expect_error('andersen', 'AndersenThermostat', thermostat=True, **ON)
if other_platform is not None:
    expect_error('platforms', 'same platform and precision', platform='Reference' if rank == 0 else other_platform)

# Checkpoint files (LBMForce.saveCheckpointFile(), openmmlbm.saveCheckpoint()) and VTK files of the fluid, written by
# every rank for its domain with MPI-IO, in the folder mpi_decomposition_files of the working directory (shared by the
# ranks).  A run of 60 steps is interrupted by a checkpoint after 30 and continued in new Contexts: with the same
# decomposition it must continue bit for bit, also with the fluctuating fluid; with another decomposition, or with one
# domain, the fluid and the particles restored must be those saved, and without fluctuations the continued run must be
# that of the uninterrupted one (it does not depend on the decomposition).
FILES = 'mpi_decomposition_files'
os.makedirs(FILES, exist_ok=True)

def snapshot(context, force):
    state = context.getState(getPositions=True, getVelocities=True)
    particles = np.concatenate([state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                                state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)])
    return (np.array(force.getFluidState(context, gather=True)).reshape(19, -1), particles,
            state.getTime().value_in_unit(unit.picosecond), state.getStepCount())

def positions_restored(loaded, saved, same):
    # With another decomposition the particles are set from their State.  In mixed and single precision OpenMM keeps
    # the positions in float, wrapped into the box (with a correction in mixed precision), and the State gives them
    # unwrapped: set again, a position outside the box is rounded to float, so it agrees to that rounding.
    if same or PROPERTIES.get('Precision', 'double') == 'double':
        return np.array_equal(loaded, saved)
    return np.array_equal(loaded[len(POSITIONS):], saved[len(POSITIONS):]) and \
        np.all(np.abs(loaded - saved) <= 1e-6*np.maximum(1.0, np.abs(saved)))

def started(case, decomposition):
    context, integrator, force, solid, _ = particle_context(case, decomposition, **ON)
    context.setPositions([mm.Vec3(*x) for x in POSITIONS])
    context.setVelocities([mm.Vec3(*v) for v in VELOCITIES])
    return context, integrator, force, solid

others = [d for d in ((size, 1, 1), (1, size, 1), (1, 1, size)) if d != (px, py, pz)][:1] + [(1, 1, 1)]
for case in ('centered', 'faces', 'thermal'):
    path = '%s/%s_%d%d%d.chk' % (FILES, case, px, py, pz)
    context, integrator, force, solid = started(case, (px, py, pz))
    integrator.step(30)
    openmmlbm.saveCheckpoint(path, context, force)
    saved = snapshot(context, force)
    integrator.step(30)
    final = snapshot(context, force)
    fluid = np.ones(np.prod(N), dtype=bool)
    fluid[solid] = False
    for decomposition in [(px, py, pz)] + others:
        context2, integrator2, force2, _, _ = particle_context(case, decomposition, **ON)
        openmmlbm.loadCheckpoint(path, context2, force2)
        loaded = snapshot(context2, force2)
        integrator2.step(30)
        continued = snapshot(context2, force2)
        if rank != 0:
            continue
        same = (decomposition == (px, py, pz))
        restored = (np.array_equal(loaded[0], saved[0]) and loaded[2:] == saved[2:] and
                    positions_restored(loaded[1], saved[1], same))
        fdiff = np.abs(continued[0][:, fluid] - final[0][:, fluid]).max()
        pdiff = np.abs(continued[1] - final[1]).max()
        if same or case != 'thermal':
            # With another decomposition the particles are set from their State, which is restored exactly; in mixed
            # and single precision OpenMM rebuilds from it its float representation of the positions (wrapped into
            # the box, with a correction in mixed precision), whose last bits may differ, so the run continues to
            # rounding.
            exact = (same or PROPERTIES.get('Precision', 'double') == 'double')
            ok = restored and continued[2:] == final[2:] and (fdiff == pdiff == 0 if exact else max(fdiff, pdiff) < 1e-5)
            result = 'continued run: fluid %s, particles %s' % (verdict(fdiff), verdict(pdiff))
        else:
            ok = restored and continued[3] == final[3]
            result = 'continued with new random numbers (fluid differs by %.1e)' % fdiff
        print('rank %d/%d %s checkpoint_%-8s loaded with %s: state %s, %s%s'
              % (rank, size, (px, py, pz), case, decomposition, 'restored' if restored else 'NOT RESTORED',
                 result, '' if ok else '  FAILED'), flush=True)

# A file of saveCheckpoint() with one domain (written by rank 0 for itself) cannot be loaded with the decomposition;
# one of saveCheckpointFile() with one domain can.
single_path = '%s/single_%d.chk' % (FILES, rank)
single_file = '%s/single_file_%d.chk' % (FILES, rank)
context, integrator, force, solid = started('centered', (1, 1, 1))
integrator.step(30)
openmmlbm.saveCheckpoint(single_path, context, force)
force.saveCheckpointFile(context, single_file)
saved = snapshot(context, force)
integrator.step(30)
final = snapshot(context, force)
context2, integrator2, force2, _, _ = particle_context('centered', (px, py, pz), **ON)
try:
    openmmlbm.loadCheckpoint('%s/single_0.chk' % FILES, context2, force2)
    message = 'no error  FAILED'
except Exception as e:
    message = ('stops with the expected error' if 'written with one domain' in str(e) else 'unexpected error: %s  FAILED' % e)
print('rank %d/%d %s %-12s %s' % (rank, size, (px, py, pz), 'single_chk', message), flush=True)
openmmlbm.loadCheckpoint('%s/single_file_0.chk' % FILES, context2, force2)
loaded = snapshot(context2, force2)
integrator2.step(30)
continued = snapshot(context2, force2)
if rank == 0:
    exact = PROPERTIES.get('Precision', 'double') == 'double'
    fdiff = np.abs(continued[0][:, fluid] - final[0][:, fluid]).max()
    pdiff = np.abs(continued[1] - final[1]).max()
    restored = np.array_equal(loaded[0], saved[0]) and positions_restored(loaded[1], saved[1], False)
    ok = restored and (fdiff == pdiff == 0 if exact else max(fdiff, pdiff) < 1e-5)
    print('rank %d/%d %s single_file   one domain loaded with the decomposition: state %s, continued run: fluid %s, '
          'particles %s%s' % (rank, size, (px, py, pz), 'restored' if restored else 'NOT RESTORED',
                              verdict(fdiff), verdict(pdiff), '' if ok else '  FAILED'), flush=True)

# VTK files of the fluid: the files written with the decomposition (every rank its domain) must be those of one domain,
# byte for byte, and only rank 0 writes the particles and the list.
for double in (False, True):
    prefixes = {}
    for decomposition, prefix in (((1, 1, 1), '%s/vtk_single_%d' % (FILES, rank)), ((px, py, pz), '%s/vtk_split' % FILES)):
        context, integrator, force, solid, sim = particle_context('faces', decomposition, simulation=True, **ON)
        context.setPositions([mm.Vec3(*x) for x in POSITIONS])
        context.setVelocities([mm.Vec3(*v) for v in VELOCITIES])
        sim.reporters.append(openmmlbm.LBMVTKReporter(prefix, 10, force, double=double))
        sim.step(20)
        prefixes[decomposition == (1, 1, 1)] = prefix
    if rank == 0:
        names = ['density_0000000020.vti', 'velocity_0000000020.vti', 'particles_0000000020.vtp']
        same = [open('%s_%s' % (prefixes[True], n), 'rb').read() == open('%s_%s' % (prefixes[False], n), 'rb').read()
                for n in names]
        print('rank %d/%d %s vtk_%-8s density, velocity and particles files identical to one domain: %s%s'
              % (rank, size, (px, py, pz), 'double' if double else 'single', same, '' if all(same) else '  FAILED'),
              flush=True)
