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
DeterministicForces, which the decomposition with coupled particles needs), and the checkpoints, not available yet
there with the decomposition, must stop every rank with an error.

Every rank runs each case twice in the same process, with one domain (no communication) and with px*py*pz domains,
and compares with the single domain: the state of the fluid nodes of its domain (getFluidState(), getLocalDomain()),
the state of the whole lattice gathered on rank 0 (gather=True), the density and velocity of its domain with the halo
(getFluidFields() with halo=True and the exchange of the halo on), those gathered on rank 0 and, with coupled
particles at T = 0, the positions and velocities of all the particles.  The initial state is set from rank 0
(scatter=True).  Without the removal of the fluid momentum they must be identical bit for bit; with it they agree to
rounding, because the sums of the ranks are added in another order.  Then the checks that must stop every rank
together: copies of the particles that differ between the ranks, an AndersenThermostat, and (with the optional
argument OpenCL) rank 0 on the Reference platform and the others on OpenCL.  A line with FAILED marks a failure."""
import hashlib
import sys
import numpy as np
import openmm as mm
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
    if case == 'fluctuating':
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
    if len(state) > 0:
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

# Coupled particles of 50 Da: near the borders of the blocks, sharing a node (0 and 6), crossing the periodic
# boundaries (2) and moving into the wall z = 0 (0 and 4).  A constant field and a soft pair force act on them, so
# that the centred drag reads other forces.
POSITIONS = [(1.70, 0.40, 0.30), (1.70, 2.00, 1.30), (0.05, 2.90, 2.95), (3.20, 0.80, 1.30), (2.40, 2.10, 0.20),
             (1.78, 2.05, 1.22), (1.72, 1.95, 1.35)]
VELOCITIES = [(1.5, 0.3, -2.0), (-1.0, 0.5, 0.2), (-0.8, 1.0, 0.6), (0.4, -0.6, 0.9), (0.0, 0.0, -1.5),
              (0.3, 0.2, -0.4), (-0.2, -0.3, 0.5)]

def run_particles(case, decomposition, steps=60, platform='Reference', thermostat=False, perturb=False, check=True,
                  properties={}, halo=False, checkpoint=False):
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
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), properties)
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
# the solid plane z = 0, each about rho c_s^2 times its area (6.0e6 kJ/mol/nm here); the ranks add the momenta in
# another order, so it agrees to the rounding of those.
PRESSURE_FORCE = 602.214*(0.5/0.01)**2/3*4*3

def verdict(diff):
    return 'bitwise identical' if diff == 0 else 'max difference %.1e' % diff

reference = (PLATFORM == 'Reference')
ON = dict(platform=PLATFORM, properties=PROPERTIES)
single_precision = (PROPERTIES.get('Precision') == 'single')
for case in ('periodic', 'bounceback', 'regularized', 'faces', 'removal', 'velocityhalo'):
    halos = dict(density_halo=(case != 'velocityhalo'))
    single = run('periodic' if case == 'velocityhalo' else case, (1, 1, 1), **ON)
    split = run('periodic' if case == 'velocityhalo' else case, (px, py, pz), **halos, **ON)
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

if not reference:
    # Not available yet on this platform with the decomposition: every rank must stop with the error.
    expect_error('checkpoint', 'checkpoints with the domain decomposition', case='fluid', density_halo=False,
                 velocity_halo=False, checkpoint=True, **ON)
    if PLATFORM in ('CUDA', 'HIP'):
        expect_error('determinism', 'DeterministicForces', platform=PLATFORM,
                     properties=dict(PROPERTIES, DeterministicForces='false'))

for case in ('explicit', 'centered', 'faces'):
    single, wall1, solid, particles1, _ = run_particles(case, (1, 1, 1), **ON)
    split, wall2, _, particles2, domain = run_particles(case, (px, py, pz), **ON)
    fluid = fluid_of_block(solid, domain)
    diff = np.abs(split[:, fluid] - state_of_block(single, domain)[:, fluid]).max()
    pdiff = np.abs(particles2 - particles1).max()
    absdiff = np.abs(wall2 - wall1).max()
    ok = diff == 0 and pdiff == 0 and absdiff < (1e-6 if single_precision else 1e-12)*PRESSURE_FORCE
    print('rank %d/%d %s particles_%-9s fluid %s, particles %s; wall force %.3e, difference %.1e = %.1e of the '
          'pressure force%s' % (rank, size, (px, py, pz), case,
                                'bitwise identical' if diff == 0 else 'max difference %.1e' % diff,
                                'bitwise identical' if pdiff == 0 else 'max difference %.1e' % pdiff,
                                np.abs(wall1).max(), absdiff, absdiff/PRESSURE_FORCE, '' if ok else '  FAILED'),
          flush=True)

# With the fluctuating fluid the ranks draw different random numbers, so the run differs from that of one domain; the
# copies of the particles must stay identical (the plugin compares them every 10 steps here, and the hash printed by
# every rank must be the same).
_, _, _, particles, _ = run_particles('thermal', (px, py, pz), steps=50, **ON)
print('rank %d/%d %s thermal      runs; hash of the particles %s'
      % (rank, size, (px, py, pz), hashlib.md5(particles.tobytes()).hexdigest()[:16]), flush=True)

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
