# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""The Reference platform with the domain decomposition against one domain, on every rank (docs/theory.md,
section 8; docs/validation.md, Domain decomposition).  It needs the plugin built with MPI (-DOPENMM_LBM_MPI=ON) and
runs under MPI, so pytest does not collect it:

    mpirun -n 4 python mpi_decomposition.py 2 2 1 [OpenCL]

Every rank runs each case twice in the same process, with one domain (no communication) and with px*py*pz domains,
and compares the populations of the fluid nodes it owns (block c of an axis of n nodes holds [n c/p, n (c + 1)/p))
and, with coupled particles at T = 0, the positions and velocities of all the particles.  Without the removal of the
fluid momentum they must be identical bit for bit; with it they agree to rounding, because the sums of the ranks are
added in another order.  Then the checks that must stop every rank together: copies of the particles that differ
between the ranks, an AndersenThermostat, and (with the optional argument OpenCL) rank 0 on the Reference platform
and the others on OpenCL.  A line with FAILED marks a failure."""
import hashlib
import sys
import numpy as np
import openmm as mm
import openmm.unit as unit
import openmmlbm
from openmmlbm import LBMForce

px, py, pz = (int(a) for a in sys.argv[1:4])
other_platform = sys.argv[4] if len(sys.argv) > 4 else None
rank, size = openmmlbm.mpiRank(), openmmlbm.mpiSize()
N = (8, 6, 6)

def owned_mask():
    p = (px, py, pz)
    coords = (rank%px, (rank//px)%py, rank//(px*py))
    mask = np.zeros(N[::-1], dtype=bool)            # [k, j, i]
    lo = [N[a]*coords[a]//p[a] for a in range(3)]
    hi = [N[a]*(coords[a] + 1)//p[a] for a in range(3)]
    mask[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = True
    return mask.reshape(-1)

def run(case, decomposition, steps=60):
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3))
    system.addParticle(1.0)
    force = LBMForce()
    force.setGridSize(*N)
    force.setKinematicViscosity((0.8 - 0.5)/3*0.25/0.01)
    force.setBodyAcceleration(mm.Vec3(0.5, -0.2, 0.1))
    force.setInitialFluidVelocity(mm.Vec3(0.5, 0.2, -0.3))
    force.setFluidMomentumRemovalFrequency(3 if case == 'removal' else 0)
    solid = []
    if case in ('bounceback', 'regularized', 'faces'):
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
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.01)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions([mm.Vec3(0.1, 0.7, 0.3)])
    state = np.array(force.getFluidState(context))
    state += 1e-3*np.sin(1.3*np.arange(len(state)))
    force.setFluidState(context, state)
    integrator.step(steps)
    wall = np.array(force.getWallForce(context).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))
    return np.array(force.getFluidState(context)).reshape(19, -1), wall, solid

# Coupled particles of 50 Da: near the borders of the blocks, sharing a node (0 and 6), crossing the periodic
# boundaries (2) and moving into the wall z = 0 (0 and 4).  A constant field and a soft pair force act on them, so
# that the centred drag reads other forces.
POSITIONS = [(1.70, 0.40, 0.30), (1.70, 2.00, 1.30), (0.05, 2.90, 2.95), (3.20, 0.80, 1.30), (2.40, 2.10, 0.20),
             (1.78, 2.05, 1.22), (1.72, 1.95, 1.35)]
VELOCITIES = [(1.5, 0.3, -2.0), (-1.0, 0.5, 0.2), (-0.8, 1.0, 0.6), (0.4, -0.6, 0.9), (0.0, 0.0, -1.5),
              (0.3, 0.2, -0.4), (-0.2, -0.3, 0.5)]

def run_particles(case, decomposition, steps=60, platform='Reference', thermostat=False, perturb=False, check=True):
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
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.01)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform))
    context.setPositions([mm.Vec3(*x) for x in POSITIONS])
    scale = 1.0 + (1e-12 if perturb and rank == size - 1 else 0.0)
    context.setVelocities([mm.Vec3(*v)*scale for v in VELOCITIES])
    integrator.step(steps)
    state = context.getState(getPositions=True, getVelocities=True)
    particles = np.concatenate([state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                                state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)])
    wall = np.array(force.getWallForce(context).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))
    return np.array(force.getFluidState(context)).reshape(19, -1), wall, solid, particles

def expect_error(name, text, **options):
    # Every rank must stop with the same error, and none may hang in a communication.
    try:
        run_particles('explicit', (px, py, pz), steps=5, **options)
        message = 'no error  FAILED'
    except Exception as e:
        message = ('stops with the expected error' if text in str(e) else 'unexpected error: %s  FAILED' % e)
    print('rank %d/%d %s %-12s %s' % (rank, size, (px, py, pz), name, message), flush=True)

# Without a body force the force on the walls is a small difference between the pressure forces on the two sides of
# the solid plane z = 0, each about rho c_s^2 times its area (6.0e6 kJ/mol/nm here); the ranks add the momenta in
# another order, so it agrees to the rounding of those.
PRESSURE_FORCE = 602.214*(0.5/0.01)**2/3*4*3

mask = owned_mask()
for case in ('periodic', 'bounceback', 'regularized', 'faces', 'removal'):
    single, wall1, solid = run(case, (1, 1, 1))
    split, wall2, _ = run(case, (px, py, pz))
    # The slots of solid nodes only hold what the fluid nodes pushed into them in the last streaming, on the rank of
    # each fluid node: they are not part of the state of the fluid.
    fluid = mask.copy()
    fluid[solid] = False
    diff = np.abs(split[:, fluid] - single[:, fluid]).max()
    walldiff = np.abs(wall2 - wall1).max()/max(1.0, np.abs(wall1).max())
    verdict = 'bitwise identical' if diff == 0 else 'max difference %.1e' % diff
    ok = (diff == 0 if case != 'removal' else diff < 1e-15) and walldiff < 1e-11
    verdict += '' if ok else '  FAILED'
    print('rank %d/%d %s %-12s %s (%d nodes); wall force relative difference %.1e'
          % (rank, size, (px, py, pz), case, verdict, fluid.sum(), walldiff), flush=True)

for case in ('explicit', 'centered', 'faces'):
    single, wall1, solid, particles1 = run_particles(case, (1, 1, 1))
    split, wall2, _, particles2 = run_particles(case, (px, py, pz))
    fluid = mask.copy()
    fluid[solid] = False
    diff = np.abs(split[:, fluid] - single[:, fluid]).max()
    pdiff = np.abs(particles2 - particles1).max()
    absdiff = np.abs(wall2 - wall1).max()
    ok = diff == 0 and pdiff == 0 and absdiff < 1e-12*PRESSURE_FORCE
    print('rank %d/%d %s particles_%-9s fluid %s, particles %s; wall force %.3e, difference %.1e = %.1e of the '
          'pressure force%s' % (rank, size, (px, py, pz), case,
                                'bitwise identical' if diff == 0 else 'max difference %.1e' % diff,
                                'bitwise identical' if pdiff == 0 else 'max difference %.1e' % pdiff,
                                np.abs(wall1).max(), absdiff, absdiff/PRESSURE_FORCE, '' if ok else '  FAILED'),
          flush=True)

# With the fluctuating fluid the ranks draw different random numbers, so the run differs from that of one domain; the
# copies of the particles must stay identical (the plugin compares them every 10 steps here, and the hash printed by
# every rank must be the same).
_, _, _, particles = run_particles('thermal', (px, py, pz), steps=50)
print('rank %d/%d %s thermal      runs; hash of the particles %s'
      % (rank, size, (px, py, pz), hashlib.md5(particles.tobytes()).hexdigest()[:16]), flush=True)

expect_error('copies', 'differ between the MPI ranks', perturb=True)
# With the check off (setParticleCopiesCheck(False)) the same copies run without an error.
try:
    run_particles('explicit', (px, py, pz), steps=5, perturb=True, check=False)
    message = 'not checked, runs'
except Exception as e:
    message = 'unexpected error: %s  FAILED' % e
print('rank %d/%d %s %-12s %s' % (rank, size, (px, py, pz), 'unchecked', message), flush=True)
expect_error('andersen', 'AndersenThermostat', thermostat=True)
if other_platform is not None:
    expect_error('platforms', 'same platform and precision', platform='Reference' if rank == 0 else other_platform)
