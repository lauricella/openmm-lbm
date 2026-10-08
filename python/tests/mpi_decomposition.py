# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""The fluid of the Reference platform with the domain decomposition against one domain, on every rank
(docs/theory.md, section 8; docs/validation.md, Domain decomposition).  It needs the plugin built with MPI
(-DOPENMM_LBM_MPI=ON) and runs under MPI, so pytest does not collect it:

    mpirun -n 4 python mpi_decomposition.py 2 2 1

Every rank runs each case twice in the same process, with one domain (no communication) and with px*py*pz domains,
and compares the populations of the fluid nodes it owns (block c of an axis of n nodes holds [n c/p, n (c + 1)/p)).
Without the removal of the fluid momentum they must be identical bit for bit; with it they agree to rounding,
because the sums of the ranks are added in another order."""
import sys
import numpy as np
import openmm as mm
import openmm.unit as unit
import openmmlbm
from openmmlbm import LBMForce

px, py, pz = (int(a) for a in sys.argv[1:4])
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
