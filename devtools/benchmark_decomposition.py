# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Time per step of the fluid with the domain decomposition, one GPU per MPI rank (docs/validation.md, Performance of
the domain decomposition).

    mpirun -n P python benchmark_decomposition.py NX NY NZ PX PY PZ [--platform CUDA] [--precision mixed] [--fluct]
           [--particles N] [--centered] [--steps 200] [--devices 4]

The fluid fills a periodic box of NX x NY x NZ nodes, 0.5 nm apart, divided into PX x PY x PZ domains (P = PX*PY*PZ
ranks), with a body force; --fluct switches the fluid fluctuations on.  --particles N adds N particles of 100 Da
coupled to the fluid at 300 K, at random positions, with a soft repulsion between them (CustomNonbondedForce, cutoff
1 nm): the time adds the forces of OpenMM, which every rank computes for all the particles, and the sum of the
coupling forces over the ranks; --centered uses the Centered drag instead of the Explicit one.  Each rank uses the
GPU of index (local rank) % devices.  After 20 steps the script times --steps steps, between two calls of
getFluidMachNumber(), which are collective and wait for the devices, and rank 0 prints the time per step and the
lattice updates per second (MLUPS), in total and per GPU.  With one rank and 1 1 1 it times one domain."""

import argparse
import time

import numpy as np
import openmm as mm
import openmmlbm
from openmmlbm import LBMForce

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('size', type=int, nargs=3, help='nodes along x, y, z')
parser.add_argument('domains', type=int, nargs=3, help='domains along x, y, z')
parser.add_argument('--platform', default='CUDA', help='OpenMM platform (default CUDA)')
parser.add_argument('--precision', default='mixed', help='precision on the GPU platforms (default mixed)')
parser.add_argument('--fluct', action='store_true', help='fluctuating fluid')
parser.add_argument('--steps', type=int, default=200, help='steps timed (default 200)')
parser.add_argument('--devices', type=int, default=4, help='GPUs per node (default 4)')
parser.add_argument('--particles', type=int, default=0, help='coupled particles (default 0)')
parser.add_argument('--centered', action='store_true', help='Centered drag (default Explicit)')
args = parser.parse_args()
nx, ny, nz = args.size
dx = 0.5
system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(nx*dx, 0, 0), mm.Vec3(0, ny*dx, 0), mm.Vec3(0, 0, nz*dx))
system.addParticle(1.0)
if args.particles > 0:
    repulsion = mm.CustomNonbondedForce('10*(1-r)^2')
    repulsion.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    repulsion.setCutoffDistance(1.0)
    repulsion.addParticle()
    for i in range(args.particles):
        system.addParticle(100.0)
        repulsion.addParticle()
    system.addForce(repulsion)
force = LBMForce()
force.setGridSize(nx, ny, nz)
force.setKinematicViscosity((0.8 - 0.5)/3*dx*dx/0.01)
force.setBodyAcceleration(mm.Vec3(0.01, 0, 0))
force.setFluidMomentumRemovalFrequency(0)
force.setTemperature(300.0)
force.setFluidFluctuations(args.fluct)
for i in range(args.particles):
    force.addParticle(i+1)
force.setFriction(5.0)
if args.centered:
    force.setDragScheme(LBMForce.Centered)
force.setDomainDecomposition(*args.domains)
system.addForce(force)
integrator = mm.VerletIntegrator(0.01)
properties = {}
if args.platform != 'Reference':
    properties = {'Precision': args.precision, 'DeviceIndex': str(openmmlbm.mpiLocalRank() % args.devices)}
    if args.platform in ('CUDA', 'HIP'):
        properties['DeterministicForces'] = 'true'    # identical copies of the particles on every rank
context = mm.Context(system, integrator, mm.Platform.getPlatformByName(args.platform), properties)
positions = np.random.default_rng(1).uniform(0, 1, (args.particles, 3))*[nx*dx, ny*dx, nz*dx]
context.setPositions([mm.Vec3(0.1, 0.1, 0.1)] + [mm.Vec3(*x) for x in positions])
integrator.step(20)
force.getFluidMachNumber(context)
start = time.perf_counter()
integrator.step(args.steps)
force.getFluidMachNumber(context)
elapsed = time.perf_counter() - start
if openmmlbm.mpiRank() == 0:
    ms = 1000*elapsed/args.steps
    mlups = nx*ny*nz/(ms*1e-3)/1e6
    print('%s %s %dx%dx%d nodes, %dx%dx%d domains, fluctuations %s, %d particles%s: %.3f ms/step, %.0f MLUPS, %.0f '
          'MLUPS per GPU' % (args.platform, args.precision, nx, ny, nz, *args.domains, 'on' if args.fluct else 'off',
                             args.particles, ' (Centered drag)' if args.centered else '', ms, mlups,
                             mlups/openmmlbm.mpiSize()))
