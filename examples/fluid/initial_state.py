# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Start the fluid from a chosen density and velocity field, and save or load its state.

A new Context starts the fluid at rest, or at the uniform velocity of setInitialFluidVelocity().  Any
other initial condition is set with setFluidState(), which takes the populations of every node as
deviations from the rest equilibrium, in lattice units (docs/theory.md, section 4).  This script builds
them from a density and a velocity field with the equilibrium of the D3Q19 lattice, here a shear wave

    u_x(y) = U sin(2 pi y / L),   u_y = u_z = 0,   uniform density,

and follows its decay, which for a Newtonian fluid is U exp(-nu k^2 t) with k = 2 pi / L.

The state and the time can be written to a NumPy file with --save and read back with --load: a run that
loads the file of another one continues it exactly.  This replaces the raw restart files of the old
plugin.
"""

import argparse
import math

import numpy as np
import openmm as mm
import openmm.unit as unit
from openmmlbm import LBMForce

# D3Q19 velocities and weights in the order used by the plugin (getFluidState: index q*numNodes + node).
CX = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1, 0, 0, 0, 0])
CZ = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1])
W = np.array([1/3] + [1/18]*6 + [1/36]*12)


def equilibrium_deviation(rho, u):
    """Equilibrium populations minus the weights, feq_q - w_q, for the lattice density rho (numNodes) and
    the lattice velocity u (numNodes x 3); returned as an array of shape (19, numNodes)."""
    cu = np.outer(CX, u[:, 0]) + np.outer(CY, u[:, 1]) + np.outer(CZ, u[:, 2])
    uu = (u**2).sum(axis=1)
    return W[:, None]*((rho - 1) + rho*(3*cu + 4.5*cu**2 - 1.5*uu))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nodes', type=int, default=32, help='nodes along x and y (default 32; 4 along z)')
    parser.add_argument('--spacing', type=float, default=0.5, help='lattice spacing in nm (default 0.5)')
    parser.add_argument('--dt', type=float, default=0.01, help='time step in ps (default 0.01)')
    parser.add_argument('--viscosity', type=float, default=1.0035, help='kinematic viscosity in nm^2/ps (default 1.0035)')
    parser.add_argument('--amplitude', type=float, default=0.05, help='amplitude U of the wave in nm/ps (default 0.05)')
    parser.add_argument('--steps', type=int, default=1000, help='number of steps (default 1000)')
    parser.add_argument('--interval', type=int, default=100, help='steps between lines of output (default 100)')
    parser.add_argument('--save', help='write the final state of the fluid and the time to this .npz file')
    parser.add_argument('--load', help='start from the state and time in this .npz file, not from the wave')
    parser.add_argument('--platform', help='OpenMM platform (default: CUDA, then OpenCL, then Reference)')
    parser.add_argument('--precision', default='mixed', help='precision on CUDA, OpenCL and HIP (default mixed)')
    return parser.parse_args()


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


def main():
    args = parse_arguments()
    nx, ny, nz = args.nodes, args.nodes, 4
    L = nx*args.spacing

    # A fluid without coupled particles.  OpenMM needs at least one particle in the System.
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0), mm.Vec3(0, L, 0), mm.Vec3(0, 0, nz*args.spacing))
    system.addParticle(1.0)
    force = LBMForce()
    force.setGridSize(nx, ny, nz)
    force.setKinematicViscosity(args.viscosity)
    force.setFluidMomentumRemovalFrequency(0)
    system.addForce(force)
    platform, properties = select_platform(args.platform, args.precision)
    context = mm.Context(system, mm.VerletIntegrator(args.dt), platform, properties)
    context.setPositions([mm.Vec3(0, 0, 0)])
    integrator = context.getIntegrator()

    # Node (i, j, k) has index i + nx*(j + ny*k); its y coordinate is j*spacing.
    j = np.arange(nx*ny*nz)//nx % ny
    if args.load:
        saved = np.load(args.load)
        state = saved['state']
        context.setTime(float(saved['time']))
        print('Fluid state read from', args.load)
    else:
        velocityScale = args.dt/args.spacing          # nm/ps -> lattice units
        u = np.zeros((nx*ny*nz, 3))
        u[:, 0] = args.amplitude*velocityScale*np.sin(2*math.pi*j/ny)
        state = equilibrium_deviation(np.ones(nx*ny*nz), u).ravel()
    force.setFluidState(context, state.tolist())

    k = 2*math.pi/L
    rate = args.viscosity*k**2
    dx, dt, tau = force.getLatticeParametersInContext(context)
    print('Platform %s, %d x %d x %d nodes of %g nm, dt %g ps, nu %g nm^2/ps (tau %.4f)' % (
        platform.getName(), nx, ny, nz, args.spacing, args.dt, args.viscosity, tau))
    print('Decay rate nu k^2 = %.5f 1/ps' % rate)
    print(' time (ps)   amplitude (nm/ps)   U exp(-nu k^2 t)')
    sine = np.sin(2*math.pi*j/ny)
    for step in range(0, args.steps + 1, args.interval):
        if step > 0:
            integrator.step(args.interval)
        rho, u = force.getFluidFields(context)
        ux = np.asarray(u.value_in_unit(unit.nanometer/unit.picosecond))[:, 0]
        amplitude = 2*(ux*sine).mean()             # projection on the sine
        t = context.getTime().value_in_unit(unit.picosecond)
        print('%9.2f   %.6e        %.6e' % (t, amplitude, args.amplitude*math.exp(-rate*t)))

    if args.save:
        np.savez(args.save, state=np.array(force.getFluidState(context)),
                 time=context.getTime().value_in_unit(unit.picosecond))
        print('Fluid state written to', args.save)


if __name__ == '__main__':
    main()
