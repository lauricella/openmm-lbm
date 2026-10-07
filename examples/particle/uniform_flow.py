# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""A bead at rest dragged by a uniform flow, with the NVE coupling scheme (zero temperature).

The fluid starts with the uniform velocity u0 along x and a bead of one lattice cell at rest.  The drag
accelerates the bead and slows the fluid down; with the removal of the fluid momentum off, the total
momentum of bead and fluid is conserved exactly, and both end up at the common velocity
u0 M/(M + m).  The kinetic energy is not conserved: the drag dissipates about gamma m |v - u|^2 per unit
time, and the viscosity damps the flow that the bead disturbs (docs/theory.md, section 2).

The default parameters are those of the old plugin's NVE test: 20^3 nodes of 0.3 nm, time step 1 fs,
tau = 1, friction 10/ps, u0 = 0.09 nm/ps.  The script prints, every --interval steps, the velocity of
the bead, the momentum of bead and fluid relative to the initial one, and the kinetic energies.  The
momentum and energy of the fluid are sums over the nodes of the density and velocity given by
getFluidFields().
"""

import argparse

import numpy as np
import openmm as mm
import openmm.unit as unit
from openmmlbm import LBMForce


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nodes', type=int, default=20, help='lattice nodes per side (default 20)')
    parser.add_argument('--spacing', type=float, default=0.3, help='lattice spacing in nm (default 0.3)')
    parser.add_argument('--dt', type=float, default=0.001, help='time step in ps (default 0.001)')
    parser.add_argument('--tau', type=float, default=1.0, help='relaxation time, sets the viscosity (default 1)')
    parser.add_argument('--density', type=float, default=993.0, help='fluid density in kg/m^3 (default 993)')
    parser.add_argument('--cells', type=float, default=1.0, help='bead mass in lattice cells (default 1)')
    parser.add_argument('--friction', type=float, default=10.0, help='friction in 1/ps (default 10)')
    parser.add_argument('--flow', type=float, default=0.09, help='initial fluid velocity along x in nm/ps (default 0.09)')
    parser.add_argument('--steps', type=int, default=1000, help='number of steps (default 1000)')
    parser.add_argument('--interval', type=int, default=50, help='steps between lines of output (default 50)')
    parser.add_argument('--drag', choices=['Explicit', 'Centered'], default='Explicit',
                        help='drag scheme of LBMForce (default Explicit)')
    parser.add_argument('--platform', help='OpenMM platform (default: CUDA, then OpenCL, then Reference)')
    parser.add_argument('--precision', default='mixed', help='precision on CUDA and OpenCL (default mixed)')
    return parser.parse_args()


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


def main():
    args = parse_arguments()
    density = (args.density*unit.kilogram/unit.meter**3*unit.AVOGADRO_CONSTANT_NA).value_in_unit(
        unit.dalton/unit.nanometer**3)
    mass = args.cells*density*args.spacing**3
    box = args.nodes*args.spacing
    cellVolume = args.spacing**3

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
    system.addParticle(mass)
    force = LBMForce()
    force.setGridSize(args.nodes, args.nodes, args.nodes)
    force.setFluidDensity(density)
    force.setKinematicViscosity((args.tau - 0.5)/3*args.spacing**2/args.dt)
    force.setFriction(args.friction)
    force.setCouplingScheme(LBMForce.NVE)
    force.setInitialFluidVelocity(mm.Vec3(args.flow, 0, 0))
    force.setFluidMomentumRemovalFrequency(0)
    force.setDragScheme(getattr(LBMForce, args.drag))
    force.addParticle(0)
    system.addForce(force)
    integrator = mm.VerletIntegrator(args.dt)
    platform, properties = select_platform(args.platform, args.precision)
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions([mm.Vec3(0, 0, 0)])
    context.setVelocities([mm.Vec3(0, 0, 0)])

    fluidMass = density*box**3
    print('Platform %s, box %.2f nm, %d^3 nodes, dt %g ps, tau %g, friction %g 1/ps (gamma dt %g)' % (
        platform.getName(), box, args.nodes, args.dt, args.tau, args.friction, args.friction*args.dt))
    print('Bead %.4f Da at rest, fluid %.1f Da at u0 = %g nm/ps: final common velocity %.6f nm/ps' % (
        mass, fluidMass, args.flow, args.flow*fluidMass/(fluidMass + mass)))
    p0 = fluidMass*args.flow
    e0 = 0.5*fluidMass*args.flow**2
    print('  step   time (ps)  v_bead/u0   p_bead/p0     p_total/p0 - 1   E_bead/E0     E_fluid/E0    E_total/E0')
    for step in range(0, args.steps + 1, args.interval):
        if step > 0:
            integrator.step(args.interval)
        v = context.getState(getVelocities=True).getVelocities()[0][0].value_in_unit(unit.nanometer/unit.picosecond)
        rho, u = force.getFluidFields(context)
        rho = np.asarray(rho.value_in_unit(unit.dalton/unit.nanometer**3))
        u = np.asarray(u.value_in_unit(unit.nanometer/unit.picosecond))
        fluidMomentum = (rho*u[:, 0]).sum()*cellVolume
        fluidEnergy = 0.5*(rho*(u**2).sum(axis=1)).sum()*cellVolume
        beadEnergy = 0.5*mass*v**2
        print('%6d   %8.3f   %.6f    %.4e    % .3e      %.4e    %.6f      %.6f' % (
            step, step*args.dt, v/args.flow, mass*v/p0, (mass*v + fluidMomentum)/p0 - 1, beadEnergy/e0,
            fluidEnergy/e0, (beadEnergy + fluidEnergy)/e0))


if __name__ == '__main__':
    main()
