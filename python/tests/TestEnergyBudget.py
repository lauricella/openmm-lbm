# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Kinetic energy budget with the NVE coupling scheme, on the Reference platform (docs/theory.md, section 2).

The kinetic energy of fluid and particles plus the energy dissipated by the viscosity and by the drag stays
constant only approximately: the viscous dissipation computed from Pi_neq is the hydrodynamic limit, valid to
O(Ma^2, Kn^2).  The tests check the size and the scaling of the residual, in lattice units.
"""

import numpy as np
import openmm as mm
from openmmlbm import LBMForce

C = np.array([[0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1],
              [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1]])
W = np.array([1/3] + [1/18]*6 + [1/36]*12)
DX, DT = 0.5, 0.01


def equilibrium_deviation(rho, u):
    cu = C.T @ u
    return W[:, None]*((rho - 1) + rho*(3*cu + 4.5*cu**2 - 1.5*(u**2).sum(0)))


def fluid(force, context):
    """Lattice density, velocity j/rho and Pi_neq of every node."""
    df = np.array(force.getFluidState(context)).reshape(19, -1)
    rho = 1 + df.sum(0)
    u = (C @ df)/rho
    fneq = df - equilibrium_deviation(rho, u)
    return rho, u, np.einsum('aq,bq,qn->abn', C, C, fneq)


def viscous_dissipation(rho, Pi, tau):
    """Energy dissipated in one step, (tau - 1/2)/(2 tau^2 rho cs^2) Pi_neq:Pi_neq summed over the nodes."""
    return ((tau - 0.5)/(2*tau**2*rho/3)*np.einsum('abn,abn->n', Pi, Pi)).sum()


def shear_wave_residual(n, tau, steps):
    """(E + dissipated)/E0 - 1 after a shear wave u_x = U sin(2 pi y/n) has decayed."""
    L = n*DX
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0), mm.Vec3(0, L, 0), mm.Vec3(0, 0, 4*DX))
    system.addParticle(1.0)
    force = LBMForce()
    force.setGridSize(n, n, 4)
    force.setKinematicViscosity((tau - 0.5)/3*DX**2/DT)
    force.setFluidMomentumRemovalFrequency(0)
    system.addForce(force)
    integrator = mm.VerletIntegrator(DT)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions([mm.Vec3(0, 0, 0)])
    j = np.arange(4*n*n)//n % n
    u = np.zeros((3, 4*n*n))
    u[0] = 0.01*np.sin(2*np.pi*j/n)
    force.setFluidState(context, equilibrium_deviation(np.ones(4*n*n), u).ravel().tolist())
    rho, u, Pi = fluid(force, context)
    E0 = 0.5*(rho*(u**2).sum(0)).sum()
    dissipated = 0.0
    for step in range(steps):
        dissipated += viscous_dissipation(rho, Pi, tau)
        integrator.step(1)
        rho, u, Pi = fluid(force, context)
    return (0.5*(rho*(u**2).sum(0)).sum() + dissipated)/E0 - 1


def kick_residual(v0, steps=200, n=12, tau=1.1, gamma=10.0, mass=100.0):
    """(E + dissipated)/E0 - 1 for a bead kicked in a fluid at rest, NVE scheme."""
    L, cellMass, velocityScale = n*DX, 602.214*DX**3, DX/DT
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0), mm.Vec3(0, L, 0), mm.Vec3(0, 0, L))
    system.addParticle(mass)
    force = LBMForce()
    force.setGridSize(n, n, n)
    force.setKinematicViscosity((tau - 0.5)/3*DX**2/DT)
    force.setFriction(gamma)
    force.setCouplingScheme(LBMForce.NVE)
    force.setFluidMomentumRemovalFrequency(0)
    force.addParticle(0)
    system.addForce(force)
    integrator = mm.VerletIntegrator(DT)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions([mm.Vec3(L/2 + 0.1, L/2, L/2)])
    context.setVelocities([mm.Vec3(v0, 0.3*v0, 0)])
    m, g = mass/cellMass, gamma*DT
    rho, u, Pi = fluid(force, context)
    state = context.getState(getPositions=True, getVelocities=True)
    v = np.array(state.getVelocities(asNumpy=True)._value[0])/velocityScale
    E0 = 0.5*m*(v**2).sum()
    dissipated = 0.0
    for step in range(steps):
        s = np.mod(np.array(state.getPositions(asNumpy=True)._value[0])/DX, n)
        i = np.floor(s + 0.5).astype(int) % n
        node = i[0] + n*(i[1] + n*i[2])
        F = -g*m*(v - u[:, node])
        rhoNode, uNode = rho[node], u[:, node].copy()
        dissipated += viscous_dissipation(rho, Pi, tau)
        integrator.step(1)
        state = context.getState(getPositions=True, getVelocities=True)
        vNew = np.array(state.getVelocities(asNumpy=True)._value[0])/velocityScale
        # The bead loses -F.(v + v')/2; the fluid, with Guo's forcing, receives -F at the velocity u - F/(2 rho).
        dissipated -= (F*((v + vNew)/2 - (uNode - F/(2*rhoNode)))).sum()
        v = vNew
        rho, u, Pi = fluid(force, context)
    E = 0.5*(rho*(u**2).sum(0)).sum() + 0.5*m*(v**2).sum()
    return (E + dissipated)/E0 - 1


def test_shear_wave_budget():
    """Fluid only: the budget closes to O(Kn^2), and the residual falls as k^2 when the wavelength doubles."""
    coarse = shear_wave_residual(16, 1.1, 400)
    fine = shear_wave_residual(32, 1.1, 1000)
    assert abs(fine) < 0.01
    assert 3 < coarse/fine < 5


def test_kick_budget():
    """A bead kicked in the fluid: the force on a single node is far from the hydrodynamic limit (Kn ~ 1), and
    the budget closes to about 5%, independently of the Mach number."""
    slow = kick_residual(1.0)
    fast = kick_residual(3.0)
    assert abs(slow) < 0.06
    assert abs(fast - slow) < 1e-3
