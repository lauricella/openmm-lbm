# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Tests of the Python API of openmm-lbm, on the Reference platform."""

import openmm as mm
import openmm.unit as unit
import pytest
from openmmlbm import LBMForce


def create_system(num_particles=10, box=4.0):
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
    force = LBMForce()
    force.setGridSize(8, 8, 8)
    for i in range(num_particles):
        system.addParticle(100.0)
        force.addParticle(i)
    system.addForce(force)
    positions = [mm.Vec3(0.37*i, 0.53*i + 0.1, 0.71*i + 0.2) for i in range(num_particles)]
    return system, force, positions


def test_parameters_with_units():
    force = LBMForce()
    force.setGridSize(10, 12, 14)
    assert force.getGridSize() == [10, 12, 14]
    force.setFluidDensity(500.0*unit.dalton/unit.nanometer**3)
    force.setKinematicViscosity(2.0*unit.nanometer**2/unit.picosecond)
    force.setFriction(5.0/unit.picosecond)
    force.setTemperature(310*unit.kelvin)
    force.setInitialFluidVelocity(mm.Vec3(0.1, 0, 0)*unit.nanometer/unit.picosecond)
    assert force.getFluidDensity() == 500.0*unit.dalton/unit.nanometer**3
    assert force.getKinematicViscosity() == 2.0*unit.nanometer**2/unit.picosecond
    assert force.getFriction() == 5.0/unit.picosecond
    assert force.getTemperature() == 310*unit.kelvin
    assert force.getInitialFluidVelocity()[0] == 0.1*unit.nanometer/unit.picosecond


def test_fluid_fields_and_state():
    system, force, positions = create_system()
    force.setInitialFluidVelocity(mm.Vec3(0.1, -0.05, 0.02))
    integrator = mm.VerletIntegrator(0.01)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions(positions)
    density, velocity = force.getFluidFields(context)
    assert len(density) == 8*8*8
    assert density[0].value_in_unit(unit.dalton/unit.nanometer**3) == pytest.approx(602.214)
    assert velocity[7][1].value_in_unit(unit.nanometer/unit.picosecond) == pytest.approx(-0.05)
    state = force.getFluidState(context)
    assert len(state) == 19*8*8*8
    force.setFluidState(context, state)
    forces = context.getState(getForces=True).getForces(asNumpy=True)
    assert abs(forces).max().value_in_unit(unit.kilojoule_per_mole/unit.nanometer) == 0.0


def test_mach_number_and_lattice_parameters():
    system, force, positions = create_system()
    # box 4 nm on an 8^3 grid: dx = 0.5 nm; with dt = 0.01 ps a lattice speed s is s*50 nm/ps
    force.setInitialFluidVelocity(mm.Vec3(0.1*50, 0, 0))
    force.setMachCheckFrequency(0)
    assert force.getMachCheckFrequency() == 0
    force.setMachNumberLimit(0.2)
    assert force.getMachNumberLimit() == 0.2
    integrator = mm.VerletIntegrator(0.01)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions(positions)
    assert force.getFluidMachNumber(context) == pytest.approx(0.1*3**0.5)
    dx, dt, tau = force.getLatticeParametersInContext(context)
    assert dx == 0.5*unit.nanometer
    assert dt == 0.01*unit.picosecond
    assert tau == pytest.approx(3*1.0035*0.01/0.25 + 0.5)


def test_solid_nodes():
    import numpy as np
    system, force, positions = create_system()
    plane = np.arange(8*8)                      # nodes of the plane k = 0 of the 8^3 grid
    force.setSolidNodes(plane)
    assert force.getSolidNodes() == list(range(64))
    integrator = mm.VerletIntegrator(0.01)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions(positions)
    density, velocity = force.getFluidFields(context)
    assert density[0].value_in_unit(unit.dalton/unit.nanometer**3) == 0.0
    assert density[64].value_in_unit(unit.dalton/unit.nanometer**3) == pytest.approx(602.214)


def test_coupling_first_step():
    # In one step the drag multiplies the velocity of a particle in a fluid at rest by 1 - friction*dt.
    system, force, positions = create_system(num_particles=1)
    force.setFriction(5.0/unit.picosecond)
    force.setTemperature(0.0)
    integrator = mm.VerletIntegrator(0.01)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions(positions)
    context.setVelocities([mm.Vec3(0.3, -0.2, 0.1)])
    integrator.step(1)
    v = context.getState(getVelocities=True).getVelocities()[0].value_in_unit(unit.nanometer/unit.picosecond)
    for found, initial in zip(v, (0.3, -0.2, 0.1)):
        assert found == pytest.approx(initial*(1 - 5.0*0.01), rel=1e-12)


def test_wall_force():
    # A fluid moving towards the solid plane k = 0 pushes it along -z.
    import numpy as np
    system, force, positions = create_system(num_particles=1)
    force.setSolidNodes(np.arange(64))
    force.setInitialFluidVelocity(mm.Vec3(0, 0, -0.5))
    force.setFluidMomentumRemovalFrequency(0)      # the default removal would stop the fluid
    integrator = mm.VerletIntegrator(0.01)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions([mm.Vec3(1.1, 1.3, 1.7)])
    assert force.getWallForce(context)[2].value_in_unit(unit.kilojoule_per_mole/unit.nanometer) == 0.0
    integrator.step(1)
    assert force.getWallForce(context)[2].value_in_unit(unit.kilojoule_per_mole/unit.nanometer) < 0.0


def test_serialization():
    system, force, positions = create_system()
    force.setFriction(7.0)
    xml = mm.XmlSerializer.serialize(system)
    system2 = mm.XmlSerializer.deserialize(xml)
    force2 = [f for f in system2.getForces() if LBMForce.isinstance(f)]
    assert len(force2) == 1
    force2 = LBMForce.cast(force2[0])
    assert force2.getFriction() == 7.0/unit.picosecond
    assert force2.getNumParticles() == 10


def test_requires_verlet():
    system, force, positions = create_system()
    integrator = mm.LangevinMiddleIntegrator(300, 1, 0.01)
    with pytest.raises(Exception):
        mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
