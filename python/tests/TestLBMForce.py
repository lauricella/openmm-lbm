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
