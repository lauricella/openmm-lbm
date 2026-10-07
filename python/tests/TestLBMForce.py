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
    assert force.getCouplingScheme() == LBMForce.EulerMaruyama
    force.setCouplingScheme(LBMForce.NVE)
    assert force.getCouplingScheme() == LBMForce.NVE
    assert force.getDragScheme() == LBMForce.Explicit
    force.setDragScheme(LBMForce.Centered)
    assert force.getDragScheme() == LBMForce.Centered


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


def test_centered_first_step():
    # With the centred drag one step multiplies the velocity of a particle in a fluid at rest by
    # (1 - a + a m/m_c)/(1 + a + a m/m_c), with a = friction*dt/2 and m_c the mass of the fluid in a cell.
    system, force, positions = create_system(num_particles=1)
    force.setFriction(50.0/unit.picosecond)
    force.setTemperature(0.0)
    force.setDragScheme(LBMForce.Centered)
    integrator = mm.VerletIntegrator(0.01)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions(positions)
    context.setVelocities([mm.Vec3(0.3, -0.2, 0.1)])
    integrator.step(1)
    v = context.getState(getVelocities=True).getVelocities()[0].value_in_unit(unit.nanometer/unit.picosecond)
    a = 0.5*50.0*0.01
    b = a*100.0/(602.214*0.5**3)
    for found, initial in zip(v, (0.3, -0.2, 0.1)):
        assert found == pytest.approx(initial*(1 - a + b)/(1 + a + b), rel=1e-12)


def test_centered_requires_last_force():
    system, force, positions = create_system()
    force.setDragScheme(LBMForce.Centered)
    system.addForce(mm.CustomExternalForce('0'))
    with pytest.raises(Exception, match='last force'):
        mm.Context(system, mm.VerletIntegrator(0.01), mm.Platform.getPlatformByName('Reference'))


@pytest.mark.parametrize('drag', ['Explicit', 'Centered'])
def test_temperature_reporter(drag, tmp_path):
    # The reporter gives the temperature of the full-step velocity with the explicit drag (the same as
    # StateDataReporter) and that of the State velocity with the centred drag.
    import numpy as np
    from openmm import app
    from openmmlbm import LBMTemperatureReporter
    system, force, positions = create_system(num_particles=20)
    force.setRandomNumberSeed(3)
    force.setDragScheme(getattr(LBMForce, drag))
    simulation = app.Simulation(app.Topology(), system, mm.VerletIntegrator(0.01), mm.Platform.getPlatformByName('Reference'))
    simulation.context.setPositions(positions)
    path = tmp_path/'temperature.txt'
    reporter = LBMTemperatureReporter(str(path), 5, force)
    simulation.reporters.append(reporter)
    simulation.step(20)
    del simulation.reporters[:]
    del reporter
    lines = path.read_text().splitlines()
    assert lines[0].startswith('#')
    step, time, temperature = (float(x) for x in lines[-1].split())
    assert step == 20 and time == pytest.approx(0.2)
    state = simulation.context.getState(getVelocities=True, getEnergy=True)
    kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
    if drag == 'Explicit':
        expected = 2*state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)/(3*20*kB)
    else:
        v = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
        expected = 100.0*np.sum(v**2)/(3*20*kB)
    assert temperature == pytest.approx(expected, abs=1e-3)       # written with three decimals


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


def test_wall_scheme_and_faces():
    # The wall scheme and the open faces, with units, survive serialization; a Couette flow between a face at rest and
    # a moving face becomes linear.
    import numpy as np
    force = LBMForce()
    assert force.getWallScheme() == LBMForce.BounceBack
    force.setWallScheme(LBMForce.Regularized)
    assert force.getWallScheme() == LBMForce.Regularized
    for face in (LBMForce.XMin, LBMForce.XMax, LBMForce.YMin, LBMForce.YMax, LBMForce.ZMin, LBMForce.ZMax):
        assert force.getFaceBoundary(face) == LBMForce.Periodic
        assert force.getFaceDensity(face) == 0*unit.dalton/unit.nanometer**3
    force.setFaceBoundary(LBMForce.ZMin, LBMForce.Velocity)
    force.setFaceBoundary(LBMForce.ZMax, LBMForce.Velocity)
    force.setFaceVelocity(LBMForce.ZMax, mm.Vec3(0.5, 0, 0)*unit.nanometer/unit.picosecond)
    force.setFaceDensity(LBMForce.XMin, 610.0*unit.dalton/unit.nanometer**3)
    assert force.getFaceVelocity(LBMForce.ZMax) == mm.Vec3(0.5, 0, 0)*unit.nanometer/unit.picosecond
    assert force.getFaceDensity(LBMForce.XMin) == 610.0*unit.dalton/unit.nanometer**3
    copy = LBMForce.cast(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(force)))
    assert copy.getWallScheme() == LBMForce.Regularized
    assert copy.getFaceBoundary(LBMForce.ZMax) == LBMForce.Velocity
    assert copy.getFaceVelocity(LBMForce.ZMax) == mm.Vec3(0.5, 0, 0)*unit.nanometer/unit.picosecond
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(1, 0, 0), mm.Vec3(0, 1, 0), mm.Vec3(0, 0, 5))
    system.addParticle(1.0)
    force.setGridSize(2, 2, 10)
    force.setFluidMomentumRemovalFrequency(0)
    force.setKinematicViscosity(5.0)
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.01)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions([mm.Vec3(0.1, 0.1, 0.1)])
    integrator.step(2000)
    density, velocity = force.getFluidFields(context)
    ux = np.array(velocity.value_in_unit(unit.nanometer/unit.picosecond))[:, 0].reshape(10, 4)
    assert np.allclose(ux, 0.5*np.arange(10)[:, None]/9, rtol=0, atol=1e-10)


def test_open_faces_need_no_momentum_removal():
    # With open faces the momentum of the fluid cannot be removed: the default removal frequency (1) is an error.
    system, force, positions = create_system(num_particles=1)
    force.setFaceBoundary(LBMForce.YMin, LBMForce.Density)
    force.setFaceBoundary(LBMForce.YMax, LBMForce.Density)
    with pytest.raises(mm.OpenMMException, match='setFluidMomentumRemovalFrequency'):
        mm.Context(system, mm.VerletIntegrator(0.01), mm.Platform.getPlatformByName('Reference'))


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


def test_fluid_fluctuations():
    # Off by default; switched on, the fluid at rest starts to move, and the setting survives serialization.
    import numpy as np
    system, force, positions = create_system(num_particles=1)
    assert not force.getFluidFluctuations()
    force.setFluidFluctuations(True)
    assert force.getFluidFluctuations()
    copy = LBMForce.cast(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(force)))
    assert copy.getFluidFluctuations()
    integrator = mm.VerletIntegrator(0.01)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions(positions[:1])
    integrator.step(5)
    density, velocity = force.getFluidFields(context)
    assert np.std([v.value_in_unit(unit.nanometer/unit.picosecond) for v in velocity]) > 0


@pytest.mark.parametrize('name', ['CUDA', 'OpenCL', 'HIP'])
def test_fluid_fluctuations_gpu(name):
    # On the GPU platforms a fluctuating fluid at rest starts to move, and the noise conserves mass and momentum.
    import numpy as np
    try:
        platform = mm.Platform.getPlatformByName(name)
    except Exception:
        pytest.skip('the %s platform is not available' % name)
    system, force, positions = create_system(num_particles=1)
    force.setFluidFluctuations(True)
    force.setFluidMomentumRemovalFrequency(0)
    integrator = mm.VerletIntegrator(0.01)
    try:
        context = mm.Context(system, integrator, platform, {'Precision': 'double'})
    except Exception as e:
        pytest.skip('no Context on the %s platform: %s' % (name, e))
    context.setPositions(positions[:1])
    integrator.step(10)
    density, velocity = force.getFluidFields(context)
    rho = np.array([d.value_in_unit(unit.dalton/unit.nanometer**3) for d in density])
    u = np.array([v.value_in_unit(unit.nanometer/unit.picosecond) for v in velocity])
    assert np.std(u) > 0
    assert rho.sum() == pytest.approx(602.214*rho.size, rel=1e-12)


@pytest.mark.parametrize('name,precision,walls', [(name, precision, walls) for name in ('CUDA', 'OpenCL', 'HIP')
                                                  for precision in ('mixed', 'double') for walls in (False, True)],
                         ids=lambda value: {False: 'periodic', True: 'walls'}.get(value, value))
def test_fluid_agrees_with_reference(name, precision, walls):
    # The fluid update of the GPU platforms has the arithmetic of the Reference platform: in double and mixed
    # precision the two agree to rounding (not bitwise, because of fused multiply-adds), here with a perturbed
    # fluid, a body force and the removal of the fluid momentum every third step, without and with solid nodes
    # (the plane j = 0 and a block), whose force on the walls must agree as well.
    import numpy as np
    try:
        platform = mm.Platform.getPlatformByName(name)
    except Exception:
        pytest.skip('the %s platform is not available' % name)
    states, wallForces = [], []
    for plat, properties in ((mm.Platform.getPlatformByName('Reference'), {}), (platform, {'Precision': precision})):
        system = mm.System()
        system.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0), mm.Vec3(0, 2.5, 0), mm.Vec3(0, 0, 2))
        system.addParticle(1.0)
        force = LBMForce()
        force.setGridSize(6, 5, 4)
        force.setKinematicViscosity((0.8 - 0.5)/3*0.25/0.01)
        force.setBodyAcceleration(mm.Vec3(0.5, -0.2, 0.1))
        force.setInitialFluidVelocity(mm.Vec3(0.5, 0.2, -0.3))
        force.setFluidMomentumRemovalFrequency(3)
        if walls:
            force.setSolidNodes([i + 6*5*k for k in range(4) for i in range(6)] +
                                [i + 6*(j + 5*k) for k in (1, 2) for j in (2, 3) for i in (2, 3)])
        system.addForce(force)
        integrator = mm.VerletIntegrator(0.01)
        try:
            context = mm.Context(system, integrator, plat, properties)
        except Exception as e:
            pytest.skip('no Context on the %s platform: %s' % (name, e))
        context.setPositions([mm.Vec3(0.1, 0.7, 0.3)])
        state = np.array(force.getFluidState(context))
        state += 1e-3*np.sin(1.3*np.arange(len(state)))
        force.setFluidState(context, state)
        integrator.step(40)
        states.append(np.array(force.getFluidState(context)))
        wallForces.append(np.array(force.getWallForce(context).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)))
    scale = abs(states[0]).max()
    assert abs(states[1] - states[0]).max() <= 1e-12*scale
    assert abs(wallForces[1] - wallForces[0]).max() <= 1e-10*max(1.0, abs(wallForces[0]).max())


@pytest.mark.parametrize('name,walls,drag', [(name, walls, drag) for name in ('CUDA', 'OpenCL', 'HIP') for walls in (False, True)
                                              for drag in ('Explicit', 'Centered')],
                         ids=lambda value: {False: 'periodic', True: 'walls'}.get(value, value))
def test_coupling_agrees_with_reference(name, walls, drag):
    # At T = 0 the coupling is deterministic: in double precision the particles and the fluid of the GPU platforms
    # follow those of the Reference platform to rounding, also with particles that share a node, cross the
    # periodic boundary or are reflected by a wall (the plane j = 0 of the 8^3 grid), with both drag schemes.
    # With the centred drag the other forces enter the drag: a constant field and a soft pair force act on the
    # particles, added to the System before the LBMForce.
    import numpy as np
    try:
        platform = mm.Platform.getPlatformByName(name)
    except Exception:
        pytest.skip('the %s platform is not available' % name)
    results = []
    for plat, properties in ((mm.Platform.getPlatformByName('Reference'), {}), (platform, {'Precision': 'double'})):
        system = mm.System()
        system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
        field = mm.CustomExternalForce('-3*x+2*y-z')
        pair = mm.CustomNonbondedForce('50*(1-r)^2')
        pair.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        pair.setCutoffDistance(1.0)
        force = LBMForce()
        force.setGridSize(8, 8, 8)
        for i in range(4):
            system.addParticle(100.0)
            field.addParticle(i)
            pair.addParticle()
            force.addParticle(i)
        system.addForce(field)
        system.addForce(pair)
        system.addForce(force)
        force.setDragScheme(getattr(LBMForce, drag))
        force.setFriction(10.0)
        force.setTemperature(0.0)
        force.setFluidMomentumRemovalFrequency(0)
        if walls:
            force.setSolidNodes([i + 64*k for k in range(8) for i in range(8)])
        integrator = mm.VerletIntegrator(0.01)
        try:
            context = mm.Context(system, integrator, plat, properties)
        except Exception as e:
            pytest.skip('no Context on the %s platform: %s' % (name, e))
        context.setPositions([mm.Vec3(1.02, 2.01, 0.98), mm.Vec3(0.97, 1.99, 1.03), mm.Vec3(3.98, 0.52, 3.96),
                              mm.Vec3(2.3, 0.6, 0.6)])
        context.setVelocities([mm.Vec3(0.5, -0.2, 0.3), mm.Vec3(-0.4, 0.1, 0.2), mm.Vec3(2.0, -1.5, 1.0),
                               mm.Vec3(0.0, -2.0, -0.6)])
        integrator.step(60)
        state = context.getState(getPositions=True, getVelocities=True)
        results.append((state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                        state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond),
                        np.array(force.getFluidState(context)),
                        np.array(force.getWallForce(context).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))))
    for a, b in zip(results[0], results[1]):
        assert abs(b - a).max() <= 1e-10*max(1.0, abs(a).max())


@pytest.mark.parametrize('name', ['Reference', 'CUDA', 'OpenCL', 'HIP'])
def test_centered_drag_sees_all_forces(name):
    # The centred drag of a step uses the other forces on the particles at the time of the force, Fc.  With one
    # particle per node, in a fluid at rest and at T = 0, the velocity after the first step is
    #   v1 = v0 + dt (Fc + F)/m,   F = -gamma m v~/(1 + a + a m/m_c),   v~ = v0 + dt Fc/(2m),   a = gamma dt/2.
    # Fc is read from the State with the force group of the other forces: a constant field and a NonbondedForce
    # with PME, whose reciprocal part the CUDA platform computes on a separate stream.  This checks that the drag
    # sees all of them, and that it respects force groups.
    import numpy as np
    try:
        platform = mm.Platform.getPlatformByName(name)
    except Exception:
        pytest.skip('the %s platform is not available' % name)
    dt, friction, mass, density, dx = 0.01, 20.0, 100.0, 602.2, 0.5
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    field = mm.CustomExternalForce('-30*x+20*y-10*z')
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(1.0)
    force = LBMForce()
    force.setGridSize(8, 8, 8)
    force.setFluidDensity(density)
    force.setFriction(friction)
    force.setTemperature(0.0)
    force.setFluidMomentumRemovalFrequency(0)
    force.setDragScheme(LBMForce.Centered)
    charges = [1.0, -1.0, 0.5, -0.5]
    for i, q in enumerate(charges):
        system.addParticle(mass)
        field.addParticle(i)
        nonbonded.addParticle(q, 0.3, 0.5)
        force.addParticle(i)
    force.setForceGroup(1)
    system.addForce(field)
    system.addForce(nonbonded)
    system.addForce(force)
    integrator = mm.VerletIntegrator(dt)
    properties = {} if name == 'Reference' else {'Precision': 'double'}
    try:
        context = mm.Context(system, integrator, platform, properties)
    except Exception as e:
        pytest.skip('no Context on the %s platform: %s' % (name, e))
    context.setPositions([mm.Vec3(0.6, 0.7, 0.4), mm.Vec3(1.9, 1.1, 0.8), mm.Vec3(2.6, 3.1, 2.4), mm.Vec3(1.1, 2.4, 3.3)])
    v0 = np.array([[0.5, -0.2, 0.3], [-0.4, 0.1, 0.2], [0.2, -0.5, -0.1], [0.0, 0.3, -0.6]])
    context.setVelocities([mm.Vec3(*v) for v in v0])
    fc = context.getState(getForces=True, groups={0}).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole/unit.nanometer)
    assert abs(fc).max() > 10.0
    integrator.step(1)
    v1 = context.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
    a = 0.5*friction*dt
    b = a*mass/(density*dx**3)
    known = v0 + 0.5*dt*fc/mass
    drag = -friction*mass*known/(1 + a + b)
    expected = v0 + dt*(fc + drag)/mass
    assert abs(v1 - expected).max() <= 1e-11*abs(expected).max()


def test_requires_verlet():
    system, force, positions = create_system()
    integrator = mm.LangevinMiddleIntegrator(300, 1, 0.01)
    with pytest.raises(Exception):
        mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
