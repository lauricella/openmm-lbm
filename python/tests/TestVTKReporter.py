# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""LBMVTKReporter: the VTK files hold the fluid and the particles of the Context, in OpenMM units."""

import os
import struct
import xml.etree.ElementTree as ET

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
from openmmlbm import LBMForce, LBMVTKReporter

TYPES = {'Float32': '<f4', 'Float64': '<f8', 'Int32': '<i4', 'Int64': '<i8', 'UInt8': 'u1'}


def read_vtk(path):
    """The XML element of the dataset and the arrays {Name: array} of a VTK XML file with raw appended data (the points
    of a PolyData under the name 'Points'; the ASCII string array of the field data, 'units', as a list of strings)."""
    data = open(path, 'rb').read()
    marker = b'<AppendedData encoding="raw">'
    head, rest = data.split(marker)
    start = rest.index(b'_') + 1
    root = ET.fromstring(head.decode('ascii') + '</VTKFile>')
    assert root.get('byte_order') == 'LittleEndian' and root.get('header_type') == 'UInt64'
    arrays = {}
    for parent in root.iter():
        for child in parent.findall('DataArray'):
            if child.get('format') == 'ascii' and child.get('type') == 'String':
                codes = [int(c) for c in child.text.split()]
                strings, current = [], ''
                for c in codes:
                    if c == 0:
                        strings.append(current)
                        current = ''
                    else:
                        current += chr(c)
                assert len(strings) == int(child.get('NumberOfTuples'))
                arrays[child.get('Name')] = strings
                continue
            offset = start + int(child.get('offset'))
            size = struct.unpack_from('<Q', rest, offset)[0]
            values = np.frombuffer(rest[offset+8:offset+8+size], dtype=TYPES[child.get('type')])
            name = 'Points' if parent.tag == 'Points' else child.get('Name')
            components = int(child.get('NumberOfComponents', '1'))
            arrays[name] = values.reshape(-1, components) if components > 1 else values
    return root, arrays


def create_simulation(platform_name='Reference'):
    """6x5x4 nodes of 0.5 nm with two solid nodes, a body force, and four particles of which three are coupled."""
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0), mm.Vec3(0, 2.5, 0), mm.Vec3(0, 0, 2))
    masses = [100.0, 120.0, 140.0, 160.0]
    for m in masses:
        system.addParticle(m)
    force = LBMForce()
    force.setGridSize(6, 5, 4)
    force.setFluidDensity(602.2)
    force.setKinematicViscosity(5.0)
    force.setFriction(5.0)
    force.setTemperature(300.0)
    force.setRandomNumberSeed(7)
    force.setBodyAcceleration(mm.Vec3(0.5, 0, 0))
    force.setFluidMomentumRemovalFrequency(0)
    force.setSolidNodes([0, 7])
    for i in (0, 1, 3):
        force.addParticle(i)
    system.addForce(force)
    topology = app.Topology()
    chain = topology.addChain()
    for i in range(len(masses)):
        topology.addAtom('CA', app.element.carbon, topology.addResidue('ALA', chain))
    simulation = app.Simulation(topology, system, mm.VerletIntegrator(0.01), mm.Platform.getPlatformByName(platform_name))
    # one particle outside the box, to check the wrapping
    simulation.context.setPositions([mm.Vec3(1.0, 1.0, 1.0), mm.Vec3(2.2, 0.4, 1.5), mm.Vec3(3.4, 1.2, 0.7),
                                     mm.Vec3(0.6, 2.0, 1.1)])
    simulation.context.setVelocitiesToTemperature(300, 3)
    return simulation, force


@pytest.mark.parametrize('double', [False, True])
def test_vtk_reporter(tmp_path, double):
    simulation, force = create_simulation()
    prefix = str(tmp_path/'run')
    simulation.reporters.append(LBMVTKReporter(prefix, 5, force, double=double))
    simulation.step(10)
    tolerance = 0 if double else 1e-6
    for step in (5, 10):
        assert os.path.exists('%s_fluid_%010d.vti' % (prefix, step))
    # fluid at step 10: the same fields as getFluidFields(), node (i, j, k) at (i dx, j dx, k dx)
    root, arrays = read_vtk('%s_fluid_%010d.vti' % (prefix, 10))
    image = root.find('ImageData')
    assert image.get('WholeExtent') == '0 5 0 4 0 3'
    assert [float(x) for x in image.get('Spacing').split()] == pytest.approx([0.5]*3, rel=1e-15)
    assert [float(x) for x in image.get('Origin').split()] == [0, 0, 0]
    density, velocity = force.getFluidFields(simulation.context)
    density = np.array(density.value_in_unit(unit.dalton/unit.nanometer**3))
    velocity = np.array(velocity.value_in_unit(unit.nanometer/unit.picosecond))
    assert np.allclose(arrays['density'], density, rtol=tolerance, atol=0)
    assert np.allclose(arrays['velocity'], velocity, rtol=tolerance, atol=tolerance*np.abs(velocity).max())
    assert np.abs(velocity).max() > 0
    assert list(np.nonzero(arrays['solid'])[0]) == [0, 7]
    assert arrays['units'] == ['Origin, Spacing: nm', 'density: Da/nm^3', 'velocity: nm/ps',
                               'solid: 1 for a solid node, 0 for a fluid node']
    assert b'OpenMM units' in open('%s_fluid_%010d.vti' % (prefix, 10), 'rb').read(200)
    # particles at step 10: positions wrapped into the box, velocities of the State, masses, indices, coupling
    root, arrays = read_vtk('%s_particles_%010d.vtp' % (prefix, 10))
    state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    assert np.allclose(arrays['Points'], positions, rtol=tolerance, atol=tolerance*3)
    assert np.all(arrays['Points'][:, 0] < 3.0)
    assert np.allclose(arrays['velocity'], state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond),
                       rtol=tolerance, atol=tolerance)
    assert list(arrays['mass']) == [100.0, 120.0, 140.0, 160.0]
    assert list(arrays['index']) == [0, 1, 2, 3]
    assert list(arrays['coupled']) == [1, 1, 0, 1]
    assert list(arrays['connectivity']) == [0, 1, 2, 3] and list(arrays['offsets']) == [1, 2, 3, 4]
    assert arrays['units'][:3] == ['Points: nm', 'velocity: nm/ps, at the half step of the leapfrog', 'mass: Da']
    # the series: two reports, fluid and particles each, at 0.05 and 0.1 ps
    collection = ET.parse(prefix + '.pvd').getroot()
    datasets = [(float(d.get('timestep')), d.get('file')) for d in collection.iter('DataSet')]
    assert datasets == [(pytest.approx(0.05), 'run_fluid_0000000005.vti'), (pytest.approx(0.05), 'run_particles_0000000005.vtp'),
                        (pytest.approx(0.1), 'run_fluid_0000000010.vti'), (pytest.approx(0.1), 'run_particles_0000000010.vtp')]


def test_vtk_reporter_does_not_change_the_run(tmp_path):
    # Writing the files neither advances the fluid nor draws random numbers: the run is the same, bit for bit.
    results = []
    for report in (False, True):
        simulation, force = create_simulation()
        if report:
            simulation.reporters.append(LBMVTKReporter(str(tmp_path/'run'), 3, force))
        simulation.step(9)
        state = simulation.context.getState(getPositions=True, getVelocities=True)
        results.append((state.getPositions(asNumpy=True)._value, np.array(force.getFluidState(simulation.context))))
    assert np.array_equal(results[0][0], results[1][0])
    assert np.array_equal(results[0][1], results[1][1])


def test_vtk_reporter_parts(tmp_path):
    # Only the fluid, or only the particles (not wrapped).
    simulation, force = create_simulation()
    simulation.reporters.append(LBMVTKReporter(str(tmp_path/'fluid'), 2, force, particles=False))
    simulation.reporters.append(LBMVTKReporter(str(tmp_path/'particles'), 2, force, fluid=False, wrap=False))
    simulation.step(2)
    files = sorted(os.listdir(tmp_path))
    assert files == ['fluid.pvd', 'fluid_fluid_0000000002.vti', 'particles.pvd', 'particles_particles_0000000002.vtp']
    root, arrays = read_vtk(str(tmp_path/'particles_particles_0000000002.vtp'))
    positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    assert np.allclose(arrays['Points'], positions, rtol=1e-6, atol=1e-6)
    assert arrays['Points'][2, 0] > 3.0


def test_vtk_reporter_append(tmp_path):
    # A reporter created with append=True keeps the files already listed in <prefix>.pvd (a continued run).
    simulation, force = create_simulation()
    prefix = str(tmp_path/'run')
    simulation.reporters.append(LBMVTKReporter(prefix, 2, force, particles=False))
    simulation.step(4)
    simulation.reporters = [LBMVTKReporter(prefix, 2, force, particles=False, append=True)]
    simulation.step(2)
    files = [d.get('file') for d in ET.parse(prefix + '.pvd').getroot().iter('DataSet')]
    assert files == ['run_fluid_%010d.vti' % step for step in (2, 4, 6)]
