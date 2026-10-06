# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Checkpoints from Python: LBMForce.createCheckpoint(), openmmlbm.saveCheckpoint(), loadCheckpoint() and
LBMCheckpointReporter, on the Reference platform."""

import io

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
import openmmlbm
from openmmlbm import LBMForce


def create_simulation(seed=5):
    """Eight beads coupled to the fluid at 300 K, with a wall and the momentum removal every 3 steps."""
    n, L = 8, 4.0
    topology = app.Topology()
    chain = topology.addChain()
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0), mm.Vec3(0, L, 0), mm.Vec3(0, 0, L))
    force = LBMForce()
    force.setGridSize(n, n, n)
    force.setFriction(10.0)
    force.setTemperature(300.0)
    force.setRandomNumberSeed(seed)
    force.setFluidMomentumRemovalFrequency(3)
    force.setSolidNodes([i + n*n*k for k in range(n) for i in range(n)])
    for i in range(8):
        topology.addAtom('B', None, topology.addResidue('BEA', chain))
        system.addParticle(100.0)
        force.addParticle(i)
    system.addForce(force)
    simulation = app.Simulation(topology, system, mm.VerletIntegrator(0.01), mm.Platform.getPlatformByName('Reference'))
    return simulation, force


def test_checkpoint_bytes():
    simulation, force = create_simulation()
    simulation.context.setPositions(np.random.default_rng(1).uniform(0.6, 3.4, (8, 3)))
    simulation.step(3)
    data = force.createCheckpoint(simulation.context)
    assert isinstance(data, bytes) and data.startswith(b'LBMCKPT1')
    force.loadCheckpoint(simulation.context, data)
    with pytest.raises(Exception):
        force.loadCheckpoint(simulation.context, b'not a checkpoint')


def test_restart_is_exact(tmp_path):
    """A Simulation restarted from the file of LBMCheckpointReporter continues exactly like the original one,
    also when a StateDataReporter at the same step has drawn the random numbers of the next step."""
    file = str(tmp_path/'run.chk')
    simulation, force = create_simulation()
    simulation.context.setPositions(np.random.default_rng(1).uniform(0.6, 3.4, (8, 3)))
    simulation.context.setVelocities(np.random.default_rng(2).normal(0, 0.3, (8, 3)))
    simulation.reporters.append(app.StateDataReporter(io.StringIO(), 5, step=True, temperature=True))
    simulation.reporters.append(openmmlbm.LBMCheckpointReporter(file, 20, force))
    simulation.step(20)
    simulation.step(15)
    positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
    fluid = np.array(force.getFluidState(simulation.context))

    restarted, restartedForce = create_simulation(seed=99)       # the seed is restored by the checkpoint
    openmmlbm.loadCheckpoint(file, restarted.context, restartedForce)
    assert restarted.currentStep == 20
    restarted.reporters.append(app.StateDataReporter(io.StringIO(), 5, step=True, temperature=True))
    restarted.step(15)
    assert np.array_equal(positions, restarted.context.getState(getPositions=True).getPositions(asNumpy=True)._value)
    assert np.array_equal(fluid, np.array(restartedForce.getFluidState(restarted.context)))


def test_save_and_load_functions(tmp_path):
    file = str(tmp_path/'state.chk')
    simulation, force = create_simulation()
    simulation.context.setPositions(np.random.default_rng(1).uniform(0.6, 3.4, (8, 3)))
    simulation.step(4)
    openmmlbm.saveCheckpoint(file, simulation.context, force)
    wall = force.getWallForce(simulation.context)
    other, otherForce = create_simulation()
    openmmlbm.loadCheckpoint(file, other.context, otherForce)
    assert otherForce.getWallForce(other.context) == wall
    assert other.context.getTime() == 4*0.01*unit.picosecond
    with open(file, 'wb') as out:
        out.write(b'garbage')
    with pytest.raises(ValueError):
        openmmlbm.loadCheckpoint(file, other.context, otherForce)
