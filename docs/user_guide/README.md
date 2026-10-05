# openmm-lbm user guide

This guide explains how to use openmm-lbm from Python with OpenMM:

- how to set up an `LBMForce` and choose its parameters;
- what each method does;
- how to run, monitor, save and restore a simulation with a lattice Boltzmann fluid.

The model itself is described in [theory.md](../theory.md).

## Contents

1. [Getting started](getting_started.md): checking the installation, a first simulation, units,
   platforms.
2. [The lattice](lattice.md): geometry, node indexing and NumPy arrays, units, relaxation time, Mach
   number, removal of the fluid momentum, initial state.
3. [API reference](api_reference.md): every method of `LBMForce`, with units, defaults and errors.
4. [Examples](examples.md): complete scripts.
   - A channel flow between two walls.
   - Monitoring a run and the Mach number check.
   - Saving and restoring the fluid.
   - Serialization.
   - `openmm.app.Simulation` with a reporter for the fluid.
5. [Troubleshooting](troubleshooting.md): error messages and common pitfalls.

## What works in this version

| Feature | Reference | CUDA, OpenCL, HIP |
|---|---|---|
| Parameters, checks at Context creation, serialization | yes | yes |
| Reading and writing the fluid: `getFluidFields()`, `getFluidState()`, `setFluidState()`, `getFluidMachNumber()` | yes | yes |
| Fluid update: collision, streaming, body force, removal of the fluid momentum, Mach number check | yes | not yet: the fluid keeps its initial state |
| Solid nodes (`setSolidNodes()`) | yes | not yet: Context creation fails |
| Particle-fluid coupling (friction and noise) | not yet | not yet |

Until the coupling is implemented, `LBMForce` applies no force to the particles and adds no energy.
The methods for the coupling already exist: `addParticle()`, `setFriction()`, `setTemperature()` and
`setRandomNumberSeed()`. Their values are checked and stored with the force.

## Conventions

The examples import

```python
import numpy as np
import openmm as mm
import openmm.unit as unit
from openmmlbm import LBMForce
```

and run on the Reference platform. Numbers without units are in OpenMM units: nm, ps, Da (g/mol), K
and kJ/mol.
