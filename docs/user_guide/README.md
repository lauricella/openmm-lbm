# openmm-lbm user guide

This guide explains how to use openmm-lbm from Python with OpenMM:

- how to set up an `LBMForce` and choose its parameters;
- what each method does;
- how to run, monitor, save and restore a simulation with a lattice Boltzmann fluid.

The model itself is described in [theory.md](../theory.md).

New to OpenMM? Start from the [installation](installation.md), which installs everything step by
step, and continue with the [tutorial](tutorial.md).

## Contents

1. [Installation](installation.md): conda, OpenMM and the plugin, step by step from nothing, with the
   tests and a first run; computing clusters; installation problems.
2. [Tutorial](tutorial.md): the pieces of an OpenMM simulation, what `LBMForce` adds, and five lessons
   with the scripts of [`examples/`](../../examples/README.md).
3. [Getting started](getting_started.md): checking the installation, a first simulation, units,
   platforms.
4. [The lattice](lattice.md): geometry, node indexing and NumPy arrays, units, relaxation time, Mach
   number, removal of the fluid momentum, initial state.
5. [API reference](api_reference.md): every method of `LBMForce`, with units, defaults and errors.
6. [Examples](examples.md): short complete scripts on specific topics (the longer scripts are in
   [`examples/`](../../examples/README.md)).
   - A channel flow between two walls.
   - A particle kicked in the fluid.
   - The temperature of coupled particles.
   - Monitoring a run and the Mach number check.
   - Saving and restoring the fluid.
   - Serialization.
   - `openmm.app.Simulation` with a reporter for the fluid.
7. [Troubleshooting](troubleshooting.md): error messages and common pitfalls.

## What works in this version

| Feature | Reference | CUDA, OpenCL, HIP |
|---|---|---|
| Parameters, checks at Context creation, serialization | yes | yes |
| Reading and writing the fluid: `getFluidFields()`, `getFluidState()`, `setFluidState()`, `getFluidMachNumber()` | yes | yes |
| Fluid update: collision, streaming, body force, removal of the fluid momentum, Mach number check | yes | yes |
| Solid nodes (`setSolidNodes()`) and the force of the fluid on the walls (`getWallForce()`) | yes | yes |
| Particle-fluid coupling: friction and random force at the nearest node, reaction on the fluid, reflection at walls | yes | yes |

Every feature runs on every platform. The random forces of the Reference platform come from a generator of
the force, those of the GPU platforms from OpenMM's generator: with the same seed the two are different
sequences with the same statistics. The coupling is dissipative: on every platform it adds no energy.

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
