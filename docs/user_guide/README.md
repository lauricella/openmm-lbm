# openmm-lbm user guide

This guide explains how to use openmm-lbm from Python with OpenMM:

- how to set up an `LBMForce` and choose its parameters;
- what each method does;
- how to run, monitor, save and restore a simulation with a lattice Boltzmann fluid.

The model itself is described in [theory.md](../theory.md).

New to OpenMM? Start from the [installation](installation.md), which installs everything step by
step, and continue with the [tutorial](tutorial.md).

## Limitations of the model

Know these before using the plugin for a study:

- **By default the fluid has no thermal fluctuations of its own.** The random force acts on the coupled particles
  only, and the fluid receives its reaction. As a consequence the diffusion coefficient of a free particle,
  or of the centre of mass of a protein, stays close to kT/(m friction), the value without hydrodynamics,
  although the fluid does carry the hydrodynamic interactions (a kick or a drag shows them): the Einstein
  relation with the hydrodynamic mobility is not satisfied. The temperature of the coupled particles is
  below the set temperature: with the explicit drag (the default) 1-2% with friction x dt = 0.1, about 13%
  with a very large friction (100/ps); with the centred drag more, 7% for SOD1 at friction x dt = 0.1
  ([choosing the drag](lattice.md#choosing-the-drag)). See [validation.md](../validation.md). A fluctuating lattice Boltzmann fluid, which removes
  these limits, is available with
  [`setFluidFluctuations(True)`](api_reference.md#setfluidfluctuationsfluctuations-getfluidfluctuations);
  its validation with particles is to come.
- **Nearest-node coupling.** Each particle is coupled to the nearest lattice node, so the forces jump when
  a particle crosses from one cell to the next, and the hydrodynamic radius of a single bead depends on the
  lattice spacing and on tau.
- **Walls.** Solid nodes have a no-slip bounce-back; faces with an imposed density or velocity are not
  available yet.

## Contents

0. [Glossary](glossary.md): the words used in this guide, explained briefly.
1. [Installation](installation.md): conda, OpenMM and the plugin, step by step from nothing, with the
   tests and a first run; computing clusters; installation problems.
2. [Tutorial](tutorial.md): the pieces of an OpenMM simulation, what `LBMForce` adds, and six lessons
   with the scripts of [`examples/`](../../examples/README.md).
3. [Getting started](getting_started.md): checking the installation, a first simulation, units,
   platforms.
4. [The lattice](lattice.md): geometry, node indexing and NumPy arrays, units, relaxation time, Mach
   number, removal of the fluid momentum, initial state.
5. [API reference](api_reference.md): every method of `LBMForce`, with units, defaults and errors.
6. [Saving and continuing a simulation](restart.md): checkpoints of a run with the fluid, a script for
   long runs split into several jobs, moving a run to another platform.
7. [Examples](examples.md): short complete scripts on specific topics (the longer scripts are in
   [`examples/`](../../examples/README.md)).
   - A channel flow between two walls.
   - A particle kicked in the fluid.
   - The temperature of coupled particles.
   - Monitoring a run and the Mach number check.
   - Saving and restoring the fluid.
   - Serialization.
   - `openmm.app.Simulation` with a reporter for the fluid.
8. [Troubleshooting](troubleshooting.md): error messages and common pitfalls.

## What works in this version

| Feature | Reference | CUDA, OpenCL, HIP |
|---|---|---|
| Parameters, checks at Context creation, serialization | yes | yes |
| Reading and writing the fluid: `getFluidFields()`, `getFluidState()`, `setFluidState()`, `getFluidMachNumber()` | yes | yes |
| Fluid update: collision, streaming, body force, removal of the fluid momentum, Mach number check | yes | yes |
| Solid nodes (`setSolidNodes()`) and the force of the fluid on the walls (`getWallForce()`) | yes | yes |
| Particle-fluid coupling: friction and random force at the nearest node, reaction on the fluid, reflection at walls | yes | yes |
| Centred drag (`setDragScheme(LBMForce.Centered)`) | yes | yes |
| Thermal fluctuations of the fluid (`setFluidFluctuations(True)`) | yes | yes |

Every feature runs on every platform. The random forces of the Reference platform, and the fluctuations of the fluid, come from a
generator of the force, those of the GPU platforms from OpenMM's generator: with the same seed the two are
different sequences with the same statistics. The coupling is dissipative: on every platform it adds no energy.

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
