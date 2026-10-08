# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""An uncaught exception on one MPI rank must stop every rank (docs/theory.md, section 8): the openmmlbm module prints
it and calls MPI_Abort, instead of leaving the other ranks waiting in a communication.  It needs the plugin built with
MPI (-DOPENMM_LBM_MPI=ON) and runs under MPI, so pytest does not collect it:

    mpirun -n 2 python mpi_abort.py           # stops at once, with a non-zero exit code
    mpirun -n 2 python mpi_abort.py nohook    # the default excepthook of Python: the job hangs

Rank 1 raises an exception while rank 0 waits for it in the collective check made when the Context is created."""
import sys
import openmm as mm
import openmmlbm
from openmmlbm import LBMForce

if len(sys.argv) > 1 and sys.argv[1] == 'nohook':
    sys.excepthook = sys.__excepthook__
if openmmlbm.mpiRank() == 1:
    raise RuntimeError('error on rank 1 only')
system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(2, 0, 0), mm.Vec3(0, 2, 0), mm.Vec3(0, 0, 2))
system.addParticle(1.0)
force = LBMForce()
force.setGridSize(4, 4, 4)
force.setDomainDecomposition(2, 1, 1)
system.addForce(force)
mm.Context(system, mm.VerletIntegrator(0.01), mm.Platform.getPlatformByName('Reference'))
print('rank 0: the Context was created, which should not happen')
