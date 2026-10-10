/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

%module openmmlbm

/*
 * The OpenMM typemaps accept NumPy arrays and call isNumpyAvailable(), which the OpenMM Python
 * wrappers define in their own header.i; a plugin has to provide it.
 */
%{
#include <numpy/arrayobject.h>
#include "openmm/Vec3.h"

// The OpenMM typemaps generate code that refers to the unqualified type Vec3.
using OpenMM::Vec3;

int isNumpyAvailable() {
    static bool initialized = false;
    static bool available = false;
    if (!initialized) {
        initialized = true;
        available = (_import_array() >= 0);
    }
    return available;
}
%}

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%fragment("Vec3_to_PyVec3");
%include <std_string.i>
%include <std_vector.i>

%{
#include "LBMForce.h"
#include "internal/LBMDecomposition.h"
#include <sstream>
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import openmm as mm
import openmm.unit as unit


def _lbmInputValue(value, expected, method):
    # A plain number is taken in the unit of the method; a Quantity is converted to that unit.  A Quantity whose
    # unit cannot be converted to it is an error: OpenMM's typemaps would otherwise strip it in the MD unit system
    # without a check, and a density in g/cm^3, for example, would become 1e-21 Da/nm^3.
    if not unit.is_quantity(value):
        return value
    if not value.unit.is_compatible(expected):
        hint = ''
        if value.unit.is_compatible(unit.gram/unit.centimeter**3):
            hint = '; a density in g/cm^3 must be multiplied by unit.AVOGADRO_CONSTANT_NA'
        raise TypeError('LBMForce.%s(): the unit %s cannot be converted to %s%s' % (method, value.unit, expected, hint))
    return value.value_in_unit(expected)
%}

/*
 * The OpenMM typemaps are written for the unqualified type Vec3; extend them to OpenMM::Vec3.
 */
%apply const Vec3& { const OpenMM::Vec3& };
%apply Vec3 { OpenMM::Vec3 };

/*
 * Output arguments of type std::vector<double>&, returned as Python lists.
 */
%typemap(in, numinputs=0) std::vector<double>& OUTPUT (std::vector<double> temp) {
    $1 = &temp;
}
%typemap(argout) std::vector<double>& OUTPUT {
    PyObject* list = PyList_New($1->size());
    for (int i = 0; i < (int) $1->size(); i++)
        PyList_SET_ITEM(list, i, PyFloat_FromDouble((*$1)[i]));
    %append_output(list);
}

/*
 * Lists of node indices: input from any Python sequence of integers (a list or a NumPy array), output as a
 * Python list.
 */
%typemap(in) const std::vector<int>& nodes (std::vector<int> temp) {
    PyObject* sequence = PySequence_Fast($input, "expected a sequence of integers");
    if (sequence == NULL)
        SWIG_fail;
    Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);
    temp.resize(size);
    for (Py_ssize_t i = 0; i < size; i++) {
        long value = PyLong_AsLong(PySequence_Fast_GET_ITEM(sequence, i));
        if (value == -1 && PyErr_Occurred()) {
            Py_DECREF(sequence);
            SWIG_fail;
        }
        temp[i] = (int) value;
    }
    Py_DECREF(sequence);
    $1 = &temp;
}
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) const std::vector<int>& nodes {
    $1 = PySequence_Check($input) ? 1 : 0;
}
%typemap(in, numinputs=0) std::vector<int>& OUTPUT (std::vector<int> temp) {
    $1 = &temp;
}
%typemap(argout) std::vector<int>& OUTPUT {
    PyObject* list = PyList_New($1->size());
    for (int i = 0; i < (int) $1->size(); i++)
        PyList_SET_ITEM(list, i, PyLong_FromLong((*$1)[i]));
    %append_output(list);
}

/*
 * Add units to function outputs.
 */
%pythonappend LBMPlugin::LBMForce::getFluidDensity() const %{
    val = unit.Quantity(val, unit.dalton/unit.nanometer**3)
%}
%pythonappend LBMPlugin::LBMForce::getKinematicViscosity() const %{
    val = unit.Quantity(val, unit.nanometer**2/unit.picosecond)
%}
%pythonappend LBMPlugin::LBMForce::getFriction() const %{
    val = unit.Quantity(val, 1/unit.picosecond)
%}
%pythonappend LBMPlugin::LBMForce::getTemperature() const %{
    val = unit.Quantity(val, unit.kelvin)
%}
%pythonappend LBMPlugin::LBMForce::getBodyAcceleration() const %{
    val = unit.Quantity(val, unit.nanometer/unit.picosecond**2)
%}
%pythonappend LBMPlugin::LBMForce::getInitialFluidVelocity() const %{
    val = unit.Quantity(val, unit.nanometer/unit.picosecond)
%}
%pythonappend LBMPlugin::LBMForce::getFaceVelocity(Face face) const %{
    val = unit.Quantity(val, unit.nanometer/unit.picosecond)
%}
%pythonappend LBMPlugin::LBMForce::getFaceDensity(Face face) const %{
    val = unit.Quantity(val, unit.dalton/unit.nanometer**3)
%}
%pythonappend LBMPlugin::LBMForce::getLatticeParametersInContext(const OpenMM::Context& context, double& dx, double& dt, double& tau) const %{
    val = (unit.Quantity(val[0], unit.nanometer), unit.Quantity(val[1], unit.picosecond), val[2])
%}
%pythonappend LBMPlugin::LBMForce::getWallForce(OpenMM::Context& context) const %{
    val = unit.Quantity(val, unit.kilojoule_per_mole/unit.nanometer)
%}
%pythonappend LBMPlugin::LBMForce::getLocalDomain(const OpenMM::Context& context, int& i0, int& j0, int& k0, int& ni, int& nj, int& nk) const %{
    val = (tuple(val[:3]), tuple(val[3:]))
%}

/*
 * getFluidFields(), getFluidState() and setFluidState() take the optional arguments gather, halo and scatter: the C++
 * methods are wrapped as _getFluidFields() and the like, with explicit arguments, and the Python methods with the
 * defaults are added to LBMForce in the Python code at the end of this file.
 */
%rename(_getFluidState) LBMPlugin::LBMForce::getFluidState;
%rename(_setFluidState) LBMPlugin::LBMForce::setFluidState;

/*
 * Check the units of the inputs (_lbmInputValue above).
 */
%pythonprepend LBMPlugin::LBMForce::setFluidDensity(double density) %{
    density = _lbmInputValue(density, unit.dalton/unit.nanometer**3, 'setFluidDensity')
%}
%pythonprepend LBMPlugin::LBMForce::setKinematicViscosity(double viscosity) %{
    viscosity = _lbmInputValue(viscosity, unit.nanometer**2/unit.picosecond, 'setKinematicViscosity')
%}
%pythonprepend LBMPlugin::LBMForce::setFriction(double friction) %{
    friction = _lbmInputValue(friction, unit.picosecond**-1, 'setFriction')
%}
%pythonprepend LBMPlugin::LBMForce::setTemperature(double temperature) %{
    temperature = _lbmInputValue(temperature, unit.kelvin, 'setTemperature')
%}
%pythonprepend LBMPlugin::LBMForce::setBodyAcceleration(const OpenMM::Vec3& acceleration) %{
    acceleration = _lbmInputValue(acceleration, unit.nanometer/unit.picosecond**2, 'setBodyAcceleration')
%}
%pythonprepend LBMPlugin::LBMForce::setInitialFluidVelocity(const OpenMM::Vec3& velocity) %{
    velocity = _lbmInputValue(velocity, unit.nanometer/unit.picosecond, 'setInitialFluidVelocity')
%}
%pythonprepend LBMPlugin::LBMForce::setFaceVelocity(Face face, const OpenMM::Vec3& velocity) %{
    velocity = _lbmInputValue(velocity, unit.nanometer/unit.picosecond, 'setFaceVelocity')
%}
%pythonprepend LBMPlugin::LBMForce::setFaceDensity(Face face, double density) %{
    density = _lbmInputValue(density, unit.dalton/unit.nanometer**3, 'setFaceDensity')
%}

/*
 * Convert C++ exceptions to Python exceptions.
 */
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

namespace LBMPlugin {

class LBMForce : public OpenMM::Force {
public:
    enum CouplingScheme {
        EulerMaruyama = 0,
        NVE = 1
    };
    enum DragScheme {
        Explicit = 0,
        Centered = 1
    };
    enum InterpolationStencil {
        NearestNode = 0,
        Trilinear = 1,
        ThreePoint = 2,
        Keys = 3
    };
    enum WallScheme {
        BounceBack = 0,
        Regularized = 1
    };
    enum Face {
        XMin = 0,
        XMax = 1,
        YMin = 2,
        YMax = 3,
        ZMin = 4,
        ZMax = 5
    };
    enum BoundaryType {
        Periodic = 0,
        Velocity = 1,
        Density = 2,
        DensityVelocity = 3
    };
    LBMForce();

    %apply int& OUTPUT {int& nx};
    %apply int& OUTPUT {int& ny};
    %apply int& OUTPUT {int& nz};
    void getGridSize(int& nx, int& ny, int& nz) const;
    %clear int& nx;
    %clear int& ny;
    %clear int& nz;
    void setGridSize(int nx, int ny, int nz);

    %apply int& OUTPUT {int& px};
    %apply int& OUTPUT {int& py};
    %apply int& OUTPUT {int& pz};
    void getDomainDecomposition(int& px, int& py, int& pz) const;
    %clear int& px;
    %clear int& py;
    %clear int& pz;
    void setDomainDecomposition(int px, int py, int pz);
    static bool isMPIAvailable();
    static int getMPIRank();
    static int getMPISize();
    static int getMPILocalRank();
    static void abortMPI(int errorCode);
    bool getParticleCopiesCheck() const;
    void setParticleCopiesCheck(bool check);
    bool getDensityHaloExchange() const;
    void setDensityHaloExchange(bool exchange);
    bool getVelocityHaloExchange() const;
    void setVelocityHaloExchange(bool exchange);

    double getFluidDensity() const;
    void setFluidDensity(double density);
    double getKinematicViscosity() const;
    void setKinematicViscosity(double viscosity);
    double getFriction() const;
    void setFriction(double friction);
    double getTemperature() const;
    void setTemperature(double temperature);
    CouplingScheme getCouplingScheme() const;
    void setCouplingScheme(CouplingScheme scheme);
    DragScheme getDragScheme() const;
    void setDragScheme(DragScheme scheme);
    InterpolationStencil getInterpolationStencil() const;
    void setInterpolationStencil(InterpolationStencil stencil);
    bool getFluidFluctuations() const;
    void setFluidFluctuations(bool fluctuations);
    WallScheme getWallScheme() const;
    void setWallScheme(WallScheme scheme);
    BoundaryType getFaceBoundary(Face face) const;
    void setFaceBoundary(Face face, BoundaryType type);
    OpenMM::Vec3 getFaceVelocity(Face face) const;
    void setFaceVelocity(Face face, const OpenMM::Vec3& velocity);
    double getFaceDensity(Face face) const;
    void setFaceDensity(Face face, double density);
    int getRandomNumberSeed() const;
    void setRandomNumberSeed(int seed);
    OpenMM::Vec3 getBodyAcceleration() const;
    void setBodyAcceleration(const OpenMM::Vec3& acceleration);
    OpenMM::Vec3 getInitialFluidVelocity() const;
    void setInitialFluidVelocity(const OpenMM::Vec3& velocity);
    int getFluidMomentumRemovalFrequency() const;
    void setFluidMomentumRemovalFrequency(int frequency);
    %apply std::vector<int>& OUTPUT {std::vector<int>& nodes};
    void getSolidNodes(std::vector<int>& nodes) const;
    %clear std::vector<int>& nodes;
    void setSolidNodes(const std::vector<int>& nodes);
    int getMachCheckFrequency() const;
    void setMachCheckFrequency(int frequency);
    double getMachNumberLimit() const;
    void setMachNumberLimit(double limit);

    int getNumParticles() const;
    int addParticle(int particle);
    int getParticle(int index) const;
    void setParticle(int index, int particle);


    %apply int& OUTPUT {int& i0};
    %apply int& OUTPUT {int& j0};
    %apply int& OUTPUT {int& k0};
    %apply int& OUTPUT {int& ni};
    %apply int& OUTPUT {int& nj};
    %apply int& OUTPUT {int& nk};
    void getLocalDomain(const OpenMM::Context& context, int& i0, int& j0, int& k0, int& ni, int& nj, int& nk) const;
    %clear int& i0;
    %clear int& j0;
    %clear int& k0;
    %clear int& ni;
    %clear int& nj;
    %clear int& nk;
    %apply std::vector<double>& OUTPUT {std::vector<double>& state};
    void getFluidState(OpenMM::Context& context, std::vector<double>& state, bool gather) const;
    %clear std::vector<double>& state;
    void setFluidState(OpenMM::Context& context, const std::vector<double>& state, bool scatter);
    double getFluidMachNumber(OpenMM::Context& context) const;
    OpenMM::Vec3 getWallForce(OpenMM::Context& context) const;
    void saveCheckpointFile(OpenMM::Context& context, const std::string& file) const;
    void loadCheckpointFile(OpenMM::Context& context, const std::string& file);
    void writeFluidFile(OpenMM::Context& context, const std::string& file, const std::string& head,
            const std::string& tail, const std::string& arrays, bool doublePrecision) const;

    %apply double& OUTPUT {double& dx};
    %apply double& OUTPUT {double& dt};
    %apply double& OUTPUT {double& tau};
    void getLatticeParametersInContext(const OpenMM::Context& context, double& dx, double& dt, double& tau) const;
    %clear double& dx;
    %clear double& dt;
    %clear double& tau;

    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const;

    /*
     * Add methods for casting a Force to an LBMForce.
     */
    %extend {
        /*
         * The density and velocity of the fluid, as the tuple (density, velocity) of two lists (LBMForce.getFluidFields()
         * adds the units and the defaults of gather and halo).
         */
        PyObject* _getFluidFields(OpenMM::Context& context, bool gather, bool halo) {
            std::vector<double> density;
            std::vector<OpenMM::Vec3> velocity;
            self->getFluidFields(context, density, velocity, gather, halo);
            PyObject* densityList = PyList_New(density.size());
            for (int i = 0; i < (int) density.size(); i++)
                PyList_SET_ITEM(densityList, i, PyFloat_FromDouble(density[i]));
            PyObject* velocityList = PyList_New(velocity.size());
            for (int i = 0; i < (int) velocity.size(); i++)
                PyList_SET_ITEM(velocityList, i, Vec3_to_PyVec3(velocity[i]));
            PyObject* fields = PyTuple_New(2);
            PyTuple_SET_ITEM(fields, 0, densityList);
            PyTuple_SET_ITEM(fields, 1, velocityList);
            return fields;
        }

        /*
         * Write a checkpoint of the fluid and of the coupling (see the C++ documentation of
         * createCheckpoint()) and return it as bytes.
         */
        PyObject* createCheckpoint(OpenMM::Context& context) {
            std::stringstream stream(std::ios_base::out | std::ios_base::binary);
            self->createCheckpoint(context, stream);
            std::string data = stream.str();
            return PyBytes_FromStringAndSize(data.c_str(), data.size());
        }

        /*
         * Load a checkpoint returned by createCheckpoint(), given as bytes.
         */
        void loadCheckpoint(OpenMM::Context& context, PyObject* checkpoint) {
            char* data;
            Py_ssize_t length;
            if (!PyBytes_Check(checkpoint) || PyBytes_AsStringAndSize(checkpoint, &data, &length) != 0)
                throw OpenMM::OpenMMException("LBMForce.loadCheckpoint: the checkpoint must be bytes");
            std::stringstream stream(std::string(data, length), std::ios_base::in | std::ios_base::binary);
            self->loadCheckpoint(context, stream);
        }

        static LBMPlugin::LBMForce& cast(OpenMM::Force& force) {
            return dynamic_cast<LBMPlugin::LBMForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<LBMPlugin::LBMForce*>(&force) != NULL);
        }
    }
};

}

// Not part of the API: called at exit by the Python module (_finalizeMPIAtExit below).
%rename(_finalizeMPI) lbmFinalizeMPI;
%inline %{
void lbmFinalizeMPI() {
    LBMPlugin::LBMDecomposition::finalizeMPI();
}
%}

%pythoncode %{
import atexit as _atexit
import os as _os
import struct as _struct
import sys as _sys

_CHECKPOINT_TAG = b'OPENMMLBM-CHECKPOINT-1\n'
_CHECKPOINT_FILE_TAG = b'OPENMMLBM-CHECKPOINT-2\n'


def mpiRank():
    """The rank of this process in MPI_COMM_WORLD (0 without MPI): LBMForce.getMPIRank()."""
    return LBMForce.getMPIRank()


def mpiSize():
    """The number of MPI ranks (1 without MPI): LBMForce.getMPISize()."""
    return LBMForce.getMPISize()


def mpiLocalRank():
    """The rank among the processes on the same node (0 without MPI), to choose the GPU: LBMForce.getMPILocalRank()."""
    return LBMForce.getMPILocalRank()


def _LBMForce_getFluidFields(self, context, gather=False, halo=False):
    """Get the density (Da/nm^3) and velocity (nm/ps) of the fluid at the nodes of the domain of this MPI rank, the
    whole lattice with one domain, as the tuple (density, velocity) of two Quantities: node (i0 + i, j0 + j, k0 + k)
    of the domain ((i0, j0, k0), (ni, nj, nk)) = getLocalDomain(context) has index i + ni*(j + nj*k).

    gather=True: rank 0 receives the fields of the whole lattice, node (i, j, k) at index i + nx*(j + ny*k), and the
    other ranks empty lists; every rank must call it.  halo=True: the domain with a layer one node thick around it,
    node (i0 + i, j0 + j, k0 + k) for i from -1 to ni and so on at index (i + 1) + (ni + 2)*((j + 1) + (nj + 2)*(k + 1));
    the layer holds the values of the neighbouring nodes for the fields whose halo is exchanged
    (setDensityHaloExchange(), setVelocityHaloExchange()), and NaN for the others and beyond the open faces."""
    density, velocity = self._getFluidFields(context, gather, halo)
    return (unit.Quantity(density, unit.dalton/unit.nanometer**3), unit.Quantity(velocity, unit.nanometer/unit.picosecond))


def _LBMForce_getFluidState(self, context, gather=False):
    """Get the complete state of the fluid of the domain of this MPI rank (the whole lattice with one domain), to
    restore it with setFluidState().  gather=True: rank 0 receives the state of the whole lattice and the other ranks
    an empty list; every rank must call it."""
    return self._getFluidState(context, gather)


def _LBMForce_setFluidState(self, context, state, scatter=False):
    """Set the complete state of the fluid, as returned by getFluidState().  With the domain decomposition every rank
    must call it, with the state of its domain or, with scatter=True, rank 0 with the state of the whole lattice (the
    state passed by the other ranks, for example None, is ignored)."""
    self._setFluidState(context, [] if state is None else state, scatter)


LBMForce.getFluidFields = _LBMForce_getFluidFields
LBMForce.getFluidState = _LBMForce_getFluidState
LBMForce.setFluidState = _LBMForce_setFluidState


def _abortMPIOnException(excType, value, traceback, _previous=_sys.excepthook):
    # With several MPI ranks an exception that stops one rank would leave the others waiting in the next
    # communication, and the exit of this process would wait for them in MPI_Finalize: the job would hang.  Print the
    # exception, then abort every rank (MPI_Abort), as python -m mpi4py does.  With one process, or without MPI,
    # abortMPI() does nothing.
    _previous(excType, value, traceback)
    try:
        _sys.stderr.flush()
    except Exception:
        pass
    LBMForce.abortMPI(1)


_sys.excepthook = _abortMPIOnException


def _finalizeMPIAtExit():
    # Finalize MPI, if the plugin initialized it, before the objects of the script are deleted: the functions
    # registered with atexit run first.  The MPI library may hold registrations of host memory made in the CUDA
    # context of a Context (UCX does for the populations exchanged through the host between nodes), and OpenMM
    # destroys that context with the Context: finalizing MPI later, from the plugin's own handler at the exit of the
    # process, made UCX print hundreds of errors at the end of every run on several nodes (docs/validation.md).
    _finalizeMPI()


_atexit.register(_finalizeMPIAtExit)


def _isDecomposed(context, force):
    """True if the lattice of the Context is divided into domains (setDomainDecomposition())."""
    return force.getLocalDomain(context)[1] != tuple(force.getGridSize())


def saveCheckpoint(file, context, force):
    """Save the complete state of a Context with an LBMForce to a file.

    With one domain the file holds an OpenMM checkpoint (Context.createCheckpoint(): positions, velocities,
    periodic box, time, step count, state of the integrator and of OpenMM's random number generators) and the
    checkpoint of the force (LBMForce.createCheckpoint(): fluid, random numbers already drawn for the next step, force
    on the walls of the last step).  With the domain decomposition (setDomainDecomposition()) every MPI rank must call
    it, and the file is that of LBMForce.saveCheckpointFile(): the fluid of the whole lattice, written by every rank
    for its domain, and the OpenMM checkpoint of every rank; it can be loaded with any decomposition.  Loading it with
    loadCheckpoint() continues the run exactly, bit for bit (with the decomposition, if it is the same).  The data are
    written to file + '.tmp' and then renamed, so an interrupted write never replaces a valid checkpoint with a
    damaged one.  Like OpenMM checkpoints, the file is specific to the platform, the precision and the System.  To
    continue a run of one domain with several, write its checkpoint with LBMForce.saveCheckpointFile().

    Parameters
    ----------
    file : str
        path of the file to write
    context : openmm.Context
        the Context to save, for example simulation.context
    force : LBMForce
        the LBMForce of the System of the Context
    """
    if _isDecomposed(context, force):
        force.saveCheckpointFile(context, file)
        return
    openmmData = context.createCheckpoint()
    forceData = force.createCheckpoint(context)
    temporary = file + '.tmp'
    with open(temporary, 'wb') as out:
        out.write(_CHECKPOINT_TAG)
        out.write(_struct.pack('<qq', len(openmmData), len(forceData)))
        out.write(openmmData)
        out.write(forceData)
    _os.replace(temporary, file)


def loadCheckpoint(file, context, force):
    """Load a file written by saveCheckpoint() or LBMForce.saveCheckpointFile() into a Context.

    The Context must be built from the same System (same particles, forces and LBMForce parameters) on the
    same platform and precision.  It restores positions, velocities, box, time and step count as
    Context.loadCheckpoint() does, and the fluid and the random numbers of the force.  With the domain decomposition
    every MPI rank must call it; a file written with another decomposition gives the same fluid and particles, and the
    run continues with the random number generators of the new Context (LBMForce.saveCheckpointFile()).

    Parameters
    ----------
    file : str
        path of the file to read
    context : openmm.Context
        the Context to restore, for example simulation.context
    force : LBMForce
        the LBMForce of the System of the Context
    """
    with open(file, 'rb') as stream:
        tag = stream.read(len(_CHECKPOINT_TAG))
    if tag not in (_CHECKPOINT_TAG, _CHECKPOINT_FILE_TAG):
        raise ValueError('%s is not a checkpoint written by openmmlbm.saveCheckpoint()' % file)
    if tag == _CHECKPOINT_FILE_TAG or _isDecomposed(context, force):
        force.loadCheckpointFile(context, file)
        return
    with open(file, 'rb') as stream:
        data = stream.read()
    start = len(_CHECKPOINT_TAG) + 16
    openmmLength, forceLength = _struct.unpack_from('<qq', data, len(_CHECKPOINT_TAG))
    if len(data) != start + openmmLength + forceLength:
        raise ValueError('%s is truncated or damaged' % file)
    context.loadCheckpoint(data[start:start+openmmLength])
    force.loadCheckpoint(context, data[start+openmmLength:])


class LBMCheckpointReporter(object):
    """A reporter for openmm.app.Simulation that saves a checkpoint of the run, fluid included, at regular
    intervals: the counterpart of openmm.app.CheckpointReporter for Systems with an LBMForce.

    Every reportInterval steps it calls saveCheckpoint(file, simulation.context, force), overwriting the
    same file.  To continue the run, build the Simulation again and call
    loadCheckpoint(file, simulation.context, force).
    """

    def __init__(self, file, reportInterval, force):
        """
        Parameters
        ----------
        file : str
            path of the checkpoint file
        reportInterval : int
            steps between checkpoints
        force : LBMForce
            the LBMForce of the System of the Simulation
        """
        self._file = file
        self._reportInterval = reportInterval
        self._force = force

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return {'steps': steps, 'periodic': None, 'include': []}

    def report(self, simulation, state):
        saveCheckpoint(self._file, simulation.context, self._force)


class LBMTemperatureReporter(object):
    """A reporter for openmm.app.Simulation that writes the temperature of the particles coupled to an LBMForce,
    measured with the velocity that has the right temperature for the drag scheme of the force (docs/theory.md,
    section 2).

    With the Explicit drag it is the full-step velocity v + dt F/(2m), from the velocities and forces of the State:
    the temperature that StateDataReporter reports for the same particles.  With the Centered drag it is the
    velocity of the State, half a step after the force, while StateDataReporter reports the full-step temperature,
    which is lower, by 1/(1 + friction*dt/2) for a particle in a fluid at rest.  The temperature counts three degrees
    of freedom per coupled particle.  Each line holds the step, the time (ps) and the temperature (K).
    """

    def __init__(self, file, reportInterval, force):
        """
        Parameters
        ----------
        file : str or file
            path of the file to write, or an open file (for example sys.stdout)
        reportInterval : int
            steps between reports
        force : LBMForce
            the LBMForce of the System of the Simulation
        """
        self._reportInterval = reportInterval
        self._force = force
        self._openedFile = isinstance(file, str)
        self._out = open(file, 'w') if self._openedFile else file
        self._particles = None

    def __del__(self):
        if self._openedFile:
            self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        include = ['velocities'] if self._force.getDragScheme() == LBMForce.Centered else ['velocities', 'forces']
        return {'steps': steps, 'periodic': None, 'include': include}

    def report(self, simulation, state):
        import numpy as np
        if self._particles is None:
            self._particles = [self._force.getParticle(i) for i in range(self._force.getNumParticles())]
            self._masses = np.array([simulation.system.getParticleMass(i).value_in_unit(unit.dalton) for i in self._particles])
            print('#"Step","Time (ps)","Temperature (K)"', file=self._out)
        velocities = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)[self._particles]
        if self._force.getDragScheme() != LBMForce.Centered:
            forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)[self._particles]
            dt = simulation.integrator.getStepSize().value_in_unit(unit.picosecond)
            velocities = velocities + forces*(0.5*dt/self._masses[:, None])
        energy = 0.5*np.sum(self._masses[:, None]*velocities**2)
        kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
        temperature = 2*energy/(3*len(self._particles)*kB)
        print('%d %.4f %.3f' % (simulation.currentStep, state.getTime().value_in_unit(unit.picosecond), temperature),
              file=self._out)
        self._out.flush()


def _vtkUnits(units):
    """The field data of a VTK XML dataset with the string array "units", one line "array: unit" per array, which
    ParaView shows in its Information panel.  VTK writes the strings of an ASCII array as the codes of their
    characters, each string ended by 0."""
    codes = ' '.join(' '.join(str(ord(c)) for c in line) + ' 0' for line in units)
    return ('    <FieldData>\n'
            '      <DataArray type="String" Name="units" NumberOfTuples="%d" format="ascii">\n'
            '        %s\n'
            '      </DataArray>\n'
            '    </FieldData>\n' % (len(units), codes))


def _vtkHead(kind, opening, closing, sizes):
    """The text of a VTK XML file before its data arrays, appended in binary (raw, little endian, UInt64 sizes).

    opening and closing are the XML lines before and after the data, with {offset<i>} where the i-th array goes;
    sizes are the numbers of bytes of the arrays, in order."""
    offsets, offset = {}, 0
    for i, size in enumerate(sizes):
        offsets['offset%d' % i] = offset
        offset += 8 + size
    return ('<?xml version="1.0"?>\n<!-- openmmlbm.LBMVTKReporter, in OpenMM units: see the field data array "units" -->\n'
            '<VTKFile type="%s" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n' % kind) + \
           opening.format(**offsets) + closing + '  <AppendedData encoding="raw">\n_'


_VTK_TAIL = '\n  </AppendedData>\n</VTKFile>\n'


def _writeVTKFile(path, kind, opening, closing, arrays):
    """Write a VTK XML file whose data arrays, the numpy arrays given in order, are appended in binary (_vtkHead())."""
    import numpy as np
    blocks = [np.ascontiguousarray(array).astype(array.dtype.newbyteorder('<'), copy=False).tobytes() for array in arrays]
    header = _vtkHead(kind, opening, closing, [len(data) for data in blocks])
    with open(path, 'wb') as out:
        out.write(header.encode('ascii'))
        for data in blocks:
            out.write(_struct.pack('<Q', len(data)))
            out.write(data)
        out.write(_VTK_TAIL.encode('ascii'))


class LBMVTKReporter(object):
    """A reporter for openmm.app.Simulation that writes the fluid of an LBMForce and the particles of the System in
    VTK files, which ParaView and VisIt read, in OpenMM units.

    Every reportInterval steps it writes, with the given prefix and the step number:
      <prefix>_density_<step>.vti    the density of the fluid in Da/nm^3 (VTK XML ImageData): point (i, j, k) is the
                                     lattice node at (i dx, j dx, k dx), in nm;
      <prefix>_velocity_<step>.vti   the velocity of the fluid in nm/ps, on the same points; both as
                                     LBMForce.getFluidFields() returns them, and with solid nodes both also have
                                     "solid", 1 for a solid node and 0 for a fluid one, so that each opens alone;
      <prefix>_particles_<step>.vtp  the particles (VTK XML PolyData): positions in nm (with each molecule
                                     wrapped into the periodic box, as State does with enforcePeriodicBox, unless
                                     wrap=False), "velocity" in nm/ps (the velocities of the State, at
                                     the half step of the leapfrog), "mass" in Da, "index" in the System and
                                     "coupled", 1 for the particles coupled to the fluid and 0 for the others;
      <prefix>.pvd                   the list of all the files written, with their times in ps: open it in
                                     ParaView to load the whole series.
    The numbers are binary, in single precision unless double=True.  The reporter does not change the run: like
    getState(), getFluidFields() neither advances the fluid nor draws random numbers.  With the domain decomposition
    (setDomainDecomposition()) every MPI rank must have the reporter: the files of the fluid are written together, each
    rank writing the nodes of its domain (LBMForce.writeFluidFile(), MPI-IO), and rank 0 writes the particles and the
    list.
    """

    def __init__(self, prefix, reportInterval, force, fluid=True, particles=True, double=False, wrap=True, append=False,
                 density=True, velocity=True):
        """
        Parameters
        ----------
        prefix : str
            prefix of the files, which may contain a directory
        reportInterval : int
            steps between reports
        force : LBMForce
            the LBMForce of the System of the Simulation
        fluid : bool
            write the fluid (default True); False writes neither the density nor the velocity
        particles : bool
            write the particles (default True)
        double : bool
            write in double precision instead of single (default False)
        wrap : bool
            wrap the positions of the particles into the periodic box, so that they overlay the lattice (default
            True)
        append : bool
            keep the files already listed in <prefix>.pvd, for a run continued from a checkpoint (default False)
        density : bool
            write the density of the fluid (default True)
        velocity : bool
            write the velocity of the fluid (default True)
        """
        self._prefix = prefix
        self._reportInterval = reportInterval
        self._force = force
        self._density = fluid and density
        self._velocity = fluid and velocity
        self._particles = particles
        self._double = double
        self._wrap = wrap
        self._hasSolid = None
        self._datasets = []
        if append and _os.path.exists(prefix + '.pvd'):
            with open(prefix + '.pvd') as pvd:
                self._datasets = [line for line in pvd if '<DataSet ' in line]

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        include = ['positions', 'velocities'] if self._particles else []
        return {'steps': steps, 'periodic': (self._wrap if self._particles else None), 'include': include}

    def report(self, simulation, state):
        import numpy as np
        real = np.float64 if self._double else np.float32
        name = 'Float64' if self._double else 'Float32'
        step = simulation.currentStep
        time = state.getTime().value_in_unit(unit.picosecond)
        files = []
        nx, ny, nz = self._force.getGridSize()
        if self._hasSolid is None:
            self._hasSolid = len(self._force.getSolidNodes()) > 0
        for field, components, unitName in (('density', 1, 'Da/nm^3'), ('velocity', 3, 'nm/ps')):
            if not (self._density if field == 'density' else self._velocity):
                continue
            dx = self._force.getLatticeParametersInContext(simulation.context)[0]
            dx = dx.value_in_unit(unit.nanometer) if unit.is_quantity(dx) else dx
            extent = '0 %d 0 %d 0 %d' % (nx - 1, ny - 1, nz - 1)
            units = ['Origin, Spacing: nm', '%s: %s' % (field, unitName)]
            sizes = [nx*ny*nz*components*np.dtype(real).itemsize]
            arrays = field
            if self._hasSolid:
                units.append('solid: 1 for a solid node, 0 for a fluid node')
                sizes.append(nx*ny*nz)
                arrays += ' solid'
            opening = ('  <ImageData WholeExtent="%s" Origin="0 0 0" Spacing="%.17g %.17g %.17g">\n' % (extent, dx, dx, dx) +
                       _vtkUnits(units) +
                       '    <Piece Extent="%s">\n'
                       '      <PointData %s="%s">\n'
                       '        <DataArray type="%s" Name="%s"%s format="appended" offset="{offset0}"/>\n' %
                       (extent, 'Scalars' if components == 1 else 'Vectors', field, name, field,
                        ' NumberOfComponents="3"' if components == 3 else ''))
            if self._hasSolid:
                opening += '        <DataArray type="UInt8" Name="solid" format="appended" offset="{offset1}"/>\n'
            closing = '      </PointData>\n    </Piece>\n  </ImageData>\n'
            path = '%s_%s_%010d.vti' % (self._prefix, field, step)
            self._force.writeFluidFile(simulation.context, path, _vtkHead('ImageData', opening, closing, sizes),
                                       _VTK_TAIL, arrays, self._double)
            files.append(path)
        if LBMForce.getMPIRank() != 0:
            return
        if self._particles:
            system = simulation.system
            n = system.getNumParticles()
            coupled = np.zeros(n, dtype=np.uint8)
            coupled[[self._force.getParticle(i) for i in range(self._force.getNumParticles())]] = 1
            positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            velocities = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
            masses = [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n)]
            arrays = [np.asarray(positions, dtype=real).reshape(-1), np.asarray(velocities, dtype=real).reshape(-1),
                      np.asarray(masses, dtype=real), np.arange(n, dtype=np.int32), coupled,
                      np.arange(n, dtype=np.int64), np.arange(1, n + 1, dtype=np.int64)]
            units = ['Points: nm', 'velocity: nm/ps, at the half step of the leapfrog', 'mass: Da',
                     'index: index of the particle in the System', 'coupled: 1 for a particle coupled to the fluid']
            opening = ('  <PolyData>\n' + _vtkUnits(units) +
                       '    <Piece NumberOfPoints="%d" NumberOfVerts="%d" NumberOfLines="0" NumberOfStrips="0" '
                       'NumberOfPolys="0">\n'
                       '      <Points>\n'
                       '        <DataArray type="%s" NumberOfComponents="3" format="appended" offset="{offset0}"/>\n'
                       '      </Points>\n'
                       '      <PointData Vectors="velocity">\n'
                       '        <DataArray type="%s" Name="velocity" NumberOfComponents="3" format="appended" '
                       'offset="{offset1}"/>\n'
                       '        <DataArray type="%s" Name="mass" format="appended" offset="{offset2}"/>\n'
                       '        <DataArray type="Int32" Name="index" format="appended" offset="{offset3}"/>\n'
                       '        <DataArray type="UInt8" Name="coupled" format="appended" offset="{offset4}"/>\n'
                       '      </PointData>\n'
                       '      <Verts>\n'
                       '        <DataArray type="Int64" Name="connectivity" format="appended" offset="{offset5}"/>\n'
                       '        <DataArray type="Int64" Name="offsets" format="appended" offset="{offset6}"/>\n'
                       '      </Verts>\n' % (n, n, name, name, name))
            closing = '    </Piece>\n  </PolyData>\n'
            path = '%s_particles_%010d.vtp' % (self._prefix, step)
            _writeVTKFile(path, 'PolyData', opening, closing, arrays)
            files.append(path)
        for part, path in enumerate(files):
            self._datasets.append('    <DataSet timestep="%.17g" part="%d" file="%s"/>\n' % (time, part,
                                                                                        _os.path.basename(path)))
        with open(self._prefix + '.pvd', 'w') as out:
            out.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
                      '  <Collection>\n' + ''.join(self._datasets) + '  </Collection>\n</VTKFile>\n')
%}

