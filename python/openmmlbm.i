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
%pythonappend LBMPlugin::LBMForce::getLatticeParametersInContext(const OpenMM::Context& context, double& dx, double& dt, double& tau) const %{
    val = (unit.Quantity(val[0], unit.nanometer), unit.Quantity(val[1], unit.picosecond), val[2])
%}
%pythonappend LBMPlugin::LBMForce::getWallForce(OpenMM::Context& context) const %{
    val = unit.Quantity(val, unit.kilojoule_per_mole/unit.nanometer)
%}
%pythonappend LBMPlugin::LBMForce::getFluidFields(OpenMM::Context& context) %{
    val = (unit.Quantity(val[0], unit.dalton/unit.nanometer**3), unit.Quantity(val[1], unit.nanometer/unit.picosecond))
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
    LBMForce();

    %apply int& OUTPUT {int& nx};
    %apply int& OUTPUT {int& ny};
    %apply int& OUTPUT {int& nz};
    void getGridSize(int& nx, int& ny, int& nz) const;
    %clear int& nx;
    %clear int& ny;
    %clear int& nz;
    void setGridSize(int nx, int ny, int nz);

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
    bool getFluidFluctuations() const;
    void setFluidFluctuations(bool fluctuations);
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


    %apply std::vector<double>& OUTPUT {std::vector<double>& state};
    void getFluidState(OpenMM::Context& context, std::vector<double>& state) const;
    %clear std::vector<double>& state;
    void setFluidState(OpenMM::Context& context, const std::vector<double>& state);
    double getFluidMachNumber(OpenMM::Context& context) const;
    OpenMM::Vec3 getWallForce(OpenMM::Context& context) const;

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
         * Get the density and velocity of the fluid at every lattice node, as the tuple
         * (density, velocity) of two lists.
         */
        PyObject* getFluidFields(OpenMM::Context& context) {
            std::vector<double> density;
            std::vector<OpenMM::Vec3> velocity;
            self->getFluidFields(context, density, velocity);
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

%pythoncode %{
import os as _os
import struct as _struct

_CHECKPOINT_TAG = b'OPENMMLBM-CHECKPOINT-1\n'


def saveCheckpoint(file, context, force):
    """Save the complete state of a Context with an LBMForce to a file.

    The file holds an OpenMM checkpoint (Context.createCheckpoint(): positions, velocities, periodic box,
    time, step count, state of the integrator and of OpenMM's random number generators) and the checkpoint
    of the force (LBMForce.createCheckpoint(): fluid, random numbers already drawn for the next step, force on
    the walls of the last step).  Loading it with loadCheckpoint() continues the run exactly, bit for bit.
    The data are written to file + '.tmp' and then renamed, so an interrupted write never replaces a valid
    checkpoint with a damaged one.  Like OpenMM checkpoints, the file is specific to the platform, the
    precision and the System.

    Parameters
    ----------
    file : str
        path of the file to write
    context : openmm.Context
        the Context to save, for example simulation.context
    force : LBMForce
        the LBMForce of the System of the Context
    """
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
    """Load a file written by saveCheckpoint() into a Context.

    The Context must be built from the same System (same particles, forces and LBMForce parameters) on the
    same platform and precision.  It restores positions, velocities, box, time and step count as
    Context.loadCheckpoint() does, and the fluid and the random numbers of the force.

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
        data = stream.read()
    if not data.startswith(_CHECKPOINT_TAG):
        raise ValueError('%s is not a checkpoint written by openmmlbm.saveCheckpoint()' % file)
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
%}

