#ifndef OPENMM_LBMFORCEIMPL_H_
#define OPENMM_LBMFORCEIMPL_H_

/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "LBMForce.h"
#include "LBMKernels.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace LBMPlugin {

/**
 * This is the internal implementation of LBMForce.  It checks the setup and converts the
 * parameters to lattice units (LBMLatticeParameters) once for all platforms.
 */

class OPENMM_EXPORT_LBM LBMForceImpl : public OpenMM::ForceImpl {
public:
    LBMForceImpl(const LBMForce& owner);
    ~LBMForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const LBMForce& getOwner() const {
        return owner;
    }
    /**
     * Called by the integrator at the start of every step: the fluid advances only on the force evaluation of
     * an integration step, never on other requests for forces or energy.
     */
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid);
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>(); // This force does not define any global parameters.
    }
    std::vector<std::string> getKernelNames();
    void updateParametersInContext(OpenMM::ContextImpl& context);
    void getFluidFields(OpenMM::ContextImpl& context, std::vector<double>& density, std::vector<OpenMM::Vec3>& velocity);
    void getFluidState(OpenMM::ContextImpl& context, std::vector<double>& state);
    void setFluidState(OpenMM::ContextImpl& context, const std::vector<double>& state);
    void createCheckpoint(OpenMM::ContextImpl& context, std::ostream& stream);
    void loadCheckpoint(OpenMM::ContextImpl& context, std::istream& stream);
    double getFluidMachNumber(OpenMM::ContextImpl& context);
    OpenMM::Vec3 getWallForce(OpenMM::ContextImpl& context);
    void getLatticeParameters(double& dx, double& dt, double& tau) const;
    /**
     * Compute the lattice parameters for a force, a System and an integrator step size, checking
     * that the setup is valid.  Throws an OpenMMException otherwise.
     */
    static LBMLatticeParameters computeLatticeParameters(const LBMForce& force, const OpenMM::System& system, double stepSize);
private:
    const LBMForce& owner;
    OpenMM::Kernel kernel;
    LBMLatticeParameters lattice;
};

} // namespace LBMPlugin

#endif /*OPENMM_LBMFORCEIMPL_H_*/
