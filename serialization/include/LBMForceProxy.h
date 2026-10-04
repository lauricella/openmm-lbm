#ifndef OPENMM_LBM_FORCE_PROXY_H_
#define OPENMM_LBM_FORCE_PROXY_H_

/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "internal/windowsExportLBM.h"
#include "openmm/serialization/SerializationProxy.h"

namespace OpenMM {

/**
 * This is a proxy for serializing LBMForce objects.  Only the parameters are serialized: the state
 * of the fluid is saved and restored with LBMForce::getFluidState() and setFluidState().
 */

class OPENMM_EXPORT_LBM LBMForceProxy : public SerializationProxy {
public:
    LBMForceProxy();
    void serialize(const void* object, SerializationNode& node) const;
    void* deserialize(const SerializationNode& node) const;
};

} // namespace OpenMM

#endif /*OPENMM_LBM_FORCE_PROXY_H_*/
