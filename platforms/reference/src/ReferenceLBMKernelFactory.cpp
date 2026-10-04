/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "ReferenceLBMKernelFactory.h"
#include "ReferenceLBMKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace LBMPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferenceLBMKernelFactory* factory = new ReferenceLBMKernelFactory();
            platform.registerKernelFactory(CalcLBMForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerLBMReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferenceLBMKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    if (name == CalcLBMForceKernel::Name())
        return new ReferenceCalcLBMForceKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
