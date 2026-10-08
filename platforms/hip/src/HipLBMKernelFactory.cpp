/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include <exception>

#include "HipLBMKernelFactory.h"
#include "CommonLBMKernels.h"
#include "openmm/hip/HipContext.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace LBMPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("HIP");
        HipLBMKernelFactory* factory = new HipLBMKernelFactory();
        platform.registerKernelFactory(CalcLBMForceKernel::Name(), factory);
    }
    catch (const std::exception& ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerLBMHipKernelFactories() {
    try {
        Platform::getPlatformByName("HIP");
    }
    catch (...) {
        Platform::registerPlatform(new HipPlatform());
    }
    registerKernelFactories();
}

KernelImpl* HipLBMKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    HipPlatform::PlatformData* data = static_cast<HipPlatform::PlatformData*>(context.getPlatformData());
    HipContext& cc = *data->contexts[0];
    if (name == CalcLBMForceKernel::Name()) {
        CommonCalcLBMForceKernel* kernel = new CommonCalcLBMForceKernel(name, platform, cc, context.getSystem());
        kernel->setDeterministicForces(data->deterministicForces);
        return kernel;
    }
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
