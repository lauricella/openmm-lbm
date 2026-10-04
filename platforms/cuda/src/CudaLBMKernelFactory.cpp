/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include <exception>

#include "CudaLBMKernelFactory.h"
#include "CommonLBMKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace LBMPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaLBMKernelFactory* factory = new CudaLBMKernelFactory();
        platform.registerKernelFactory(CalcLBMForceKernel::Name(), factory);
    }
    catch (const std::exception& ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerLBMCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaLBMKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cc = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcLBMForceKernel::Name())
        return new CommonCalcLBMForceKernel(name, platform, cc, context.getSystem());
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
