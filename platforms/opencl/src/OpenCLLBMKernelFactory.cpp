/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include <exception>

#include "OpenCLLBMKernelFactory.h"
#include "CommonLBMKernels.h"
#include "openmm/opencl/OpenCLContext.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace LBMPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("OpenCL");
        OpenCLLBMKernelFactory* factory = new OpenCLLBMKernelFactory();
        platform.registerKernelFactory(CalcLBMForceKernel::Name(), factory);
    }
    catch (const std::exception& ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerLBMOpenCLKernelFactories() {
    try {
        Platform::getPlatformByName("OpenCL");
    }
    catch (...) {
        Platform::registerPlatform(new OpenCLPlatform());
    }
    registerKernelFactories();
}

KernelImpl* OpenCLLBMKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    OpenCLContext& cc = *static_cast<OpenCLPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcLBMForceKernel::Name())
        return new CommonCalcLBMForceKernel(name, platform, cc, context.getSystem());
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
