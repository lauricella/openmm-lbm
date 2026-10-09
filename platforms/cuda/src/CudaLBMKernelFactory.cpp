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
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace LBMPlugin;
using namespace OpenMM;

/**
 * The kernel of the common platforms, with the addresses of its arrays in device memory, so that an MPI library that
 * reads and writes device memory (CUDA-aware MPI) exchanges the populations of the domain decomposition from device to
 * device, over NVLink within a node.
 */
class CudaCalcLBMForceKernel : public CommonCalcLBMForceKernel {
public:
    CudaCalcLBMForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) :
            CommonCalcLBMForceKernel(name, platform, cu, system) {
    }
protected:
    char* getDeviceAddress(ComputeArray& array) {
        return (char*) dynamic_cast<CudaArray&>(array.getArray()).getDevicePointer();
    }
};

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
    CudaPlatform::PlatformData* data = static_cast<CudaPlatform::PlatformData*>(context.getPlatformData());
    CudaContext& cc = *data->contexts[0];
    if (name == CalcLBMForceKernel::Name()) {
        CommonCalcLBMForceKernel* kernel = new CudaCalcLBMForceKernel(name, platform, cc, context.getSystem());
        kernel->setDeterministicForces(data->deterministicForces);
        return kernel;
    }
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
