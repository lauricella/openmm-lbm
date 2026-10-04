/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * Derived from the OpenMM example plugin (openmm/openmmexampleplugin),       *
 * portions copyright (c) 2014 Stanford University and the Authors.           *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
#include <windows.h>
#include <sstream>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <cstdlib>
#endif

#include "LBMForce.h"
#include "LBMForceProxy.h"
#include "openmm/serialization/SerializationProxy.h"

#if defined(WIN32)
    #include <windows.h>
    extern "C" OPENMM_EXPORT_LBM void registerLBMSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            registerLBMSerializationProxies();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerLBMSerializationProxies();
#endif

using namespace LBMPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT_LBM void registerLBMSerializationProxies() {
    SerializationProxy::registerProxy(typeid(LBMForce), new LBMForceProxy());
}
