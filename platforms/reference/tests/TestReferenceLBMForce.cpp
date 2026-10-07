/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "TestLBMForce.h"
#include "TestLBMFluid.h"
#include "TestLBMCoupling.h"
#include "TestLBMCentered.h"
#include "TestLBMFluctuations.h"

extern "C" OPENMM_EXPORT void registerLBMReferenceKernelFactories();

int main(int argc, char* argv[]) {
    try {
        registerLBMReferenceKernelFactories();
        runPlatformTests(Platform::getPlatformByName("Reference"));
        runFluidTests(Platform::getPlatformByName("Reference"));
        runWallTests(Platform::getPlatformByName("Reference"));
        runCouplingTests(Platform::getPlatformByName("Reference"));
        runCenteredTests(Platform::getPlatformByName("Reference"));
        runFluctuationTests(Platform::getPlatformByName("Reference"));
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
