/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "TestLBMForce.h"
#include "TestLBMFluid.h"

extern "C" OPENMM_EXPORT void registerLBMHipKernelFactories();

int main(int argc, char* argv[]) {
    try {
        registerLBMHipKernelFactories();
        Platform& platform = Platform::getPlatformByName("HIP");
        if (argc > 1)
            platform.setPropertyDefaultValue("Precision", string(argv[1]));
        runPlatformTests(platform);
        runFluidTests(platform);
        runWallTests(platform);
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
