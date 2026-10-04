/* -------------------------------------------------------------------------- *
 *                                 openmm-lbm                                 *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2026 the Authors (see README.md).                            *
 * SPDX-License-Identifier: MIT                                               *
 * -------------------------------------------------------------------------- */

#include "TestLBMForce.h"

extern "C" OPENMM_EXPORT void registerLBMOpenCLKernelFactories();

int main(int argc, char* argv[]) {
    try {
        registerLBMOpenCLKernelFactories();
        Platform& platform = Platform::getPlatformByName("OpenCL");
        if (argc > 1)
            platform.setPropertyDefaultValue("Precision", string(argv[1]));
        runPlatformTests(platform);
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
