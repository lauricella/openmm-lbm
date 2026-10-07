#---------------------------------------------------
# openmm-lbm: lattice Boltzmann fluid plugin for OpenMM
#----------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

PROJECT(OpenMMLBM)

# The version number is set here and copied to anywhere it is needed.
SET(OPENMM_LBM_VERSION 0.2.0 CACHE STRING "The version of openmm-lbm that will be built." FORCE)

# We need to know where OpenMM is installed so we can access the headers and libraries.
SET(OPENMM_DIR "/usr/local/openmm" CACHE PATH "Where OpenMM is installed")
INCLUDE_DIRECTORIES("${OPENMM_DIR}/include")
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

# Supported OpenMM versions: 8.3 to 8.6 (see CONTRIBUTING.md). The minimum is 8.3 because the
# per-cell reduction of the coupling forces sorts on the device with ComputeSort, which entered
# the OpenMM common compute layer in 8.3.0.  The headers of OpenMM do not contain its version: a small
# program linked to the OpenMM library in OPENMM_DIR prints Platform::getOpenMMVersion().
SET(LBM_MINIMUM_OPENMM_VERSION 8.3)
SET(LBM_NEWEST_TESTED_OPENMM_VERSION 8.6)
IF(NOT EXISTS "${OPENMM_DIR}/include/openmm/Platform.h")
    MESSAGE(FATAL_ERROR "OpenMM was not found in ${OPENMM_DIR}: set OPENMM_DIR to the directory where OpenMM is installed (with conda: -DOPENMM_DIR=$CONDA_PREFIX)")
ENDIF()
FILE(WRITE "${CMAKE_BINARY_DIR}/CMakeFiles/openmm_version.cpp"
    "#include \"openmm/Platform.h\"\n#include <iostream>\nint main() {\n    std::cout << OpenMM::Platform::getOpenMMVersion();\n    return 0;\n}\n")
# The library of OPENMM_DIR must be the one that runs, also when the run path of the active environment
# (which conda adds to the linker flags) or the library path point to another OpenMM: the program runs with
# that library preloaded.
SET(OPENMM_LIBRARY_FOR_VERSION "${OPENMM_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}OpenMM${CMAKE_SHARED_LIBRARY_SUFFIX}")
SET(OPENMM_VERSION_OUTPUT "")
IF(EXISTS "${OPENMM_LIBRARY_FOR_VERSION}")
    SET(OPENMM_VERSION_PROGRAM "${CMAKE_BINARY_DIR}/CMakeFiles/print_openmm_version${CMAKE_EXECUTABLE_SUFFIX}")
    TRY_COMPILE(OPENMM_VERSION_COMPILED "${CMAKE_BINARY_DIR}/CMakeFiles/openmm_version_check"
        SOURCES "${CMAKE_BINARY_DIR}/CMakeFiles/openmm_version.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${OPENMM_DIR}/include"
        LINK_LIBRARIES "${OPENMM_LIBRARY_FOR_VERSION}"
        CXX_STANDARD 17
        COPY_FILE "${OPENMM_VERSION_PROGRAM}")
    IF(OPENMM_VERSION_COMPILED)
        IF(APPLE)
            SET(OPENMM_VERSION_PRELOAD "DYLD_INSERT_LIBRARIES=${OPENMM_LIBRARY_FOR_VERSION}")
        ELSEIF(UNIX)
            SET(OPENMM_VERSION_PRELOAD "LD_PRELOAD=${OPENMM_LIBRARY_FOR_VERSION}")
        ELSE()
            SET(OPENMM_VERSION_PRELOAD "")
        ENDIF()
        EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -E env ${OPENMM_VERSION_PRELOAD} "${OPENMM_VERSION_PROGRAM}"
            OUTPUT_VARIABLE OPENMM_VERSION_OUTPUT ERROR_QUIET)
    ENDIF()
ENDIF()
STRING(REGEX MATCH "^[0-9]+\\.[0-9]+(\\.[0-9]+)?" OPENMM_VERSION "${OPENMM_VERSION_OUTPUT}")
IF(OPENMM_VERSION)
    MESSAGE(STATUS "OpenMM ${OPENMM_VERSION} in ${OPENMM_DIR}")
    IF(OPENMM_VERSION VERSION_LESS LBM_MINIMUM_OPENMM_VERSION)
        MESSAGE(FATAL_ERROR "openmm-lbm requires OpenMM ${LBM_MINIMUM_OPENMM_VERSION} or later, but the OpenMM in ${OPENMM_DIR} is version ${OPENMM_VERSION}. Install a supported version (8.3 to 8.6), for example with conda: conda install -c conda-forge openmm=8.6")
    ENDIF()
    IF(OPENMM_VERSION VERSION_GREATER_EQUAL 8.7)
        MESSAGE(WARNING "OpenMM ${OPENMM_VERSION} is newer than the versions openmm-lbm is tested with (8.3 to ${LBM_NEWEST_TESTED_OPENMM_VERSION}): build and run the tests (ctest) before using it")
    ENDIF()
ELSE()
    MESSAGE(WARNING "The version of the OpenMM in ${OPENMM_DIR} could not be determined; checking for the headers of OpenMM 8.3")
ENDIF()
IF(NOT EXISTS "${OPENMM_DIR}/include/openmm/common/ComputeSort.h")
    MESSAGE(FATAL_ERROR "openmm-lbm requires OpenMM 8.3 or later: openmm/common/ComputeSort.h was not found in ${OPENMM_DIR}/include")
ENDIF()

# Specify the C++ version we are building for.
SET(CMAKE_CXX_STANDARD 17)

# Debug diagnostics of the fluid (docs/theory.md, section 6): warning when the Mach number exceeds 0.1 and
# lattice parameters printed at Context creation.
OPTION(LBM_DEBUG "Print debug diagnostics of the lattice Boltzmann fluid" OFF)
IF(LBM_DEBUG)
    ADD_DEFINITIONS(-DLBM_DEBUG)
ENDIF(LBM_DEBUG)

# Set flags for linking on mac
IF(APPLE)
    SET(CMAKE_INSTALL_NAME_DIR "@rpath")
    SET(EXTRA_COMPILE_FLAGS "-stdlib=libc++")
ENDIF(APPLE)

# Select where to install
IF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    IF(WIN32)
        SET(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/OpenMM" CACHE PATH "Where to install the plugin" FORCE)
    ELSE(WIN32)
        SET(CMAKE_INSTALL_PREFIX "/usr/local/openmm" CACHE PATH "Where to install the plugin" FORCE)
    ENDIF(WIN32)
ENDIF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})

# Put all the tests and libraries in a single output directory.
IF(NOT EXECUTABLE_OUTPUT_PATH)
    SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}
        CACHE INTERNAL "Single output directory for building all executables.")
ENDIF()
IF(NOT LIBRARY_OUTPUT_PATH)
    SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}
        CACHE INTERNAL "Single output directory for building all libraries.")
ENDIF()

# The source is organized into subdirectories, but we handle them all from
# this CMakeLists file rather than letting CMake visit them as SUBDIRS.
SET(LBM_PLUGIN_SOURCE_SUBDIRS openmmapi serialization)

# Set the library name
SET(LBM_LIBRARY_NAME OpenMMLBM)
SET(SHARED_LBM_TARGET ${LBM_LIBRARY_NAME})

# These are all the places to search for header files which are to be part of the API.
SET(API_INCLUDE_DIRS "openmmapi/include" "openmmapi/include/internal")

# Locate header files.
SET(API_INCLUDE_FILES)
FOREACH(dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)
    SET(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
ENDFOREACH(dir)

# Collect up source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)
FOREACH(subdir ${LBM_PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})   #append
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})

    ## Make sure we find these locally before looking in OpenMM/include if
    ## OpenMM was previously installed there.
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

# Create the library.

ADD_LIBRARY(${SHARED_LBM_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
SET_TARGET_PROPERTIES(${SHARED_LBM_TARGET}
    PROPERTIES COMPILE_FLAGS "-DLBM_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTRA_COMPILE_FLAGS}")
TARGET_LINK_LIBRARIES(${SHARED_LBM_TARGET} OpenMM)
INSTALL(TARGETS ${SHARED_LBM_TARGET} DESTINATION lib)

# install headers
FILE(GLOB API_ONLY_INCLUDE_FILES "openmmapi/include/*.h")
INSTALL(FILES ${API_ONLY_INCLUDE_FILES} DESTINATION include)
FILE(GLOB API_ONLY_INCLUDE_FILES_INTERNAL "openmmapi/include/internal/*.h")
INSTALL(FILES ${API_ONLY_INCLUDE_FILES_INTERNAL} DESTINATION include/internal)

# Tests shared by all platforms live in tests/.
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/tests)

# Enable testing

ENABLE_TESTING()
ADD_SUBDIRECTORY(serialization/tests)

# Build the implementations for different platforms

ADD_SUBDIRECTORY(platforms/reference)

# A part of openmm-lbm is built only if everything it needs is installed: a GPU platform needs that platform in
# the OpenMM of OPENMM_DIR (its header and its library: an OpenMM compiled from source can be built without
# some platforms) and the toolkit that compiles it; the Python wrapper needs Python with the OpenMM module,
# NumPy, setuptools and pip, SWIG, and the SWIG files and plugin headers of OpenMM.  The checks set the default of each option and
# say why a part is not built.  An option set to ON for a part that cannot be built stops the configuration
# with an error, instead of the build.

# LBM_OPTION(<option> <description> <part> <reason>): the option is ON by default if reason is empty, and must be
# OFF otherwise.
FUNCTION(LBM_OPTION OPTION_NAME DESCRIPTION PART REASON)
    IF(REASON)
        OPTION(${OPTION_NAME} "${DESCRIPTION}" OFF)
        IF(${OPTION_NAME})
            MESSAGE(FATAL_ERROR "${OPTION_NAME} is ON, but ${REASON}: ${PART} cannot be built. Configure with -D${OPTION_NAME}=OFF, or install what is missing.")
        ENDIF()
        MESSAGE(STATUS "${REASON}: ${PART} is not built.")
    ELSE()
        OPTION(${OPTION_NAME} "${DESCRIPTION}" ON)
    ENDIF()
ENDFUNCTION()

# LBM_OPENMM_PLATFORM_REASON(<platform> <header> <library> <result>): empty if the OpenMM in OPENMM_DIR has the
# platform, that is its header in include/openmm and its library in lib or lib/plugins, else why not.
FUNCTION(LBM_OPENMM_PLATFORM_REASON PLATFORM HEADER LIBRARY RESULT)
    SET(names "${CMAKE_SHARED_LIBRARY_PREFIX}${LIBRARY}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    IF(CMAKE_IMPORT_LIBRARY_SUFFIX)
        LIST(APPEND names "${CMAKE_IMPORT_LIBRARY_PREFIX}${LIBRARY}${CMAKE_IMPORT_LIBRARY_SUFFIX}")
    ENDIF()
    SET(library_found OFF)
    FOREACH(dir lib lib/plugins)
        FOREACH(name ${names})
            IF(EXISTS "${OPENMM_DIR}/${dir}/${name}")
                SET(library_found ON)
            ENDIF()
        ENDFOREACH()
    ENDFOREACH()
    IF(NOT EXISTS "${OPENMM_DIR}/include/openmm/${HEADER}" OR NOT library_found)
        SET(${RESULT} "OpenMM in ${OPENMM_DIR} was built without the ${PLATFORM} platform (include/openmm/${HEADER} and the ${LIBRARY} library in lib or lib/plugins are required)" PARENT_SCOPE)
    ELSE()
        SET(${RESULT} "" PARENT_SCOPE)
    ENDIF()
ENDFUNCTION()

FIND_PACKAGE(OpenCL QUIET)
LBM_OPENMM_PLATFORM_REASON(OpenCL opencl/OpenCLContext.h OpenMMOpenCL LBM_OPENCL_REASON)
IF(NOT LBM_OPENCL_REASON AND NOT (OpenCL_FOUND AND TARGET OpenCL::OpenCL))
    SET(LBM_OPENCL_REASON "OpenCL (headers and library) was not found on this system")
ENDIF()
LBM_OPTION(LBM_BUILD_OPENCL_LIB "Build implementation for OpenCL" "the OpenCL plugin of openmm-lbm" "${LBM_OPENCL_REASON}")
IF(LBM_BUILD_OPENCL_LIB)
    ADD_SUBDIRECTORY(platforms/opencl)
ENDIF(LBM_BUILD_OPENCL_LIB)

FIND_PACKAGE(CUDAToolkit QUIET)
LBM_OPENMM_PLATFORM_REASON(CUDA cuda/CudaContext.h OpenMMCUDA LBM_CUDA_REASON)
IF(NOT LBM_CUDA_REASON AND NOT (CUDAToolkit_FOUND AND TARGET CUDA::cuda_driver))
    SET(LBM_CUDA_REASON "the CUDA toolkit with the CUDA driver library was not found on this system")
ENDIF()
LBM_OPTION(LBM_BUILD_CUDA_LIB "Build implementation for CUDA" "the CUDA plugin of openmm-lbm" "${LBM_CUDA_REASON}")
IF(LBM_BUILD_CUDA_LIB)
    ADD_SUBDIRECTORY(platforms/cuda)
ENDIF(LBM_BUILD_CUDA_LIB)

LIST(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)
FIND_PACKAGE(HIP CONFIG QUIET)
LBM_OPENMM_PLATFORM_REASON(HIP hip/HipContext.h OpenMMHIP LBM_HIP_REASON)
IF(NOT LBM_HIP_REASON AND NOT (HIP_FOUND AND TARGET hip::host))
    SET(LBM_HIP_REASON "HIP was not found on this system")
ENDIF()
LBM_OPTION(LBM_BUILD_HIP_LIB "Build implementation for HIP" "the HIP plugin of openmm-lbm" "${LBM_HIP_REASON}")
IF(LBM_BUILD_HIP_LIB)
    ADD_SUBDIRECTORY(platforms/hip)
ENDIF(LBM_BUILD_HIP_LIB)

IF(LBM_BUILD_OPENCL_LIB OR LBM_BUILD_CUDA_LIB OR LBM_BUILD_HIP_LIB)
    ADD_SUBDIRECTORY(platforms/common)
ENDIF()

# Build the Python API

FIND_PROGRAM(PYTHON_EXECUTABLE python)
FIND_PROGRAM(SWIG_EXECUTABLE swig)
SET(LBM_PYTHON_REASON "")
IF(NOT PYTHON_EXECUTABLE)
    SET(LBM_PYTHON_REASON "python was not found")
ELSEIF(NOT SWIG_EXECUTABLE)
    SET(LBM_PYTHON_REASON "SWIG was not found")
ELSE()
    # make PythonInstall compiles and installs the module with pip and setuptools.
    FOREACH(module openmm numpy setuptools pip)
        IF(NOT LBM_PYTHON_REASON)
            EXECUTE_PROCESS(COMMAND "${PYTHON_EXECUTABLE}" -c "import ${module}"
                RESULT_VARIABLE LBM_PYTHON_IMPORT_RESULT OUTPUT_QUIET ERROR_QUIET)
            IF(NOT LBM_PYTHON_IMPORT_RESULT EQUAL 0)
                SET(LBM_PYTHON_REASON "${PYTHON_EXECUTABLE} cannot import the Python module ${module}")
            ENDIF()
        ENDIF()
    ENDFOREACH()
    # The wrapper imports the SWIG interface of OpenMM and includes the headers of its plugins.
    FOREACH(file swig/OpenMMSwigHeaders.i swig/typemaps.i OpenMMAmoeba.h OpenMMDrude.h openmm/RPMDIntegrator.h
            openmm/RPMDMonteCarloBarostat.h)
        IF(NOT LBM_PYTHON_REASON AND NOT EXISTS "${OPENMM_DIR}/include/${file}")
            SET(LBM_PYTHON_REASON "OpenMM in ${OPENMM_DIR} has no include/${file} (it was built without its Python wrapper or plugins)")
        ENDIF()
    ENDFOREACH()
ENDIF()
LBM_OPTION(LBM_BUILD_PYTHON_WRAPPERS "Build wrappers for Python" "the Python wrapper of openmm-lbm" "${LBM_PYTHON_REASON}")
IF(LBM_BUILD_PYTHON_WRAPPERS)
    ADD_SUBDIRECTORY(python)
ENDIF(LBM_BUILD_PYTHON_WRAPPERS)

# Summary of what is built.
SET(LBM_PLATFORMS_BUILT Reference)
FOREACH(platform CUDA OpenCL HIP)
    STRING(TOUPPER ${platform} upper)
    IF(LBM_BUILD_${upper}_LIB)
        LIST(APPEND LBM_PLATFORMS_BUILT ${platform})
    ENDIF()
ENDFOREACH()
STRING(REPLACE ";" ", " LBM_PLATFORMS_BUILT "${LBM_PLATFORMS_BUILT}")
IF(LBM_BUILD_PYTHON_WRAPPERS)
    SET(LBM_PYTHON_BUILT "yes")
ELSE()
    SET(LBM_PYTHON_BUILT "no")
ENDIF()
MESSAGE(STATUS "openmm-lbm ${OPENMM_LBM_VERSION}: platforms to build: ${LBM_PLATFORMS_BUILT}; Python wrapper: ${LBM_PYTHON_BUILT}")
