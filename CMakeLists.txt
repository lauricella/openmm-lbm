#---------------------------------------------------
# openmm-lbm: lattice Boltzmann fluid plugin for OpenMM
#----------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

PROJECT(OpenMMLBM)

# The version number is set here and copied to anywhere it is needed.
SET(OPENMM_LBM_VERSION 0.1.0 CACHE STRING "The version of openmm-lbm that will be built." FORCE)

# We need to know where OpenMM is installed so we can access the headers and libraries.
SET(OPENMM_DIR "/usr/local/openmm" CACHE PATH "Where OpenMM is installed")
INCLUDE_DIRECTORIES("${OPENMM_DIR}/include")
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

# Supported OpenMM versions: 8.3 to 8.6 (see CONTRIBUTING.md). The minimum is 8.3 because the
# per-cell reduction of the coupling forces sorts on the device with ComputeSort, which entered
# the OpenMM common compute layer in 8.3.0.
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

FIND_PACKAGE(OpenCL QUIET)
IF(OpenCL_FOUND)
    SET(LBM_BUILD_OPENCL_LIB ON CACHE BOOL "Build implementation for OpenCL")
ELSE(OpenCL_FOUND)
    SET(LBM_BUILD_OPENCL_LIB OFF CACHE BOOL "Build implementation for OpenCL")
ENDIF(OpenCL_FOUND)
IF(LBM_BUILD_OPENCL_LIB)
    ADD_SUBDIRECTORY(platforms/opencl)
ENDIF(LBM_BUILD_OPENCL_LIB)

FIND_PACKAGE(CUDAToolkit QUIET)
IF(CUDAToolkit_FOUND)
    SET(LBM_BUILD_CUDA_LIB ON CACHE BOOL "Build implementation for CUDA")
ELSE(CUDAToolkit_FOUND)
    SET(LBM_BUILD_CUDA_LIB OFF CACHE BOOL "Build implementation for CUDA")
ENDIF(CUDAToolkit_FOUND)
IF(LBM_BUILD_CUDA_LIB)
    ADD_SUBDIRECTORY(platforms/cuda)
ENDIF(LBM_BUILD_CUDA_LIB)

LIST(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)
FIND_PACKAGE(HIP CONFIG QUIET)
IF(HIP_FOUND)
    SET(LBM_BUILD_HIP_LIB ON CACHE BOOL "Build implementation for HIP")
ELSE(HIP_FOUND)
    SET(LBM_BUILD_HIP_LIB OFF CACHE BOOL "Build implementation for HIP")
ENDIF(HIP_FOUND)
IF(LBM_BUILD_HIP_LIB)
    ADD_SUBDIRECTORY(platforms/hip)
ENDIF(LBM_BUILD_HIP_LIB)

IF(LBM_BUILD_OPENCL_LIB OR LBM_BUILD_CUDA_LIB OR LBM_BUILD_HIP_LIB)
    ADD_SUBDIRECTORY(platforms/common)
ENDIF()

# Build the Python API

FIND_PROGRAM(PYTHON_EXECUTABLE python)
FIND_PROGRAM(SWIG_EXECUTABLE swig)
IF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(LBM_BUILD_PYTHON_WRAPPERS ON CACHE BOOL "Build wrappers for Python")
ELSE(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(LBM_BUILD_PYTHON_WRAPPERS OFF CACHE BOOL "Build wrappers for Python")
ENDIF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
IF(LBM_BUILD_PYTHON_WRAPPERS)
    ADD_SUBDIRECTORY(python)
ENDIF(LBM_BUILD_PYTHON_WRAPPERS)
