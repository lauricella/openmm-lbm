set(WRAP_FILE LBMPluginWrapper.cpp)
set(MODULE_NAME openmmlbm)

# Execute SWIG to generate source code for the Python module.

add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/openmmlbm.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/openmmlbm.i"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Compile the Python module.

add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}" "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
set(LBM_PLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(LBM_PLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
add_custom_command(TARGET PythonInstall POST_BUILD
    COMMAND "${PYTHON_EXECUTABLE}" -m pip install --no-deps --no-build-isolation .
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
