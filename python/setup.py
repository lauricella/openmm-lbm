# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# Derived from the OpenMM example plugin (openmm/openmmexampleplugin),
# portions copyright (c) 2014 Stanford University and the Authors.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

from setuptools import setup, Extension
import numpy
import os
import platform

version = '@OPENMM_LBM_VERSION@'
openmm_dir = '@OPENMM_DIR@'
lbm_plugin_header_dir = '@LBM_PLUGIN_HEADER_DIR@'
# The libraries are linked by full path: with -l the linker would take the first library of that name in its
# search path, where the linker flags of an active conda environment put the environment, which may hold
# another OpenMM or an older openmm-lbm (docs/user_guide/installation.md).
openmm_library = '@OPENMM_LIBRARY@'
lbm_plugin_library = '@LBM_PLUGIN_LIBRARY@'

extra_compile_args = ['-std=c++17']
extra_link_args = []
runtime_library_dirs = [os.path.join(openmm_dir, 'lib')]

# setup extra compile and link arguments on Mac
if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.13']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.13']

extension = Extension(name='_openmmlbm',
                      sources=['LBMPluginWrapper.cpp'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), lbm_plugin_header_dir, numpy.get_include()],
                      extra_objects=[openmm_library, lbm_plugin_library],
                      runtime_library_dirs=runtime_library_dirs,
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='openmmlbm',
      version=version,
      py_modules=['openmmlbm'],
      ext_modules=[extension],
      install_requires=['openmm', 'numpy']
     )
