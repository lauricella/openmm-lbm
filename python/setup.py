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
lbm_plugin_library_dir = '@LBM_PLUGIN_LIBRARY_DIR@'

extra_compile_args = ['-std=c++17']
extra_link_args = []
runtime_library_dirs = [os.path.join(openmm_dir, 'lib')]

# setup extra compile and link arguments on Mac
if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.13']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.13']

extension = Extension(name='_openmmlbm',
                      sources=['LBMPluginWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMLBM'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), lbm_plugin_header_dir, numpy.get_include()],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), lbm_plugin_library_dir],
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
