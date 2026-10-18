# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
# Meson build for munkres (parallel to the setuptools build in pyproject.toml).
#   meson setup build && meson compile -C build          # native extension, in-tree check
#   tools/build_meson.sh                                  # wheel through meson-python
# The C extension is optional: without a C compiler (or with -Dnative=disabled) a pure-Python
# package is installed and everything still works.
project(
  'munkres',
  version: '3.0.0',  # must equal munkres.__version__ (tests/test_meson_build.py enforces it)
  license: 'Apache-2.0',
  meson_version: '>=1.1.0',
  default_options: ['c_std=c99', 'buildtype=release', 'optimization=3', 'b_ndebug=if-release', 'warning_level=2'],
)

have_c = add_languages('c', required: get_option('native'), native: false)
py = import('python').find_installation(pure: not have_c)

py_sources = [
  'src/munkres/__init__.py',
  'src/munkres/__main__.py',
  'src/munkres/_algebra.py',
  'src/munkres/_algebraic.py',
  'src/munkres/_analysis.py',
  'src/munkres/_api.py',
  'src/munkres/_batch.py',
  'src/munkres/_bounds.py',
  'src/munkres/_components.py',
  'src/munkres/_core.py',
  'src/munkres/_dense.py',
  'src/munkres/_dynamic.py',
  'src/munkres/_enterprise.py',
  'src/munkres/_extras.py',
  'src/munkres/_gibbs.py',
  'src/munkres/_landscape.py',
  'src/munkres/_landscape2.py',
  'src/munkres/_lazy.py',
  'src/munkres/_market.py',
  'src/munkres/_minimize.py',
  'src/munkres/_more.py',
  'src/munkres/_native.py',
  'src/munkres/_parametric.py',
  'src/munkres/_plan.py',
  'src/munkres/_popular.py',
  'src/munkres/_proof.py',
  'src/munkres/_qubo.py',
  'src/munkres/_rankmax.py',
  'src/munkres/_risk.py',
  'src/munkres/_robust.py',
  'src/munkres/_sensitivity.py',
  'src/munkres/_sparse.py',
  'src/munkres/_structure.py',
  'src/munkres/_structure2.py',
  'src/munkres/_temporal.py',
  'src/munkres/_tools.py',
  'src/munkres/_topology.py',
  'src/munkres/_trace.py',
  'src/munkres/_verify.py',
  'src/munkres/cli.py',
]
py.install_sources(py_sources, 'src/munkres/py.typed', subdir: 'munkres')

if have_c
  cc = meson.get_compiler('c')
  # Portable flags: no -march=native (wheels must run on any CPU) and no -ffast-math (IEEE NaN semantics kept).
  add_project_arguments(
    cc.get_supported_arguments('-fno-trapping-math', '-fno-math-errno', '-funroll-loops'),
    language: 'c',
  )
  py.extension_module(
    '_lapc',
    'src/munkres/_native/lapmodule.c',  # includes lap.c
    dependencies: py.dependency(),
    install: true,
    subdir: 'munkres',
  )
endif
