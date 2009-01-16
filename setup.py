#!/usr/bin/python
# coding:utf-8: © Arno Pähler, 2007-08

from distutils.core import setup

setup(
    name='python-cuda',
    version='2.0_42',
    author='Arno Pähler',
    author_email='paehler@graviscom.de',
    url='ftp://ftp.graviscom.de/pub/python-cuda/',
    description='ctypes Python bindings for NVidia CUDA',
    long_description = """\
The ctypes Python bindings are created from the header files,
distributed with NVidia's CUDA SDK. They implement both the
driver and the runtime API as well as part of the BLAS library.
Some SDK examples have been ported and other test examples been
created and tested with GeForce 8500GT and GeForce 8600GTS cards
under CUDA versions 1.0 and 1.1.
The latest version have support for CUBLAS (except complex cases)
and CUFFT. Support for the CUDA Application Interface is still
incomplete (things like something<<<dim,dim>>> not supported).
Python bindings to FFTW 2.x are also supplied to use in testing
CUFFT examples.
""",
    download_url='ftp://ftp.graviscom.de/pub/python-cuda/',
    license='LGPL',
    package_dir = {'cuda':'src/cuda'},
    packages=['cuda','cuda.utils','cuda.examples'],
    package_data = {'cuda':['MANIFEST.in','README'],
                    'cuda.examples':
                    ['*.so','*.c','*.h','*.cu*','*.ptx','compile*']}
)
