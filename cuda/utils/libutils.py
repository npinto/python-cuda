#!/usr/bin/env python
from ctypes import CDLL
import platform

OSNAME = platform.system()

def get_lib(name, cdll_opts = None):
    libname = None
    if OSNAME == "Linux": 
        libname = "lib" + name + ".so"
    elif OSNAME == "Darwin": 
        libname = "lib" + name + ".dylib"
    elif OSNAME == "Windows": 
        libname = None
    if cdll_opts:
        lib = CDLL(libname, cdll_opts)
    else: 
        lib = CDLL(libname)
    return lib

# NP: do we need this?
# def get_cuda(cdll_opts = None):
#     """libcuda.so"""
#     return get_lib("cuda", cdll_opts)

# def get_cublas(cdll_opts = None):
#     """libcublas.so"""
#     return get_lib("cublas", cdll_opts)

# def get_cudart(cdll_opts = None):
#     "libcudart.so"
#     return get_lib("cudart", cdll_opts)

# def get_kernelGL(cdll_opts = None):
#     "libkernelGL.so"
#     return get_lib("kernelGL", cdll_opts)

# def get_cufft(cdll_opts = None):
#     "libcufft.so"
#     return get_lib("cufft", cdll_opts)

# def get_fftw(cdll_opts = None):
#     "libfftw.so"
#     return get_lib("fftw", cdll_opts)

# def get_rfftw(cdll_opts = None):
#     "librfftw.so"
#     return get_lib("rfftw", cdll_opts)

# def get_sfftw(cdll_opts = None):
#     "libsfftw.so"
#     return get_lib("sfftw", cdll_opts)

# def get_srfftw(cdll_opts = None):
#     "libsrfftw.so"
#     return get_lib("srfftw", cdll_opts)

# if __name__ == "__main__":
#     try:
#         print "Loading libcuda..."
#         get_cuda()
#         print "Loading libcublas..."
#         get_cublas()
#         print "Loading libcudart..."
#         get_cudart()
#         #print "Loading libkernelGL..."
#         #print get_kernelGL()
#         print "Loading libcufft..."
#         get_cufft()
#         print "Loading libfftw..."
#         get_fftw()
#         #print "Loading librfftw..."
#         #get_rfftw()
#         #print "Loading libsfftw..."
#         #get_sfftw()
#         #print "Loading librsfftw..."
#         #get_rsfftw()
#         print "Test PASSED"
#     except:
#         print "Test FAILED"
