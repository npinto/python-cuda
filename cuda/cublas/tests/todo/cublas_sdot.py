#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08

from ctypes import *
from time import time

from cuda.cublas_api import *
from cuda.cublas_defs import *
from cuda.cuda_api import cudaThreadSynchronize

from cpuFunctions import vectorInit,cpuSDOT
from ctypes_array import *

useSciPy = True
if useSciPy:
    from scipy.lib.blas.fblas import sdot as _sdot

    def sdot(x,y):
        nx = convert(x)
        ny = convert(y)
        rv = _sdot(nx,ny)
        return c_float(rv)
else:
    sdot = cpuSDOT

def main(vlength = 128,loops = 1):
    print "+-----------------------+"
    print "|   CUBLAS SAXPY Test   |"
    print "|   using CUBLAS API    |"
    print "+-----------------------+\n"
    print "Parameters: %d %d\n" % (vlength,loops)
    runTest(vlength,loops)

def runTest(vlength = 128,loops = 1):
    n2 = vlength*vlength
    alfa = c_float(.5)

    cublasInit()
    cublasFree(0)

    h_X = (c_float*n2)()
    h_Y = (c_float*n2)()
    vectorInit(h_X)
    vectorInit(h_Y)

    d_X = c_void_p()
    d_Y = c_void_p()
    cublasAlloc(n2, sizeof(c_float), byref(d_X))
    cublasAlloc(n2, sizeof(c_float), byref(d_Y))

    cublasSetVector(n2, sizeof(c_float), h_X, 1, d_X, 1)
    cublasSetVector(n2, sizeof(c_float), h_Y, 1, d_Y, 1)

    flops = (2.e-9*n2)*float(loops)
    s0 = 0.
    t0 = time()
    for i in range(loops):
        s0 += cublasSdot(n2, d_X, 1, d_Y, 1)
    cudaThreadSynchronize()
    t0 = time()-t0

    print "Processing time: %.3g sec" % t0
    print "Gigaflops GPU: %.2f (%d)" % (flops/t0,n2)

    s1 = 0.
    t1 = time()
    for i in range(loops):
        s1 += cpuSDOT(h_X,h_Y)
    t1 = time()-t1
    print "\nProcessing time: %.3g sec" % t1
    print "Gigaflops CPU: %.2f" % (flops/t1)
    print "GPU vs. CPU  : %.2f" % (t1/t0)

    sx = max(1.e-7,max(abs(s0),abs(s1)))
    err = abs(s1-s0)/sx
    print "\nError = %.2e" % err

    cublasFree(d_X)
    cublasFree(d_Y)

    cublasShutdown()

if __name__ == "__main__":
    import sys

##  square root of vector length
    vlength,loops = 1024,10
    if len(sys.argv) > 1:
        vlength = int(sys.argv[1])
    if len(sys.argv) > 2:
        loops = int(sys.argv[2])
    main(vlength,loops)
