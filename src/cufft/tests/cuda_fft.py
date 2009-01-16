#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08

from ctypes import *
from math import log
from time import time

from cuda.cufft_defs import *
from cuda.cufft_api import *
from cuda.cuda_utils import *

import xfft as xf
import gfft_cuda as gf
from cpuFunctions import arrayInit,checkError,scale
from cpuFunctions import ReadTimestampCounter

def main(check=False,doComplex=False,dims=(128,)):
    print "+------------------------+"
    print "| Fast Fourier Transform |"
    print "| using CUDA runtime API |"
    print "+------------------------+\n"
    dims = tuple(dims)
    ndim = len(dims)
    v = ("","NX = %d","NX = %d NY = %d","NX = %d NY = %d NZ = %d")
    SC = reduce(lambda x,y:x*y,dims)
    SR = reduce(lambda x,y:x*y,dims[:-1],1)
    SR *= 2*(dims[-1]/2+1)

    print v[ndim] % dims
    print "< doComplex: %s >\n" % doComplex

    rz = 1./float(SC)
    flops = 2.*5.*SC*log(SC)/log(2.)*1.e-9
    if doComplex:
        SC *= 2
    S4 = sizeof(c_float)

    if doComplex:
        sz = S4*(SC+SC)/(1024*1024)
    else:
        sz = S4*(SC+SR)/(1024*1024)

    h_A = (c_float*SC)()
    g_A = (c_float*SC)()
    arrayInit(h_A)

    d_A = getMemory(h_A)
    allocate = True

    if doComplex:
        d_B = getMemory(SC)
    elif allocate:
        d_B = getMemory(SR)

    if doComplex:
        plan = gf.makePlan(dims,CUFFT_C2C)
    else:
        plan1 = gf.makePlan(dims,CUFFT_R2C)
        plan2 = gf.makePlan(dims,CUFFT_C2R)

    t0 = time()
    x0 = ReadTimestampCounter()
    cudaThreadSynchronize()

    if doComplex:
        d_B = gf.ccfft(plan,d_A,None,d_B)
        d_A = gf.icfft(plan,d_B,None,d_A)
    else:
        if allocate:
            d_B = gf.rcfft(plan1,d_A,None,d_B)
            d_A = gf.crfft(plan2,d_B,None,d_A)
        else:
            d_B = gf.rcfft(plan1,d_A,SR)
            cuMemFree(d_A)
            d_A = gf.crfft(plan2,d_B,SR)

    cudaThreadSynchronize()
    t0 = time()-t0
    x1 = ReadTimestampCounter()
    fc = 1.e-3/2.8
    print "RDTSC: %.0f µs" % ((x1-x0)*fc)

    cudaMemcpy(g_A,d_A,S4*SC,cudaMemcpyDeviceToHost)

    cudaFree(d_A)
    cudaFree(d_B)

    if doComplex:
        cufftDestroy(plan)
    else:
        cufftDestroy(plan1)
        cufftDestroy(plan2)

    cudaThreadExit()
    scale(g_A,rz)

    print "\nProcessing time: %.3g sec" % t0
    print "Gigaflops GPU  : %.2f" % (flops/t0)
    gflops = (flops/t0,)

    print "\nError CPU initial vs GPU"
    err,mxe = checkError(h_A,g_A)
    stats = err,mxe
    print "Avg and max rel error = %.2e %.2e\n" % (err,mxe)

    if check:
        t1 = time()
        if doComplex:
            h_B = xf.ccfft(h_A,dims)
            h_B = xf.icfft(h_B,dims)
        else:
            h_B = xf.rcfft(h_A,dims)
            h_B = xf.crfft(h_B,dims)
        t1 = time()-t1
        print "Processing time: %.3g sec" % t1
        print "Gigaflops CPU  : %.2f" % (flops/t1)
        print "Speedup GPU/CPU: %.2f" % (t1/t0)

        print "\nError CPU final vs CPU initial"
        err,mxe = checkError(h_B,h_A)
        print "Avg and max rel error = %.2e %.2e" % (err,mxe)

        print "\nError CPU final vs GPU"
        err,mxe = checkError(h_B,g_A)
        print "Avg and max rel error = %.2e %.2e" % (err,mxe)
    f = (-1.,)
    if check:
        f = (t1/t0,)
    fmt = "\n## "+" ".join(len(dims)*["%3d"])+" : %.1f %.1f: %.2e %.2e"
    print fmt % (dims+gflops+f+stats)

if __name__ == "__main__":
    import sys

    cudaSetDevice(0)

    check = False
    doComplex = False
    dims = (256,128,256)
    if len(sys.argv) > 1:
        if sys.argv[1] == "-c":
            check = True
        elif sys.argv[1] == "-cx":
            check = True
            doComplex = True
        elif sys.argv[1] == "-x":
            doComplex = True
        else:
            xyz = sys.argv[1].split(",")
            dims = tuple(int(x) for x in xyz)
    if len(sys.argv) > 2:
        if sys.argv[2] == "-c":
            check = True
        elif sys.argv[2] == "-cx":
            check = True
            doComplex = True
        elif sys.argv[2] == "-x":
            doComplex = True
    main(check,doComplex,dims)
