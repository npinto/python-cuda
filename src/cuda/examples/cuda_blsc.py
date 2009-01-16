#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *
from time import time

from cuda.cuda_defs import *
from cuda.cuda_api import *
from cuda.cuda_utils import *

from cpuFunctions import randInit,checkError
from gpuFunctions import gpuBLSC

UseVML = False
if UseVML:
    from mklMath import cpuBLSC
else:
    from cpuFunctions import cpuBLSC

BLOCK_SIZE = 128
GRID_SIZE  = 192
checkErrorFlag = False

S4 = sizeof(c_float)

def main(vlength = 128,loops = 1):

    n2 = vlength ## Vector length

    h_S = (c_float*n2)()
    h_X = (c_float*n2)()
    h_T = (c_float*n2)()
    h_C = (c_float*n2)()
    h_P = (c_float*n2)()


    randInit(h_S,5.,30.)
    randInit(h_X,1.,100.)
    randInit(h_T,.25,10.)
    R,V = .03,.3

    d_S = getMemory(h_S)
    d_X = getMemory(h_X)
    d_T = getMemory(h_T)
    d_C = getMemory(h_C)
    d_P = getMemory(h_P)

    blockDim  = dim3(BLOCK_SIZE,1,1)
    gridDim   = dim3(GRID_SIZE,1,1)

    cudaThreadSynchronize()
    t0 = time()
    for i in range(loops):
        cudaConfigureCall(gridDim,blockDim,0,0)
        gpuBLSC(d_C,d_P,d_S,d_X,d_T,R,V,n2)
    cudaThreadSynchronize()
    t0 = time()-t0

    flops = (2.e-6*n2)*float(loops)
    g_C = (c_float*n2)()
    g_P = (c_float*n2)()
    cudaMemcpy(g_C,d_C,S4*n2,cudaMemcpyDeviceToHost)
    cudaMemcpy(g_P,d_P,S4*n2,cudaMemcpyDeviceToHost)
    cudaThreadSynchronize()

    cudaFree(d_S)
    cudaFree(d_X)
    cudaFree(d_T)
    cudaFree(d_C)
    cudaFree(d_P)

    cudaThreadExit()
    t1 = time()
    for i in range(loops):
        cpuBLSC(h_C,h_P,h_S,h_X,h_T,R,V,n2)
    t1 = time()-t1
    print "%10d%10.2f%10.2f" % (vlength,flops/t1,flops/t0)

    if checkErrorFlag:
        err,mxe = checkError(h_C,g_C)
        print "Avg rel error (call) = %.2e" % (err,)
        err,mxe = checkError(h_P,g_P)
        print "Avg rel error (put)  = %.2e" % (err,)

if __name__ == "__main__":
    import sys

    cudaSetDevice(0)

    lmin,lmax = 7,23
    if len(sys.argv) > 1:
        lmin = lmax = int(sys.argv[1])
    lmax = min(max(0,lmax),23)
    lmin = min(max(0,lmin),lmax)
    for l in range(lmin,lmax+1):
        if l < 10:
            loops = 1000
        elif l < 13:
            loops = 500
        elif l < 17:
            loops = 100
        elif l < 21:
            loops = 10
        else:
            loops = 5
        loops = 2
        vlength = 1 << l
        print "%5d %5d" % (l,loops),
        main(vlength,loops)
