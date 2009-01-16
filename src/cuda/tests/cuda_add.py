#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *
from time import time

from cuda.cuda_defs import *
from cuda.cuda_api import *
from cuda.cuda_utils import *

from cpuFunctions import fixedInit,cpuVADD,checkError
from gpuFunctions import gpuVADD

BLOCK_SIZE = 256
GRID_SIZE  = 256
S4 = sizeof(c_float)
checkErrorFlag = False

def main(vlength = 128,loops = 1):

    n2 = vlength ## Vector length

    h_X = (c_float*n2)()
    h_Y = (c_float*n2)()
    g_Y = (c_float*n2)()

    fixedInit(h_X)

    d_X = getMemory(h_X)
    d_Y = getMemory(h_Y)

    blockDim  = dim3(BLOCK_SIZE,1,1)
    gridDim   = dim3(GRID_SIZE,1,1)

    t0 = time()
    cudaThreadSynchronize()
    for i in range(loops):
        cudaConfigureCall(gridDim,blockDim,0,0)
        gpuVADD(d_X,d_Y,n2)
    cudaThreadSynchronize()
    t0 = time()-t0

    flops = (1.e-9*n2)*float(loops)
    g_Y = (c_float*n2)()
    cudaMemcpy(g_Y,d_Y,S4*n2,cudaMemcpyDeviceToHost)
    cudaThreadSynchronize()

    cudaFree(d_X)
    cudaFree(d_Y)

    cudaThreadExit()
    t1 = time()
    for i in range(loops):
        cpuVADD(h_X,h_Y)
    t1 = time()-t1
    print "%10d%6.2f%6.2f" % (vlength,flops/t1,flops/t0)

    if checkErrorFlag:
        err,mxe = checkError(h_Y,g_Y)
        print "Avg and max rel error = %.2e %.2e" % (err,mxe)

if __name__ == "__main__":
    import sys

    cudaSetDevice(0)

    lmin,lmax = 7,24
    if len(sys.argv) > 1:
        lmin = lmax = int(sys.argv[1])
    loopx = -1
    if len(sys.argv) > 2:
        loopx = int(sys.argv[2])
    lmax = min(max(0,lmax),24)
    lmin = min(max(0,lmin),lmax)
    for l in range(lmin,lmax+1):
        if l < 10:
            loops = 25000
        elif l < 17:
            loops = 10000
        elif l < 21:
            loops = 250
        else:
            loops = 25
        vlength = 1 << l
        if loopx > 0:
            loops = loopx
        print "%5d %5d" % (l,loops),
        main(vlength,loops)
