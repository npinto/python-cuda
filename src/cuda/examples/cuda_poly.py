#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *
from time import time

from cuda.cuda_defs import *
from cuda.cuda_api import *
from cuda.cuda_utils import *

from cpuFunctions import vectorInit,checkError
from cpuFunctions import cpuPOLY5,cpuPOLY10,cpuPOLY20,cpuPOLY40
from gpuFunctions import gpuPOLY5,gpuPOLY10,gpuPOLY20,gpuPOLY40

BLOCK_SIZE = 144
GRID_SIZE  = 192
checkErrorFlag = False

S4 = sizeof(c_float)
psize = 5

def main(vlength = 128,loops = 1,m1 = 1):
    print "%5d %5d %5d" % (l,loops,m1),

    alfa = c_float(.5)
    n2 = vlength ## Vector length

    mp = 1 << (m1-1)
    print "%5d" % (mp*psize),
    gpuPOLY = eval("gpuPOLY%d"%(mp*psize))
    h_X = (c_float*n2)()
    h_Y = (c_float*n2)()
    g_Y = (c_float*n2)()

    vectorInit(h_X)

    d_X = getMemory(h_X)
    d_Y = getMemory(h_Y)

    blockDim  = dim3(BLOCK_SIZE,1,1)
    gridDim   = dim3(GRID_SIZE,1,1)

    t0 = time()
    cudaThreadSynchronize()
    for i in range(loops):
        cudaConfigureCall(gridDim,blockDim,0,0)
        gpuPOLY(d_X,d_Y,n2)
    cudaThreadSynchronize()
    t0 = time()-t0

    flops = (2.e-9*m1*n2*(psize-1))*float(loops)
    cudaMemcpy(g_Y,d_Y,S4*n2,cudaMemcpyDeviceToHost)
    cudaThreadSynchronize()

    cudaFree(d_X)
    cudaFree(d_Y)

    cudaThreadExit()
    cpuPOLY = eval("cpuPOLY%d" % (mp*psize))
    t1 = time()
    for i in range(loops):
        cpuPOLY(h_X,h_Y)
    t1 = time()-t1
    print "%10d%6.2f%6.2f" % (vlength,flops/t1,flops/t0)

    if checkErrorFlag:
        err,mxe = checkError(h_Y,g_Y)
        print "Avg and max rel error = %.2e %.2e" % (err,mxe)

if __name__ == "__main__":
    import sys

    cudaSetDevice(0)

    lmin,lmax = 7,23
    if len(sys.argv) > 1:
        lmin = lmax = int(sys.argv[1])
    loopx = -1
    if len(sys.argv) > 2:
        loopx = int(sys.argv[2])
    m1 = 4
    if len(sys.argv) > 3:
        m1 = min(4,int(sys.argv[3]))
    lmax = min(max(0,lmax),23)
    lmin = min(max(0,lmin),lmax)

    for l in range(lmin,lmax+1):
        if l < 10:
            loops = 10000/m1
        elif l < 13:
            loops = 5000/m1
        elif l < 17:
            loops = 500/m1
        elif l < 21:
            loops = 250/m1
        else:
            loops = 100/m1
        vlength = 1 << l
        if loopx > 0:
            loops = loopx
        main(vlength,loops,m1)
