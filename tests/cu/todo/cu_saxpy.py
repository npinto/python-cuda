#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *
from time import time

from cu.cu_defs import *
from cu.cu_api import *
from utils.cu_utils import *

from cpuFunctions import fixedInit,cpuSAXPY,checkError

BLOCK_SIZE = 64
GRID_SIZE  = 256
S4 = sizeof(c_float)
checkErrorFlag = False

def main(device,vlength = 128,loops = 1):

    alfa = c_float(.5)
    n2 = vlength ## Vector length
    gpuSAXPY = device.functions["gpuSAXPY"]

    h_X = (c_float*n2)()
    h_Y = (c_float*n2)()
    g_Y = (c_float*n2)()

    fixedInit(h_X)

    d_X = getMemory(h_X)
    d_Y = getMemory(h_Y)

    cuFuncSetBlockShape(gpuSAXPY,BLOCK_SIZE,1,1)
    cuParamSetf(gpuSAXPY,0,alfa)
    cuParamSeti(gpuSAXPY,4,d_X)
    cuParamSeti(gpuSAXPY,8,d_Y)
    cuParamSeti(gpuSAXPY,12,n2)
    cuParamSetSize(gpuSAXPY,16)

    cuCtxSynchronize()
    t0 = time()
    for i in range(loops):
        cuLaunchGrid(gpuSAXPY,GRID_SIZE,1)
    cuCtxSynchronize()
    t0 = time()-t0

    flops = (2.e-9*n2)*float(loops)
    cuMemcpyDtoH(g_Y,d_Y,n2*S4)
    cuCtxSynchronize()

    cuMemFree(d_X)
    cuMemFree(d_Y)

    t1 = time()
    for i in range(loops):
        cpuSAXPY(alfa,h_X,h_Y)
    t1 = time()-t1
    print "%10d%6.2f%6.2f" % (vlength,flops/t1,flops/t0)

    if checkErrorFlag:
        err,mxe = checkError(h_Y,g_Y)
        print "Avg and max rel error = %.2e %.2e" % (err,mxe)

if __name__ == "__main__":
    import sys

    device = cu_CUDA()
    device.getSourceModule("gpuFunctions.cubin")
    device.getFunction("gpuSAXPY")

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
        main(device,vlength,loops)
    cuCtxDetach(device.context)
