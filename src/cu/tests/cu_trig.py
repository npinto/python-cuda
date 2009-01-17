#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *
from time import time

from cu.cu_defs import *
from cu.cu_api import *
from utils.cu_utils import *

from cpuFunctions import vectorInit,checkError

UseVML = True
if UseVML:
    from mklMath import cpuTRIG
else:
    from cpuFunctions import cpuTRIG

BLOCK_SIZE = 128
GRID_SIZE  = 192
checkErrorFlag = False

S4 = sizeof(c_float)

def main(device,vlength = 128,loops = 1):

    n2 = vlength ## Vector length
    gpuTRIG = device.functions["gpuTRIG"]

    h_X = (c_float*n2)()
    h_Y = (c_float*n2)()
    h_Z = (c_float*n2)()

    vectorInit(h_X)

    d_X = getMemory(h_X)
    d_Y = getMemory(h_Y)
    d_Z = getMemory(h_Z)

    cuFuncSetBlockShape(gpuTRIG,BLOCK_SIZE,1,1)
    cuParamSeti(gpuTRIG,0,d_Y)
    cuParamSeti(gpuTRIG,4,d_Z)
    cuParamSeti(gpuTRIG,8,d_X)
    cuParamSeti(gpuTRIG,12,n2)
    cuParamSetSize(gpuTRIG,16)

    cuCtxSynchronize()
    t0 = time()
    for i in range(loops):
        cuLaunchGrid(gpuTRIG,GRID_SIZE,1)
    cuCtxSynchronize()
    t0 = time()-t0

    flops = (8.e-9*n2)*float(loops)
    g_Y = (c_float*n2)()
    g_Z = (c_float*n2)()
    cuMemcpyDtoH(g_Y,d_Y,S4*n2)
    cuMemcpyDtoH(g_Z,d_Z,S4*n2)
    cuCtxSynchronize()

    cuMemFree(d_X)
    cuMemFree(d_Y)
    cuMemFree(d_Z)

    t1 = time()
    for i in range(loops):
        cpuTRIG(h_Y,h_Z,h_X)
    t1 = time()-t1
    print "%10d%6.2f%6.2f GFlops" % (vlength,flops/t1,flops/t0)

    if checkErrorFlag:
        err,mxe = checkError(h_Y,g_Y,n2)
        print "Avg and max rel error (cos) = %.2e %.2e" % (err,mxe)
        err,mxe = checkError(h_Z,g_Z,n2)
        print "Avg and max rel error (sin) = %.2e %.2e" % (err,mxe)

if __name__ == "__main__":
    import sys

    device = cu_CUDA()
    device.getSourceModule("gpuFunctions.cubin")
    device.getFunction("gpuTRIG")

    lmin,lmax = 7,23
    if len(sys.argv) > 1:
        lmin = lmax = int(sys.argv[1])
    lmax = min(max(0,lmax),23)
    lmin = min(max(0,lmin),lmax)
    for l in range(lmin,lmax+1):
        if l < 10:
            loops = 10000
        elif l < 13:
            loops = 2000
        elif l < 17:
            loops = 250
        elif l < 21:
            loops = 100
        else:
            loops = 50
        vlength = 1 << l
        print "%5d %5d" % (l,loops),
        main(device,vlength,loops)
    cuCtxDetach(device.context)
