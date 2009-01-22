#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *
from time import time

from cu.cu_defs import *
from cu.cu_api import *
from utils.cu_utils import *

from cpuFunctions import randInit,checkError

UseVML = False
if UseVML:
    from mklMath import cpuBLSC
else:
    from cpuFunctions import cpuBLSC

BLOCK_SIZE = 128
GRID_SIZE  = 192
checkErrorFlag = False

S4 = sizeof(c_float)

def main(device,vlength = 128,loops = 1):

    n2 = vlength ## Vector length

    gpuBLSC = device.functions["gpuBLSC"]

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

    cuFuncSetBlockShape(gpuBLSC,BLOCK_SIZE,1,1)
    cuParamSeti(gpuBLSC, 0,d_C)
    cuParamSeti(gpuBLSC, 4,d_P)
    cuParamSeti(gpuBLSC, 8,d_S)
    cuParamSeti(gpuBLSC,12,d_X)
    cuParamSeti(gpuBLSC,16,d_T)
    cuParamSetf(gpuBLSC,20,R)
    cuParamSetf(gpuBLSC,24,V)
    cuParamSeti(gpuBLSC,28,n2)
    cuParamSetSize(gpuBLSC,32)

    cuCtxSynchronize()
    t0 = time()
    for i in range(loops):
        cuLaunchGrid(gpuBLSC,GRID_SIZE,1)
    cuCtxSynchronize()
    t0 = time()-t0

    flops = (2.e-6*n2)*float(loops)
    g_C = (c_float*n2)()
    g_P = (c_float*n2)()
    cuMemcpyDtoH(g_C,d_C,n2*S4)
    cuMemcpyDtoH(g_P,d_P,n2*S4)
    cuCtxSynchronize()

    cuMemFree(d_S)
    cuMemFree(d_X)
    cuMemFree(d_T)
    cuMemFree(d_C)
    cuMemFree(d_P)

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

    device = cu_CUDA()
    device.getSourceModule("gpuFunctions.cubin")
    device.getFunction("gpuBLSC")

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
        main(device,vlength,loops)
    cuCtxDetach(device.context)
