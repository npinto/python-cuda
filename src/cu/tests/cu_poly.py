#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *
from time import time

from cu.cu_defs import *
from cu.cu_api import *
from utils.cu_utils import *

from cpuFunctions import cpuPOLY5,cpuPOLY10,cpuPOLY20,cpuPOLY40

BLOCK_SIZE = 144
GRID_SIZE  = 192
##BLOCK_SIZE = 320
##GRID_SIZE  = 8
checkErrorFlag = False

S4 = sizeof(c_float)
psize = 5

def main(device,vlength = 128,loops = 1,m1 = 1):
    print "%5d %5d %5d" % (l,loops,m1),

    alfa = c_float(.5)
    n2 = vlength ## Vector length

    mp = 1 << (m1-1)
    print "%5d" % (mp*psize),
    fcn = "gpuPOLY%d"%(mp*psize)
    gpuPOLY = device.functions[fcn]
    h_X = (c_float*n2)()
    h_Y = (c_float*n2)()
    g_Y = (c_float*n2)()

    vectorInit(h_X)

    d_X = getMemory(h_X)
    d_Y = getMemory(h_Y)

    cuFuncSetBlockShape(gpuPOLY,BLOCK_SIZE,1,1)
    cuParamSeti(gpuPOLY,0,d_X)
    cuParamSeti(gpuPOLY,4,d_Y)
    cuParamSeti(gpuPOLY,8,n2)
    cuParamSetSize(gpuPOLY,12)

    cuCtxSynchronize()
    cuLaunchGrid(gpuPOLY,GRID_SIZE,1)
    t0 = time()
    for i in range(loops):
        cuLaunchGrid(gpuPOLY,GRID_SIZE,1)
    cuCtxSynchronize()
    t0 = time()-t0

    flops = (2.e-9*m1*n2*(psize-1))*float(loops)
    cuMemcpyDtoH(g_Y,d_Y,n2*S4)
    cuCtxSynchronize()

    cuMemFree(d_X)
    cuMemFree(d_Y)

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

    mp = 1 << (m1-1)
    device = cu_CUDA()
    device.getSourceModule("gpuFunctions.cubin")
    fcn = "gpuPOLY%d"%(mp*psize)
    device.getFunction(fcn)

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
        main(device,vlength,loops,m1)
    cuCtxDetach(device.context)
