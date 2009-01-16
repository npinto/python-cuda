#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *
from time import time

from cuda.cu_defs import *
from cuda.cu_api import *
from cuda.cu_utils import *

from cpuFunctions import cpuGFLOPS

BLOCK_SIZE_C = 192
ITERATIONS_C = 512

BLOCK_SIZE_G = 512
GRID_SIZE_G  = 512
ITERATIONS_G = 512

def main(device,loops = 1):

    gpuGFLOPS = device.functions["gpuGFLOPS"]

    cuFuncSetBlockShape(gpuGFLOPS,BLOCK_SIZE_G,1,1)

    t0 = time()
    for i in range(loops):
        cuCtxSynchronize()
        cuLaunchGrid(gpuGFLOPS,GRID_SIZE_G,1)
        cuCtxSynchronize()
    t0 = time()-t0

    flopsc = 4096.*ITERATIONS_C*BLOCK_SIZE_C
    flopsg = 4096.*ITERATIONS_G*BLOCK_SIZE_G*GRID_SIZE_G

    flopsc *= 1.e-9*float(loops)
    flopsg *= 1.e-9*float(loops)

    t1 = time()
    for i in range(loops):
        cpuGFLOPS()
    t1 = time()-t1
#    peakg = 4.*8.*2.*1.458 # 4MP*8SP/MP*2flops/SP/clock*clock[GHz] (8600GTS)
    peakg = 14.*8.*2.*1.512 # 14MP*8SP/MP*2flops/SP/clock*clock[GHz] (9800GT)
    print "%8.3f%8.2f%8.3f%8.2f [%.2f]" % (
        t1,flopsc/t1,t0,flopsg/t0,peakg)

if __name__ == "__main__":
    import sys

    device = cu_CUDA()
    device.getSourceModule("gpuFunctions.cubin")
    device.getFunction("gpuGFLOPS")

    loops = 1
    if len(sys.argv) > 1:
        loops = int(sys.argv[1])
    print "%5d" % (loops),
    main(device,loops)
    cuCtxDetach(device.context)
