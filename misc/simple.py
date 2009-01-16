#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *
from time import time

from cuda.cu_defs import *
from cuda.cu_api import *
from cuda.cu_utils import *

from cpuFunctions import checkError,checkTrig,vectorInit

UseVML = True
if UseVML:
    from mklMath import cpuTRIG
else:
    from cpuFunctions import cpuTRIG

BLOCK_SIZE = 320
GRID_SIZE  = 8

S4 = sizeof(c_float)

def main(device,vlength = 128,loops = 1):
    print "+-----------------------+"
    print "|   Simple  TRIG Test   |"
    print "| using CUDA driver API |"
    print "+-----------------------+"
    print "params: %2d %5dK %3d\n" % (log2n,vlength >> 10,loops),

    n2 = vlength ## Vector length

    # TRIGTex is about 1.5x faster than TRIG
#    name = "TRIG"
    name = "TRIGTex"

    TRIG = device.functions[name]
    mod0 = device.modules[0]

    sizeV = S4*n2
    h_Arg = (c_float*n2)()
    h_Cos = (c_float*n2)()
    h_Sin = (c_float*n2)()

    vectorInit(h_Arg)

    d_Arg = getMemory(h_Arg)
    d_Cos = getMemory(n2)
    d_Sin = getMemory(n2)

    tex = devMemToTex(mod0,"Arg",d_Arg,sizeV)

    cuFuncSetBlockShape(TRIG,BLOCK_SIZE,1,1)
    cuParamSeti(TRIG,0,d_Cos)
    cuParamSeti(TRIG,4,d_Sin)
    if name != "TRIGTex":
        cuParamSeti(TRIG,8,d_Arg)
        cuParamSeti(TRIG,12,n2)
        cuParamSetSize(TRIG,16)
    else:
        cuParamSetTexRef(TRIG,CU_PARAM_TR_DEFAULT,tex)
        cuParamSeti(TRIG,8,n2)
        cuParamSetSize(TRIG,12)
    cuCtxSynchronize()

    t0 = time()
    for i in range(loops):
        cuLaunchGrid(TRIG,GRID_SIZE,1)
    cuCtxSynchronize()
    t0 = time()-t0

    g_Cos = (c_float*n2)()
    g_Sin = (c_float*n2)()
    cuMemcpyDtoH(g_Cos,d_Cos,sizeV)
    cuMemcpyDtoH(g_Sin,d_Sin,sizeV)
    cuCtxSynchronize()

    cuMemFree(d_Arg)
    cuMemFree(d_Cos)
    cuMemFree(d_Sin)

    t1 = time()
    for i in range(loops):
        cpuTRIG(h_Cos,h_Sin,h_Arg)
    t1 = time()-t1

    flopsg = (2.e-6*n2)*float(loops)
    flopsc = flopsg

    t0 *= 1.e3;
    t1 *= 1.e3;
    print "\n       time[msec]    GFlops\n"
    print "GPU: %12.1f%10.2f" % (t0,flopsg/t0)
    print "CPU: %12.1f%10.2f" % (t1,flopsc/t1)
    print "     %12.1f" % (t1/t0)

    x = float(1 << 23)
    e,m = checkTrig(g_Cos,g_Sin)
    print "\n",name, "internal check GPU"
    print "%8.1e %8.1e" % (e,m)
    print "%8.1f %8.1f" % (e*x,m*x)

    e,m = checkTrig(h_Cos,h_Sin)
    print "\n",name, "internal check CPU"
    print "%8.1e %8.1e" % (e,m)
    print "%8.1f %8.1f" % (e*x,m*x)

    print "\n","check between CPU and GPU"
    err,mxe = checkError(h_Cos,g_Cos)
    print "Avg and max abs error (cos) = %8.1e %8.1e" % (err,mxe)
    print "                              %8.1f %8.1f" % (err*x,mxe*x)
    err,mxe = checkError(h_Sin,g_Sin)
    print "Avg and max abs error (sin) = %8.1e %8.1e" % (err,mxe)
    print "                              %8.1f %8.1f" % (err*x,mxe*x)

if __name__ == "__main__":
    import sys
    device = cu_CUDA()
    device.getSourceModule("simple.cubin")
    device.getFunction("TRIG")
    device.getFunction("TRIGTex")

    log2n,loops = 15,1
    if len(sys.argv) > 1:
        log2n = int(sys.argv[1])
    log2n = max(0,min(log2n,25))
    if len(sys.argv) > 2:
        loops = int(sys.argv[2])
    vlength = 1 << log2n
    main(device,vlength,loops)
    cuCtxDetach(device.context)
