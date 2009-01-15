#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08

from ctypes import *
from cuda.cu_defs import CUdevprop
from cuda.cu_api import *

if __name__ == "__main__":
    print "+-----------------------+"
    print "| CUDA Device Info      |"
    print "| using CUDA driver API |"
    print "+-----------------------+\n"
    cuInit(0)
    count = c_int()
    cuDeviceGetCount(byref(count))
    device = CUdevice()
    name = (c_char*256)()
    cuDeviceGet(byref(device),0)
    cuDeviceGetName(name,256,device)
    memsize = c_uint()
    cuDeviceTotalMem(byref(memsize),device)
    major,minor = c_int(),c_int()
    cuDeviceComputeCapability(byref(major),byref(minor),device)
    props = CUdevprop()
    cuDeviceGetProperties(byref(props),device)

    cuContext = CUcontext()
    cuCtxCreate(byref(cuContext),0,device)
    free,total = c_uint(),c_uint()
    cuMemGetInfo(byref(free),byref(total))
    free = free.value
    cuCtxDetach(cuContext)

    print "%-19s = %d" % ("number of devices",count.value)
    print "%-19s = %s" % ("device name =",name.value)
    print "%-19s = %.f MB" % ("memory size",memsize.value/1024.**2)
    print "%-19s = %.f MB" % ("memory free",free/1024.**2)
    print "%-19s = %.f MHz" % ("clock rate",props.clockRate/1000.)
    print "%-19s = %d" % ("major",major.value)
    print "%-19s = %d" % ("minor",minor.value)
    print 21*"-"
    print props
