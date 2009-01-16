#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *

from cuda.cu_defs import *
from cuda.cu_api import *
from cuda.cu_utils import *

from ctypes_array import *
from numpy import all,int32,zeros

MB = 1024*1024
SI = sizeof(c_int)

def check_results(a,n,c):
    u = (c_int*n).from_address(a.value)
    a = convert(u,(n,))
    c = c.value
    return all(a==c)

def main(device):
    nstreams = 8
    nreps = 10
    n = 16*MB
    nbytes = n*SI

    count = c_int()
    cuDeviceGetCount(byref(count))
    if count == 0:
        print "no CUDA capable device found"
        return

    major = c_int()
    minor = c_int()
    cuDeviceComputeCapability(byref(major),byref(minor),device.device)
    if major.value == 1 and minor.value < 1:
        print "%s does not support streams" % props.name
        return

    init_array = device.functions["init_array"]
    u = zeros((n,),dtype=int32)+5
    x = convert(u)
    c = c_int(x[0])
    a = c_void_p()
    cuMemAllocHost(byref(a),nbytes)

    d_a = getMemory(n)
    d_c = getMemory(x)

    streams = (CUstream*nstreams)()
    for i in range(nstreams):
        stream = CUstream()
        cuStreamCreate(byref(stream),0)
        streams[i] = stream

    ev_start = CUevent()
    ev_stop = CUevent()
    cuEventCreate(byref(ev_start),0)
    cuEventCreate(byref(ev_stop),0)

    cuEventRecord(ev_start,streams[0])
    cuMemcpyDtoHAsync(a,d_a,nbytes,streams[0])
    cuEventRecord(ev_stop,streams[0])
    cuEventSynchronize(ev_stop)
    t_copy = c_float()
    cuEventElapsedTime(byref(t_copy),ev_start,ev_stop)
    t_copy = t_copy.value

    cuFuncSetBlockShape(init_array,512,1,1)
    cuParamSeti(init_array,0,d_a)
    cuParamSeti(init_array,4,d_c)
    cuParamSetSize(init_array,8)

    cuEventRecord(ev_start,streams[0])
    cuLaunchGrid(init_array,n/512,1)
    cuEventRecord(ev_stop,streams[0])
    cuEventSynchronize(ev_stop)
    t_kernel = c_float()
    cuEventElapsedTime(byref(t_kernel),ev_start,ev_stop)
    t_kernel = t_kernel.value

    cuFuncSetBlockShape(init_array,512,1,1)
    cuParamSeti(init_array,0,d_a)
    cuParamSeti(init_array,4,d_c)
    cuParamSetSize(init_array,8)

    cuEventRecord(ev_start,streams[0])
    for i in range(nreps):
        cuLaunchGrid(init_array,n/512,1)
        cuMemcpyDtoH(a,d_a,nbytes)
    cuEventRecord(ev_stop,streams[0])
    cuEventSynchronize(ev_stop)
    elapsed0 = c_float()
    cuEventElapsedTime(byref(elapsed0),ev_start,ev_stop)
    elapsed0 = elapsed0.value

    memset(a,255,nbytes)
    cuMemsetD32(d_a,0,n)
    cuEventRecord(ev_start,streams[0])
    a_0 = a.value
    off = n*SI/nstreams
    for k in range(nreps):
        for i in range(nstreams):
            d_ai = d_a+i*n*SI/nstreams
            cuParamSeti(init_array,0,d_ai)
            cuLaunchGridAsync(init_array,n/(nstreams*512),1,streams[i])
        for i in range(nstreams):
            ai = a_0+i*off
            di = d_c+i*off
            cuMemcpyDtoHAsync(ai,di,nbytes/nstreams,streams[i])
    cuEventRecord(ev_stop,streams[0])
    cuEventSynchronize(ev_stop)
    elapsed1 = c_float()
    cuEventElapsedTime(byref(elapsed1),ev_start,ev_stop)
    elapsed1 = elapsed1.value

    passed = check_results(a,n,c)

    for i in range(nstreams):
        cuStreamDestroy(streams[i])
    cuEventDestroy(ev_start)
    cuEventDestroy(ev_stop)

    cuMemFree(d_a)
    cuMemFree(d_c)
    cuMemFreeHost(a)

    print "memcopy:\t%.2f" % t_copy
    print "kernel:\t\t%.2f" % t_kernel
    print "non-streamed:\t%.2f (%.2f expected)" % (
        elapsed0/nreps,t_kernel+t_copy)
    print "%d streams:\t%.2f (%.2f expected)" % (
        nstreams,elapsed1/nreps,t_kernel+t_copy/nstreams)

    print "-------------------------------"
    if passed:
        print "Test PASSED"
    else:
        print "Test FAILED"

if __name__ == "__main__":
    device = cu_CUDA()
    device.getSourceModule("gpuFunctions.cubin")
    device.getFunction("init_array")
    main(device)
    cuCtxDetach(device.context)
