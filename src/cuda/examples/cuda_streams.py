#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *

from cuda.cuda_defs import *
from cuda.cuda_api import *
from cuda.cuda_utils import *

from ctypes_array import *
from numpy import all,int32,zeros

from gpuFunctions import init_array

MB = 1024*1024
SI = sizeof(c_int)

def check_results(a,n,c):
    u = (c_int*n).from_address(a.value)
    a = convert(u,(n,))
    c = c.value
    return all(a==c)

def main():
    nstreams = 8
    nreps = 10
    n = 16*MB
    nbytes = n*SI

    count = c_int()
    cudaGetDeviceCount(byref(count))
    if count == 0:
        print "no CUDA capable device found"
        return

    props = cudaDeviceProp()
    cudaGetDeviceProperties(byref(props),0)
    if props.major == 1 and props.minor < 1:
        print "%s does not support streams" % props.name
        return

    u = zeros((n,),dtype=int32)+5
    x = convert(u)
    c = c_int(x[0])
    a = c_void_p()
    cudaMallocHost(byref(a),nbytes)

    d_a = getMemory(n)
    d_c = getMemory(x)

    streams = (cudaStream_t*nstreams)()
    for i in range(nstreams):
        stream = cudaStream_t()
        cudaStreamCreate(byref(stream))
        streams[i] = stream.value

    ev_start = cudaEvent_t()
    ev_stop = cudaEvent_t()
    cudaEventCreate(byref(ev_start))
    cudaEventCreate(byref(ev_stop))

    cudaEventRecord(ev_start,0)
    cudaMemcpyAsync(a,d_a,nbytes,cudaMemcpyDeviceToHost,streams[0])
    cudaEventRecord(ev_stop,0)
    cudaEventSynchronize(ev_stop)
    t_copy = c_float()
    cudaEventElapsedTime(byref(t_copy),ev_start,ev_stop)
    t_copy = t_copy.value

    threads=dim3(512,1,1)
    blocks=dim3(n/threads.x,1,1)
    cudaEventRecord(ev_start,0)
    cudaConfigureCall(blocks,threads,0,streams[0])
    init_array(d_a,d_c)
    cudaEventRecord(ev_stop,0)
    cudaEventSynchronize(ev_stop)
    t_kernel = c_float()
    cudaEventElapsedTime(byref(t_kernel),ev_start,ev_stop)
    t_kernel = t_kernel.value

    threads=dim3(512,1,1)
    blocks=dim3(n/threads.x,1,1)
    cudaEventRecord(ev_start,0)
    for i in range(nreps):
        cudaConfigureCall(blocks,threads,0,0)
        init_array(d_a,d_c)
        cudaMemcpy(a,d_a,nbytes,cudaMemcpyDeviceToHost)
    cudaEventRecord(ev_stop,0)
    cudaEventSynchronize(ev_stop)
    elapsed0 = c_float()
    cudaEventElapsedTime(byref(elapsed0),ev_start,ev_stop)
    elapsed0 = elapsed0.value

    threads = dim3(512,1,1)
    blocks = dim3(n/(nstreams*threads.x),1,1)
    memset(a,255,nbytes)
    cudaMemset(d_a,0,nbytes)
    cudaEventRecord(ev_start,0)
    a_0 = a.value
    off = n*SI/nstreams
    for k in range(nreps):
        for i in range(nstreams):
            cudaConfigureCall(blocks,threads,0,streams[i])
            init_array(d_a+i*n*SI/nstreams,d_c)
        for i in range(nstreams):
            ai = a_0+i*off
            di = d_c+i*off
            cudaMemcpyAsync(ai,di,nbytes/nstreams,
                cudaMemcpyDeviceToHost,streams[i])
    cudaEventRecord(ev_stop,0)
    cudaEventSynchronize(ev_stop)
    elapsed1 = c_float()
    cudaEventElapsedTime(byref(elapsed1),ev_start,ev_stop)
    elapsed1 = elapsed1.value

    passed = check_results(a,n,c)

    for i in range(nstreams):
        cudaStreamDestroy(streams[i])
    cudaEventDestroy(ev_start)
    cudaEventDestroy(ev_stop)

    cudaFree(d_a)
    cudaFree(d_c)
    cudaFreeHost(a)
    cudaThreadExit()

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
    cudaSetDevice(0)
    main()
