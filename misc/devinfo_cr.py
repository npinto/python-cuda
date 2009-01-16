#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08

from ctypes import *
from cuda.cuda_api import *

if __name__ == "__main__":
    print "+------------------------+"
    print "| CUDA Device Info       |"
    print "| using CUDA runtime API |"
    print "+------------------------+\n"
    count = c_int()
    cudaGetDeviceCount(byref(count))
    print "number of devices  =", count.value
    props = cudaDeviceProp()
    cudaGetDeviceProperties(props, 0)
    print props
