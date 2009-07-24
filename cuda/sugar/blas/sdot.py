#!/bin/env python

from ctypes import c_float
from time import time

import cuda.cublas as cublas
import cuda.cuda as cuda
from cuda.sugar.memory import Linear

import numpy
from numpy.random import randn

def gpu_sdot(a,b):
    assert a.size == b.size
    assert a.shape[0] == b.shape[1]
    cublas.cublasInit()
    cublas.cublasFree(0)
    d_X = Linear(a.shape).from_numpy(a)
    d_Y = Linear(b.shape).from_numpy(b)
    gpu_result = cublas.cublasSdot(a.shape[1], d_X.ref, 1, d_Y.ref, 1)
    cuda.cudaThreadSynchronize()
    cublas.cublasShutdown()
    return gpu_result

def test():
    vlength = 1024

    n2 = vlength*vlength

    h_X = randn(1,n2).astype('float32')
    h_Y = randn(1,n2).astype('float32')

    print "-"*80
    print "h_X:"
    print h_X
    print "-"*80

    print "-"*80
    print "h_Y:"
    print h_Y
    print "-"*80

    print "-"*80
    print numpy.dot(h_X,h_Y.transpose())[0][0]
    print "-"*80

    print "-"*80
    print "cublasSdot(d_X,d_Y):"
    print gpu_sdot(h_X, h_Y.transpose())
    print "-"*80

if __name__ == "__main__":
    test()
