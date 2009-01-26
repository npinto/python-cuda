#!/bin/env python

from time import time
from ctypes import c_float

from numpy import empty_like,dot
from numpy.random import randn

from cuda.cublas import *
from cuda.cuda import cudaThreadSynchronize
from cuda.array import CudaArrayFromArray

def cpuSaxpy(a,b, alpha):
    return (alpha*a+b)

vlength = 8192
alpha = 1

# init cublas lib
cublasInit()

# allocate host vectors
h_X = randn(1,vlength).astype('float32')
h_Y = randn(1,vlength).astype('float32')

# allocate device vectors from host
d_X = CudaArrayFromArray(h_X)
d_Y = CudaArrayFromArray(h_Y)

print "-"*80
print 'h_X:'
print h_X
print "-"*80

print "-"*80
print 'h_Y:'
print h_Y
print "-"*80

print "-"*80
print 'CPU RESULT:'
print cpuSaxpy(h_X,h_Y,alpha)
print "-"*80

# execute cublasSaxpy and sync threads
cublasSaxpy(vlength,alpha,d_X.data,1,d_Y.data,1)
cudaThreadSynchronize()

print "-"*80
print 'GPU RESULT:'
print d_Y.toArray()
print "-"*80
