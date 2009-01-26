#!/bin/env python

from ctypes import *
from time import time

from cuda.cublas import *
from cuda.array import CudaArrayFromArray
from cuda.cuda import cudaThreadSynchronize

from numpy import dot
from numpy.random import randn

vlength = 1024

n2 = vlength*vlength
alpha = c_float(.5)

cublasInit()
cublasFree(0)

h_X = randn(1,n2).astype('float32')
h_Y = randn(1,n2).astype('float32')

d_X = CudaArrayFromArray(h_X)
d_Y = CudaArrayFromArray(h_Y)

gpu_result = cublasSdot(n2, d_X.ref, 1, d_Y.ref, 1)
cudaThreadSynchronize()

print "-"*80
print "h_X:"
print h_X
print "-"*80

print "-"*80
print "h_Y:"
print h_Y
print "-"*80

print "-"*80
print "numpy.dot(h_X,h_Y):"
print dot(h_X,h_Y.transpose())[0][0]
print "-"*80

print "-"*80
print "cublasSdot(d_X,d_Y):"
print gpu_result
print "-"*80

cublasShutdown()
