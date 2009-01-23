#!/usr/bin/env python

import sys 
from cuda.cuda import *
from cuda.utils.cuda_utils import mallocHost
from cuda.utils.ctypes_array import convert
from cuda.cublas import *
from cuda.utils.cudaarray import CublasArray
from cuda.utils.cudaarray import CudaArrayFromArray
import numpy

def contig(array):
    return numpy.ascontiguousarray(array, array.dtype)

# Size of square matrix
N = 4096

# init cublas and arrays
cublasInit()

# allocate host matrices
A = numpy.random.randn(N,N).astype(numpy.float32)
B = numpy.random.randn(N,N).astype(numpy.float32)
C = numpy.random.randn(N,N).astype(numpy.float32)

# allocate device matrices from host
dA = CublasArray(A)
dB = CublasArray(B)
dC = CublasArray(C)
print dA.toArray()

cublas_result = numpy.empty(N*N)
print cublas_result

transa = 'N'
transb = 'N'

print "\ntesting sgemm( '%s', '%s', n, n, n, ... )" % (transa, transb) 

# compute with CUBLAS
cublasSetMatrix( N , N, sizeof( c_float ), contig(C).ctypes.data, N, dC.data, N ) 
cublasSgemm( transa, transb, N, N, N, 1, dA.data, N, dB.data, N, -1, dC.data, N )
print cublasGetError()
cublasGetMatrix( N, N, sizeof( c_float ), dC.data, N, cublas_result.ctypes.data, N )

print cublas_result
print numpy.dot(A,B)

# shutdown
dA.free()
dB.free()
dC.free()

cublasShutdown() 
