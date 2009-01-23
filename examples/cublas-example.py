#!/usr/bin/env python

import sys 
from cuda.cuda import *
from cuda.utils.cuda_utils import mallocHost
from cuda.utils.ctypes_array import convert
from cuda.cublas import *
from cuda.utils.cudaarray import CublasArray
from cuda.utils.cudaarray import CudaArrayFromArray
from numpy.random import randn
from numpy import empty_like,dot

# Size of square matrix
N = 4096

# init cublas
cublasInit()

# allocate host matrices
A = randn(N,N).astype('float32')
B = randn(N,N).astype('float32')
C = empty_like(A).astype('float32')

# allocate device matrices from host
dA = CublasArray(A)
dB = CublasArray(B)
dC = CublasArray(C)

# transpose a/b ? t = yes, n = no
transa = 'n'
transb = 'n'

# compute with CUBLAS
cublasSgemm( transa, transb, N, N, N, 1, dA.data, N, dB.data, N, 0, dC.data, N )

# retrieve results
C = dC.toArray()

print '-'*80
print C
print '-'*80

print '-'*80
print dot(A,B)
print '-'*80

# shutdown
cublasShutdown() 
