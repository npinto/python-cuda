#!/usr/bin/env python
from cuda.cuda import cudaThreadSynchronize
from cuda.cublas import cublasInit, cublasShutdown, cublasSgemm
from cuda.array import CublasArray

from numpy import empty_like,dot
from numpy.random import randn

# Size of square matrix
N = 1024

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
cublasSgemm( transa, transb, N, N, N, 1, dA.ref, N, dB.ref, N, 0, dC.ref, N )
cudaThreadSynchronize()

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
