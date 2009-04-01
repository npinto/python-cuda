#!/usr/bin/env python
from cuda.cuda import cudaThreadSynchronize
from cuda.cublas import cublasInit, cublasShutdown, cublasSgemm
from cuda.memory import Linear

from numpy import empty_like,dot
from numpy.random import randn

# Size of square matrix
N = 2

# init cublas
cublasInit()

# allocate host matrices
A = randn(N,N).astype('float32')
B = randn(N,N).astype('float32')
C = empty_like(A).astype('float32')

# allocate device matrices from host
dA = Linear(A.shape, order='F').from_numpy(A)

dB = Linear(B.shape, order='F').from_numpy(B)

dC = Linear(C.shape, order='F').from_numpy(C)

# transpose a/b ? t = yes, n = no
transa = 'n'
transb = 'n'

# compute with CUBLAS
cublasSgemm( transa, transb, N, N, N, 1, dA.ref, N, dB.ref, N, 0, dC.ref, N )
cudaThreadSynchronize()

# retrieve results
C = dC.to_numpy()

# compute the cpu reference
ref = dot(A,B)

print '-'*80
print C 
print '-'*80

print '-'*80
print ref 
print '-'*80

# shutdown
cublasShutdown() 
