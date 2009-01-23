#!/usr/bin/env python

import sys 
from cuda.cuda import *
from cuda.utils.cuda_utils import mallocHost
from cuda.utils.ctypes_array import convert
from cuda.cublas import *
from cuda.utils.cudaarray import CublasArray
from cuda.utils.cudaarray import CudaArrayFromArray
import numpy

# Size of square matrix
N = 64

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

cublas_result = numpy.empty(N*N)
print cublas_result
# bench square matrices
for i in range(2): 
    transa = 'N'
    transb = None

    # transb = i ? 'T' : 'N'
    if i == 0:
        transb = 'T'
    else: 
        transb = 'N'


    print "\ntesting sgemm( '%s', '%s', n, n, n, ... )" % (transa, transb) 
    nb = 64

    print "   n   CUBLAS,Gflop/s   we,Gflop/s   \"error\"\n"
    idim = 1 
    while (idim <= N/nb):
        dim = idim*nb

        # set up the parameters
        m = dim
        n = dim
        k = dim
        lda = dim
        ldb = dim
        ldc = dim
        alpha = 1
        beta = -1

        # compute with CUBLAS
        cublasSetMatrix( m, n, sizeof( c_float ), C.ctypes.data, ldc, dC.data, ldc ) 
        cublasSgemm( transa, transb, m, n, k, alpha, dA.data, lda, dB.data, ldb, beta, dC.data, ldc )
        cublasGetError()
        cublasGetMatrix( m, n, sizeof( c_float ), dC.data, ldc, cublas_result.ctypes.data, ldc )
        print cublas_result
        print numpy.dot(A,B)
        
        idim = int((idim+1))*1.1


# shutdown
dA.free()
dB.free()
dC.free()

cublasShutdown() 
