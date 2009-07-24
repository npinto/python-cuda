#!/usr/bin/env python
from cuda.cuda import cudaThreadSynchronize
from cuda.cublas import cublasInit, cublasShutdown, cublasSgemm
from cuda.sugar.memory import Linear

import numpy 
from numpy.random import randn

def gpu_sgemm(a,b, alpha=1):
    """ Single Precision Matrix Multiplication on GPU, expects two, two-dimensional numpy arrays as input. Arrays must be such that a.shape[1] == b.shape[0]. Optionally specify alpha for scalar multiplication"""
    # init cublas
    cublasInit()

    assert a.shape[1] == b.shape[0]

    c_shape = (a.shape[0], b.shape[1])
    # allocate device matrices from host
    dA = Linear(a.shape, order='F').from_numpy(a)
    dB = Linear(b.shape, order='F').from_numpy(b)
    dC = Linear(c_shape, order='F')

    # transpose a/b ? t = yes, n = no
    transa = 'n'
    transb = 'n'

    # compute with CUBLAS
    cublasSgemm( transa, transb, a.shape[0], b.shape[1], a.shape[1], alpha, dA.ref, a.shape[0], dB.ref, b.shape[0], 0, dC.ref, a.shape[0] )
    cudaThreadSynchronize()
    # shutdown
    cublasShutdown() 
    return dC.to_numpy()



def test():
    # Size of square matrix
    N = 2

    # allocate host matrices
    A = randn(3,N).astype('float32')
    B = randn(3,5).astype('float32')

    # compute the cpu reference
    ref = numpy.dot(A,B)

    print '-'*80
    print ref 
    print '-'*80

    print '-'*80
    print gpu_sgemm(A,B) 
    print '-'*80


if __name__ == "__main__":
    test()
