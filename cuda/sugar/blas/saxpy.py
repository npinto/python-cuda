#!/bin/env python
from time import time
from ctypes import cast,c_float, POINTER

from numpy import empty_like,dot
from numpy.random import randn

from cuda.cublas import *
from cuda.cuda import cudaThreadSynchronize
from cuda.sugar.memory import Linear

def embed_ipython():
    from IPython.Shell import IPShellEmbed
    ipshell = IPShellEmbed(user_ns = dict())
    ipshell()

def cpu_saxpy(a,b, alpha):
    return (alpha*a+b)

def gpu_saxpy(a,b,alpha):
    # init cublas lib
    cublasInit()

    # allocate device vectors from host
    d_X = Linear(a.shape).from_numpy(a)
    d_Y = Linear(b.shape).from_numpy(b)

    # execute cublasSaxpy and sync threads
    cublasSaxpy(a.shape[1],alpha,d_X.ref,1,d_Y.ref,1)
    cudaThreadSynchronize()

    return d_Y.to_numpy()

def test():
    vlength = 8192
    alpha = 1

    # allocate host vectors
    h_X = randn(1,vlength).astype('float32')
    h_Y = randn(1,vlength).astype('float32')

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
    print cpu_saxpy(h_X,h_Y,alpha)
    print "-"*80

    print "-"*80
    print 'GPU RESULT:'
    print gpu_saxpy(h_X, h_Y, alpha)
    print "-"*80

if __name__ == "__main__":
    test()
