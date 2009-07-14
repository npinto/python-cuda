#!/bin/env python
from time import time
from ctypes import cast,c_float, POINTER

from numpy import empty_like,dot
from numpy.random import randn

from cuda.cublas import *
from cuda.cuda import cudaThreadSynchronize
from cuda.memory import Linear

class TestCublas:
    def embed_ipython():
        from IPython.Shell import IPShellEmbed
        ipshell = IPShellEmbed(user_ns = dict())
        ipshell()

    def cpuSaxpy(a,b, alpha):
        return (alpha*a+b)

    def test_saxpy(self):
        vlength = 8192
        alpha = 1

        # init cublas lib
        cublasInit()

        # allocate host vectors
        h_X = randn(1,vlength).astype('float32')
        h_Y = randn(1,vlength).astype('float32')

        # allocate device vectors from host
        d_X = Linear(h_X.shape).from_numpy(h_X)
        d_Y = Linear(h_Y.shape).from_numpy(h_Y)

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
        #embed_ipython()
        cublasSaxpy(vlength,alpha,d_X.ref,1,d_Y.ref,1)
        cudaThreadSynchronize()

        print "-"*80
        print 'GPU RESULT:'
        print d_Y.to_numpy()
        print "-"*80
