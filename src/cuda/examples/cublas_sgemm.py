#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08

from ctypes import *
from time import time

from cuda.cublas_api import *
from cuda.cublas_defs import *
from cuda.cuda_api import cudaThreadSynchronize

from cpuFunctions import arrayInit,cpuSGEMM,checkError
from ctypes_array import *

useSciPy = True
if useSciPy:
    from scipy.lib.blas.fblas import sgemm as _sgemm
    def sgemm(z,x,y,m,n,k):
        nx = convert(x,(m,k),"F")
        ny = convert(y,(k,n),"F")
        nz = _sgemm(1.,nx,ny)
        convert(nz,out=z)
        return z
else:
    # this will give incorrect results
    # cpuSGEMM expects row-major (C) order
    print "\n < CPU results will be wrong! >\n"
    sgemm = cpuSGEMM

def main(N = 1024,L = 100):
    M = N
    K = N >> 1
    N = N << 1
    flops = (2.e-9*M*N)*float(K*L)
    print "M = %d, N = %d, K = %d, L = %d; GFlops = %.1f\n" % (M,N,K,L,flops)
    na,nb,nc,alfa,beta = M*K,K*N,M*N,1.,0.

    h_A = (c_float*na)()
    h_B = (c_float*nb)()
    h_C = (c_float*nc)()
    g_C = (c_float*nc)()

    arrayInit(h_A,na)
    arrayInit(h_B,nb)
    arrayInit(h_C,nc)

    cublasInit()
    cublasFree(0)
    t0 = time()

    d_A = c_void_p()
    d_B = c_void_p()
    d_C = c_void_p()

    cublasAlloc(na, sizeof(c_float), byref(d_A))
    cublasAlloc(nb, sizeof(c_float), byref(d_B))
    cublasAlloc(nc, sizeof(c_float), byref(d_C))

    cublasSetVector(na, sizeof(c_float), h_A, 1, d_A, 1)
    cublasSetVector(nb, sizeof(c_float), h_B, 1, d_B, 1)
    cublasSetVector(nc, sizeof(c_float), h_C, 1, d_C, 1)
    tt = time()-t0
    print "Overhead CUBLAS: %.3f sec\n" % tt

    cublasSgemm('n', 'n', M, N, K, alfa, d_A, M, d_B, K, beta, d_C, M)
    t0 = time()
    for i in range(L):
        cublasSgemm('n', 'n', M, N, K, alfa, d_A, M, d_B, K, beta, d_C, M)
    cudaThreadSynchronize()
    t0 = time()-t0

    tt += t0

    print "Processing time: %.3g (%.3g) sec" % (t0,tt)
    print "Gigaflops GPU: %.2f (%.2f)" % (flops/t0,flops/tt)

    t1 = time()
    for i in range(L):
        sgemm(h_C,h_A,h_B,M,N,K)
    t1 = time()-t1
    print "\nProcessing time: %.3g sec" % t1
    print "Gigaflops CPU  : %.2f" % (flops/t1)
    print "Speedup GPU/CPU: %.2f" % (t1/t0)

    cublasGetVector(nc, sizeof(c_float), d_C, 1, g_C, 1)
    err,mxe = checkError(h_C,g_C,nc)
    print "\nAvg and max rel error = %.2e %.2e" % (err,mxe)

    cublasFree(d_A)
    cublasFree(d_B)
    cublasFree(d_C)

    cublasShutdown()

if __name__ == "__main__":
    import sys

    M, L = 1024, 100
    if len(sys.argv) > 1:
        M = int(sys.argv[1])
    if len(sys.argv) > 2:
        L = int(sys.argv[2])
    print "+-----------------------+"
    print "|   CUBLAS SGEMM Test   |"
    print "|   using CUBLAS API    |"
    print "+-----------------------+\n"
    main(M,L)
