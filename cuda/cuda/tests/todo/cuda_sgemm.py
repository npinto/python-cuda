#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *
from time import time

from cuda.cuda_api import *
from cuda.cuda_defs import *
from cuda.cuda_utils import *

from cpuFunctions import arrayInit,cpuSGEMM,checkError
from gpuFunctions import gpuSGEMM
from ctypes_array import *

useSciPy = True
if useSciPy:
    from scipy.lib.blas.fblas import sgemm as _sgemm
    # C : A*B (on the GPU)
    # F : (A*B).T = B.T * A.T (scipy)
    def sgemm(z,x,y,m,n,k):
        nx = convert(x,(m,k)).T
        ny = convert(y,(k,n)).T
        nz = _sgemm(1.,ny,nx)
        convert(nz,out=z)
        return z
else:
    # C : A*B (on the CPU) (in C)
    sgemm = cpuSGEMM

BLOCK_SIZE  = 1 << 4
S4 = sizeof(c_float)

def main(N = 1024,L = 100):
    M = N
    K = N >> 1
    N = N << 1
    flops = (2.e-9*M*N)*float(K*L)
    print "M = %d, N = %d, K = %d, L = %d; GFlops = %.1f\n" % (M,N,K,L,flops)
    na,nb,nc,alfa,beta = M*K,K*N,M*N,1.,0.

    t0 = time()

    sizeA = M*K
    sizeB = K*N
    sizeC = M*N

    h_A = (c_float*sizeA)()
    h_B = (c_float*sizeB)()

    arrayInit(h_A)
    arrayInit(h_B)

    d_A = getMemory(h_A)
    d_B = getMemory(h_B)
    d_C = getMemory(sizeC)

    blockDim  = dim3(BLOCK_SIZE,BLOCK_SIZE,1)
    gridDim   = dim3(N/BLOCK_SIZE,M/BLOCK_SIZE,1)
    sharedMem = S4*2*BLOCK_SIZE*BLOCK_SIZE
    tt = t0 = time()-t0
    print "Overhead runtime API: %.3f sec\n" % t0

    t0 = time()
    cudaThreadSynchronize()
    for i in range(L):
        cudaConfigureCall(gridDim,blockDim,sharedMem,0)
        gpuSGEMM(d_C,d_A,d_B,K,N)
    cudaThreadSynchronize()
    t0 = time()-t0
    tt += t0

    h_C = (c_float*sizeC)()
    cudaMemcpy(h_C,d_C,S4*sizeC,cudaMemcpyDeviceToHost)

    cudaThreadSynchronize()

    cudaFree(d_A)
    cudaFree(d_B)
    cudaFree(d_C)

    cudaThreadExit()
    print "Processing time: %.3g (%.3g) sec" % (t0,tt)
    print "Gigaflops GPU: %.2f (%.2f)" % (flops/t0,flops/tt)

    ref = (c_float*sizeC)()

    t1 = time()
    for i in range(L):
        sgemm(ref,h_A,h_B,M,N,K)
    t1 = time()-t1
    print "\nProcessing time: %.3g sec" % t1
    print "Gigaflops CPU  : %.2f" % (flops/t1)
    print "Speedup GPU/CPU: %.2f" % (t1/t0)

    err,mxe = checkError(ref,h_C)
    print "\nAvg and max rel error = %.2e %.2e" % (err,mxe)

if __name__ == "__main__":
    import sys

    cudaSetDevice(0)

    M, L = 1024, 100
    if len(sys.argv) > 1:
        M = int(sys.argv[1])
    M = (M >> 5) << 5 # multiple of (2*BLOCK_SIZE)
    if len(sys.argv) > 2:
        L = int(sys.argv[2])

    print "+-----------------------+"
    print "| Matrix Multiplication |"
    print "|     using CUDA API    |"
    print "+-----------------------+\n"
    main(M,L)
