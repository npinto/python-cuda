#!/bin/env python

import numpy as np
from cuda.sugar.memory import Linear
import cuda.sugar.blas as blas 

class TestCublas:

    def embed_ipython():
        from IPython.Shell import IPShellEmbed
        ipshell = IPShellEmbed(user_ns = dict())
        ipshell()

    def cpu_saxpy(self, a, b, alpha):
        return (alpha*a+b)

    def test_saxpy(self):
        vlength = 8192
        alpha = 1
        a = np.random.randn(1,vlength).astype('float32')
        b = np.random.randn(1,vlength).astype('float32')
        cpu_result = self.cpu_saxpy(a,b,alpha)
        gpu_result = blas.gpu_saxpy(a,b,alpha)

        print cpu_result   
        print gpu_result

        assert np.allclose(cpu_result, gpu_result) == True

    def test_sdot(self):
        vlength = 1024
        n2 = vlength*vlength
        a = np.random.randn(1,n2).astype('float32')
        b = np.random.randn(1,n2).astype('float32')
        cpu_result = np.dot(a,b.transpose())[0][0]
        gpu_result = blas.gpu_sdot(a, b.transpose())

        print cpu_result
        print gpu_result

        assert np.allclose([cpu_result], [gpu_result], atol=1e-1) == True

    def test_sgemm(self):
        M=7; N=5; P=3;
        a = np.random.randn(M,N).astype('float32')
        b = np.random.randn(N,P).astype('float32')
        cpu_result = np.dot(a,b)
        gpu_result = blas.gpu_sgemm(a,b)
        print cpu_result
        print gpu_result
        assert np.allclose(cpu_result, gpu_result)
        
if __name__ == "__main__":
    test_cublas = TestCublas()
    test_cublas.test_saxpy()
    test_cublas.test_sdot()
    test_cublas.test_sgemm()
