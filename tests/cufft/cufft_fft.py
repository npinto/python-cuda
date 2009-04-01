#!/usr/bin/env python
import sys
import numpy
from ctypes import *

from numpy import zeros,complex,allclose
from numpy.random import randn,random_integers
from numpy.fft import fft,ifft

from cuda.cuda import * 
from cuda.cufft import *

def get_cufft_signal(numpy_array):
    print "[*] Creating device signal..."
    size = numpy_array.shape[1]
    hsignal = (float2*size)()
    for i in range(size): 
        hsignal[i].x = numpy_array[0,i].real
        hsignal[i].y = numpy_array[0,i].imag
    dsignal = c_void_p()
    cudaMalloc(byref(dsignal), sizeof(hsignal))
    cudaMemcpy(dsignal, hsignal, sizeof(hsignal), cudaMemcpyHostToDevice)
    return cast(dsignal,POINTER(cufftComplex))
    #return dsignal

def get_data(device_ptr,size):
    data = (cufftComplex*size)()
    cudaMemcpy(data, device_ptr, sizeof(data), cudaMemcpyDeviceToHost)
    result = zeros((1,size)).astype(complex)
    for i in range(size):
        result[0,i] = complex(data[i].x,data[i].y)
    return result

def get_inverse_data(device_ptr,size):
    data = (cufftComplex*size)()
    cudaMemcpy(data, device_ptr, sizeof(data), cudaMemcpyDeviceToHost)
    result = zeros((1,size)).astype(complex)
    print "(*) Dividing by num of signal elements to get back original data"
    for i in range(size):
        result[0,i] = complex(data[i].x/float(size),data[i].y/float(size))
    return result

def python_cuda_fft(numpy_array, leave_on_device=False):
    dsignal = get_cufft_signal(numpy_array)
    size = numpy_array.shape[1]
    plan = get_1dplan(size)
    print "[*] Using the CUFFT plan to forward transform the signal in place..."
    print "(*) cufftExecC2C note: Identical pointers to input and output arrays "
    print "    implies in-place transformation"
    cufftExecC2C(plan, dsignal, dsignal, CUFFT_FORWARD)
    print "[*] Destroying CUFFT plan..."
    cufftDestroy(plan)
    if not leave_on_device:
        result = get_data(dsignal,size)
        cudaFree(dsignal)
        return result

def get_1dplan(size,batch=1):
    print "[*] Creating a 1D FFT plan..."
    plan = cufftHandle()
    cufftPlan1d(plan, size, CUFFT_C2C, batch)
    return plan

def python_cuda_ifft(numpy_array, leave_on_device=False):
    dsignal = get_cufft_signal(numpy_array)
    size = numpy_array.shape[1]
    plan = get_1dplan(size)
    print "[*] Using the CUFFT plan to inverse transform the signal in place..."
    cufftExecC2C(plan, dsignal, dsignal, CUFFT_INVERSE)
    print "[*] Destroying CUFFT plan..."
    cufftDestroy(plan)
    if not leave_on_device:
        result = get_inverse_data(dsignal,size)
        cudaFree(dsignal)
        return result
        
def main():
    print "-"*55
    print "--                                                   --"
    print "--    python-cuda versions of numpy.fft.{fft,ifft}   --"
    print "--                                                   --"
    print "-"*55
    print
    print ">>> Creating host signal..."

    try:
        size = int(sys.argv[1])
    except Exception,e:
        size = 10
    print "size = %s" % size
    numpy_array = randn(1,size).astype(complex)
    numpy_array -= numpy_array.mean()
    numpy_array /= numpy_array.std()
    print numpy_array.mean()
    print numpy_array.std()
    print numpy_array

    print
    print ">>> Forward fft on gpu ..."
    fft_res = python_cuda_fft(numpy_array)

    print
    print ">>> Inverse fft on gpu ..."
    ifft_res = python_cuda_ifft(fft_res) 

    print
    print ">>> Computing references with numpy..."
    print "[*] Forward fft"
    forward_ref = fft(numpy_array)
    print "[*] Inverse fft"
    inverse_ref = ifft(forward_ref)
    
    print "l2norm fft: ", numpy.linalg.norm(fft_res - forward_ref)

    print "l2norm ifft: ", numpy.linalg.norm(ifft_res - inverse_ref)

    print
    print ">>> Forward transform:"
    #print "[*] GPU: \n%s" % fft_res
    #print "[*] CPU: \n%s" % forward_ref
    print "(*) GPU/CPU close? ", allclose(fft_res,forward_ref)

    print
    print ">>> Inverse transform:"
    #print "[*] GPU: \n%s" % ifft_res
    #print "[*] CPU: \n%s" % inverse_ref
    print "(*) GPU/CPU close? ",allclose(ifft_res,inverse_ref)

if __name__ == "__main__":
    main()
