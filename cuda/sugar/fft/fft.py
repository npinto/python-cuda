#!/usr/bin/env python
import sys
import numpy
from ctypes import *

from numpy import zeros,complex64, allclose
from numpy.random import randn,random_integers
import numpy.fft

from cuda.cuda import * 
from cuda.cufft import *

def _get_cufft_signal(numpy_array):
    dsignal = c_void_p()
    cudaMalloc(byref(dsignal), numpy_array.nbytes)
    cudaMemcpy(dsignal, numpy_array.ctypes.data, numpy_array.nbytes, cudaMemcpyHostToDevice)
    return cast(dsignal,POINTER(cufftComplex))

def _get_plan(shape):
    ndims = len(shape)
    if ndims == 1:
        return _get_1dplan(shape)
    elif ndims == 2:
        return _get_2dplan(shape)
    elif ndims == 3:
        return _get_3dplan(shape)
    else:
        print '_get_plan: invalid size (todo: throw exception)'
        
def _get_1dplan(shape,batch=1):
    #print "[*] Creating a 1D FFT plan..."
    plan = cufftHandle()
    cufftPlan1d(plan, shape[0], CUFFT_C2C, batch)
    return plan

def _get_2dplan(shape):
    #print "[*] Creating a 2D FFT plan..."
    plan = cufftHandle()
    cufftPlan2d(plan, shape[0], shape[1], CUFFT_C2C)
    return plan

def _get_3dplan(shape):
    #print "[*] Creating a 3D FFT plan..."
    plan = cufftHandle()
    cufftPlan3d(plan, shape[0], shape[1], shape[2], CUFFT_C2C)
    return plan

def _get_data(device_ptr,numpy_array):
    result = numpy.empty_like(numpy_array)
    cudaMemcpy(result.ctypes.data, device_ptr, numpy_array.nbytes, cudaMemcpyDeviceToHost)
    return result

def _get_inverse_data(device_ptr,numpy_array):
    result = _get_data(device_ptr, numpy_array)
    return result/float(numpy_array.size)

def _cuda_fft(numpy_array, leave_on_device=False):
    dsignal = _get_cufft_signal(numpy_array)
    plan = _get_plan(numpy_array.shape)
    #print "[*] Using the CUFFT plan to forward transform the signal in place..."
    #print "(*) cufftExecC2C note: Identical pointers to input and output arrays "
    #print "    implies in-place transformation"
    cufftExecC2C(plan, dsignal, dsignal, CUFFT_FORWARD)
    #print "[*] Destroying CUFFT plan..."
    cufftDestroy(plan)
    if not leave_on_device:
        result = _get_data(dsignal, numpy_array)
        #result = result.reshape(numpy_array.shape)
        cudaFree(dsignal)
        return result

def _cuda_ifft(numpy_array, leave_on_device=False):
    dsignal = _get_cufft_signal(numpy_array)
    plan = _get_plan(numpy_array.shape)
    #print "[*] Using the CUFFT plan to inverse transform the signal in place..."
    cufftExecC2C(plan, dsignal, dsignal, CUFFT_INVERSE)
    #print "[*] Destroying CUFFT plan..."
    cufftDestroy(plan)
    if not leave_on_device:
        result = _get_inverse_data(dsignal, numpy_array)
        #result = result.reshape(numpy_array.shape)
        cudaFree(dsignal)
        return result

def fft(numpy_array, leave_on_device=False):
    if numpy_array.ndim == 1:
        return _cuda_fft(numpy_array, leave_on_device)
    else:
        print 'cuda.sugar.fft.fft: ndim != 1, throw exception '

def fft2(numpy_array, leave_on_device=False):
    if numpy_array.ndim == 2:
        return _cuda_fft(numpy_array, leave_on_device)
    else:
        print 'cuda.sugar.fft.fft2: ndim !=2, throw exception'

def fftn(numpy_array, leave_on_device=False):
    if numpy_array.ndim > 3:
        print 'cuda.sugar.fft.fftn: ndim > 3, throw exception'
    else:
        return _cuda_fft(numpy_array, leave_on_device)

def ifft(numpy_array, leave_on_device=False):
    if numpy_array.ndim == 1:
        return _cuda_ifft(numpy_array, leave_on_device)
    else:
        print 'cuda.sugar.fft.ifft: ndim != 1, throw exception '

def ifft2(numpy_array, leave_on_device=False):
    if numpy_array.ndim == 2:
        return _cuda_ifft(numpy_array, leave_on_device)
    else:
        print 'cuda.sugar.fft.ifft2: ndim != 2, throw exception '

def ifftn(numpy_array, leave_on_device=False):
    if numpy_array.ndim > 3:
        print 'cuda.sugar.fft.ifftn: ndim > 3, throw exception '
    else:
        return _cuda_ifft(numpy_array, leave_on_device)

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

    numpy_array = randn(size).astype(complex64)
    numpy_array -= numpy_array.mean()
    numpy_array /= numpy_array.std()

    print numpy_array.mean()
    print numpy_array.std()
    #print numpy_array

    print
    print ">>> Forward fft on gpu ..."
    fft_res = fft(numpy_array)

    print
    print ">>> Inverse fft on gpu ..."
    ifft_res = ifft(fft_res) 

    print
    print ">>> Computing references with numpy..."
    print "[*] Forward fft"
    forward_ref = numpy.fft.fft(numpy_array)
    print "[*] Inverse fft"
    inverse_ref = numpy.fft.ifft(forward_ref)

    print "l2norm fft: ", numpy.linalg.norm(fft_res - forward_ref)

    print "l2norm ifft: ", numpy.linalg.norm(ifft_res - inverse_ref)

    #print print ">>> Forward transform:"
    #print "[*] GPU: \n%s" % fft_res
    #print "[*] CPU: \n%s" % forward_ref
    #print "(*) GPU/CPU close? ", allclose(fft_res,forward_ref)

    #print
    #print ">>> Inverse transform:"
    #print "[*] GPU: \n%s" % ifft_res
    #print "[*] CPU: \n%s" % inverse_ref
    #print "(*) GPU/CPU close? ",allclose(ifft_res,inverse_ref)

if __name__ == "__main__":
    main()
