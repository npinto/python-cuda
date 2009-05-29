#!/usr/bin/env python
import os
import sys
import ctypes

import numpy as np

import cuda.cuda as cuda
import cuda.cufft as cufft
import logging 

logger = logging.getLogger(os.path.basename(__file__))
info = logger.info
debug = logger.debug
warn = logger.warn
error = logger.error

def _get_cufft_signal(numpy_array):
    dsignal = ctypes.c_void_p()
    cuda.cudaMalloc(ctypes.byref(dsignal), numpy_array.nbytes)
    cuda.cudaMemcpy(dsignal, numpy_array.ctypes.data, numpy_array.nbytes, cuda.cudaMemcpyHostToDevice)
    return ctypes.cast(dsignal,ctypes.POINTER(cufft.cufftComplex))

def _get_plan(shape):
    ndims = len(shape)
    if ndims == 1:
        return _get_1dplan(shape)
    elif ndims == 2:
        return _get_2dplan(shape)
    elif ndims == 3:
        return _get_3dplan(shape)
    else:
        error('_get_plan: invalid size (todo: throw exception)')

def _get_1dplan(shape,batch=1):
    debug("[*] Creating a 1D FFT plan...")
    plan = cufft.cufftHandle()
    cufft.cufftPlan1d(plan, shape[0], cufft.CUFFT_C2C, batch)
    return plan

def _get_2dplan(shape):
    debug("[*] Creating a 2D FFT plan...")
    plan = cufft.cufftHandle()
    cufft.cufftPlan2d(plan, shape[0], shape[1], cufft.CUFFT_C2C)
    return plan

def _get_3dplan(shape):
    debug("[*] Creating a 3D FFT plan...")
    plan = cufft.cufftHandle()
    cufft.cufftPlan3d(plan, shape[0], shape[1], shape[2], cufft.CUFFT_C2C)
    return plan

def _get_data(device_ptr,numpy_array):
    result = np.empty_like(numpy_array)
    cuda.cudaMemcpy(result.ctypes.data, device_ptr, numpy_array.nbytes, cuda.cudaMemcpyDeviceToHost)
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
    cufft.cufftExecC2C(plan, dsignal, dsignal, cufft.CUFFT_FORWARD)
    debug("[*] Destroying CUFFT plan...")
    cufft.cufftDestroy(plan)
    if not leave_on_device:
        result = _get_data(dsignal, numpy_array)
        #result = result.reshape(numpy_array.shape)
        cuda.cudaFree(dsignal)
        return result
    else:
        return dsignal

def _cuda_ifft(numpy_array, leave_on_device=False):
    dsignal = _get_cufft_signal(numpy_array)
    plan = _get_plan(numpy_array.shape)
    debug("[*] Using the CUFFT plan to inverse transform the signal in place...")
    cufft.cufftExecC2C(plan, dsignal, dsignal, cufft.CUFFT_INVERSE)
    debug("[*] Destroying CUFFT plan...")
    cufft.cufftDestroy(plan)
    if not leave_on_device:
        result = _get_inverse_data(dsignal, numpy_array)
        #result = result.reshape(numpy_array.shape)
        cuda.cudaFree(dsignal)
        return result
    else:
        return dsignal

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

    numpy_array = np.random.randn(size).astype('complex64')
    numpy_array -= numpy_array.mean()
    numpy_array /= numpy_array.std()

    print ">>> Computing ffts with GPU..."
    print "[*] Forward fft on gpu ..."
    fft_res = fft(numpy_array)

    print "[*] Inverse fft on gpu ..."
    ifft_res = ifft(fft_res) 

    print ">>> Computing references with numpy..."

    print "[*] Forward fft"
    forward_ref = np.fft.fft(numpy_array)

    print "[*] Inverse fft"
    inverse_ref = np.fft.ifft(forward_ref)

    print "l2norm fft: ", np.linalg.norm(fft_res - forward_ref)

    print "l2norm ifft: ", np.linalg.norm(ifft_res - inverse_ref)

if __name__ == "__main__":
    main()
