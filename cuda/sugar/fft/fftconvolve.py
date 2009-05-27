import os
import sys
from ctypes import *
import numpy
from scipy.signal import fftconvolve, convolve2d
from conv_gold import get_convolution_cpu, get_check_results,centered

from cuda.cufft import *
from cuda.cuda import * 
from cuda.kernel.kernelfactoryrt import SourceModule

import logging
logging.basicConfig(level=logging.INFO)

def info(msg):
    logging.info(" %s" % msg)

def warn(msg):
    logging.warning(" %s" % msg)

def error(msg):
    logging.error(" %s" % msg)

def debug(msg):
    logging.debug(" %s" % msg)

def check_results(h_ResultCPU, h_ResultGPU, DATA_W, DATA_H, FFT_W):
    ResultGPU = h_ResultGPU[0:DATA_W,0:DATA_H]
    max_delta_ref = numpy.sqrt(numpy.abs(h_ResultCPU-ResultGPU)**2/numpy.abs(h_ResultCPU)**2).max()
    L2norm = numpy.linalg.norm(h_ResultCPU - ResultGPU)/numpy.linalg.norm(h_ResultCPU)
    info("Max delta / CPU value %s" % max_delta_ref)
    info('L2 norm: %s' % L2norm)
    if L2norm < 1e-6:
        info("TEST PASSED")
    else:
        info("TEST FAILED")

def iDivUp(a, b):
    """ Round a / b to nearest higher integer value """
    if a % b !=0:
        return (a / b + 1)
    else:
        return (a / b)

def iAlignUp(a, b):
    """ Align a to nearest higher multiple of b """
    if a % b != 0:
        return (a - a % b + b)
    else:
        return a

def cudaCheckError(status):
    try:
        assert status == 0 
    except AssertionError,e:
        error('ERROR: status = %s' % status)

def calculateFFTsize(dataSize):
    # Highest non-zero bit position of dataSize
    # Neares lower and higher powers of two numbers for dataSize
    lowPOT = 0
    hiPOT = 0

    # Align data size to a multiple of half-warp
    # in order to have each line starting at properly aligned addresses
    # for coalesced global memory writes in padKernel() and padData()
    dataSize = iAlignUp(dataSize, 16)

    # Find highest non-zero bit
    counter = range(0,32)
    counter.reverse()
    for hiBit in counter:
        if dataSize & (1 << hiBit):
            break

    # No need to align, if already power of two
    lowPOT = 1 << hiBit
    if lowPOT == dataSize:
        return dataSize

    # Align to a nearest higher power of two, if the size is small
    # enough,
    # else align only to a nearest higher multiple of 512,
    # in order to save computation and memory bandwidth
    hiPOT = 1 << (hiBit + 1)
    if hiPOT <= 1024:
        return hiPOT
    else:
        return iAlignUp(dataSize, 512)

def _get_cufft_signal(numpy_array):
    dsignal = c_void_p()
    cudaMalloc(byref(dsignal), numpy_array.nbytes)
    cudaMemcpy(dsignal, numpy_array.ctypes.data, numpy_array.nbytes, cudaMemcpyHostToDevice)
    return cast(dsignal,POINTER(cufftComplex))

def _get_cuda_array(numpy_array, float2tex):
    dsignal = cast(c_void_p(),POINTER(cudaArray))
    cudaMallocArray(dsignal, float2tex, numpy_array.shape[0], numpy_array.shape[1])
    cudaMemcpyToArray(dsignal, 0, 0, numpy_array.ctypes.data, numpy_array.nbytes, cudaMemcpyHostToDevice)
    return cast(dsignal,POINTER(cudaArray))

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
    plan = cufftHandle()
    cufftPlan1d(plan, shape[0], CUFFT_C2C, batch)
    return plan

def _get_2dplan(shape):
    debug("[*] Creating a 2D FFT plan...")
    plan = cufftHandle()
    cufftPlan2d(plan, shape[0], shape[1], CUFFT_C2C)
    return plan

def _get_3dplan(shape):
    debug("[*] Creating a 3D FFT plan...")
    plan = cufftHandle()
    cufftPlan3d(plan, shape[0], shape[1], shape[2], CUFFT_C2C)
    return plan

def get_float2_ptr(numpy_array):
    return numpy_array.ctypes.data_as(POINTER(float2))

# //////////////////////////////////////////////////////////////////////////////
# Main program
# //////////////////////////////////////////////////////////////////////////////
def fftconvolve2d(data, kernel):

    # alias Complex type to float2
    Complex = float2

    # Kernel dimensions
    KERNEL_W = kernel.shape[0]
    KERNEL_H = kernel.shape[1]

    # Kernel center position
    KERNEL_X = KERNEL_W/2
    KERNEL_Y = KERNEL_H/2

    # Width and height of padding for "clamp to border" addressing mode
    PADDING_W = KERNEL_W - 1
    PADDING_H = KERNEL_H - 1

    # Input data dimension
    DATA_W = data.shape[0] 
    DATA_H = data.shape[1]

    # Derive FFT size from data and kernel dimensions
    FFT_W = calculateFFTsize(DATA_W + PADDING_W)
    FFT_H = calculateFFTsize(DATA_H + PADDING_H)
    FFT_SIZE = FFT_W * FFT_H * sizeof(Complex)
    KERNEL_SIZE = KERNEL_W * KERNEL_H * sizeof(Complex)
    DATA_SIZE = DATA_W * DATA_H * sizeof(Complex)

    e = sizeof(c_float) * 8
    float2tex = cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat)

    info("Input data size           : %i x %i" % (DATA_W, DATA_H))
    info("Convolution kernel size   : %i x %i" % (KERNEL_W, KERNEL_H))
    info("Padded image size         : %i x %i" % (DATA_W + PADDING_W, DATA_H + PADDING_H))
    info("Aligned padded image size : %i x %i" % (FFT_W, FFT_H))

    info(">>> Loading Kernels...")
    kernel_src = os.path.join(os.path.dirname(__file__), 'fftconvolve2d_kernel.cu')
    fftconvolve2d = SourceModule(open(kernel_src,'r').read(), no_extern_c=True)

    info(">>> Extracting functions from Kernel...")
    info("[*] Configuring Block/Grid dimensions...")
    # Block width should be a multiple of maximum coalesced write size 
    # for coalesced memory writes in padKernel() and padData()
    threadBlock = dim3(16, 12, 1)
    kernelBlockGrid = dim3(iDivUp(KERNEL_W, threadBlock.x), iDivUp(KERNEL_H, threadBlock.y),1)
    dataBlockGrid = dim3(iDivUp(FFT_W, threadBlock.x),iDivUp(FFT_H, threadBlock.y),1)
    sixteen = dim3(16,1,1)
    onetwentyeight = dim3(128,1,1)
    # Extract kernel functions from SourceModule
    info("[*] Loading padKernel...")
    padKernel = fftconvolve2d.padKernel(kernelBlockGrid, threadBlock)
    info("[*] Loading padData...")
    padData = fftconvolve2d.padData(dataBlockGrid, threadBlock)
    info("[*] Loading modulateAndNormalize...")
    modulateAndNormalize = fftconvolve2d.modulateAndNormalize(sixteen, onetwentyeight)

    info(">>> Allocating memory...")

    info("[*] Generating random input data...")
    h_Kernel = kernel
    h_Data = data
    #h_Kernel = numpy.random.uniform(0,1,(KERNEL_W,KERNEL_H)).astype(numpy.complex64)
    #h_Data = numpy.random.uniform(0,1,(DATA_W,DATA_H)).astype(numpy.complex64)

    info("[*] Allocating host memory for results...")
    h_ResultCPU = numpy.zeros((DATA_W,DATA_H)).astype(numpy.complex64)
    h_ResultGPU = numpy.zeros((FFT_W,FFT_H)).astype(numpy.complex64)

    info("[*] Allocating linear device memory (Complex)...")
    d_PaddedKernel = _get_cufft_signal(numpy.zeros((FFT_W,FFT_H)).astype(numpy.complex64))
    d_PaddedData = _get_cufft_signal(numpy.zeros((FFT_W,FFT_H)).astype(numpy.complex64))

    info("[*] Allocating cuda array device memory...")
    a_Kernel = _get_cuda_array(h_Kernel,float2tex)
    a_Data = _get_cuda_array(h_Data, float2tex)

    info("[*] Binding textures...")
    texKernel = cast(c_void_p(), POINTER(textureReference))
    cudaCheckError(cudaGetTextureReference(texKernel,'texKernel')) 
    texData = cast(c_void_p(), POINTER(textureReference))
    cudaCheckError(cudaGetTextureReference(texData,'texData'))

    fdesc = cudaChannelFormatDesc()
    cudaCheckError(cudaGetChannelDesc(fdesc, a_Kernel))
    cudaCheckError(cudaBindTextureToArray(texKernel, a_Kernel, fdesc))

    fdesc2 = cudaChannelFormatDesc()
    cudaCheckError(cudaGetChannelDesc(fdesc2, a_Data))
    cudaCheckError(cudaBindTextureToArray(texData, a_Data, fdesc2))

    info('>>> Calling kernels')
    info("[*] Padding convolution kernel")
    padKernel(d_PaddedKernel, FFT_W, FFT_H, KERNEL_W, KERNEL_H, KERNEL_X, KERNEL_Y)

    info("[*] Padding input data array")
    padData(d_PaddedData, FFT_W, FFT_H, DATA_W, DATA_H, KERNEL_W, KERNEL_H, KERNEL_X, KERNEL_Y)

    # Not including kernel transformation into time measurement,
    # since convolution kernel is not changed very frequently
    info('>>> Calling CUFFT')
    info("[*] Transforming convolution kernel (CUFFT)...")
    FFTplan = _get_plan(h_ResultGPU.shape)
    cudaCheckError(cufftExecC2C(FFTplan, d_PaddedKernel, d_PaddedKernel, CUFFT_FORWARD))
    info("[*] Transforming data (CUFFT)...")
    cudaCheckError(cudaThreadSynchronize())
    cudaCheckError(cufftExecC2C(FFTplan, d_PaddedData, d_PaddedData, CUFFT_FORWARD))

    info('>>> Calling kernel')
    info("[*] modulateAndNormalize()")
    modulateAndNormalize(d_PaddedData, d_PaddedKernel, FFT_W * FFT_H)
    info('>>> Calling CUFFT')
    info("[*] Inverse transforming data (CUFFT)...")
    cudaCheckError(cufftExecC2C(FFTplan, d_PaddedData, d_PaddedData, CUFFT_INVERSE))
    cudaCheckError(cudaThreadSynchronize())

    info(">>> Copying results from GPU...")
    cudaCheckError(cudaMemcpy(h_ResultGPU.ctypes.data, d_PaddedData, FFT_SIZE, cudaMemcpyDeviceToHost))

    info(">>> Checking GPU results...")
    info("[*] running reference CPU convolution...")
    conv_gold = get_convolution_cpu() 
    
    conv_gold(get_float2_ptr(h_ResultCPU), get_float2_ptr(h_Data), get_float2_ptr(h_Kernel), DATA_W, DATA_H, KERNEL_W, KERNEL_H, KERNEL_X, KERNEL_Y)

    info( "[*] comparing the results...")
    
    check_results(h_ResultCPU.real, h_ResultGPU.real, h_ResultCPU.shape[0], h_ResultCPU.shape[1], FFT_W)

    info( ">>> Shutting down...")

    info( "[*] Destroying FFT plans...")
    cudaCheckError(cufftDestroy(FFTplan))

    info( "[*] Unbinding textures...")
    cudaCheckError(cudaUnbindTexture(texData))
    cudaCheckError(cudaUnbindTexture(texKernel))

    info( "[*] Freeing device memory...")
    cudaCheckError(cudaFree(d_PaddedData))
    cudaCheckError(cudaFree(d_PaddedKernel))
    cudaCheckError(cudaFreeArray(a_Data))
    cudaCheckError(cudaFreeArray(a_Kernel))

    info( "[*] CUDA Thread Exit")
    cudaThreadExit()

    s1 = numpy.array(data.shape)
    s2 = numpy.array(kernel.shape)
    dh, dw = data.shape
    return centered(h_ResultGPU.real[0:dh,0:dw], abs(s2-s1)+1)

if __name__ == "__main__":
    kernel = numpy.random.uniform(0,1,(7,7)).astype(numpy.complex64)
    data = numpy.random.uniform(0,1,(512,512)).astype(numpy.complex64)
    fftconvolve2d(data, kernel)
