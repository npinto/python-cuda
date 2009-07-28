import os
import ctypes
import numpy
import scipy.signal

import logging
log = logging.getLogger('python-cuda')

#from conv_gold import get_convolution_cpu, get_check_results,centered

import fft 
import cuda.cufft as cufft
import cuda.cuda as cuda
from cuda.sugar.kernel.kernelfactoryrt import SourceModule

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = numpy.asarray(newsize)
    currsize = numpy.array(arr.shape)
    startind = (currsize - newsize) / 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def check_results(h_ResultCPU, h_ResultGPU):
    L2norm = numpy.linalg.norm(h_ResultCPU - h_ResultGPU)/numpy.linalg.norm(h_ResultCPU)
    log.info('L2 norm: %s' % L2norm)
    if L2norm < 1e-6:
        log.info("TEST PASSED")
    else:
        log.info("TEST FAILED")

def _i_div_up(a, b):
    """ Round a / b to nearest higher integer value """
    if a % b !=0:
        return (a / b + 1)
    else:
        return (a / b)

def _i_align_up(a, b):
    """ Align a to nearest higher multiple of b """
    if a % b != 0:
        return (a - a % b + b)
    else:
        return a

def cuda_check_error(status):
    try:
        assert status == 0 
    except AssertionError,e:
        log.error('ERROR: status = %s' % status)

def _calc_fft_size(dataSize):
    # Highest non-zero bit position of dataSize
    # Neares lower and higher powers of two numbers for dataSize
    lowPOT = 0
    hiPOT = 0

    # Align data size to a multiple of half-warp
    # in order to have each line starting at properly aligned addresses
    # for coalesced global memory writes in padKernel() and padData()
    dataSize = _i_align_up(dataSize, 16)

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
        return _i_align_up(dataSize, 512)

def _get_cuda_array(numpy_array, float2tex):
    dsignal = ctypes.cast(ctypes.c_void_p(),ctypes.POINTER(cuda.cudaArray))
    cuda.cudaMallocArray(dsignal, float2tex, numpy_array.shape[0], numpy_array.shape[1])
    cuda.cudaMemcpyToArray(dsignal, 0, 0, numpy_array.ctypes.data, numpy_array.nbytes, cuda.cudaMemcpyHostToDevice)
    return ctypes.cast(dsignal,ctypes.POINTER(cuda.cudaArray))

def _get_float2_ptr(numpy_array):
    return numpy_array.ctypes.data_as(ctypes.POINTER(cuda.float2))

# //////////////////////////////////////////////////////////////////////////////
# Main program
# //////////////////////////////////////////////////////////////////////////////
def fftconvolve2d(data, kernel, test=False):
    s1 = numpy.array(data.shape)
    s2 = numpy.array(kernel.shape)
    dh, dw = data.shape

    h_Kernel = kernel
    h_Data = data

    # alias Complex type to float2
    Complex = cuda.float2

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
    FFT_W = _calc_fft_size(DATA_W + PADDING_W)
    FFT_H = _calc_fft_size(DATA_H + PADDING_H)
    FFT_SIZE = FFT_W * FFT_H * ctypes.sizeof(Complex)
    KERNEL_SIZE = KERNEL_W * KERNEL_H * ctypes.sizeof(Complex)
    DATA_SIZE = DATA_W * DATA_H * ctypes.sizeof(Complex)

    e = ctypes.sizeof(ctypes.c_float) * 8
    float2tex = cuda.cudaCreateChannelDesc(e, e, 0, 0, cuda.cudaChannelFormatKindFloat)

    log.debug("Input data size           : %i x %i" % (DATA_W, DATA_H))
    log.debug("Convolution kernel size   : %i x %i" % (KERNEL_W, KERNEL_H))
    log.debug("Padded image size         : %i x %i" % (DATA_W + PADDING_W, DATA_H + PADDING_H))
    log.debug("Aligned padded image size : %i x %i" % (FFT_W, FFT_H))

    log.debug("Loading Kernels...")
    kernel_src = os.path.join(os.path.dirname(__file__), 'fftconvolve2d_kernel.cu')
    fftconvolve2d = SourceModule(open(kernel_src,'r').read(), no_extern_c=True)

    log.debug("Extracting functions from Kernel...")
    log.debug("[*] Configuring Block/Grid dimensions...")
    # Block width should be a multiple of maximum coalesced write size 
    # for coalesced memory writes in padKernel() and padData()
    threadBlock = cuda.dim3(16, 12, 1)
    kernelBlockGrid = cuda.dim3(_i_div_up(KERNEL_W, threadBlock.x), _i_div_up(KERNEL_H, threadBlock.y),1)
    dataBlockGrid = cuda.dim3(_i_div_up(FFT_W, threadBlock.x),_i_div_up(FFT_H, threadBlock.y),1)
    sixteen = cuda.dim3(16,1,1)
    onetwentyeight = cuda.dim3(128,1,1)
    # Extract kernel functions from SourceModule
    log.debug("[*] Loading padKernel...")
    padKernel = fftconvolve2d.padKernel(kernelBlockGrid, threadBlock)
    log.debug("[*] Loading padData...")
    padData = fftconvolve2d.padData(dataBlockGrid, threadBlock)
    log.debug("[*] Loading modulateAndNormalize...")
    modulateAndNormalize = fftconvolve2d.modulateAndNormalize(sixteen, onetwentyeight)

    log.debug("Allocating memory...")

    #log.debug("[*] Generating random input data...")
    #h_Kernel = numpy.random.uniform(0,1,(KERNEL_W,KERNEL_H)).astype(numpy.complex64)
    #h_Data = numpy.random.uniform(0,1,(DATA_W,DATA_H)).astype(numpy.complex64)

    log.debug("[*] Allocating host memory for results...")
    h_ResultGPU = numpy.zeros((FFT_W,FFT_H)).astype(numpy.complex64)

    log.debug("[*] Allocating linear device memory (Complex)...")
    d_PaddedKernel = fft._get_cufft_signal(numpy.zeros((FFT_W,FFT_H)).astype(numpy.complex64))
    d_PaddedData = fft._get_cufft_signal(numpy.zeros((FFT_W,FFT_H)).astype(numpy.complex64))

    log.debug("[*] Allocating cuda array device memory...")
    a_Kernel = _get_cuda_array(h_Kernel,float2tex)
    a_Data = _get_cuda_array(h_Data, float2tex)

    log.debug("[*] Binding textures...")
    texKernel = ctypes.cast(ctypes.c_void_p(), ctypes.POINTER(cuda.textureReference))
    cuda_check_error(cuda.cudaGetTextureReference(texKernel,'texKernel')) 
    texData = ctypes.cast(ctypes.c_void_p(), ctypes.POINTER(cuda.textureReference))
    cuda_check_error(cuda.cudaGetTextureReference(texData,'texData'))

    fdesc = cuda.cudaChannelFormatDesc()
    cuda_check_error(cuda.cudaGetChannelDesc(fdesc, a_Kernel))
    cuda_check_error(cuda.cudaBindTextureToArray(texKernel, a_Kernel, fdesc))

    fdesc2 = cuda.cudaChannelFormatDesc()
    cuda_check_error(cuda.cudaGetChannelDesc(fdesc2, a_Data))
    cuda_check_error(cuda.cudaBindTextureToArray(texData, a_Data, fdesc2))

    log.debug('Calling kernels')
    log.debug("[*] Padding convolution kernel")
    padKernel(d_PaddedKernel, FFT_W, FFT_H, KERNEL_W, KERNEL_H, KERNEL_X, KERNEL_Y)

    log.debug("[*] Padding input data array")
    padData(d_PaddedData, FFT_W, FFT_H, DATA_W, DATA_H, KERNEL_W, KERNEL_H, KERNEL_X, KERNEL_Y)

    # Not including kernel transformation into time measurement,
    # since convolution kernel is not changed very frequently
    log.debug('Calling CUFFT')
    log.debug("[*] Transforming convolution kernel (CUFFT)...")
    FFTplan = fft._get_plan(h_ResultGPU.shape)
    cuda_check_error(cufft.cufftExecC2C(FFTplan, d_PaddedKernel, d_PaddedKernel, cufft.CUFFT_FORWARD))
    log.debug("[*] Transforming data (CUFFT)...")
    cuda_check_error(cuda.cudaThreadSynchronize())
    cuda_check_error(cufft.cufftExecC2C(FFTplan, d_PaddedData, d_PaddedData, cufft.CUFFT_FORWARD))

    log.debug('Calling kernel')
    log.debug("[*] modulateAndNormalize()")
    modulateAndNormalize(d_PaddedData, d_PaddedKernel, FFT_W * FFT_H)
    log.debug('Calling CUFFT')
    log.debug("[*] Inverse transforming data (CUFFT)...")
    cuda_check_error(cufft.cufftExecC2C(FFTplan, d_PaddedData, d_PaddedData, cufft.CUFFT_INVERSE))
    cuda_check_error(cuda.cudaThreadSynchronize())

    log.debug("Copying results from GPU...")
    cuda_check_error(cuda.cudaMemcpy(h_ResultGPU.ctypes.data, d_PaddedData, FFT_SIZE, cuda.cudaMemcpyDeviceToHost))
    h_ResultGPU = _centered(h_ResultGPU.real[0:dh,0:dw], abs(s2-s1)+1)

    if test:
        log.info("Checking GPU results...")
        log.info("[*] running reference CPU convolution...")
        #conv_gold = get_convolution_cpu() 
        #conv_gold(_get_float2_ptr(h_ResultCPU), _get_float2_ptr(h_Data), _get_float2_ptr(h_Kernel), DATA_W, DATA_H, KERNEL_W, KERNEL_H, KERNEL_X, KERNEL_Y)
        h_ResultCPU = scipy.signal.fftconvolve(h_Data.real, h_Kernel.real, mode='valid')
        log.info( "[*] comparing the results...")
        check_results(h_ResultCPU, h_ResultGPU)

    log.debug( "Shutting down...")

    log.debug( "[*] Destroying FFT plans...")
    cuda_check_error(cufft.cufftDestroy(FFTplan))

    log.debug( "[*] Unbinding textures...")
    cuda_check_error(cuda.cudaUnbindTexture(texData))
    cuda_check_error(cuda.cudaUnbindTexture(texKernel))

    log.debug( "[*] Freeing device memory...")
    cuda_check_error(cuda.cudaFree(d_PaddedData))
    cuda_check_error(cuda.cudaFree(d_PaddedKernel))
    cuda_check_error(cuda.cudaFreeArray(a_Data))
    cuda_check_error(cuda.cudaFreeArray(a_Kernel))

    log.debug( "[*] CUDA Thread Exit")
    cuda.cudaThreadExit()

    return h_ResultGPU

if __name__ == "__main__":
    kernel = numpy.random.uniform(0,1,(7,7)).astype(numpy.complex64)
    data = numpy.random.uniform(0,1,(512,512)).astype(numpy.complex64)
    fftconvolve2d(data, kernel, True)
