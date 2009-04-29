import sys
from ctypes import *
import numpy
from scipy.signal import fftconvolve
from conv_gold import get_convolution_cpu, get_check_results

from cuda.cufft import *
from cuda.cuda import * 
from cuda.kernel.kernelfactoryrt import SourceModule

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
        print 'ERROR: status = %s' % status

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

def get_float2_ptr(numpy_array):
    return numpy_array.ctypes.data_as(POINTER(float2))

# //////////////////////////////////////////////////////////////////////////////
# Main program
# //////////////////////////////////////////////////////////////////////////////
def main():
    # alias Complex type to float2
    Complex = float2

    # Kernel dimensions
    KERNEL_W = 7
    KERNEL_H = 7

    # Kernel center position
    KERNEL_X = 1
    KERNEL_Y = 6

    # Width and height of padding for "clamp to border" addressing mode
    PADDING_W = KERNEL_W - 1
    PADDING_H = KERNEL_H - 1

    # Input data dimension
    DATA_W = 1000 
    DATA_H = 1000

    # Derive FFT size from data and kernel dimensions
    FFT_W = calculateFFTsize(DATA_W + PADDING_W)
    FFT_H = calculateFFTsize(DATA_H + PADDING_H)
    FFT_SIZE = FFT_W * FFT_H * sizeof(Complex)
    KERNEL_SIZE = KERNEL_W * KERNEL_H * sizeof(Complex)
    DATA_SIZE = DATA_W * DATA_H * sizeof(Complex)

    e = sizeof(c_float) * 8
    float2tex = cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat)

    print "Input data size           : %i x %i" % (DATA_W, DATA_H)
    print "Convolution kernel size   : %i x %i" % (KERNEL_W, KERNEL_H)
    print "Padded image size         : %i x %i" % (DATA_W + PADDING_W, DATA_H + PADDING_H)
    print "Aligned padded image size : %i x %i" % (FFT_W, FFT_H)

    print ">>> Loading Kernels..."
    fftconvolve2d = SourceModule(open('fftconvolve2d_kernel.cu','r').read(), no_extern_c=True)

    print ">>> Extracting functions from Kernel..."
    print "[*] Configuring Block/Grid dimensions..."
    # Block width should be a multiple of maximum coalesced write size 
    # for coalesced memory writes in padKernel() and padData()
    threadBlock = dim3(16, 12, 1)
    kernelBlockGrid = dim3(iDivUp(KERNEL_W, threadBlock.x), iDivUp(KERNEL_H, threadBlock.y),1)
    dataBlockGrid = dim3(iDivUp(FFT_W, threadBlock.x),iDivUp(FFT_H, threadBlock.y),1)
    sixteen = dim3(16,1,1)
    onetwentyeight = dim3(128,1,1)
    # Extract kernel functions from SourceModule
    print "[*] Loading padKernel..."
    padKernel = fftconvolve2d.padKernel(kernelBlockGrid, threadBlock)
    print "[*] Loading padData..."
    padData = fftconvolve2d.padData(dataBlockGrid, threadBlock)
    print "[*] Loading modulateAndNormalize..."
    modulateAndNormalize = fftconvolve2d.modulateAndNormalize(sixteen, onetwentyeight)

    print ">>> Allocating memory..."

    print "[*] Generating random input data..."
    #h_Kernel = numpy.ones((KERNEL_W,KERNEL_H)).astype(numpy.complex64)
    #h_Data = numpy.ones((DATA_W,DATA_H)).astype(numpy.complex64)
    h_Kernel = numpy.random.randn(KERNEL_W,KERNEL_H).astype(numpy.complex64)
    h_Data = numpy.random.randn(DATA_W,DATA_H).astype(numpy.complex64)

    print "[*] Allocating host memory for results..."
    h_ResultCPU = numpy.zeros((DATA_W, DATA_H)).astype(numpy.complex64)
    h_ResultGPU = numpy.zeros((FFT_W,FFT_H)).astype(numpy.complex64)

    print "[*] Allocating linear device memory (Complex)..."
    d_PaddedKernel = _get_cufft_signal(numpy.zeros((FFT_W,FFT_H)).astype(numpy.complex64))
    d_PaddedData = _get_cufft_signal(numpy.zeros((FFT_W,FFT_H)).astype(numpy.complex64))

    print "[*] Allocating cuda array device memory..."
    a_Kernel = _get_cuda_array(h_Kernel,float2tex)
    a_Data = _get_cuda_array(h_Data, float2tex)

    print "[*] Binding textures..." 
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

    print '>>> Calling kernels'
    print "[*] Padding convolution kernel"
    padKernel(d_PaddedKernel, FFT_W, FFT_H, KERNEL_W, KERNEL_H, KERNEL_X, KERNEL_Y)

    print "[*] Padding input data array"
    padData(d_PaddedData, FFT_W, FFT_H, DATA_W, DATA_H, KERNEL_W, KERNEL_H, KERNEL_X, KERNEL_Y)

    # Not including kernel transformation into time measurement,
    # since convolution kernel is not changed very frequently
    print '>>> Calling CUFFT'
    print "[*] Transforming convolution kernel (CUFFT)..."
    FFTplan = _get_plan(h_ResultGPU.shape)
    cudaCheckError(cufftExecC2C(FFTplan, d_PaddedKernel, d_PaddedKernel, CUFFT_FORWARD))
    print "[*] Transforming data (CUFFT)..."
    cudaCheckError(cudaThreadSynchronize())
    cudaCheckError(cufftExecC2C(FFTplan, d_PaddedData, d_PaddedData, CUFFT_FORWARD))

    print '>>> Calling kernel'
    print "[*] modulateAndNormalize()"
    modulateAndNormalize(d_PaddedData, d_PaddedKernel, FFT_W * FFT_H)
    print '>>> Calling CUFFT'
    print "[*] Inverse transforming data (CUFFT)..."
    cudaCheckError(cufftExecC2C(FFTplan, d_PaddedData, d_PaddedData, CUFFT_INVERSE))
    cudaCheckError(cudaThreadSynchronize())

    print ">>> Copying results from GPU..."
    cudaCheckError(cudaMemcpy(h_ResultGPU.ctypes.data, d_PaddedData, FFT_SIZE, cudaMemcpyDeviceToHost))
    #print h_ResultGPU

    print ">>> Checking GPU results..."
    print "[*] running reference CPU convolution..."
    conv_gold = get_convolution_cpu() 
    check_results = get_check_results()
    result = numpy.zeros_like(h_Data)
    conv_gold(get_float2_ptr(result), get_float2_ptr(h_Data), get_float2_ptr(h_Kernel), DATA_W, DATA_H, KERNEL_W, KERNEL_H, KERNEL_X, KERNEL_Y)
    result =  result.astype('complex64')
    #print result

    print "[*] comparing the results..."
    check_results(get_float2_ptr(result), get_float2_ptr(h_ResultGPU), DATA_W, DATA_H, FFT_W)
    #print "[*] l2norm = %s" % numpy.linalg.norm(result-h_ResultGPU)

    print ">>> Shutting down..."

    print "[*] Destroying FFT plans..."
    cudaCheckError(cufftDestroy(FFTplan))

    print "[*] Unbinding textures..."
    cudaCheckError(cudaUnbindTexture(texData))
    cudaCheckError(cudaUnbindTexture(texKernel))

    print "[*] Freeing device memory..."
    cudaCheckError(cudaFree(d_PaddedData))
    cudaCheckError(cudaFree(d_PaddedKernel))
    cudaCheckError(cudaFreeArray(a_Data))
    cudaCheckError(cudaFreeArray(a_Kernel))

    print "[*] CUDA Thread Exit"
    cudaThreadExit()

if __name__ == "__main__":
    main()
