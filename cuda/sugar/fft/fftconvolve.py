import sys
import numpy
from ctypes import *

from cuda.cufft import *
from cuda.cuda import * 
from cuda.sugar.fft import *
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
    ndims = len(numpy_array.shape)
    if ndims == 1:
        return _get_cufft_1dsignal(numpy_array)
    elif ndims == 2:
        return _get_cufft_2dsignal(numpy_array)
    elif ndims == 3:
        return _get_cufft_3dsignal(numpy_array)
    else:
        print '_get_cufft_signal: invalid size (todo: throw exception)'

def _get_cufft_1dsignal(numpy_array):
    #print "[*] Creating 1d device signal..."
    size = numpy_array.size
    hsignal = (float2*size)()
    for i in range(size): 
        hsignal[i].x = numpy_array[i].real
        hsignal[i].y = numpy_array[i].imag
    dsignal = c_void_p()
    cudaMalloc(byref(dsignal), sizeof(hsignal))
    cudaMemcpy(dsignal, hsignal, sizeof(hsignal), cudaMemcpyHostToDevice)
    return cast(dsignal,POINTER(cufftComplex))
    
def _get_cufft_2dsignal(numpy_array):
    #print "[*] Creating 2d device signal..."
    size = numpy_array.size
    height = numpy_array.shape[0]
    width = numpy_array.shape[1]
    hsignal = (float2*size)()
    for i in range(height): 
        for j in range(width): 
            hsignal[i*width+j].x = numpy_array[i,j].real
            hsignal[i*width+j].y = numpy_array[i,j].imag
    dsignal = c_void_p()
    cudaMalloc(byref(dsignal), sizeof(hsignal))
    cudaMemcpy(dsignal, hsignal, sizeof(hsignal), cudaMemcpyHostToDevice)
    return cast(dsignal,POINTER(cufftComplex))

def _get_cufft_3dsignal(numpy_array):
    #print "[*] Creating 2d device signal..."
    size = numpy_array.size
    height = numpy_array.shape[0]
    width = numpy_array.shape[1]
    length = numpy_array.shape[2]
    hsignal = (float2*size)()
    for i in range(height): 
        for j in range(width): 
            for k in range(length):
                hsignal[i*length*width + j*width + k].x = numpy_array[i,j,k].real
                hsignal[i*length*width + j*width + k].y = numpy_array[i,j,k].imag
    dsignal = c_void_p()
    cudaMalloc(byref(dsignal), sizeof(hsignal))
    cudaMemcpy(dsignal, hsignal, sizeof(hsignal), cudaMemcpyHostToDevice)
    return cast(dsignal,POINTER(cufftComplex))

def _get_cuda_array(numpy_array, float2tex):
    size = numpy_array.size
    hsignal = (float2*size)()
    height = numpy_array.shape[0]
    width = numpy_array.shape[1]
    for i in range(height): 
        for j in range(width): 
            hsignal[i*width + j].x = numpy_array[i,j].real
            hsignal[i*width + j].y = numpy_array[i,j].imag
    dsignal = cast(c_void_p(),POINTER(cudaArray))
    cudaMallocArray(dsignal,float2tex, numpy_array.shape[0],numpy_array.shape[1])
    cudaMemcpyToArray(dsignal, 0,0, hsignal, sizeof(hsignal), cudaMemcpyHostToDevice)
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

    e = sizeof(c_float) * 8;
    float2tex = cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);

    print "Input data size           : %i x %i" % (DATA_W, DATA_H)
    print "Convolution kernel size   : %i x %i" % (KERNEL_W, KERNEL_H)
    print "Padded image size         : %i x %i" % (DATA_W + PADDING_W, DATA_H + PADDING_H)
    print "Aligned padded image size : %i x %i" % (FFT_W, FFT_H)

    print "KERNEL_SIZE = ", KERNEL_SIZE

    print ">>> Loading Kernel..."
    fftconvolve2d = SourceModule(open('fftconvolve2d_kernel.cu','r').read(), no_extern_c=True)

    print ">>> Allocating memory..."

    print "[*] Generating random input data..."
    h_Kernel = numpy.random.randn(KERNEL_W,KERNEL_H).astype(numpy.complex)
    h_Data = numpy.random.randn(DATA_W,DATA_H).astype(numpy.complex)

    print "[*] Allocating host memory for results..."
    h_ResultCPU = numpy.zeros((DATA_W, DATA_H)).astype(numpy.complex)
    h_ResultGPU = numpy.zeros((FFT_W,FFT_H)).astype(numpy.complex)

    print "[*] Allocating linear device memory (Complex)..."
    d_PaddedKernel = _get_cufft_2dsignal(numpy.zeros((FFT_W,FFT_H)).astype(numpy.complex))
    d_PaddedData = _get_cufft_2dsignal(numpy.zeros((FFT_W,FFT_H)).astype(numpy.complex))

    print "[*] Allocating cuda array device memory..."
    a_Kernel = _get_cuda_array(h_Kernel,float2tex)
    a_Data = _get_cuda_array(h_Data, float2tex)

    print "[*] Binding textures..." 
    texKernel = cast(c_void_p(), POINTER(textureReference))
    assert cudaGetTextureReference(texKernel,'texKernel') == 0
    texData = cast(c_void_p(), POINTER(textureReference))
    assert cudaGetTextureReference(texData,'texData') == 0
    cudaBindTextureToArray(texKernel, a_Kernel, cudaChannelFormatDesc())
    cudaBindTextureToArray(texData, a_Data, cudaChannelFormatDesc())

    print ">>> Configuring Block/Grid dimensions..."
    ## Block width should be a multiple of maximum coalesced write size 
    ## for coalesced memory writes in padKernel() and padData()
    import cuda.cuda
    threadBlock = cuda.cuda.dim3(16, 12, 1)
    kernelBlockGrid = cuda.cuda.dim3(iDivUp(KERNEL_W, threadBlock.x), iDivUp(KERNEL_H, threadBlock.y),1)
    dataBlockGrid = cuda.cuda.dim3(iDivUp(FFT_W, threadBlock.x),iDivUp(FFT_H, threadBlock.y),1)
    sixteen = cuda.cuda.dim3(16,1,1)
    onetwentyeight = cuda.cuda.dim3(128,1,1)

    print ">>> Extracting functions from Kernel..."
    # Extract kernel functions from SourceModule
    print "[*] Loading padKernel..."
    padKernel = fftconvolve2d.padKernel(kernelBlockGrid, threadBlock)
    print "[*] Loading padData..."
    padData = fftconvolve2d.padData(dataBlockGrid, threadBlock)
    print "[*] Loading modulateAndNormalize..."
    modulateAndNormalize = fftconvolve2d.modulateAndNormalize(sixteen, onetwentyeight)


    print '>>> Calling kernels'
    print "[*] Padding convolution kernel"
    padKernel(d_PaddedKernel, FFT_W, FFT_H, KERNEL_W, KERNEL_H, KERNEL_X, KERNEL_Y)

    print "[*] Padding input data array"
    padData(d_PaddedData, FFT_W, FFT_H, DATA_W, DATA_H, KERNEL_W, KERNEL_H, KERNEL_X, KERNEL_Y)

    ## Not including kernel transformation into time measurement,
    ## since convolution kernel is not changed very frequently
    print ">>> Transforming convolution kernel (CUFFT)..."
    FFTplan = _get_plan(h_ResultGPU.shape)
    cufftExecC2C(FFTplan, d_PaddedKernel, d_PaddedKernel, CUFFT_FORWARD)

    print ">>> Running GPU FFT convolution (CUFFT)..."
    cudaThreadSynchronize()
    cufftExecC2C(FFTplan, d_PaddedData, d_PaddedData, CUFFT_FORWARD)
    print '>>> Calling kernel'
    print "[*] modulateAndNormalize()"
    modulateAndNormalize(d_PaddedData, d_PaddedKernel, FFT_W * FFT_H)
    print ">>> Running GPU FFT convolution (CUFFT)..."
    cufftExecC2C(FFTplan, d_PaddedData, d_PaddedData, CUFFT_INVERSE) 
    cudaThreadSynchronize()

    print ">>> Reading back GPU FFT results..."
    cudaMemcpy(h_ResultGPU.ctypes.data, d_PaddedData, FFT_SIZE, cudaMemcpyDeviceToHost)
    print h_ResultGPU

    print ">>> Checking GPU results..."
    print "[*] running reference CPU convolution..."
    #scipy.signal.fftconvolve(h_ResultCPU)
    print "[*] comparing the results..."
    #todo

    print ">>> Shutting down..."
    print "[*] Unbinding textures..."
    cudaUnbindTexture(texData)
    cudaUnbindTexture(texKernel)
    print "[*] Destroying FFT plans..."
    cufftDestroy(FFTplan)
    print "[*] Freeing device memory..."
    cudaFree(d_PaddedData)
    cudaFree(d_PaddedKernel)
    cudaFreeArray(a_Data)
    cudaFreeArray(a_Kernel)
    #free(h_ResultGPU)
    #free(h_ResultCPU)
    #free(h_Data)
    #free(h_Kernel)
    print "[*] cudaThreadExit()"

    cudaThreadExit()

if __name__ == "__main__":
    main()