import sys
import os
from ctypes import *
import numpy
from scipy.signal import fftconvolve, convolve2d
from conv_gold import get_convolution_cpu, get_check_results,centered

from cuda.cufft import *
from cuda.cuda import * 
from cuda.kernel.kernelfactoryrt import SourceModule

def convolutionCPU(h_Data, h_Kernel, dataW, dataH, kernelW, kernelH, kernelX, kernelY):
    h_Result = numpy.zeros_like(h_Data).astype('complex64')
    for y in range(0,dataH):
        for x in range(0,dataW):
            sum = 0j
            for ky in range(-(kernelH-kernelY-1), kernelY+1):
                for kx in range(-(kernelW-kernelX-1), kernelX+1):
                    dx = x + kx
                    dy = y + ky
                    if dx < 0: dx = 0
                    if dy < 0: dy = 0
                    if dx >= dataW: dx = dataW - 1
                    if dy >= dataH: dy = dataH - 1
                    sum += h_Data[dx,dy]*h_Kernel[(kernelX-kx),(kernelY-ky)]
            h_Result[x,y] = sum
    return h_Result

def convolve(image1, image2, MinPad=True, pad=True):
    from numpy.fft import fft2, ifft2 
    from numpy import log, max
    """ Not so simple convolution """

    #Just for comfort:
    FFt = fft2
    iFFt = ifft2

    #The size of the images:
    r1,c1 = image1.shape
    r2,c2 = image2.shape

    #MinPad results simpler padding,smaller images:
    if MinPad:
        r = r1+r2
        c = c1+c2
    else:
    #if the Numerical Recipies says so:
        r = 2*max(r1,r2)
        c = 2*max(c1,c2)

    #For nice FFT, we need the power of 2:
    if pad:
        pr2 = int(log(r)/log(2.0) + 1.0 )
        pc2 = int(log(c)/log(2.0) + 1.0 )
        rOrig = r
        cOrig = c
        r = 2**pr2
        c = 2**pc2

    #numpy fft has the padding built in, which can save us some steps
    #here. The thing is the s(hape) parameter:
    fftimage = FFt(image1,s=(r,c)) * FFt(image2,s=(r,c))
    #fftimage = FFt(image1, s=(r,c))*FFt(image2[::-1,::-1],s=(r,c))

    if pad:
        return (iFFt(fftimage))[:rOrig,:cOrig].real
    else:
        return (iFFt(fftimage)).real

def check_results(h_ResultCPU, h_ResultGPU, DATA_W, DATA_H, FFT_W):
    ResultGPU = h_ResultGPU[0:DATA_W,0:DATA_H]
    max_delta_ref = numpy.sqrt(numpy.abs(h_ResultCPU-ResultGPU)**2/numpy.abs(h_ResultCPU)**2).max()
    L2norm = numpy.linalg.norm(h_ResultCPU - ResultGPU)/numpy.linalg.norm(h_ResultCPU)
    print "Max delta / CPU value ", max_delta_ref
    print 'L2 norm: ', L2norm
    if L2norm < 1e-6:
        print "TEST PASSED"
    else:
        print "TEST FAILED"

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
def fftconvolve2d(data, kernel):

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
<<<<<<< HEAD:cuda/sugar/fft/fftconvolve.py
    DATA_W = data.shape[0] 
    DATA_H = data.shape[1]
=======
    DATA_W = 512 
    DATA_H = 512
>>>>>>> 0f090d8cc2ada9949809f38bca555ba2b2a79382:cuda/sugar/fft/fftconvolve.py

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
    kernel_src = os.path.join(os.path.dirname(__file__), 'fftconvolve2d_kernel.cu')
    fftconvolve2d = SourceModule(open(kernel_src,'r').read(), no_extern_c=True)

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
    h_Kernel = kernel
    h_Data = data
    #h_Kernel = numpy.random.uniform(0,1,(KERNEL_W,KERNEL_H)).astype(numpy.complex64)
    #h_Data = numpy.random.uniform(0,1,(DATA_W,DATA_H)).astype(numpy.complex64)

    print "[*] Allocating host memory for results..."
    h_ResultCPU = numpy.zeros((DATA_W,DATA_H)).astype(numpy.complex64)
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

    print ">>> Checking GPU results..."
    print "[*] running reference CPU convolution..."
    conv_gold = get_convolution_cpu() 
    #check_results = get_check_results()

    
    conv_gold(get_float2_ptr(h_ResultCPU), get_float2_ptr(h_Data), get_float2_ptr(h_Kernel), DATA_W, DATA_H, KERNEL_W, KERNEL_H, KERNEL_X, KERNEL_Y)
    #h_ResultCPU = fftconvolve(h_Data.real, h_Kernel.real, mode="valid")
    #print h_ResultCPU.real

    print "[*] comparing the results..."
    
    #get_check_results()(get_float2_ptr(h_ResultCPU), get_float2_ptr(h_ResultGPU), DATA_W, DATA_H, FFT_W)
    check_results(h_ResultCPU.real, h_ResultGPU.real, h_ResultCPU.shape[0], h_ResultCPU.shape[1], FFT_W)

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
    return h_ResultGPU.real

if __name__ == "__main__":
    kernel = numpy.random.uniform(0,1,(7,7)).astype(numpy.complex64)
    data = numpy.random.uniform(0,1,(512,512)).astype(numpy.complex64)
    fftconvolve2d(data, kernel)
