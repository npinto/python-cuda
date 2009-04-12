import random, math
import sys
from ctypes import sizeof
from cuda.cuda import float2


# //////////////////////////////////////////////////////////////////////////////
# Helper functions
# //////////////////////////////////////////////////////////////////////////////

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

# //////////////////////////////////////////////////////////////////////////////
# Padding kernels
# //////////////////////////////////////////////////////////////////////////////
#include "convolutionFFT2D_kernel.cu"

# //////////////////////////////////////////////////////////////////////////////
# Data configuration
# //////////////////////////////////////////////////////////////////////////////

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
Complex = float2
FFT_SIZE = FFT_W * FFT_H * sizeof(Complex)
KERNEL_SIZE = KERNEL_W * KERNEL_H * sizeof(Complex)
DATA_SIZE = DATA_W * DATA_H * sizeof(Complex)



# //////////////////////////////////////////////////////////////////////////////
# Main program
# //////////////////////////////////////////////////////////////////////////////
def main():
    #cudaArray      *a_Kernel,
    #               *a_Data

    #cudaChannelFormatDesc float2tex = cudaCreateChannelDesc < float2 > ()

    #Complex        *d_PaddedKernel,
    #               *d_PaddedData

    #cufftHandle     FFTplan

    #Complex         rCPU,
    #                rGPU



    #unsigned hTimer


    # use command-line specified CUDA device, otherwise use device with
    # highest Gflops/s

    #if cutCheckCmdLineFlag(argc, (const : argv, "device"))
	#cutilDeviceInit(argc, argv)
    #else:
	#cudaSetDevice(cutGetMaxGflopsDeviceId())

    #cutilCheckError(cutCreateTimer(&hTimer))

    print "Input data size           : %i x %i" % (DATA_W, DATA_H)
    print "Convolution kernel size   : %i x %i" % (KERNEL_W, KERNEL_H)
    print "Padded image size         : %i x %i" % (DATA_W + PADDING_W, DATA_H + PADDING_H)
    print "Aligned padded image size : %i x %i" % (FFT_W, FFT_H)

    print "Allocating memory..."
    print "KERNEL_SIZE = ", KERNEL_SIZE
    #h_Kernel = (Complex *) malloc(KERNEL_SIZE)
    #h_Data = (Complex *) malloc(DATA_SIZE)
    #h_ResultCPU = (Complex *) malloc(DATA_SIZE)
    #h_ResultGPU = (Complex *) malloc(FFT_SIZE)
    #cutilSafeCall(cudaMallocArray
	#	  (&a_Kernel, &float2tex, KERNEL_W, KERNEL_H))
    #cutilSafeCall(cudaMallocArray(&a_Data, &float2tex, DATA_W, DATA_H))
    #cutilSafeCall(cudaMalloc(() &d_PaddedKernel, FFT_SIZE))
    #cutilSafeCall(cudaMalloc(() &d_PaddedData, FFT_SIZE))

    #printf("Generating random input data...\n")
    #random.seed(2007)
    #for i in range((KERNEL_W * KERNEL_H)):
	#h_Kernel[i].x = (float) rand() / (float) RAND_MAX
	#h_Kernel[i].y = 0
    #for i in range((DATA_W * DATA_H)):
	#h_Data[i].x = (float) rand() / (float) RAND_MAX
	#h_Data[i].y = 0

    #printf("Creating FFT plan for %i x %i...\n" % (FFT_W, FFT_H))
    #cufftSafeCall(cufftPlan2d(&FFTplan, FFT_H, FFT_W, CUFFT_C2C))

    #printf
	#("Uploading to GPU and padding convolution kernel and input data...\n")
    #printf
	#("...initializing padded kernel and data storage with zeroes...\n")
    #cutilSafeCall(cudaMemset(d_PaddedKernel, 0, FFT_SIZE))
    #cutilSafeCall(cudaMemset(d_PaddedData, 0, FFT_SIZE))
    #printf
	#("...copying input data and convolution kernel from host to CUDA arrays\n")
    #cutilSafeCall(cudaMemcpyToArray
	#	  (a_Kernel, 0, 0, h_Kernel, KERNEL_SIZE,
	#	   cudaMemcpyHostToDevice))
    #cutilSafeCall(cudaMemcpyToArray
	#	  (a_Data, 0, 0, h_Data, DATA_SIZE,
	#	   cudaMemcpyHostToDevice))
    #printf("...binding CUDA arrays to texture references\n")
    #cutilSafeCall(cudaBindTextureToArray(texKernel, a_Kernel))
    #cutilSafeCall(cudaBindTextureToArray(texData, a_Data))

    ## Block width should be a multiple of maximum coalesced write size 
    ## for coalesced memory writes in padKernel() and padData()
    #dim3            threadBlock(16, 12)
    #dim3            kernelBlockGrid(iDivUp(KERNEL_W, threadBlock.x),
	#			    iDivUp(KERNEL_H, threadBlock.y))
    #dim3            dataBlockGrid(iDivUp(FFT_W, threadBlock.x),
	#			  iDivUp(FFT_H, threadBlock.y))

    #printf("...padding convolution kernel\n")
    #padKernel <<< kernelBlockGrid, threadBlock >>> (d_PaddedKernel,
	#					    FFT_W,
	#					    FFT_H,
	#					    KERNEL_W,
	#					    KERNEL_H,
	#					    KERNEL_X, KERNEL_Y)
    #cutilCheckMsg("padKernel() execution failed\n")

    #printf("...padding input data array\n")
    #padData <<< dataBlockGrid, threadBlock >>> (d_PaddedData,
	#					FFT_W,
	#					FFT_H,
	#					DATA_W,
	#					DATA_H,
	#					KERNEL_W,
	#					KERNEL_H,
	#					KERNEL_X, KERNEL_Y)
    #cutilCheckMsg("padData() execution failed\n")

    ## Not including kernel transformation into time measurement,
    ## since convolution kernel is not changed very frequently
    #printf("Transforming convolution kernel...\n")
    #cufftSafeCall(cufftExecC2C
	#	  (FFTplan, (cufftComplex *) d_PaddedKernel,
	#	   (cufftComplex *) d_PaddedKernel, CUFFT_FORWARD))

    #printf("Running GPU FFT convolution...\n")
    #cutilSafeCall(cudaThreadSynchronize())
    #cutilCheckError(cutResetTimer(hTimer))
    #cutilCheckError(cutStartTimer(hTimer))
    #cufftSafeCall(cufftExecC2C
	#	  (FFTplan, (cufftComplex *) d_PaddedData,
	#	   (cufftComplex *) d_PaddedData, CUFFT_FORWARD))
    #modulateAndNormalize <<< 16, 128 >>> (d_PaddedData, d_PaddedKernel,
	#				  FFT_W * FFT_H)
    #cutilCheckMsg("modulateAndNormalize() execution failed\n")
    #cufftSafeCall(cufftExecC2C
	#	  (FFTplan, (cufftComplex *) d_PaddedData,
	#	   (cufftComplex *) d_PaddedData, CUFFT_INVERSE))
    #cutilSafeCall(cudaThreadSynchronize())
    #cutilCheckError(cutStopTimer(hTimer))
    #gpuTime = cutGetTimerValue(hTimer)
    #printf("GPU time: %f msecs. //%f MPix/s\n" % (gpuTime,
	#   DATA_W * DATA_H * 1e-6 / (gpuTime * 0.001)))

    #printf("Reading back GPU FFT results...\n")
    #cutilSafeCall(cudaMemcpy
	#	  (h_ResultGPU, d_PaddedData, FFT_SIZE,
	#	   cudaMemcpyDeviceToHost))


    #printf("Checking GPU results...\n")
    #printf("...running reference CPU convolution\n")
    #convolutionCPU(h_ResultCPU,
	#	   h_Data,
	#	   h_Kernel,
	#	   DATA_W, DATA_H, KERNEL_W, KERNEL_H, KERNEL_X, KERNEL_Y)

    #printf("...comparing the results\n")
    #sum_delta2 = 0
    #sum_ref2 = 0
    #max_delta_ref = 0
    #for y in range(0, DATA_H):
	#for x in range(DATA_W):
	#    rCPU = h_ResultCPU[y * DATA_W + x]
	#    rGPU = h_ResultGPU[y * FFT_W + x]
	#    delta =
	#	(rCPU.x - rGPU.x) * (rCPU.x - rGPU.x) + (rCPU.y -
	#						 rGPU.y) *
	#	(rCPU.y - rGPU.y)
	#    ref = rCPU.x * rCPU.x + rCPU.y * rCPU.y
	#    if (delta / ref) > max_delta_ref:
	#	max_delta_ref = delta / ref
	#    sum_delta2 += delta
	#    sum_ref2 += ref
    #L2norm = math.sqrt(sum_delta2 / sum_ref2)
    #printf("Max delta / CPU value %E\n" % (math.sqrt(max_delta_ref)))
    #printf("L2 norm: %E\n" % (L2norm))
    #printf((L2norm < 1e-6) ? "TEST PASSED\n" : "TEST FAILED\n")


    #printf("Shutting down...\n")
    #cutilSafeCall(cudaUnbindTexture(texData))
    #cutilSafeCall(cudaUnbindTexture(texKernel))
    #cufftSafeCall(cufftDestroy(FFTplan))
    #cutilSafeCall(cudaFree(d_PaddedData))
    #cutilSafeCall(cudaFree(d_PaddedKernel))
    #cutilSafeCall(cudaFreeArray(a_Data))
    #cutilSafeCall(cudaFreeArray(a_Kernel))
    #free(h_ResultGPU)
    #free(h_ResultCPU)
    #free(h_Data)
    #free(h_Kernel)

    #cudaThreadExit()

    #cutilExit(argc, argv)

if __name__ == "__main__":
    main()
