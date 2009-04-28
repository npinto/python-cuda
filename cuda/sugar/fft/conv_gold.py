import os
import ctypes
import numpy as np
from scipy.signal import convolve2d, fftconvolve

source = '''
#include <stdio.h>
#include <math.h>

typedef struct{
    float x, y;
} Complex;

const Complex CPLX_ZERO = {0, 0};

//a += b * c
extern "C" void complexMAD(Complex& a, Complex b, Complex c){
    Complex t = {a.x + b.x * c.x - b.y * c.y, a.y + b.x * c.y + b.y * c.x};
    a = t;
}

extern "C" void printComplexArray(Complex * arr, int rows, int cols) {
    printf("arr[%d,%d] = %f \\n", 0, 0, arr[0].x);
    for(int i=0; i < rows; ++i) {
        for(int j=0; j < cols; ++j) {
            //printf("arr[%d,%d] = %f+i%f \\n", i, j, arr[i*cols+j].x, arr[i*cols+j].y);
        }
    }
}

extern "C" int checkResults(Complex *h_ResultCPU, Complex *h_ResultGPU, int DATA_W, int DATA_H, int FFT_W) {
    Complex rCPU, rGPU;
    double max_delta_ref, delta, ref, sum_delta2, sum_ref2, L2norm;

    sum_delta2 = 0;
    sum_ref2   = 0;
    max_delta_ref = 0;

    for(int y = 0; y < DATA_H; y++)
        for(int x = 0; x < DATA_W; x++){
            rCPU = h_ResultCPU[y * DATA_W + x];
            rGPU = h_ResultGPU[y * FFT_W  + x];
            delta = (rCPU.x - rGPU.x) * (rCPU.x - rGPU.x) + (rCPU.y - rGPU.y) * (rCPU.y - rGPU.y);
            ref   = rCPU.x * rCPU.x + rCPU.y * rCPU.y;
            if((delta / ref) > max_delta_ref) max_delta_ref = delta / ref;
            sum_delta2 += delta;
            sum_ref2   += ref;
        }
    L2norm = sqrt(sum_delta2 / sum_ref2);
    printf("Max delta / CPU value %E\\n", sqrt(max_delta_ref));
    printf("L2 norm: %E\\n", L2norm);
    printf((L2norm < 1e-6) ? "TEST PASSED\\n" : "TEST FAILED\\n");
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Reference straightfroward CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionCPU(
    Complex *h_Result,
    Complex *h_Data,
    Complex *h_Kernel,
    int dataW,
    int dataH,
    int kernelW,
    int kernelH,
    int kernelX,
    int kernelY
){
    for(int y = 0; y < dataH; y++)
        for(int x = 0; x < dataW; x++){
            Complex sum = CPLX_ZERO;

            for(int ky = -(kernelH - kernelY - 1); ky <= kernelY; ky++)
                for(int kx = -(kernelW - kernelX - 1); kx <= kernelX; kx++){
                    int dx = x + kx;
                    int dy = y + ky;
                    if(dx < 0) dx = 0;
                    if(dy < 0) dy = 0;
                    if(dx >= dataW) dx = dataW - 1;
                    if(dy >= dataH) dy = dataH - 1;

                    complexMAD(
                        sum,
                        h_Data[dy * dataW + dx],
                        h_Kernel[(kernelY - ky) * kernelW + (kernelX - kx)]
                    );
                }

            h_Result[y * dataW + x] = sum;
        }
}
'''

file = open('conv_gold.cpp','w')
file.write(source)
file.close()

os.system('rm -f conv_gold.so')
os.system('g++ -fPIC -shared -o conv_gold.so conv_gold.cpp')
conv_gold = ctypes.cdll.LoadLibrary('./conv_gold.so')

print_complex = conv_gold.printComplexArray
convolutionCPU = conv_gold.convolutionCPU
check_results = conv_gold.checkResults

#data = np.ones((3,3)).astype('complex64')
data = np.random.randn(3,3).astype('complex64')
#kernel = np.ones((3,3)).astype('complex64')
kernel = np.random.randn(3,3).astype('complex64')
result = np.zeros_like(data)

class float2(ctypes.Structure):
    pass
float2._fields_ = [
    ('x', ctypes.c_float),
    ('y', ctypes.c_float),
]

def get_float2_ptr(numpy_array):
    return numpy_array.ctypes.data_as(ctypes.POINTER(float2))

def run():
    convolutionCPU(get_float2_ptr(result), get_float2_ptr(data), get_float2_ptr(kernel), data.shape[1], data.shape[0], kernel.shape[1], kernel.shape[0], 1, 6)
    print result
    print
    print fftconvolve(data.real, kernel.real, mode='same').astype('complex64')
    print
    print result[kernel.shape[0]/2:-(kernel.shape[0]/2),kernel.shape[1]/2:-(kernel.shape[1]/2)].astype('complex64')
    print 
    print fftconvolve(data.real, kernel.real, mode='valid').astype('complex64')
    print
    #print convolve2d(data, kernel, mode='full').astype('complex64')
 
run()
