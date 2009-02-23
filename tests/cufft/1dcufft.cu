#define NX 3
#include <stdio.h>
#include <cuda.h>
#include <cufft.h>


int main(int argc, char** argv) {
    cufftHandle plan;
    cufftComplex *data;

    int mem_size = sizeof(cufftComplex)*NX;
    cudaMalloc((void**)&data, mem_size);

    float2 * h_signal = (float2*)malloc(NX*sizeof(float2));

    for (int i = 0; i < NX ; ++i) {
        h_signal[i].x = 1;
        h_signal[i].y = 0;
    }

    printf(">>> memsize = %d\n", mem_size);
    cudaMemcpy(data, h_signal, mem_size, cudaMemcpyHostToDevice);

    printf(">>> Create a 1D FFT plan.\n");
    cufftPlan1d(&plan, NX, CUFFT_C2C, 1);

    printf(">>> Use the CUFFT plan to transform the signal in place.\n");
    cufftExecC2C(plan, data, data, CUFFT_FORWARD);

    cudaMemcpy(h_signal, data, mem_size, cudaMemcpyHostToDevice);

    float2 * h_signal_fft;

    cudaMemcpy(h_signal_fft, data, mem_size, cudaMemcpyDeviceToHost);
    //printf("x = %f, y = %f\n", h_signal_fft[1].x, h_signal_fft[1].y);

    for (int i=0; i < NX; ++i){
        printf("h_signal_fft[%d] = %f + j%f\n", i, h_signal_fft[i].x, h_signal_fft[i].y); //h_signal_fft[i]);
    }

    //printf(">>> Inverse transform the signal in place.\n");
    //cufftExecC2C(plan, data, data, CUFFT_INVERSE);

    printf(">>> Note:\n");
    printf("(1) Divide by number of elements in data set to get back original data\n");
    printf("(2) Identical pointers to input and output arrays implies in-place\n");
    printf("    transformation\n");

    printf(">>> Destroy the CUFFT plan.\n");
    cufftDestroy(plan);
    cudaFree(data);
    return 0;
}
