// © Arno Pähler, 2007-08
// gflops example: Simon Green (?)
// blsc from NVIDIA SDK
extern "C" {

typedef const float _F;

// 128 MAD instructions
#define FMAD128(a, b) \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \
     a = b * a + b; \
     b = a * b + a; \

#define NUM_THREADS_PER_BLOCK 768
#define NUM_ITERATIONS 512

__shared__ float result[NUM_THREADS_PER_BLOCK];

__global__ void gpuGFLOPS()
{
// this ensures the mads don't get compiled out
   float a = result[threadIdx.x];
   float b = 1.01f;

   for (int i = 0; i < NUM_ITERATIONS; i++)
   {
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);

       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);

       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);

       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
       FMAD128(a, b);
   }
   result[threadIdx.x] = a + b;
}
////////////////////////////////////////////////////////////////////////
#define A1 0.31938153f
#define A2 -0.356563782f
#define A3 1.781477937f
#define A4 -1.821255978f
#define A5 1.330274429f
#define RSQRT2PI 0.3989422804f

//Polynomial approx. of cumulative normal distribution function
__device__ float CND(
    float d){
    float K, cnd;
    K = 1.0f / (1.0f + 0.2316419f * fabsf(d));
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
    (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if(d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}

__device__ void BlackScholesBody(
    float& Call, float& Put,
    float S, float X, float T, float R, float V){
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = sqrtf(T);
    d1 = (__logf(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = CND(d1);
    CNDD2 = CND(d2);

    expRT = __expf(- R * T);
    Call = S * CNDD1 - X * expRT * CNDD2;
    Put  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}

__global__ void gpuBLSC(
    float *d_Calls, float *d_Puts,
    float *d_S, float *d_X, float *d_T,
    float R, float V, int OptN){
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int THREAD_N = blockDim.x * gridDim.x;

    for(int opt = tid; opt < OptN; opt += THREAD_N)
        BlackScholesBody(d_Calls[opt], d_Puts[opt],
            d_S[opt], d_X[opt], d_T[opt], R, V);
}
////////////////////////////////////////////////////////////////////////
__global__ void gpuPOLY5(
    float *d_In1, float *d_Out1, int size ){
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tsz = blockDim.x * gridDim.x;

    float a0 = 1.f;
    float a1 = 2.f;
    float a2 = 3.f;
    float a3 = 4.f;
    float a4 = 5.f;
    float p,q;
    for (int i = tid; i < size; i += tsz) {
        q = d_In1[i];
        p = (((a0*q+a1)*q+a2)*q+a3)*q+a4;
        d_Out1[i] = p;
        }
}
__global__ void gpuPOLY10(
    float *d_In1, float *d_Out1, int size ){
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tsz = blockDim.x * gridDim.x;

    float a0 = 1.f;
    float a1 = 2.f;
    float a2 = 3.f;
    float a3 = 4.f;
    float a4 = 5.f;
    float p,q;
    for (int i = tid; i < size; i += tsz) {
        q = d_In1[i];
        p = (((a0*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        d_Out1[i] = p;
        }
}
__global__ void gpuPOLY20(
    float *d_In1, float *d_Out1, int size ){
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tsz = blockDim.x * gridDim.x;

    float a0 = 1.f;
    float a1 = 2.f;
    float a2 = 3.f;
    float a3 = 4.f;
    float a4 = 5.f;
    float p,q;
    for (int i = tid; i < size; i += tsz) {
        q = d_In1[i];
        p = (((a0*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        d_Out1[i] = p;
        }
}
__global__ void gpuPOLY40(
    float *d_In1, float *d_Out1, int size ){
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tsz = blockDim.x * gridDim.x;

    float a0 = 1.f;
    float a1 = 2.f;
    float a2 = 3.f;
    float a3 = 4.f;
    float a4 = 5.f;
    float p,q;
    for (int i = tid; i < size; i += tsz) {
        q = d_In1[i];
        p = (((a0*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        d_Out1[i] = p;
        }
}
////////////////////////////////////////////////////////////////////////
__global__ void gpuSAXPY(
    float Factor, float *d_In1, float *d_In2, int size ){
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tsz = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += tsz)
        d_In2[i] = d_In2[i] + d_In1[i] * Factor;
}

__global__ void gpuVADD(
    float *d_In1, float *d_In2, int size ){
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tsz = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += tsz)
        d_In2[i] = d_In2[i] + d_In1[i];
}

__global__ void gpuSGEMM(
    float* C, float* A, float* B, int wA, int wB ){
#define BLOCK_SIZE 16

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;

    float Cs   = 0;

    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            Cs += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Cs;
}

__global__ void gpuTRIG(
    float *d_Out1, float *d_Out2, float *d_In1, int size ){
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tsz = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += tsz) {
        d_Out1[i] = cosf(d_In1[i]);
        d_Out2[i] = sinf(d_In1[i]);
    }
}

__global__ void gpuScale(
    float *d_Out1, _F *d_In1, _F scale, int size ){
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tsz = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += tsz) {
        d_Out1[i] = d_In1[i]*scale;
    }
}

// for streams example
__global__ void init_array(
    int *g_data, int *factor){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = *factor;
}
}
