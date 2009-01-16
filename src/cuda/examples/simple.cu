// © Arno Pähler, 2007-08
extern "C" {
typedef float _F;
typedef const float _cF;
typedef const unsigned int _cI;

texture<float,1,cudaReadModeElementType> Arg;

__global__ void TRIG
    (_F *d_Out1, _F *d_Out2, _cF *d_In1, _cI size )
{
    _cI tid = blockDim.x * blockIdx.x + threadIdx.x;
    _cI tsz = blockDim.x * gridDim.x;
    int i;

    for (i = tid; i < size; i += tsz)
    {
        d_Out1[i] = cosf(d_In1[i]);
        d_Out2[i] = sinf(d_In1[i]);
    }
}

__global__ void TRIGTex
    (_F *d_Out1, _F *d_Out2, _cI size )
{
    _cI tid = blockDim.x * blockIdx.x + threadIdx.x;
    _cI tsz = blockDim.x * gridDim.x;
    int i;
    __shared__ float x;

    for (i = tid; i < size; i += tsz)
    {
        x = tex1Dfetch(Arg,i);
        d_Out1[i] = cosf(x);
        d_Out2[i] = sinf(x);
    }
}
}
