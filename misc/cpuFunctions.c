//  © Arno Pähler, 2007-08
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//  © Arno Pähler, 2007-08
#include <sys/time.h>

typedef const unsigned int _T;

/*
 * Utility functions
*/
unsigned long long int rdtsc(void)
{
   unsigned long long int x;

   __asm__ volatile(".byte 0x0f,0x31" : "=A" (x));
   return x;
}

long int microtime(void)
{
#define usec 1000000
    long int ts;
    struct timeval t;
    gettimeofday(&t,0);
    ts = t.tv_sec*usec+t.tv_usec;
    return ts;
}

void arrayInit
    (float *a, _T size)
{
    int i;
    float f;
    for (i = 0; i < size; i++){
        f = (float)drand48();
        if (f <= 0.f)
            f = 2.e-23f;
        a[i] = f;
    }
}

void fixedInit
    (float *a, _T size)
{
    int i;
    float f = 2.e-3f;
    for (i = 0; i < size; i++){
        a[i] = (float)(size-i)*f;
    }
}

void randInit
    (float *a, _T size, float low, float high)
{
    int i;
    float f;
    for (i = 0; i < size; i++){
        f = (float)drand48();
        if (f <= 0.f)
            f = 2.e-23f;
        a[i] = (1.0f - f) * low + f * high;
    }
}

void setZero
    (float *a, _T size)
{
    int i;
    for (i = 0; i < size; i++)
        a[i] = 0.f;
}

void checkError
    (float *a, float *b, _T size, float *err, float *mxe)
{
    int i;
    float ae,e,m;
    e = 0.f;
    m = -1.f;
    for (i = 0; i < size; i++)
    {
        ae = fabs(a[i]-b[i]);
        e += ae;
        if (m < ae)
            m = ae;
    }
    e /= (float) size;
    *err = e;
    *mxe = m;
}

void checkTrig
    (float *e, float *m, float *c, float *s, _T size)
{
    int i;
    float a,t,x;

    a = 0.f;
    t = 0.f;
    x = 0.f;
    for (i = 0; i < size; i++)
    {
        a = fabs(1.f-sqrtf(c[i]*c[i]+s[i]*s[i]));
        t += a;
        if (x < a)
            x = a;
    }
    *e = t/(float)size;
    *m = x;
}

/*
 * Math functions
*/
////////////////////////////////////////////////////////////////////////

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

#define NUM_THREADS_PER_BLOCK 192
#define NUM_ITERATIONS 512

static float result[NUM_THREADS_PER_BLOCK];

void gflops()
{
// this ensures the mads don't get compiled out
    float a = result[0];
    float b = 1.01f;
    int i,j;

    for (j = 0; j < NUM_THREADS_PER_BLOCK; j++)
    {
        for (i = 0; i < NUM_ITERATIONS; i++)
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
        result[j] = a + b;
    }
}
////////////////////////////////////////////////////////////////////////
#define A1 0.31938153f
#define A2 -0.356563782f
#define A3 1.781477937f
#define A4 -1.821255978f
#define A5 1.330274429f
#define RSQRT2PI 0.3989422804f

//Polynomial approximation of cumulative normal distribution function
float CND
    (float d)
{
    float K, cnd;
    K = 1.0f / (1.0f + 0.2316419f * fabsf(d));
    cnd = RSQRT2PI * expf(- 0.5f * d * d) *
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    if(d > 0)
        cnd = 1.0f - cnd;
    return cnd;
}

// Calculate Black-Scholes formula for both call and put
//     float S, //Stock price
//     float X, //Option strike
//     float T, //Option years
//     float R, //Riskless rate
//     float V  //Volatility rate
void BlackScholesBody
    (float *Calls, float *Puts,
    float S, float X, float T, float R,float V)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = sqrtf(T);
    d1 = (logf(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = CND(d1);
    CNDD2 = CND(d2);

    expRT = expf(- R * T);
    *Calls = S * CNDD1 - X * expRT * CNDD2;
    *Puts  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}
void blsc
    (float *h_Calls, float *h_Puts,
    float *h_S, float *h_X, float *h_T, float R, float V, int OptN)
{
    int opt;
    for(opt = 0; opt < OptN; opt++)
        BlackScholesBody(&h_Calls[opt], &h_Puts[opt],
            h_S[opt], h_X[opt], h_T[opt], R, V);
}
////////////////////////////////////////////////////////////////////////
void poly5
    (const float *X, float *Y, _T size)
{
    float a0 = 1.f;
    float a1 = 2.f;
    float a2 = 3.f;
    float a3 = 4.f;
    float a4 = 5.f;
    float p,q;
    unsigned int i, j;
    for (i = 0; i < size; ++i)
    {
        q = X[i];
        p = (((a0*q+a1)*q+a2)*q+a3)*q+a4;
        Y[i] = p;
    }
}
void poly10
    (const float *X, float *Y, _T size)
{
    float a0 = 1.f;
    float a1 = 2.f;
    float a2 = 3.f;
    float a3 = 4.f;
    float a4 = 5.f;
    float p,q;
    unsigned int i, j;
    for (i = 0; i < size; ++i)
    {
        q = X[i];
        p = (((a0*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        Y[i] = p;
    }
}
void poly20
    (const float *X, float *Y, _T size)
{
    float a0 = 1.f;
    float a1 = 2.f;
    float a2 = 3.f;
    float a3 = 4.f;
    float a4 = 5.f;
    float p,q;
    unsigned int i, j;
    for (i = 0; i < size; ++i)
    {
        q = X[i];
        p = (((a0*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        Y[i] = p;
    }
}
void poly40
    (const float *X, float *Y, _T size)
{
    float a0 = 1.f;
    float a1 = 2.f;
    float a2 = 3.f;
    float a3 = 4.f;
    float a4 = 5.f;
    float p,q;
    unsigned int i, j;
    for (i = 0; i < size; ++i)
    {
        q = X[i];
        p = (((a0*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        p = ((( p*q+a1)*q+a2)*q+a3)*q+a4;
        Y[i] = p;
    }
}
////////////////////////////////////////////////////////////////////////
void saxpy
    (const float alpha, const float *X, float *Y, _T size)
{
    unsigned int i;
    for (i = 0; i < size; ++i)
        Y[i] = Y[i] + alpha * X[i];
}

void vadd
    (const float *X, float *Y, _T size)
{
    unsigned int i;
    for (i = 0; i < size; ++i)
        Y[i] = Y[i] + X[i];
}

float sdot
    (const float *X, float *Y, _T size)
{
    unsigned int i;
    float result = 0.f;
    for (i = 0; i < size; ++i)
        result += X[i] * Y[i];
    return result;
}

void sgemm
    (float *C, const float *A, const float *B, _T hA, _T wA, _T wB)
{
    unsigned int i, j, k, iA, iB;
    double a, b, sum;
    for (i = 0; i < hA; ++i) {
        iA = i * wA;
        iB = i * wB;
        for (j = 0; j < wB; ++j) {
            sum = 0.;
            for (k = 0; k < wA; ++k) {
                a = A[iA + k];
                b = B[k * wB + j];
                sum += a * b;
            }
            C[iB + j] = (float)sum;
        }
    }
}

void trig
    (float *Y, float *Z, const float *X, _T size)
{
    unsigned int i;
    double x;
    for (i = 0; i < size; ++i) {
        x = (double)X[i];
        Y[i] = (float)cos(x);
        Z[i] = (float)sin(x);
    }
}

void scale
    (float *D, const float scale, _T size)
{
    unsigned int i;
    for (i = 0; i < size; ++i) {
        D[i] *= scale;
    }
}

float l1norm
    (const float *A, const float *B, _T size)
{
    unsigned int i;
    float s = 0.f;
    for (i = 0; i < size; ++i) {
        s += fabsf(A[i]-B[i]);
    }
    return s;
}
