//
// (c) January 24, 2008 Vasily Volkov @ UC Berkeley
//
// Other credits:
// - Paul Leventis @ Altera Corp. for prefetching and -maxrregcount techniques
// - many thanks to Wladimir J. van der Laan @ the University of Groningen
// for his cubin disassembler (http://www.cs.rug.nl/~wladimir/decuda/)
//
// Compile with -maxrregcount 32
//

//#include <windows.h>
#include <sys/time.h>
#include <stdio.h>
#include "cuda.h"
#include "cublas.h"

//
//	generic functions first
//
#define BEGIN_TIMING(time)	\
{	\
	unsigned int n_iterations;	\
	for( n_iterations = 1; n_iterations < 0x80000000; n_iterations *= 2 )	\
	{	\
		cudaThreadSynchronize();		\
		time = read_timer( );	\
		for( unsigned int iteration = 0; iteration < n_iterations; iteration++ ){

#define END_TIMING(time,timer_tol) }\
		cudaThreadSynchronize();		\
		time = read_timer( ) - time;	\
		if (time >= timer_tol)	\
			break;	\
	}	\
	time /= n_iterations;\
}

double read_timer( )
{	
/*
	static bool initialized = false;
	static LARGE_INTEGER time0;
	static double dfreq;
	
	LARGE_INTEGER temp;
	
	if( !initialized )
	{
		QueryPerformanceFrequency (&temp);
		dfreq = 1.0/temp.QuadPart;
		QueryPerformanceCounter (&time0);
		initialized = true;
	}
	QueryPerformanceCounter (&temp);
	double time = (temp.QuadPart - time0.QuadPart)*dfreq;
*/
   struct timeval tbf;
   int failed;
   double time;
   failed = gettimeofday(&tbf, 0);
   if (failed)
       return 0.;
   time = (double)tbf.tv_sec + 1.e-6*(double)tbf.tv_usec;
	return time;
}	

void error( char *message )
{
	fprintf( stderr, "ERROR: %s\n", message );
	exit (1);
}

#define assert( condition, ... ) { if( !( condition ) ) error( __VA_ARGS__ ); }

inline void Q( cudaError_t status ) { assert( status == cudaSuccess, "CUDA fails" ); }
inline void Q( cublasStatus status ){ assert( status == CUBLAS_STATUS_SUCCESS, "CUBLAS fails" ); }

//
//	SGEMM routines
//
__device__ void saxpy( float a, float *b, float *c )
{
	c[0] += a*b[0];
	c[1] += a*b[1];
	c[2] += a*b[2];
	c[3] += a*b[3];
	c[4] += a*b[4];
	c[5] += a*b[5];
	c[6] += a*b[6];
	c[7] += a*b[7];
	c[8] += a*b[8];
	c[9] += a*b[9];
	c[10] += a*b[10];
	c[11] += a*b[11];
	c[12] += a*b[12];
	c[13] += a*b[13];
	c[14] += a*b[14];
	c[15] += a*b[15];
}

__global__ void sgemmNT( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{
	const int inx = threadIdx.x;
	const int iny = threadIdx.y;
	const int ibx = blockIdx.x * 64;
	const int iby = blockIdx.y * 16;
	const int id  = inx + iny*16;

	A += ibx + id;
	B += iby + inx + __mul24( iny, ldb );
	C += ibx + id  + __mul24( iby, ldc );
	
	float a[4] = {A[0], A[lda], A[2*lda], A[3*lda]};
	float b = B[0];
	
	const float *Blast = B + k*ldb;

	A += 4*lda;
	B += 4*ldb;
    
	__shared__ float bs[4][16];
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
	do
	{
		float as[4] = {a[0], a[1], a[2], a[3]};
		
		bs[iny][inx] = b;
		__syncthreads();
		
		a[0] = A[0*lda];
		a[1] = A[1*lda];
		a[2] = A[2*lda];
		a[3] = A[3*lda];
		b    = B[0];
		
		saxpy( as[0], &bs[0][0], c );
		saxpy( as[1], &bs[1][0], c );
		saxpy( as[2], &bs[2][0], c );
		saxpy( as[3], &bs[3][0], c );
		
		A += 4*lda;
		B += 4*ldb;
		__syncthreads();
		
	} while( B < Blast );
	
	bs[iny][inx] = b;
	__syncthreads();
	
	saxpy( a[0], &bs[0][0], c );
	saxpy( a[1], &bs[1][0], c );
	saxpy( a[2], &bs[2][0], c );
	saxpy( a[3], &bs[3][0], c );

	for( int i = 0; i < 16; i++, C += ldc )
		C[0] = alpha*c[i] + beta*C[0];
}	

__global__ void sgemmNN( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{
	const int inx = threadIdx.x;
	const int iny = threadIdx.y;
	const int ibx = blockIdx.x * 64;
	const int iby = blockIdx.y * 16;
	const int id = inx + iny*16;
	
	A += ibx + id;
	B += inx + __mul24( iby + iny, ldb );
	C += ibx + id  + __mul24( iby, ldc );
	
	const float *Blast = B + k;
	
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
	do
	{
		float a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

		__shared__ float bs[16][17];
		bs[inx][iny]    = B[0*ldb];
		bs[inx][iny+4]  = B[4*ldb];
		bs[inx][iny+8]  = B[8*ldb];
		bs[inx][iny+12] = B[12*ldb];
		__syncthreads();

		A += 4*lda;
		saxpy( a[0], &bs[0][0], c );		a[0] = A[0*lda];
		saxpy( a[1], &bs[1][0], c );		a[1] = A[1*lda];
		saxpy( a[2], &bs[2][0], c );		a[2] = A[2*lda];
		saxpy( a[3], &bs[3][0], c );		a[3] = A[3*lda];	

		A += 4*lda;
		saxpy( a[0], &bs[4][0], c );		a[0] = A[0*lda];
		saxpy( a[1], &bs[5][0], c );		a[1] = A[1*lda];
		saxpy( a[2], &bs[6][0], c );		a[2] = A[2*lda];
		saxpy( a[3], &bs[7][0], c );		a[3] = A[3*lda];
		
		A += 4*lda;
		saxpy( a[0], &bs[8][0], c );		a[0] = A[0*lda];
		saxpy( a[1], &bs[9][0], c );		a[1] = A[1*lda];
		saxpy( a[2], &bs[10][0], c );		a[2] = A[2*lda];
		saxpy( a[3], &bs[11][0], c );		a[3] = A[3*lda];
		
		A += 4*lda;
		saxpy( a[0], &bs[12][0], c );
		saxpy( a[1], &bs[13][0], c );
		saxpy( a[2], &bs[14][0], c );
		saxpy( a[3], &bs[15][0], c );
		
		B += 16;
		__syncthreads();
	} while( B < Blast );
	
	for( int i = 0; i < 16; i++, C += ldc )
		C[0] = alpha*c[i] + beta*C[0]; 
}	

extern "C" void ourSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{	
	assert( transa == 'N' || transa == 'n', "unsupported value of 'transa' in ourSgemm()" );
	assert( transb == 'N' || transb == 'n' || transb == 'T' || transb == 't' || transb == 'C' || transb == 'c',
		"invalid value of 'transb' in ourSgemm()" );
	assert( (m%64) == 0 && (n%16) == 0, "unsupported dimensions of matrix C in ourSgemm()" );
	
	dim3 grid( m/64, n/16 ), threads( 16, 4 );
	if( transb == 'N' || transb == 'n' )
	{
		assert( (k%16) == 0 && k > 0, "unsupported shared dimension in ourSgemm( 'N', 'N', ... )" );
		sgemmNN<<<grid, threads>>>( A, lda, B, ldb, C, ldc, k, alpha, beta );
	}
	else
	{
		assert( (k%4) == 0 && k > 4, "unsupported shared dimension in ourSgemm( 'N', 'T', ... )" );
		sgemmNT<<<grid, threads>>>( A, lda, B, ldb, C, ldc, k, alpha, beta );
	}
}	

//
//	auxiliary routines
//	
void fill( float *A, int n, int maxi )
{	
	for( int j = 0; j < n; j++ )
		A[j] = float( (rand()%(maxi*2+1)) - maxi ) / ( maxi + 1. );
}	

float diff( int m, int n, float *A, int lda, float *B, int ldb )
{
	float err = 0;
	for( int j = 0; j < n; j++ )
		for( int i = 0; i < m; i++ )
			err = max( err, fabs( A[i+j*lda] - B[i+j*ldb] ) );
	return err;
}

//
//	main()
//
int main()
{	
	const int N = 4096;
	
	//
	//	init cublas and arrays
	//
	
	Q( cublasInit( ) );
	
	float *dA, *dB, *dC;
	Q( cublasAlloc( N*N, sizeof(float), (void**)&dA ) );
	Q( cublasAlloc( N*N, sizeof(float), (void**)&dB ) );
	Q( cublasAlloc( N*N, sizeof(float), (void**)&dC ) );
	
	float *A = (float*)malloc( N*N*sizeof( float ) );
	float *B = (float*)malloc( N*N*sizeof( float ) );
	float *C = (float*)malloc( N*N*sizeof( float ) );
	
	assert( A != NULL && B != NULL && C != NULL, "memory allocation error" );
	
	fill( A, N*N, 31 );
	fill( B, N*N, 31 );
	fill( C, N*N, 31 );
	
	Q( cudaMemcpy( dA, A, N*N*sizeof(float), cudaMemcpyHostToDevice ) );
	Q( cudaMemcpy( dB, B, N*N*sizeof(float), cudaMemcpyHostToDevice ) );
	
	float *cublas_result = (float*)malloc( N*N*sizeof( float ) );
	float *our_result = (float*)malloc( N*N*sizeof( float ) );
	
	assert( cublas_result != NULL && our_result != NULL, "memory allocation error" );
	
	//
	//	bench square matrices
	//
	for( int i = 0; i < 2; i++ )
	{
		const char transa = 'N';
		const char transb = i ? 'T' : 'N';

		printf( "\ntesting sgemm( '%c', '%c', n, n, n, ... )\n\n", transa, transb );
		
		const int nb = 64;
		printf( "   n   CUBLAS,Gflop/s   we,Gflop/s   \"error\"\n" );
		for( int idim = 1; idim <= N/nb; idim = int((idim+1)*1.1) )
		{
			int dim = idim*nb;
			
			//
			//	set up the parameters
			//
			const int m = dim;
			const int n = dim;
			const int k = dim;
			const int lda = dim;
			const int ldb = dim;
			const int ldc = dim;
			const float alpha = 1;
			const float beta = -1;

			//
			// compute with CUBLAS
			//
			Q( cublasSetMatrix( m, n, sizeof( float ), C, ldc, dC, ldc ) );
			cublasSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
			Q( cublasGetError( ) );
			Q( cublasGetMatrix( m, n, sizeof( float ), dC, ldc, cublas_result, ldc ) );
			
			//
			// compute with our routine
			//
			Q( cublasSetMatrix( m, n, sizeof( float ), C, ldc, dC, ldc ) );
			ourSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
			Q( cublasGetMatrix( m, n, sizeof( float ), dC, ldc, our_result, ldc ) );
			
			//
			//	check the difference in results
			//
			float difference = diff( m, n, cublas_result, ldc, our_result, ldc );
			
			//
			//	bench cublas
			//
			double cublas_time;
			cublasSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
			BEGIN_TIMING( cublas_time );
				cublasSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
			END_TIMING( cublas_time, 0.1 );
			
			double cublas_gflops = 2.*m*n*k/cublas_time/1e9;
			
			//
			//	bench our routine
			//
			double our_time;
			ourSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
			BEGIN_TIMING( our_time );
				ourSgemm( transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc );
			END_TIMING( our_time, 0.1 );
			
			double our_gflops = 2.*m*n*k/our_time/1e9;
			
			//
			//	report the results
			//
			printf( "%5d %11.2f %14.2f %8g\n", n, cublas_gflops, our_gflops, difference );
		}
	}
	
	//
	//	shutdown
	//
	
	cublasFree( dA );
	cublasFree( dB );
	cublasFree( dC );
	
	free( A );
	free( B );
	free( C );
	
	free( cublas_result );
	free( our_result );
		
	Q( cublasShutdown( ) );
	
	return 0;
}	
