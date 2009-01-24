#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cutil.h>
#define  NUM    (512)		//NUM must be mutiple of 16 to obtain the best performance

#include "cuSVD_kernel.cu"

int main(int argc, char** argv)
{
    CUT_DEVICE_INIT();
	unsigned int num2 = NUM * NUM;
	unsigned int iteration2 = 1*(128 - 1);
	unsigned int iteration3 = 1*(256 - 1);
	unsigned int iteration4 = 2*(NUM - 1);
	    
    printf("\n");
    
    dim3 grid	(NUM>>1, 1, 1);
	dim3 grid1	((NUM>>1)-1, 1, 1);
	dim3 grid2  ((NUM>>1)-64, 1, 1);
	dim3 grid3  ((NUM>>1)-128, 1, 1);
	dim3 threads(256, 1, 1);	
	float * w;
	float * u;
	float * orign;
	float * unit;
	
	float * d_w;
	float * d_u;
	float * d_w_temp;
	float * d_u_temp;

	w = (float*)malloc(num2 * sizeof(float)); 
	u = (float*)malloc(num2 * sizeof(float));
	orign = (float*)malloc(num2 * sizeof(float));
	unit  = (float*)malloc(num2 * sizeof(float));
	
	FILE *fp;

    if((fp=fopen("C:\\b.dat","rb"))==NULL)
	{
		printf("cannot open file\n");
	}
    for(int i = 0; i < NUM; i++)
    {
    for(int j = 0; j < NUM; j++)
    {
    //orign[i*NUM + j] = 0.001f * (j + 1.0f) + (i + 1.0f);
	//orign[i*NUM + j] = (float)rand()/(float)RAND_MAX;
    fread(&orign[i*NUM + j],sizeof(float),1,fp);
    w[i*NUM + j] = unit[i*NUM + j] = 0.0f;
    //printf("%3.3f	", orign[i*NUM + j]);
    }
    unit[i*NUM + i] = 1.0f;
    //printf("\n");
    }
    fclose(fp);
    
	unsigned int timer = 0;
    float elapsedTimeInMs = 0.0f;
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_w, sizeof(float) * num2));
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_u, sizeof(float) * num2));
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_w_temp, sizeof(float) * num2));
	CUDA_SAFE_CALL( cudaMalloc((void**)&d_u_temp, sizeof(float) * num2));

	CUDA_SAFE_CALL( cudaMemcpy(d_u, unit, sizeof(float) * num2, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy(d_w, orign, sizeof(float) * num2, cudaMemcpyHostToDevice));

	CUT_SAFE_CALL( cutCreateTimer( &timer ) );
    CUT_SAFE_CALL( cutStartTimer( timer));

	for(int i=0;i < 2;i++)
	{
	bjrot8<<<grid, threads, 0>>>(d_w, d_u, 0, 0);
	bjrot8<<<grid, threads, 0>>>(d_w, d_u, 1, 0);
	bjrot8<<<grid, threads, 0>>>(d_w, d_u, 2, 0);
	bjrot8<<<grid, threads, 0>>>(d_w, d_u, 3, 0);
	bjrot8<<<grid, threads, 0>>>(d_w, d_u, 4, 0);
	bjrot8<<<grid, threads, 0>>>(d_w, d_u, 5, 0);
	bjrot8<<<grid, threads, 0>>>(d_w, d_u, 6, 0);

	bjrot8<<<grid1, threads, 0>>>(d_w, d_u, 0, 1);
	bjrot8<<<grid1, threads, 0>>>(d_w, d_u, 1, 1);
	bjrot8<<<grid1, threads, 0>>>(d_w, d_u, 2, 1);
	bjrot8<<<grid1, threads, 0>>>(d_w, d_u, 3, 1);
	bjrot8<<<grid1, threads, 0>>>(d_w, d_u, 4, 1);
	bjrot8<<<grid1, threads, 0>>>(d_w, d_u, 5, 1);
	bjrot8<<<grid1, threads, 0>>>(d_w, d_u, 6, 1);
	}

	for(unsigned int j = 0; j < 1; j++)
	{
		for(int i=0;i < iteration2;i++)
		{
		bjrot<<<grid, threads, 0>>>(d_w_temp, d_w, d_u_temp, d_u, 7, 0);
		bjrot<<<grid, threads, 0>>>(d_w, d_w_temp, d_u, d_u_temp, 7, 0);
		}

		for(int i=0;i < iteration2;i++)
		{
		bjrot<<<grid2, threads>>>(d_w_temp, d_w, d_u_temp, d_u, 7, 128);
		bjrot<<<grid2, threads>>>(d_w, d_w_temp, d_u, d_u_temp, 7, 128);
		}
	}

	for(unsigned int j = 0; j < 1; j++)
	{
		for(int i=0;i < iteration3;i++)
		{
		bjrot<<<grid, threads, 0>>>(d_w_temp, d_w, d_u_temp, d_u, 8, 0);
		bjrot<<<grid, threads, 0>>>(d_w, d_w_temp, d_u, d_u_temp, 8, 0);
		}

		for(int i=0;i < iteration3;i++)
		{
		bjrot<<<grid3, threads>>>(d_w_temp, d_w, d_u_temp, d_u, 8, 256);
		bjrot<<<grid3, threads>>>(d_w, d_w_temp, d_u, d_u_temp, 8, 256);
		}
	}

	for(int i=0;i < iteration4;i++)
	{
	bjrot<<<grid, threads, 0>>>(d_w_temp, d_w, d_u_temp, d_u, 9, 0);
	bjrot<<<grid, threads, 0>>>(d_w, d_w_temp, d_u, d_u_temp, 9, 0);
	}
	
	CUT_SAFE_CALL( cutStopTimer( timer));
	elapsedTimeInMs = cutGetTimerValue( timer);		
	CUDA_SAFE_CALL( cudaMemcpy(u, d_u, sizeof(float) * num2, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL( cudaMemcpy(w, d_w, sizeof(float) * num2, cudaMemcpyDeviceToHost));	

    CUDA_SAFE_CALL( cudaFree(d_w));
    CUDA_SAFE_CALL( cudaFree(d_u));
	CUDA_SAFE_CALL( cudaFree(d_w_temp));
	CUDA_SAFE_CALL( cudaFree(d_u_temp));

	if((fp=fopen("C:\\result.dat","wb"))==NULL)
	{
		printf("cannot open file\n");
	}
	fwrite(w,sizeof(float),num2,fp);
	fclose(fp);

	float wi[NUM];
	float sorttemp;

    for(int i = 0; i < NUM; i ++)
		{
		wi[i]=0.0f;
		for( int j = 0; j < NUM; j++)
		{
		wi[i] +=  w[i*NUM + j] * w[i*NUM + j];
		}
		wi[i] = sqrt(wi[i]);
		}

	for(int i=0;i<NUM; i++)
		for(int j=0; j < NUM; j++)
			if(wi[i]> wi[j])
			{
				sorttemp = wi[i];
				wi[i] = wi[j];
				wi[j] = sorttemp;
			}

	for(int i=0; i<NUM; i++)
	printf("%3.8f	", wi[i]);

	printf("\n");

	free(w);
	free(u);
	free(orign);
	free(unit);
	printf("\n");
	printf("%f", elapsedTimeInMs);
    CUT_EXIT(argc, argv);
}
