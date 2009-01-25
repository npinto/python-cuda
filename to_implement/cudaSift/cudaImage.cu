//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include <stdio.h>
#include <cutil.h>

#include "cudaImage.h"

#undef VERBOSE

void StartTimer(unsigned int *hTimer)
{
  CUT_SAFE_CALL(cutCreateTimer(hTimer));
  CUT_SAFE_CALL(cutResetTimer(*hTimer));
  CUT_SAFE_CALL(cutStartTimer(*hTimer));
}

double StopTimer(unsigned int hTimer)
{
  CUT_SAFE_CALL(cutStopTimer(hTimer));
  double gpuTime = cutGetTimerValue(hTimer);	
  CUT_SAFE_CALL(cutDeleteTimer(hTimer));
  return gpuTime;
}

int iDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

int iDivDown(int a, int b) {
  return a / b;
}

int iAlignUp(int a, int b) {
  return (a % b != 0) ?  (a - a % b + b) : a;
}

int iAlignDown(int a, int b) {
  return a - a % b;
}

double AllocCudaImage(CudaImage *img, int w, int h, int p, bool host, bool dev)
{
  unsigned int hTimer;
  StartTimer(&hTimer);
  int sz = sizeof(float)*p*h;
  img->width = w;
  img->height = h;
  img->pitch = p;
  img->h_data = NULL;
  if (host) {
    //printf("Allocating host data...\n");
    img->h_data = (float *)malloc(sz);
    //CUDA_SAFE_CALL(cudaMallocHost((void **)&img->h_data, sz));
  }
  img->d_data = NULL;
  if (dev) {
    //printf("Allocating device data...\n");
    CUDA_SAFE_CALL(cudaMalloc((void **)&img->d_data, sz));
    if (img->d_data==NULL) 
      printf("Failed to allocate device data\n");
  }
  img->t_data = NULL;
  double gpuTime = StopTimer(hTimer);
#ifdef VERBOSE
  printf("AllocCudaImage time =         %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double FreeCudaImage(CudaImage *img)
{
  unsigned int hTimer;
  StartTimer(&hTimer);
  if (img->d_data!=NULL) {
    //printf("Freeing device data...\n");
    CUDA_SAFE_CALL(cudaFree(img->d_data));
  }
  img->d_data = NULL;
  if (img->h_data!=NULL) {
    //printf("Freeing host data...\n");
    free(img->h_data);
    //CUDA_SAFE_CALL(cudaFreeHost(img->h_data));
  }
  img->h_data = NULL;
  if (img->t_data!=NULL) {
    //printf("Freeing texture data...\n");
    CUDA_SAFE_CALL(cudaFreeArray((cudaArray *)img->t_data));
  }
  img->t_data = NULL;
  double gpuTime = StopTimer(hTimer);
#ifdef VERBOSE
  printf("FreeCudaImage time =          %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double Download(CudaImage *img)
{
  unsigned int hTimer;
  StartTimer(&hTimer);
  if (img->d_data!=NULL && img->h_data!=NULL) 
    CUDA_SAFE_CALL(cudaMemcpy(img->d_data, img->h_data, 
      sizeof(float)*img->pitch*img->height, cudaMemcpyHostToDevice));
  double gpuTime = StopTimer(hTimer);
#ifdef VERBOSE
  printf("Download time =               %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double Readback(CudaImage *img, int w, int h)
{
  unsigned int hTimer;
  StartTimer(&hTimer);
  int p = sizeof(float)*img->pitch;
  w = sizeof(float)*(w<0 ? img->width : w);
  h = (h<0 ? img->height : h); 
  CUDA_SAFE_CALL(cudaMemcpy2D(img->h_data, p, img->d_data, p, 
    w, h, cudaMemcpyDeviceToHost));
  //CUDA_SAFE_CALL(cudaMemcpy(img->h_data, img->d_data, 
  //  sizeof(float)*img->pitch*img->height, cudaMemcpyDeviceToHost));
  double gpuTime = StopTimer(hTimer);
#ifdef VERBOSE
  printf("Readback time =               %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}

double InitTexture(CudaImage *img)
{
  unsigned int hTimer;
  StartTimer(&hTimer);
  cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>(); 
  CUDA_SAFE_CALL(cudaMallocArray((cudaArray **)&img->t_data, &t_desc, 
    img->pitch, img->height)); 
  //printf("InitTexture(%d, %d)\n", img->pitch, img->height); 
  if (img->t_data==NULL)
    printf("Failed to allocated texture data\n");
  double gpuTime = StopTimer(hTimer);
#ifdef VERBOSE
  printf("InitTexture time =            %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}
 
double CopyToTexture(CudaImage *src, CudaImage *dst, bool host)
{
  unsigned int hTimer;
  if (dst->t_data==NULL) {
    printf("Error CopyToTexture: No texture data\n");
    return 0.0;
  }
  if ((!host || src->h_data==NULL) && (host || src->d_data==NULL)) {
    printf("Error CopyToTexture: No source data\n");
    return 0.0;
  }
  StartTimer(&hTimer);
  if (host)
    CUDA_SAFE_CALL(cudaMemcpyToArray((cudaArray *)dst->t_data, 0, 0, 
      src->h_data, sizeof(float)*src->pitch*dst->height, 
      cudaMemcpyHostToDevice));
  else
    CUDA_SAFE_CALL(cudaMemcpyToArray((cudaArray *)dst->t_data, 0, 0, 
      src->d_data, sizeof(float)*src->pitch*dst->height, 
      cudaMemcpyDeviceToDevice));
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  double gpuTime = StopTimer(hTimer);
#ifdef VERBOSE
  printf("CopyToTexture time =          %.2f msec\n", gpuTime);
#endif
  return gpuTime;
}
