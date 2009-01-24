//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//              celle @ nada.kth.se                       //
//********************************************************//  
 
#include <cutil.h>
#include <iostream>  
#include <cmath>

#include "tpimage.h"
#include "tpimageutil.h"

#include "cudaImage.h"
#include "cudaSift.h"


///////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
///////////////////////////////////////////////////////////////////////////////
extern "C" void ConvRowCPU(float *h_Result, float *h_Data, 
  float *h_Kernel, int w, int h, int kernelR);   
extern "C" void ConvColCPU(float *h_Result, float *h_Data,
  float *h_Kernel, int w, int h, int kernelR);
extern "C" double Find3DMinMaxCPU(CudaImage *res, CudaImage *data1, 
  CudaImage *data2, CudaImage *data3);

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) 
{     
  CudaImage img1, img2;
  Image<float> limg(1280, 960);
  Image<float> rimg(1280, 960);

  limg.Load("data/left2.pgm");
  rimg.Load("data/righ2.pgm");
  ReScale(limg, 1.0/256.0f);
  ReScale(rimg, 1.0/256.0f);
  unsigned int w = limg.GetWidth();
  unsigned int h = limg.GetHeight();

  std::cout << "Image size = (" << w << "," << h << ")" << std::endl;
      
  InitCuda();
  std::cout << "Initializing data..." << std::endl;
  AllocCudaImage(&img1, w, h, w, false, true);
  img1.h_data = limg.GetData();
  AllocCudaImage(&img2, w, h, w, false, true);
  img2.h_data = rimg.GetData(); 
  Download(&img1);
  Download(&img2);

  SiftData siftData1, siftData2;
  InitSiftData(&siftData1, 128, true, true); 
  ExtractSift(&siftData1, &img1, 3, 3, 0.3f, 0.04);
  InitSiftData(&siftData2, 128, true, true);
  ExtractSift(&siftData2, &img2, 3, 3, 0.3f, 0.04);
  std::cout << "Number of original features: " <<  siftData1.numPts << " " 
	    << siftData2.numPts << std::endl;
  MatchSiftData(&siftData1, &siftData2);
  float homography[9];
  int numMatches;
  FindHomography(&siftData1, homography, &numMatches, 1000, 0.85f, 0.95f, 5.0);

  int numPts = siftData1.numPts;
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
  float *h_img = img1.h_data; 
  for (int j=0;j<numPts;j++) { 
    int k = sift1[j].match;
    float x = sift1[j].xpos;
    float y = sift1[j].ypos;
    float den = homography[6]*x + homography[7]*y + homography[8]; 
    float x2 = (homography[0]*x + homography[1]*y + homography[2]) / den;
    float y2 = (homography[3]*x + homography[4]*y + homography[5]) / den;
    float erx = x2 - sift2[k].xpos;
    float ery = y2 - sift2[k].ypos;
    float er2 = erx*erx + ery*ery;
    if (er2<25.0f && sift1[j].score>0.85f && sift1[j].ambiguity<0.95f) {
      float dx = sift2[k].xpos - sift1[j].xpos;
      float dy = sift2[k].ypos - sift1[j].ypos;
      int len = (int)(fabs(dx)>fabs(dy) ? fabs(dx) : fabs(dy));
      for (int l=0;l<len;l++) {
        int x = (int)(sift1[j].xpos + dx*l/len);
        int y = (int)(sift1[j].ypos + dy*l/len);
        h_img[y*w+x] = 1.0f;
      }
    }
    int p = (int)(sift1[j].ypos+0.5)*w + (int)(sift1[j].xpos+0.5);
    p += (w+1);
    for (int k=0;k<(int)(1.41*sift1[j].scale);k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] =h_img[p+k*w] = 0.0f;
    p -= (w+1);
    for (int k=0;k<(int)(1.41*sift1[j].scale);k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] =h_img[p+k*w] = 1.0f;
  }

  FreeSiftData(&siftData1);
  FreeSiftData(&siftData2);
  limg.Store("data/limg_pts.pgm", true, false);

  img1.h_data = NULL;
  FreeCudaImage(&img1);
  img2.h_data = NULL;
  FreeCudaImage(&img2);
  CUT_EXIT(argc, argv);
}

