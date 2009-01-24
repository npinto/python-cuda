#ifndef TPIMAGEUTIL_H
#define TPIMAGEUTIL_H

#include <typeinfo>
#include <cstring>
#include <cmath>
#include <cassert>
#include "tpimage.h"

/** @file tpimageutil.h
    Templated image utility functions */

//=========================================================================

/// Gaussian low-pass filtering, with computational cost proportional to \b dw
/** @param dw filter width (an odd value, typ. \f$ 4 \sigma+1 \f$)
    @param limg source image
    @param variance filter variance \f$ \sigma^2 \f$ 
    @retval oimg filtered image */
template<int dw> void GaussLowPass(Image<float> &limg, Image<float> &oimg, 
  float variance);  
/// Low-pass filtering using Deriche's recursive method, with constant cost for all filter sizes. Suitable for \f$ \sigma \f$ up to about 5. For floating point images, an assembly implementation is used     
/** @param src source image
    @param sigma filter standard deviation \f$ \sigma \f$ 
    @param zeroboard whether to assume outer area is zero or mirrored 
    @retval dst filtered image */
template <class T> void DericheLowPass(Image<T> &src, Image<T> &dst, 
  double sigma, bool zeroboard);

/// Convert from RGB (three value per pixel) to grey-level image
/** @param src RGB source image
    @retval dst grey-level image */
template <class S, class T> void RGBToGrey(Image<S> &src, Image<T> &dst);
/// Convert from UYVY (two value per pixel) to grey-level image
/** @param src UYVY source image
    @retval dst grey-level image */
template <class S, class T> void YUVToGrey(Image<S> &src, Image<T> &dst);
/// Convert from RGB to YUV 
/** @param src RGB source image
    @retval yimg luminance component image 
    @retval uimg u colour component image 
    @retval vimg v colour component image */
template <class S, class T> void RGBToYUV(Image<S> &src, Image<T> &yimg, 
  Image<T> &uimg, Image<T> &vimg);
/// Convert from YUV to RGB 
/** @param yimg luminance component image 
    @param uimg u colour component image 
    @param vimg v colour component image 
    @retval dst RGB image*/
template <class S, class T> void YUVToRGB(Image<T> &yimg, Image<T> &uimg, 
  Image<T> &vimg, Image<S> &dst);

/// Point-wise multiply two images
/** @param src source image
    @retval dst product image */
template <class T> void operator*=(Image<T> &dst, Image<T> &src);
/// Point-wise divide two images
/** @param src source image
    @retval dst fraction image */
template <class T> void operator/=(Image<T> &dst, Image<T> &src);
/// Point-wise add two images
/** @param src source image
    @retval dst sum image */
template <class T> void operator+=(Image<T> &dst, Image<T> &src);
/// Point-wise subtract two images
/** @param src source image
    @retval dst difference image */
template <class T> void operator-=(Image<T> &dst, Image<T> &src);
/// Compute Laplacian of image
/** @param src source image
    @retval dst Laplacian image */
template <class T> void Laplace(Image<T> &src, Image<T> &dst);
/// Compute absolute difference between two images
/** @param src source image
    @retval dst absolute difference image */
template <class T> void AbsDiff(Image<T> &src, Image<T> &dst);
/// Compute absolute value of image
/** @param src source image
    @retval dst absolute value image */
template <class T> void Abs(Image<T> &src, Image<T> &dst);
/// Sum up values within rectangular regions (quick implementation)
/** @param src source image
    @param dw region widths
    @param dh region heights
    @retval dst image of summed up values */
template <class T> void RotatingSum(Image<T> &src, Image<T> &dst, 
  int dw, int dh);

/// Rescale image values
/** @param scale rescaling factor 
    @retval img image to be rescales */
template <class T> void ReScale(Image<T> &img, float scale);
/// Copy an data array to image 
/** @param indat array of image data
    @retval img destination image */
template <class T, class S> void Copy(T* indat, Image<S> &img);
/// Copy data from one image to another
/** @param src source image
    @retval dst destination image */
template <class T, class S> void Copy(Image<T> &src, Image<S> &dst);
/// Copy a sub-window from an image
/** @param src source image
    @param x center x-position of window in source image
    @param y center y-position of window in source image 
    @param dst destination sub-window */
template <class T> void SubCopy(Image<T> &src, Image<T> &dst, int x, int y);
/// Clear image data
/** @retval img image to be cleared */
template <class T> void Clear(Image<T> &img);
/// Fill image data with a certain value
/** @param value value to be used for filling
    @retval img image to be filled */
template <class T> void Fill(Image<T> &img, T value);

/// Scale-up an image to twice the size
/** @param src source image
    @retval dst destination image of twice the size */
template <class T> void ScaleUp(Image<T> &src, Image<T> &dst);
/// Low-pass and scale-down an image to half the size
/** @param src source image
    @retval dst destination image of half the size */
template <class T> void ScaleDown(Image<T> &src, Image<T> &dst);
/// Scale-down an image to a smaller size
/** @param src source image
    @param res scale-down factor
    @retval dst scale-downed destination image */
template<int res, class T> void SubSample(Image<T> &src, Image<T> &dst);

/// Low-pass an image with variance \f$\sigma^2\f$ = 1.0
/** @param img source image (repeated boundary assumed)
    @retval out blurred destination image */
template <class T> void LowPass(Image<T> &img, Image<T> &out);
/// Low-pass an image with variance \f$\sigma^2\f$ = 1.0
/** @param img source image (zero boundary assumed)
    @retval out blurred destination image */
template <class T> void LowPassZero(Image<T> &img, Image<T> &out);
/// Low-pass an image with variance \f$\sigma^2\f$ = 0.5
/** @param img source image (repeated boundary assumed)
    @retval out blurred destination image */
template <class T> void LowPass3(Image<T> &img, Image<T> &out);

/// Low-pass an image x-wise with variance \f$\sigma^2\f$ = 1.0
/** @param img source image
    @retval out blurred destination image */
template <class T> void LowPassX(Image<T> &img, Image<T> &out);
/// Low-pass an image y-wise with variance \f$\sigma^2\f$ = 1.0
/** @param img source image
    @retval out blurred destination image */
template <class T> void LowPassY(Image<T> &img, Image<T> &out);
/// Low-pass an image x-wise with variance \f$\sigma^2\f$ = 0.5
/** @param img source image
    @retval out blurred destination image */
template <class T> void LowPassX3(Image<T> &img, Image<T> &out);
/// Low-pass an image y-wise with variance \f$\sigma^2\f$ = 0.5
/** @param img source image
    @retval out blurred destination image */
template <class T> void LowPassY3(Image<T> &img, Image<T> &out);
/// Compute x-wise derivative
/** @param img source image
    @retval out x-wise derivative */
template <class T> void HighPassX3(Image<T> &img, Image<T> &out);
/// Compute y-wise derivative
/** @param img source image
    @retval out y-wise derivative */
template <class T> void HighPassY3(Image<T> &img, Image<T> &out);

/// Recify (rotate around y-axis and translate) a sub-window
/** @param img image to be rectified
    @param angle rotation angle around y-axis
    @param focal focal length (in pixels)
    @param xp x-position of sub-window 
    @param yp y-position of sub-window
    @param sourcepos whether positions defined in source image 
    @retval out rectifed sub-window */
template <class T, class S> void SubRectify(Image<T> &img, Image<S> &out, 
  float angle, float focal, float xp, float yp, bool sourcepos = true);
/// Recify (rotate around y-axis and translate) an image
/** @param img image to be rectified
    @param angle rotation angle around y-axis
    @param focal focal length (in pixels)
    @param xshift x-wise translation
    @param yshift y-wise translation
    @retval out rectifed image */
template <class T, class S> void Rectify(Image<T> &img, Image<S> &out, 
  float angle, float focal, float xshift = 0.0, float yshift = 0.0);
/// Correct an image for radial distortion
template <class T> void RadialCorrect(Image<T> &img, Image<T> &out, 
  float factor);

typedef double (*DoubleFunc1)(double);
typedef double (*DoubleFunc2)(double, double);
template <class T> void ImageFunc(DoubleFunc1, Image<T> &img, Image<T> &out);
template <class T> void ImageFunc(DoubleFunc2, Image<T> &img1, Image<T> &img2, 
  Image<T> &out);

//=========================================================================

/// @cond
typedef unsigned char UCHAR;
void operator+=(Image<UCHAR> &dst, Image<UCHAR> &src);
template<> void LowPassX<UCHAR>(Image<UCHAR> &img, Image<UCHAR> &out);
template<> void LowPassY<UCHAR>(Image<UCHAR> &img, Image<UCHAR> &out);
template<> void LowPass<float>(Image<float> &img, Image<float> &out);
/// @endcond

//=========================================================================

/// @cond
template<int dw>
void GaussLowPass(Image<float> &limg, Image<float> &oimg, float variance)
{
  float filter[dw];
  const int m = (dw-1) / 2;
  float sum = 0.0;
  for (int i=0;i<dw;i++) {
    filter[i] = exp(-(i-m)*(i-m)/(2.0*variance));
    sum += filter[i];
  }
  for (int i=0;i<dw;i++) 
    filter[i] /= sum;
  
#ifdef PYRAASM
  if (dw==5) {
    SymmFilter5(limg, oimg, filter[2], filter[1], filter[0]);
    return;
  }
#endif
  int w = limg.GetWidth();
  int h = limg.GetHeight();
  Image<float> temp(w, h);
  float *limd = limg.GetData();
  float *temd = temp.GetData();
  for (int y=0;y<h;y++) {
    for (int x=0;x<m;x++) {
      float sum = 0.0;
      for (int d=-m;d<-x;d++) 
	sum += limd[y*w]*filter[m+d];
      for (int d=-x;d<=m;d++) 
	sum += limd[y*w+x+d]*filter[m+d];
      temd[y*w+x] = sum;
    }
    for (int x=m;x<w-m;x++) {
      float sum = 0.0;
      for (int d=-m;d<=m;d++) 
	sum += limd[y*w+x+d]*filter[m+d];
      temd[y*w+x] = sum;
    }
    for (int x=w-m;x<w;x++) {
      float sum = 0.0;
      for (int d=-m;d<w-x;d++) 
	sum += limd[y*w+x+d]*filter[m+d];
      for (int d=w-x;d<=m;d++) 
	sum += limd[y*w+w-1]*filter[m+d];
      temd[y*w+x] = sum;
    }
  }
  float *oimd = oimg.GetData();
  for (int x=0;x<w;x++) {
    for (int y=0;y<m;y++) {
      float sum = 0.0;
      for (int d=-m;d<-y;d++) 
	sum += temd[x]*filter[m+d];
      for (int d=-y;d<=m;d++) 
	sum += temd[(y+d)*w+x]*filter[m+d];
      oimd[y*w+x] = sum;
    }
    for (int y=m;y<h-m;y++) {
      float sum = 0.0;
      for (int d=-m;d<=m;d++) 
	sum += temd[(y+d)*w+x]*filter[m+d];
      oimd[y*w+x] = sum;
    }
    for (int y=h-m;y<h;y++) {
      float sum = 0.0;
      for (int d=-m;d<h-y;d++) 
	sum += temd[(y+d)*w+x]*filter[m+d];
      for (int d=h-y;d<m;d++) 
	sum += temd[(h-1)*w+x]*filter[m+d];
      oimd[y*w+x] = sum;
    }
  }
}

template <class T>
void DericheLowPassXRow(T *srcd, float *tmpd, float *facd, int w)
{
  float t1 = facd[24] * srcd[0];
  float t2 = t1;
  float *td = &tmpd[1]; 
  T *sd = &srcd[1];
  td[-1] = t1;
  for (int x=1;x<w;x++,td++,sd++) {
    td[0] = facd[0]*(float)sd[0] + facd[4]*(float)sd[-1] - 
      facd[16]*t1 - facd[20]*t2;
    t2 = t1;
    t1 = td[0];
  }
  t1 = t2 = facd[28]  * srcd[w-1]; 
  sd = &srcd[w-2];
  td = &tmpd[w-3];
  td[2] += t1;
  td[1] += t1;
  for (int x=w-3;x>=0;x--,td--,sd--) { 
    float t3 = facd[8]*(float)sd[0] + facd[12]*(float)sd[1] - 
      facd[16]*t1 - facd[20]*t2;
    td[0] += t3;
    t2 = t1;
    t1 = t3;
  }
}
 
template <class T>
void DericheLowPassYRowD(float *td1, float *td2, float *tm1, float *tm2, 
  T *dd, float *fd, int w)
{
  for (int x=0;x<w;x++) {
    float t3 = fd[0]*td1[x] + fd[4]*td2[x] - fd[16]*tm1[x] - fd[20]*tm2[x];
    dd[x] = (T)t3;
    tm2[x] = t3;
  }
}

template <class T>
void DericheLowPassYRowU(float *td1, float *td2, float *tm1, float *tm2, 
  T *dd, float *fd, int w)
{
  for (int x=0;x<w;x++) {
    float t3 = fd[8]*td1[x] + fd[12]*td2[x] - fd[16]*tm1[x] - fd[20]*tm2[x];
    dd[x] += (T)t3;
    tm2[x] = t3;
  }
}

template <class T>
void DericheLowPass(Image<T> &src, Image<T> &dst, double sigma, 
  bool zeroboard = false) 
{
  // L(z) = (a0 + a1*z^-1) / (1.0 + b1*z^-1 + b2*z^-2) * X(z)
  // R(z) = (a2 + a3*z^+1) / (1.0 + b1*z^+1 + b2*z^+2) * X(z)
  // Y(z) = L(z) + R(z) 

  //sigma /= sqrt(2.0); //%%%%  Uncertain
#ifdef PYRAASM
  if (typeid(T)==typeid(float) && !(src.GetWidth()%4) && 
      !(src.GetHeight()%4)) {
    DericheLowPass(src.GetData(), dst.GetData(), src.GetWidth(), 
      src.GetHeight(),sigma, zeroboard);
    return;
  }
#endif 
  Image<float> factors(4, 8);
  float *facd = factors.GetData();

  float alpha = 5.0 / (2.0*sqrt(3.1415)*sigma);
  float ea = exp(-alpha);
  float e2a = exp(-2.0*alpha);
  float c0 = (1.0-ea)*(1.0-ea);
  float k = c0 / (1.0+2.0*alpha*ea-e2a);
   
  for (int i=0;i<4;i++) {
    float a0 = facd[i] = k;                               // a0 
    float a1 = facd[i+4] = k*ea*(alpha-1);                // a1
    float a2 = facd[i+8] = k*ea*(alpha+1);                // a2
    float a3 = facd[i+12] = -k*e2a;                       // a3
    facd[i+16] = -2.0*ea;                                 // b1
    facd[i+20] = e2a;                                     // b2
    facd[i+24] = (zeroboard ? 0.0 : 1.0) * (a0+a1) / c0;  // L(-1)
    facd[i+28] = (zeroboard ? 0.0 : 1.0) * (a2+a3) / c0;  // R(w)
  }  

  int w = src.GetWidth();
  int h = src.GetHeight();
  Buffer<float> tmp(w, 2);
  Buffer<float> buf(w, 2);
  T *srcd = src.GetData();
  float *tmpd = tmp.GetData();
  float *bufd = buf.GetData();
  DericheLowPassXRow(src[0], tmp[-1], facd, w);
  DericheLowPassXRow(src[0], tmp[0], facd, w);
  for (int i=0;i<2*w;i++)
    bufd[i] = facd[24]*tmpd[i];
  for (int y=0;y<h;y++) {
    DericheLowPassXRow(src[y], tmp[y], facd, w);
    DericheLowPassYRowD(tmp[y], tmp[y-1], buf[y], buf[y-1], dst[y], facd, w);
  }
  DericheLowPassXRow(src[h-1], tmp[h], facd, w);
  DericheLowPassXRow(src[h-1], tmp[h-1], facd, w);
  for (int i=0;i<2*w;i++) 
    bufd[i] = facd[28]*tmpd[i];
  DericheLowPassYRowU(tmp[h-1], tmp[h], buf[h-1], buf[h], dst[h-1], facd, w);
  for (int y=h-2;y>=0;y--) {
    DericheLowPassXRow(src[y+1], tmp[y], facd, w);
    DericheLowPassYRowU(tmp[y], tmp[y+1], buf[y], buf[y+1], dst[y], facd, w);
  }
}

template <class T>
void DericheLowPassOld(Image<T> &src, Image<T> &dst, double sigma, 
  bool zeroboard = false)     
{
  T *srcd = src.GetData();
  T *dstd = dst.GetData();
  int w = src.GetWidth();
  int h = src.GetHeight();
  if (w!=dst.GetWidth() && h!=dst.GetHeight()) {
    std::cout << "ERROR: DericheLowPass requires images of the same sizes" 
	      << std::endl;
    return;
  }

  Image<float> temp(w, h);
  float *temd = temp.GetData();
  float *tmpX = new float[w];
  float *tmpY = new float[h];
 
  float alpha = 5.0 / (2.0*sqrt(3.1415)*sigma);
  float ea = exp(-alpha);
  float e2a = exp(-2.0*alpha);
  float c0 = (1.0-ea)*(1.0-ea);
  float k = c0 / (1.0+2.0*alpha*ea-e2a);
  
  float b1 = -2.0*ea;
  float b2 = e2a;
  float a0 = k;
  float a1 = k*ea*(alpha-1);
  float a2 = k*ea*(alpha+1);
  float a3 = -k*e2a;
  
  float t1, t2;
  float boardfac = (zeroboard ? 0.0 : 1.0);

  for (int y=0;y<h;y++) {
    t1 = t2 = boardfac * srcd[y*w] * (a0+a1) / c0;
    tmpX[0] = t1;
    for (int x=1;x<w; x++) {
      tmpX[x] = a0*srcd[y*w+x] + a1*srcd[y*w+x-1] - b1*t1 - b2*t2;
      t2 = t1;
      t1 = tmpX[x];
    }
    t1 = t2 = boardfac * srcd[y*w+w-1] * (a2+a3) / c0; 
    temd[y*w+w-2] = tmpX[w-2] + t1;
    temd[y*w+w-1] = tmpX[w-1] + t1;
    for (int x=w-3;x>=0;x--) { 
      float t3 = a2*srcd[y*w+x+1] + a3*srcd[y*w+x+2] - b1*t1 - b2*t2;
      temd[y*w+x] = tmpX[x] + t3;
      t2 = t1;
      t1 = t3;
    }
  }

  for (int x=0;x<w;x++) {
    t1 = t2 = boardfac * temd[x] * (a0+a1) / c0;
    tmpY[0] = t1;
    for (int y=1;y<h;y++) {
      tmpY[y] = a0*temd[y*w+x] + a1*temd[(y-1)*w+x] - b1*t1 - b2*t2;
      t2 = t1;
      t1 = tmpY[y];
    }
    t1 = t2 = boardfac * temd[(h-1)*w+x] * (a2+a3) / c0;
    dstd[(h-2)*w+x] = tmpY[h-2] + t1;
    dstd[(h-1)*w+x] = tmpY[h-1] + t1;
    for (int y=h-3;y>=0;y--) { 
      float t3  = a2*temd[(y+1)*w+x] + a3*temd[(y+2)*w+x] - b1*t1 - b2*t2;
      dstd[y*w+x] = tmpY[y] + t3;
      t2 = t1;
      t1 = t3;
    }
  }

  delete [] tmpY;  
  delete [] tmpX;
}

template <class S, class T>
void RGBToGrey(Image<S> &src, Image<T> &dst)
{
  S *srcd = src.GetData();
  T *dstd = dst.GetData();
  int w = dst.GetWidth();
  int h = dst.GetHeight(); 
  assert((3*w)==src.GetWidth());
#ifdef PYRAASM 
  if (typeid(S)==typeid(unsigned char) && typeid(T)==typeid(unsigned char) && 
      !(src.GetWidth()%4)) {
    RGBToGreyCCASM((unsigned char*)srcd, (unsigned char*)dstd, w, h);
    return;
  }
#endif 
  for (int i=0;i<w*h;i++) 
    dstd[i] = (T)(0.1159*srcd[3*i+2] + 0.5849*srcd[3*i+1] + 0.2991*srcd[3*i]);
}

template <class S, class T>
void YUVToGrey(Image<S> &src, Image<T> &dst)
{
  S *srcd = src.GetData();
  T *dstd = dst.GetData();
  int w = dst.GetWidth();
  int h = dst.GetHeight();
  assert((2*w)==src.GetWidth());
#ifdef PYRAASM 
  if (typeid(S)==typeid(unsigned char) && typeid(T)==typeid(unsigned char) && 
      !(src.GetWidth()%4)) {
    YUVToGreyASM((unsigned char*)srcd, (unsigned char*)dstd, w, h);
    return;
  }
#endif 
  for (int i=0;i<w*h;i++) 
    dstd[i] = (T)(srcd[2*i+1]);
}

template <class S, class T> 
void RGBToYUV(Image<S> &src, Image<T> &yimg, Image<T> &uimg, Image<T> &vimg)
{
  S *srcd = src.GetData();
  T *yimd = yimg.GetData();
  T *uimd = uimg.GetData();
  T *vimd = vimg.GetData();
  int w = yimg.GetWidth();
  int h = yimg.GetHeight();
  assert((3*w)==src.GetWidth());
  for (int i=0;i<w*h;i++) {
    float r = srcd[3*i+0];
    float g = srcd[3*i+1];
    float b = srcd[3*i+2];
    yimd[i] = (T)( 0.299*r + 0.587*g + 0.114*b);
    uimd[i] = (T)(-0.146*r - 0.288*g + 0.434*b);
    vimd[i] = (T)( 0.617*r - 0.517*g - 0.100*b);
  }
}

template <class S, class T>
void YUVToRGB(Image<T> &yimg, Image<T> &uimg, Image<T> &vimg, Image<S> &dst)
{
  T *yimd = yimg.GetData();
  T *uimd = uimg.GetData();
  T *vimd = vimg.GetData();
  S *dstd = dst.GetData();
  int w = yimg.GetWidth();
  int h = yimg.GetHeight();
  assert((3*w)==dst.GetWidth());
  for (int i=0;i<w*h;i++) {
    float y = yimd[i];
    float u = uimd[i];
    float v = vimd[i];
    dstd[3*i+0] = (T)(1.0000*y - 0.0009*u + 1.1359*v);
    dstd[3*i+1] = (T)(1.0000*y - 0.3959*u - 0.5783*v);
    dstd[3*i+2] = (T)(1.0000*y + 2.0411*u - 0.0016*v);
  }
}

template <class T>
void ScaleUpRow(T *src, T *dst, int w1, int w2)
{
  dst[0] = (7*src[0] + src[1]) / 8;
  dst[1] = (src[0] + src[1]) / 2;
  for (int x=1;x<(w2-1);x++) {
    dst[2*x] = (src[x-1] + 6*src[x] + src[x+1]) / 8;
    dst[2*x+1] = (src[x] + src[x+1]) / 2;
  }
  dst[2*w2-2] = (src[w2-2] + 7*src[w2-1]) / 8;
  dst[2*w2-1] = src[w2-1];
  if (w1>(2*w2)) dst[2*w2] = src[w2-1];
}


template <class T>
void ScaleUp(Image<T> &src, Image<T> &dst)
{
  int w1 = dst.GetWidth();
  int h1 = dst.GetHeight();
  int w2 = src.GetWidth();
  int h2 = src.GetHeight();
  Buffer<T> buf(w1, 3);
  ScaleUpRow(src[0], buf[-1], w1, w2);
  ScaleUpRow(src[0], buf[0], w1, w2);
  for (int y=0;y<(h2-1);y++) {
    ScaleUpRow(src[y+1], buf[y+1], w1, w2);
    T *row1 = buf[y-1], *row2 = buf[y], *row3 = buf[y+1];
    T *des1 = dst[2*y], *des2 = dst[2*y+1];
    for (int x=0;x<w1;x++) {
      des1[x] = (row1[x] + 6*row2[x] + row3[x]) / 8;
      des2[x] = (row2[x] + row3[x]) / 2;
    }
  }
  T *row1 = buf[h2-2], *row2 = buf[h2-1];
  T *des1 = dst[2*h2-2], *des2 = dst[2*h2-1], *des3 = dst[2*h2];
  for (int x=0;x<w1;x++) {
    des1[x] = (row1[x] + 7*row2[x]) / 8;
    des2[x] = row2[x];
  }
  if (h1>(2*h2))
    for (int x=0;x<w1;x++) des3[x] = row2[x];
}

template <class T>
void ScaleDownRow(T *src, T *dst, int w1, int w2)
{
  dst[0] = (11*src[0] + 4*src[1] + src[2]) / 16;
  for (int x=1;x<(w2-1);x++) 
    dst[x] = (src[2*x-2] + 4*src[2*x-1] + 6*src[2*x] + 4*src[2*x+1] + src[2*x+2]) / 16;
  dst[w2-1] = src[2*w2-4] + 4*src[2*w2-3] + 6*src[2*w2-2] + 4*src[2*w2-1];
  if (w1>(2*w2)) dst[w2-1] += src[2*w2];
  else dst[w2-1] += src[2*w2-1];
  dst[w2-1] /= 16;
}

template <class T>
void ScaleDown(Image<T> &src, Image<T> &dst)
{
  int w1 = src.GetWidth();
  int h1 = src.GetHeight();
  int w2 = dst.GetWidth();
  int h2 = dst.GetHeight();
  Buffer<T> buf(w2, 5);
  ScaleDownRow(src[0], buf[-2], w1, w2);
  ScaleDownRow(src[0], buf[-1], w1, w2);
  ScaleDownRow(src[0], buf[0], w1, w2);
  for (int y=0;y<(h2-1);y++) {
    ScaleDownRow(src[2*y+1], buf[2*y+1], w1, w2);
    ScaleDownRow(src[2*y+2], buf[2*y+2], w1, w2);
    T *row1 = buf[2*y-2], *row2 = buf[2*y-1], *row3 = buf[2*y];
    T *row4 = buf[2*y+1], *row5 = buf[2*y+2], *dest = dst[y];
    for (int x=0;x<w2;x++) 
      dest[x] = (row1[x] + 4*row2[x] + 6*row3[x] + 4*row4[x] + row5[x]) / 16;
  }
  ScaleDownRow(src[2*h2-1], buf[2*h2-1], w1, w2);
  T *row1 = buf[2*h2-4], *row2 = buf[2*h2-3], *row3 = buf[2*h2-2];
  T *row4 = buf[2*h2-1], *row5 = buf[2*h2], *dest = dst[h2-1];
  if (h1>(2*h2)) ScaleDownRow(src[2*h2], buf[2*h2], w1, w2);
  else ScaleDownRow(src[2*h2-1], buf[2*h2], w1, w2);
  for (int x=0;x<w2;x++) 
    dest[x] = (row1[x] + 4*row2[x] + 6*row3[x] + 4*row4[x] + row5[x]) / 16;
}

template<int res, class T>
void SubSample(Image<T> &src, Image<T> &dst)
{
  T *srcd = src.GetData();
  T *dstd = dst.GetData();
  int sw = src.GetWidth();
  int sh = src.GetHeight();
  int dw = dst.GetWidth();
  int dh = dst.GetHeight();
  int w = (sw<dw*res ? sw / res : dw); 
  int h = (sh<dh*res ? sh / res : dh); 
  for (int y=0;y<h;y++) {
    int sx = 0;
    for (int dx=0;dx<w;dx++,sx+=res) 
      dstd[dx] = srcd[sx];
    for (int dx=w;dx<dw;dx++)
      dstd[dx] = dstd[w-1];
    srcd += sw*res;
    dstd += dw;
  }
  for (int y=h;y<dh;y++) {
    for (int dx=0;dx<dw;dx++)
      dstd[dx] = dstd[dx-dw];
    dstd += dw;
  }
}

template <class T>
void Laplace(Image<T> &src, Image<T> &dst)
{
  T *srcd = src.GetData();
  T *dstd = dst.GetData();
  int w = src.GetWidth();
  int h = src.GetHeight();
  T *s = srcd;
  T *d = dstd;
  d[0] = d[w-1] = d[(h-1)*w] = d[h*w-1] = 0; 
  for (int x=1;x<w-1;x++)
    d[x] = (- s[x-1] + 2*s[x] - s[x+1]) / 2;
  for (int y=1;y<h-1;y++) {
    s = &srcd[y*w];
    d = &dstd[y*w];
    d[0] = (- s[-w] + 2*s[0] - s[w]) / 2;
    for (int x=1;x<w-1;x++)
      d[x] = (- s[x-w] - s[x-1] + 4*s[x] - s[x+1] - s[x+w]) / 2;
    d[w-1] = (- s[w-1-w] + 2*s[w-1] - s[w-1+w]) / 2;
  }
  s = &srcd[(h-1)*w];
  d = &dstd[(h-1)*w];
  for (int x=1;x<w-1;x++)
    d[x] = (- s[x-1] + 2*s[x] - s[x+1]) / 2;
}

template <class T>
void operator*=(Image<T> &dst, Image<T> &src)
{
  T *srcd = src.GetData();
  T *dstd = dst.GetData();
  int w = src.GetWidth();
  int h = src.GetHeight();
  for (int i=0;i<w*h;i++) dstd[i] *= srcd[i];
}

template <class T>
void operator/=(Image<T> &dst, Image<T> &src)
{
  T *srcd = src.GetData();
  T *dstd = dst.GetData();
  int w = src.GetWidth();
  int h = src.GetHeight();
  for (int i=0;i<w*h;i++) 
    if (srcd[i]!=(T)0) 
      dstd[i] /= srcd[i];
    else
      dstd[i] = srcd[i];
}

template <class T>
void operator+=(Image<T> &dst, Image<T> &src)
{
  T *srcd = src.GetData();
  T *dstd = dst.GetData();
  int w = src.GetWidth();
  int h = src.GetHeight();
  for (int i=0;i<w*h;i++) dstd[i] += srcd[i];
}

template <class T>
void operator-=(Image<T> &dst, Image<T> &src)
{
  T *srcd = src.GetData();
  T *dstd = dst.GetData();
  int w = src.GetWidth();
  int h = src.GetHeight();
  for (int i=0;i<w*h;i++) dstd[i] -= srcd[i];
}

template <class T>
void AbsDiff(Image<T> &src, Image<T> &dst)
{
  int w = src.GetWidth();
  int h = src.GetHeight();
  T *srcd = src.GetData();
  T *dstd = dst.GetData();
#ifdef PYRAASM 
  if (typeid(T)==typeid(float)) {
    AbsDiffASM((float*)srcd, (float*)dstd, w*h);
    return;
  }
#endif 
  for (int i=0;i<w*h;i++) {
    T diff = dstd[i] - srcd[i];
    dstd[i] = (diff>0 ? diff : -diff);
  }
}

template <class T>
void Abs(Image<T> &src, Image<T> &dst)
{
  T *srcd = src.GetData();
  T *dstd = dst.GetData();
  int w = src.GetWidth();
  int h = src.GetHeight();
  for (int i=0;i<w*h;i++) {
    T diff = srcd[i];
    dstd[i] = (diff>0 ? diff : -diff);
  }
}


template <class T> 
void RotatingSum(Image<T> &src, Image<T> &dst, int dw, int dh)
{
  T *srcd = src.GetData();
  T *dstd = dst.GetData();
  int w = src.GetWidth();
  int h = src.GetHeight();
  T *tmpd = new T[w];
  for (int i=0;i<w;i++) 
    tmpd[i] = (T)0;
  int dw2 = dw/2;
  int dh2 = dh/2;
  for (int y=0;y<dh2;y++) 
    for (int x=0;x<w;x++) 
      tmpd[x] += srcd[y*w+x];
  for (int y=dh2;y<dh;y++) {
    for (int x=0;x<w;x++) 
      tmpd[x] += srcd[y*w+x];
    int p = (y-dh2) * w;
    dstd[p] = tmpd[0];
    for (int x=-dw2+1;x<1;x++) 
      dstd[p] += tmpd[x+dw2];
    for (int x=1;x<dw;x++) 
      dstd[p+x] = dstd[p+x-1] + tmpd[x+dw2];
    for (int x=dw;x<w-dw2;x++) 
      dstd[p+x] = dstd[p+x-1] + tmpd[x+dw2] - tmpd[x+dw2-dw];
    for (int x=w-dw2;x<w;x++) 
      dstd[p+x] = dstd[p+x-1] - tmpd[x+dw2-dw];
  }
  for (int y=dh;y<h;y++) {
    for (int x=0;x<w;x++) 
      tmpd[x] += (srcd[y*w+x] - srcd[(y-dh)*w+x]);
    int p = (y-dh2) * w;
    dstd[p] = tmpd[0];
    for (int x=-dw2+1;x<1;x++) 
      dstd[p] += tmpd[x+dw2];
    for (int x=1;x<dw;x++) 
      dstd[p+x] = dstd[p+x-1] + tmpd[x+dw2];
    for (int x=dw;x<w-dw2;x++) 
      dstd[p+x] = dstd[p+x-1] + tmpd[x+dw2] - tmpd[x+dw2-dw];
    for (int x=w-dw2;x<w;x++) 
      dstd[p+x] = dstd[p+x-1] - tmpd[x+dw2-dw];
  }
  for (int y=h;y<h+dh2;y++) {
    for (int x=0;x<w;x++) 
      tmpd[x] -= srcd[(y-dh)*w+x];
    int p = (y-dh2) * w;
    dstd[p] = tmpd[0];
    for (int x=-dw2+1;x<1;x++) 
      dstd[p] += tmpd[x+dw2];
    for (int x=1;x<dw;x++) 
      dstd[p+x] = dstd[p+x-1] + tmpd[x+dw2];
    for (int x=dw;x<w-dw2;x++) 
      dstd[p+x] = dstd[p+x-1] + tmpd[x+dw2] - tmpd[x+dw2-dw];
    for (int x=w-dw2;x<w;x++) 
      dstd[p+x] = dstd[p+x-1] - tmpd[x+dw2-dw];
  }
  delete [] tmpd;
}

template <class T>
void ReScale(Image<T> &img, float scale)
{
  T *image = img.GetData();
  int w = img.GetWidth();
  int h = img.GetHeight();
  for (int i=0;i<w*h;i++) 
    image[i] = (T)(scale * image[i]);
}

template <class T, class S>
void Copy(T* indat, Image<S> &img)
{
  if (typeid(T)==typeid(S)) 
    memcpy(img.GetData(), indat, sizeof(T) * img.GetWidth() * img.GetHeight());
  else {
    S *image = img.GetData();
    int w = img.GetWidth();
    int h = img.GetHeight();
    for (int i=0;i<w*h;i++) 
      image[i] = (S)indat[i];
  }
}

template <class T, class S>
void Copy(Image<T> &src, Image<S> &dst)
{
  if (typeid(T)==typeid(S)) 
    memcpy(dst.GetData(), src.GetData(), 
      sizeof(T) * src.GetWidth() * src.GetHeight());
  else {
    S *image = dst.GetData();
    T *indat = src.GetData();
    int w = dst.GetWidth();
    int h = dst.GetHeight();
    for (int i=0;i<w*h;i++) 
      image[i] = (S)indat[i];
  }
}

template <class T> 
void SubCopy(Image<T> &src, Image<T> &dst, int xp, int yp)
{
  int sw = src.GetWidth();
  int sh = src.GetHeight();
  int dw = dst.GetWidth();
  int dh = dst.GetHeight();
  int sx = xp - dw/2;
  int sy = yp - dh/2;
  int minx = (sx<0 ? -sx : 0);
  int maxx = (dw>(sw-sx) ? sw-sx  : dw);
  int miny = (sy<0 ? -sy : 0);
  int maxy = (dh>(sh-sy) ? sh-sy  : dh);
  T *srcd = src.GetData();
  T *dstd = dst.GetData();
  for (int y=0;y<miny;y++) 
    for (int x=0;x<dw;x++) 
      dstd[y*dw+x] = (T) 0;
  for (int y=miny;y<maxy;y++)
    for (int x=0;x<minx;x++) 
      dstd[y*dw+x] = (T) 0;
  for (int y=miny;y<maxy;y++)
    for (int x=minx;x<maxx;x++) 
      dstd[y*dw+x] = srcd[(y+sy)*sw+(x+sx)];
  for (int y=miny;y<maxy;y++)
    for (int x=maxx;x<dw;x++) 
      dstd[y*dw+x] = (T) 0;
  for (int y=maxy;y<dh;y++) 
    for (int x=0;x<dw;x++) 
      dstd[y*dw+x] = (T) 0;
}

template <class T>
void Clear(Image<T> &img)
{
  T *image = img.GetData();
  int w = img.GetWidth();
  int h = img.GetHeight();
  for (int i=0;i<(w*h);i++) image[i] = (T) 0;
}

template <class T>
void Fill(Image<T> &img, T value)
{
  T *image = img.GetData();
  int w = img.GetWidth();
  int h = img.GetHeight();
  for (int i=0;i<(w*h);i++) image[i] = value;
}

template <class T> 
void LowPassRow(T *src, T *dst, int w) 
{
  dst[0] = (11*src[0] + 4*src[1] + src[2]) / 16;
  dst[1] = (5*src[0] + 6*src[1] + 4*src[2] + src[3]) / 16;
  for (int x=2;x<(w-2);x++) 
    dst[x] = (src[x-2] + 4*src[x-1] + 6*src[x] + 4*src[x+1] + src[x+2]) / 16;
  dst[w-2] = (src[w-4] + 4*src[w-3] + 6*src[w-2] + 5*src[w-1]) / 16;
  dst[w-1] = (src[w-3] + 4*src[w-2] + 11*src[w-1]) / 16;
}

template <class T> 
void LowPass(Image<T> &img, Image<T> &out)
{
  int w = img.GetWidth();
  int h = img.GetHeight();
  Buffer<T> buf(w, 5);
  LowPassRow(img[0], buf[-2], w);
  LowPassRow(img[0], buf[-1], w);
  LowPassRow(img[0], buf[0], w);
  LowPassRow(img[1], buf[1], w);
  for (int y=0;y<(h-2);y++) {
    LowPassRow(img[y+2], buf[y+2], w);
    T *row1 = buf[y-2], *row2 = buf[y-1], *row3 = buf[y];
    T *row4 = buf[y+1], *row5 = buf[y+2], *dest = out[y];
    for (int x=0;x<w;x++) 
      dest[x] = (row1[x] + 4*row2[x] + 6*row3[x] + 4*row4[x] + row5[x]) / 16;
  }
  for (int y=h-2;y<h;y++) {
    LowPassRow(img[h-1], buf[y+2], w);
    T *row1 = buf[y-2], *row2 = buf[y-1], *row3 = buf[y];
    T *row4 = buf[y+1], *row5 = buf[y+2], *dest = out[y];
    for (int x=0;x<w;x++) 
      dest[x] = (row1[x] + 4*row2[x] + 6*row3[x] + 4*row4[x] + row5[x]) / 16;
  }
}

template <class T> 
void LowPassRowZero(T *src, T *dst, int w) 
{
  dst[0] = (6*src[0] + 4*src[1] + src[2]) / 16;
  dst[1] = (4*src[0] + 6*src[1] + 4*src[2] + src[3]) / 16;
  for (int x=2;x<(w-2);x++) 
    dst[x] = (src[x-2] + 4*src[x-1] + 6*src[x] + 4*src[x+1] + src[x+2]) / 16;
  dst[w-2] = (src[w-4] + 4*src[w-3] + 6*src[w-2] + 4*src[w-1]) / 16;
  dst[w-1] = (src[w-3] + 4*src[w-2] + 6*src[w-1]) / 16;
}

template <class T> 
void LowPassZero(Image<T> &img, Image<T> &out)
{
  int w = img.GetWidth();
  int h = img.GetHeight();
  Buffer<T> buf(w, 5);
  buf.Clear();
  LowPassRowZero(img[0], buf[0], w);
  LowPassRowZero(img[1], buf[1], w);
  for (int y=0;y<(h-2);y++) {
    LowPassRowZero(img[y+2], buf[y+2], w);
    T *row1 = buf[y-2], *row2 = buf[y-1], *row3 = buf[y];
    T *row4 = buf[y+1], *row5 = buf[y+2], *dest = out[y];
    for (int x=0;x<w;x++) 
      dest[x] = (row1[x] + 4*row2[x] + 6*row3[x] + 4*row4[x] + row5[x]) / 16;
  }
  for (int y=h-2;y<h;y++) {
    T *bptr = buf[y+2];
    for (int x=0;x<w;x++) bptr[x] = (T)0;
    T *row1 = buf[y-2], *row2 = buf[y-1], *row3 = buf[y];
    T *row4 = buf[y+1], *row5 = buf[y+2], *dest = out[y];
    for (int x=0;x<w;x++) 
      dest[x] = (row1[x] + 4*row2[x] + 6*row3[x] + 4*row4[x] + row5[x]) / 16;
  }
}

template <class T> 
void LowPassRow3(T *src, T *dst, int w) 
{
  dst[0] = (3*src[0] + src[1]) / 4;
  for (int x=1;x<(w-1);x++) 
    dst[x] = (src[x-1] + 2*src[x] + src[x+1]) / 4;
  dst[w-1] = (src[w-2] + 3*src[w-1]) / 4;
}

template <class T> 
void LowPass3(Image<T> &img, Image<T> &out)
{
  int w = img.GetWidth();
  int h = img.GetHeight();
  Buffer<T> buf(w, 3);
  LowPassRow(img[0], buf[-1], w);
  LowPassRow(img[0], buf[0], w);
  for (int y=0;y<(h-1);y++) {
    LowPassRow(img[y+1], buf[y+1], w);
    T *row1 = buf[y-1], *row2 = buf[y], *row3 = buf[y+1], *dest = out[y];
    for (int x=0;x<w;x++) 
      dest[x] = (row1[x] + 2*row2[x] + row3[x]) / 4;
  }
  for (int y=h-1;y<h;y++) {
    LowPassRow(img[h-1], buf[y+1], w);
    T *row1 = buf[y-1], *row2 = buf[y], *row3 = buf[y+1], *dest = out[y];
    for (int x=0;x<w;x++) 
      dest[x] = (row1[x] + 2*row2[x] + row3[x]) / 4;
  }
}

template <class T>
void LowPassX(Image<T> &img, Image<T> &out)
{
  T *image = img.GetData();
  T *outimg = out.GetData();
  int w = img.GetWidth();
  int h = img.GetHeight();
  for (int y=0;y<h;y++) {
    T *irow = &image[y*w];
    T *orow = &outimg[y*w];
    orow[0] = (irow[2] + 4*irow[1] + 6*irow[0]) / 11;
    orow[1] = (irow[3] + 4*(irow[0] + irow[2]) + 6*irow[1]) / 15;
    for (int x=2;x<(w-2);x++) 
      orow[x] = (irow[x-2] + irow[x+2] + 4*(irow[x-1] + 
        irow[x+1]) + 6*irow[x]) / 16;
    orow[w-2] = (irow[w-4] + 4*(irow[w-3] + 
      irow[w-1]) + 6*irow[w-2]) / 15;
    orow[w-1] = (irow[w-3] + 4*irow[w-2] + 6*irow[w-1]) / 11;
  }
}

template <class T>
void LowPassY(Image<T> &img, Image<T> &out)
{
  T *image = img.GetData();
  T *outimg = out.GetData();
  int w = img.GetWidth();
  int h = img.GetHeight();
  for (int x=0;x<w;x++) {
    T *irow = &image[x];
    T *orow = &outimg[x];
    orow[0] = (irow[2*w] + 4*irow[w] + 6*irow[0]) / 11;
    orow[w] = (irow[3*w] + 4*(irow[0] + irow[2*w])+
      6*irow[w]) / 15;
    for (int y=2;y<(h-2);y++) 
      orow[y*w] = (irow[(y-2)*w] + irow[(y+2)*w] + 
	4*(irow[(y-1)*w] + irow[(y+1)*w]) + 6*irow[y*w])/16;
    orow[(h-2)*w] = (irow[(h-4)*w] + 
      4*(irow[(h-3)*w] + irow[(h-1)*w]) + 
      6*irow[(h-2)*w]) / 15;
    orow[(h-1)*w] = (irow[(h-3)*w] + 
      4*irow[(h-2)*w] + 6*irow[(h-1)*w]) / 11;
  }
}

template <class T>
void LowPassX3(Image<T> &img, Image<T> &out)
{
  T *image = img.GetData();
  T *outimg = out.GetData();
  int w = img.GetWidth();
  int h = img.GetHeight();
  for (int y=0;y<h;y++) {
    T *irow = &image[y*w];
    T *orow = &outimg[y*w];
    orow[0] = (irow[1] + 2*irow[0]) / 3;
    for (int x=1;x<(w-1);x++) 
      orow[x] = (irow[x-1] + irow[x+1] + 2*irow[x]) / 4;
    orow[w-1] = (irow[w-2] + 2*irow[w-1]) / 3;
  }
}

template <class T>
void LowPassY3(Image<T> &img, Image<T> &out)
{
  T *image = img.GetData();
  T *outimg = out.GetData();
  int w = img.GetWidth();
  int h = img.GetHeight();
  for (int x=0;x<w;x++) {
    T *irow = &image[x];
    T *orow = &outimg[x];
    orow[0] = (irow[w] + 2*irow[0]) / 3;
    for (int y=1;y<(h-1);y++) 
      orow[y*w] = (irow[(y-1)*w] + irow[(y+1)*w] + 2*irow[y*w]) / 4;
    orow[(h-1)*w] = (irow[(h-2)*w] + 2*irow[(h-1)*w]) / 3;
  }
} 

template <class T>
void HighPassX3(Image<T> &img, Image<T> &out)
{
  T *image = img.GetData();
  T *outimg = out.GetData();
  int w = img.GetWidth();
  int h = img.GetHeight();
  for (int y=0;y<h;y++) {
    T *irow = &image[y*w];
    T *orow = &outimg[y*w];
    orow[0] = irow[1] - irow[0];
    for (int x=1;x<(w-1);x++) 
      orow[x] = (irow[x+1] - irow[x-1]) / 2;
    orow[w-1] = irow[w-1] - irow[w-2];
  }
}

template <class T>
void HighPassY3(Image<T> &img, Image<T> &out)
{
  T *image = img.GetData();
  T *outimg = out.GetData();
  int w = img.GetWidth();
  int h = img.GetHeight();
#ifdef PYRAASM 
  if (typeid(T)==typeid(float) && !(w%4)) {
    HighPassY3ASM(image, outimg, w, h);
    return;
  }
#endif
  for (int x=0;x<w;x++) {
    T *irow = &image[x];
    T *orow = &outimg[x];
    orow[0] = irow[w] - irow[0];
    for (int y=1;y<(h-1);y++) 
      orow[y*w] = (irow[(y+1)*w] - irow[(y-1)*w]) / 2;
    orow[(h-1)*w] = irow[(h-1)*w] - irow[(h-2)*w];
  }
}

template <class T, class S>
void SubRectify(Image<T> &img, Image<S> &out, float angle, float focal, 
  float xp, float yp, bool sourcepos) 
{
  T *indat = img.GetData();
  S *outdat = out.GetData();
  int w = out.GetWidth();
  int h = out.GetHeight();
  int wi = img.GetWidth();                // x' =  cx / (sx + f)
  int hi = img.GetHeight();               // y' =   y / (sx + f)
  float cosa = cos(3.1415*angle/180.0);   // x =  fx' / (c - sx')
  float sina = sin(3.1415*angle/180.0);   // y = cfy' / (c - sx')

  float xval, fval, xflt;
  for (int x=0;x<w;x++) {
    if (sourcepos) {
      xval = (float)(x+xp - wi/2);  
      fval = sina * xval + focal;                              
      xflt = (focal * cosa * xval/fval + 0.5) + wi/2;  
    } else {
      xval = (float)(x - w/2);  
      fval = cosa * focal - sina * xval;                         
      xflt = (focal * xval/fval + 0.5) + w/2 - xp;  
    }
    if (xflt>=0.0 && xflt<(wi-1)) {
      int xint = (int)xflt;                                         
      float xfra = xflt - xint;                                     
      T *intmp = &indat[xint];
      S *outtmp = outdat;

      float ydel = (sourcepos ? focal / fval : cosa * focal / fval);
      float yflt = (sourcepos ? hi/2 : h/2) * (1.0 - ydel);
      if (sourcepos) 
	yflt += yp*ydel;
      else 
	yflt -= yp;
      int yfirst = 0; 
      if (yflt<0.0) {
	int num = -(int)(yflt/ydel) + 1;
	yfirst += num;
	yflt += num*ydel;
      }
      int ylast = (int)(((hi-2)-yflt) / ydel) + yfirst;
      if (ylast>=(h-1)) 
	ylast = h-1;

      for (int y=0;y<yfirst;y++) { 
	*outtmp = (S) 0; 
	outtmp = &outtmp[w]; 
      }
      int ydeli = (int)(ydel * 0x0400);
      int yflti = (int)(yflt * 0x0400);
      int xfrai = (int)(xfra * 0x0400);
      int xfram = 0x0400 - xfrai;
      for (int y=yfirst;y<ylast;y++) {
	yflti += ydeli;
	int yfrai = yflti & 0x03ff;
	T *inp = &intmp[(yflti >> 10) * wi];
	int ix1 = (int)(xfram * inp[0] + xfrai * inp[1]);
	int ix2 = (int)(xfram * inp[wi] + xfrai * inp[wi+1]);
	int yfram = 0x0400 - yfrai;
	*outtmp = (S)((yfram * ix1 + yfrai * ix2) >> 20);
	outtmp += w;
      }
      for (int y=ylast;y<h;y++) { 
	*outtmp = (S) 0; 
	outtmp = &outtmp[w]; 
      }
    } else 
      for (int y=0;y<h;y++) 
	outdat[y*w] = (S) 0; 
    outdat = &outdat[1];
  } 
}

template <class T, class S>
void Rectify(Image<T> &img, Image<S> &out, float angle, float focal, 
  float xshift = 0.0, float yshift = 0.0) 
{
  T *indat = img.GetData();
  S *outdat = out.GetData();
  int w = out.GetWidth();
  int h = out.GetHeight();
  int wi = img.GetWidth();
  int hi = img.GetHeight();
  float scale = (float) wi / w;
  float cosa = cos(3.1415*angle/180.0);
  float sina = sin(3.1415*angle/180.0);
  int yshi = (int)(yshift / scale);                                 

  for (int x=0;x<w;x++) {
    float xval = (float)(scale * x - wi/2);                         
    float fval = sina * xval + focal;                              
    float xflt = (focal * cosa * xval/fval + 0.5) + wi/2 + xshift;  
    if (xflt>=0.0 && xflt<(wi-1)) {
      int xint = (int)xflt;                                         
      float xfra = xflt - xint;                                     
      T *intmp = &indat[xint];
      S *outtmp = outdat;

      float ydel = scale * focal / fval;
      float yflt = (scale - ydel) * h/2;
      int yfirst = (int)(- h/2 * fval / focal + h/2 + 1);
      if (yfirst<0) yfirst = 0;
      else yflt += (yfirst * ydel);
      yfirst += yshi;
      if (yfirst<0) {
	yflt -= (yfirst * ydel);
	yfirst = 0;
      }
      int ylast = (int)((h-1 - h/2) * fval / focal + h/2 - 1) + yshi;
      if (ylast>=h) ylast = h-1;

      for (int y=0;y<yfirst;y++) { 
	*outtmp = (S) 0; 
	outtmp = &outtmp[w]; 
      }
      int ydeli = (int)(ydel * 0x0400);
      int yflti = (int)(yflt * 0x0400);
      int xfrai = (int)(xfra * 0x0400);
      int xfram = 0x0400 - xfrai;
      for (int y=yfirst;y<ylast;y++) {
	yflti += ydeli;
	int yfrai = yflti & 0x03ff;
	T *inp = &intmp[(yflti >> 10) * wi];
	int ix1 = (int)(xfram * inp[0] + xfrai * inp[1]);
	int ix2 = (int)(xfram * inp[wi] + xfrai * inp[wi+1]);
	int yfram = 0x0400 - yfrai;
	*outtmp = (S)((yfram * ix1 + yfrai * ix2) >> 20);
	outtmp += w;
      }
      for (int y=ylast;y<h;y++) { 
	*outtmp = (S) 0; 
	outtmp = &outtmp[w]; 
      }
    } else for (int y=0;y<h;y++) outdat[y*w] = (S) 0; 
    outdat = &outdat[1];
  } 
}


// pos_new = pos_old * (1.0 + factor*radius^2)
// radius = 1.0 for each image corner
template <class T>
void RadialCorrect(Image<T> &img, Image<T> &out, float factor)
{
  T *imgd = img.GetData();
  T *outd = out.GetData();
  int w = img.GetWidth();
  int h = img.GetHeight();
  float xc =  w / 2;
  float yc = h / 2;
  float rscale = factor / (xc*xc + yc*yc);
  for (int y=-(int)yc;y<yc;y++) {
    for (int x=-(int)xc;x<xc;x++) {
      float scale = (1 - rscale * (x*x+y*y));
      float xf = x*scale + xc;
      int xn = (int)xf;
      float xd = xf - xn;
      float yf = y*scale + yc;
      int yn = (int)yf;
      float yd = yf - yn;
      *outd++ = (T)((1.0-yd)*((1.0-xd)*imgd[yn*w+xn] + xd*imgd[yn*w+xn+1]) + 
	yd*((1.0-xd)*imgd[yn*w+xn+w] + xd*imgd[yn*w+xn+w+1]));
    }
  }  
}

template <class T>
void ImageFunc(DoubleFunc1 func, Image<T> &img, Image<T> &out)
{
  T *imgd = img.GetData();
  T *outd = out.GetData();
  int w = img.GetWidth();
  int h = img.GetHeight();
  for (int i=0;i<w*h;i++)
    outd[i] = (T)func((double)imgd[i]);
}

template <class T>
void ImageFunc(DoubleFunc2 func, Image<T> &img1, Image<T> &img2, Image<T> &out)
{
  T *im1d = img1.GetData();
  T *im2d = img2.GetData();
  T *outd = out.GetData();
  int w = img1.GetWidth();
  int h = img1.GetHeight();
  for (int i=0;i<w*h;i++)
    outd[i] = (T)func((double)im1d[i], (double)im2d[i]);
}

/// @endcond

#endif // TPIMAGEUTIL_H
