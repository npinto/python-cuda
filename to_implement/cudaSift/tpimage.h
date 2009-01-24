#ifndef TPIMAGE_H
#define TPIMAGE_H

#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <typeinfo>
#include <stdint.h>

/** @file tpimage.h
    Templated image class */

/// Templated image class
template<class T>
class Image {
  int width, height;
  T *image, *img;
  bool localalloc;
public:
  /// Empty constructor
  Image();
  /// Constructor
  /** @param w image width
      @param h image height
      @param ptr image data pointer (if NULL, allocated internally) */
  Image(int w, int h, T *ptr=NULL);
  ~Image() { if (localalloc) delete [] img; }
  /// Set new image size
  /** @param w image width
      @param h image height */
  void SetSize(int w, int h);
  /// Set image data position
  /** @param ptr image data pointer */
  void SetData(T *ptr) { image = ptr; }
  /// Load grey-level image from PGM file
  /** @param filename image file name */
  bool Load(const char *filename);
  /// Load RGB image (three values per pixel) from PPM file  
  /** @param filename image file name */
  bool LoadRGB(const char *filename);
  /// Store grey-level image as PGM file
  /** @param filename image file name 
      @param norm whether to normalize image before storage
      @param ascii whether to store in ASCII or binary format */
  void Store(const char *filename, bool norm = false, bool ascii =false) const;
  /// Store RGB image (three values per pixel) as PPM file
  /** @param filename image file name */
  void StoreRGB(const char *filename) const;
  /// Convert from UYVY (two values per pixel) to RGB and store as PPM file 
  /** @param filename image file name */
  void StoreYUV(const char *filename) const;
  /// Get image data position 
  T *GetData() const { return image; }
  /// Get image width 
  int GetWidth() const { return width; }
  /// Get image height
  int GetHeight() const { return height; }
  /// Get pointer to pixel row \b i 
  T *operator[](int i) { return &image[i*width]; }
  /// Copy image data
  void operator=(Image<T> &src);
};

/// Templated circular buffer 
template<class T>
class Buffer {
  int width, height;
  T *image, *img;
public:
  /// Constructor
  /** @param w buffer width
      @param h buffer height */
  Buffer(int w, int h);
  ~Buffer() { delete [] img; }
  /// Get buffer data position 
  T *GetData() const { return image; }
  /// Get buffer width
  int GetWidth() const { return width; }
  /// Get buffer height
  int GetHeight() const { return height; }
  /// Get pointer to pixel row \b i modulo \b height 
  T *operator[](int i) { 
    return &image[((i+height)%height)*width]; 
  }
  /// Clear buffer data
  /** @param val value to fill buffer with */
  void Clear(T val = (T) 0);
};

template<class T>
Image<T>::Image() : Image(32, 32, NULL)
{

}

template<class T>
Image<T>::Image(int w, int h, T *ptr) : width(w), height(h)
{
  int extra = 16 / sizeof(T);
  if (ptr==NULL) {
    img = new T[w*h+extra];
    localalloc = true;
    //image = (T *)((unsigned long)(img+extra-1) & (unsigned long)(~15));
#if __WORDSIZE == 64
    image = (T *)((uint64_t)(img+extra-1) & (uint64_t)(~15));
#else 
    image = (T *)((uint32_t)(img+extra-1) & (uint32_t)(~15));
#endif
  } else {
    img = ptr;
    localalloc = false;
    image = img;
  }
}

template<class T>
void Image<T>::SetSize(int w, int h)
{
  if (w==width && h==height) 
    return;
  if (localalloc) 
    delete [] img; 
  width = w;
  height = h;
  int extra = 16 / sizeof(T);
  img = new T[w*h+extra];
  localalloc = true;
  //image  = (T *)((unsigned long)(img+extra-1) & (unsigned long)(~15));
#if __WORDSIZE == 64
    image = (T *)((uint64_t)(img+extra-1) & (uint64_t)(~15));
#else 
    image = (T *)((uint32_t)(img+extra-1) & (uint32_t)(~15));
#endif
}

template<class T>
bool Image<T>::Load(const char *filename)
{
  std::ifstream imagefile;
  imagefile.open(filename, std::ios::binary);
  if (!imagefile) {
    std::cerr << "Error: couldn't find PPM file " << filename << std::endl;
    return false;
  }
  char string[80];
  imagefile >> string;
  if (strcmp(string,"P2") && strcmp(string,"P5")) {
    std::cerr << "Error: " << filename << " is not an PGM file" << std::endl;
    return false;
  }
  char comment[120];
  imagefile >> comment[0];
  while (comment[0]=='#') {
    imagefile.getline(comment, 119, '\n');
    imagefile >> comment[0];
  }
  imagefile.putback(comment[0]);
  int w, h, d;
  imagefile >> w;
  imagefile >> h;
  imagefile >> d;
  int size = w * h;
  if (w!=width || h!=height) {
    delete [] img;
    width = w;
    height = h;
    int extra = 16 / sizeof(T);
    img = new T[size+extra];
    //image  = (T *)((unsigned long)(img+extra) & (unsigned long)(~15));
#if __WORDSIZE == 64
    image = (T *)((uint64_t)(img+extra) & (uint64_t)(~15));
#else 
    image = (T *)((uint32_t)(img+extra) & (uint32_t)(~15));
#endif
    std::cout << "WARNING: The size of the loaded image was changed" << std::endl;
  }
  int value;
  if (strcmp(string,"P2")) { // not ascii, thus raw
    unsigned char *tmp = new unsigned char[size];
    imagefile.ignore(1, '\n');
    imagefile.read((char*)tmp, size);
    for(int cnt=0;cnt<size;cnt++) 
      image[cnt] = (T)tmp[cnt];
    delete [] tmp;
  } else {                   // with ascii
    for(int cnt=0;cnt<size;cnt++) {
      imagefile >> value;
      image[cnt] = (T)value;
    }
  }
  imagefile.close();
  return true;
}

template<class T>
bool Image<T>::LoadRGB(const char *filename)
{
  std::ifstream imagefile;
  imagefile.open(filename);
  if (!imagefile) {
    std::cerr << "Error: couldn't find PPM file " << filename << std::endl;
    return false;
  }
  char string[80];
  imagefile >> string;
  if (strcmp(string,"P3") && strcmp(string,"P6")) {
    std::cerr << "Error: " << filename << " is not an PPM file" << std::endl;
    return false;
  }
  char comment[120];
  imagefile >> comment[0];
  while (comment[0]=='#') {
    imagefile.getline(comment, 119, '\n');
    //std::cout << comment << std::endl;
    imagefile >> comment[0];
  }
  imagefile.putback(comment[0]);
  int w, h, d;
  imagefile >> w;
  imagefile >> h;
  imagefile >> d;
  w *= 3;
  int size = w * h;
  if (w!=width || h!=height) {
    delete [] img;
    width = w;
    height = h;
    int extra = 16 / sizeof(T);
    img = new T[size+extra];
    //image  = (T *)((unsigned long)(img+extra) & (unsigned long)(~15));
#if __WORDSIZE == 64
    image = (T *)((uint64_t)(img+extra) & (uint64_t)(~15));
#else 
    image = (T *)((uint32_t)(img+extra) & (uint32_t)(~15));
#endif
    std::cout << "WARNING: The size of the loaded image was changed" << std::endl;
  }
  if (strcmp(string,"P3")) { // not ascii, thus raw
    unsigned char *tmp = new unsigned char[size];
    imagefile.ignore(1, '\n');
    imagefile.read((char*)tmp, size);
    for (int cnt=0;cnt<size;cnt+=3) {
      image[cnt+0] = (T)tmp[cnt+0];
      image[cnt+1] = (T)tmp[cnt+1];
      image[cnt+2] = (T)tmp[cnt+2];
    }
    delete [] tmp;
  } else {
    int value;
    for (int cnt=0;cnt<size;cnt+=3) {
      imagefile >> value;
      image[cnt+0] = (T)value;
      imagefile >> value;
      image[cnt+1] = (T)value;
      imagefile >> value;
      image[cnt+2] = (T)value;
    }
  }
  imagefile.close();
  return true;
}

template<class T>
void Image<T>::Store(const char *filename, bool type, bool ascii) const
{
  std::ofstream imagefile;
  imagefile.open(filename, std::ios::binary);
  if (ascii) 
    imagefile << "P2\n";
  else 
    imagefile << "P5\n";
  imagefile << width << " " << height << "\n";
  imagefile << "255\n";
  int size = width * height;
  float small, delta, large;
  if (type) { 
    small = large = image[0];
    for (int cnt=0;cnt<size;cnt++) {
      if (small>image[cnt]) small = (float)image[cnt];
      if (large<image[cnt]) large = (float)image[cnt];
    }
    delta = (float)255.0 / (large-small);
    if (ascii) {  // with rescale, with ascii
      for (int cnt=0;cnt<size;cnt++) {
	int value = (int)(delta * ((float)image[cnt]-small));
	if (value<0) value = 0;
	else if (value>255) value = 255;
	imagefile << value;
	if ((cnt&15)==15) imagefile << "\n";
	else imagefile << " ";
      }
    } else {      // with rescale, no ascii
      unsigned char *tmp = new unsigned char[size];
      for (int cnt=0;cnt<size;cnt++) {
	int value = (int)(delta * ((float)image[cnt]-small));
	if (value<0) tmp[cnt] = 0;
	else if (value>255) tmp[cnt] = 255;
	tmp[cnt] = (unsigned char)value;
      }
      imagefile.write((char*)tmp, size);
      delete [] tmp;
    }
  } else {
    if (ascii) { // no rescale, with ascii
      for(int cnt=0;cnt<size;cnt++) {
	int value = (int)image[cnt];
	imagefile << value;
	if ((cnt&15)==15) imagefile << "\n";
	else imagefile << " ";
      }
    } else {    // no rescale, no ascii
      if (typeid(T)==typeid(unsigned char) || typeid(T)==typeid(char))
	imagefile.write((char*)image, size);
      else {
	unsigned char *tmp = new unsigned char[size];
	for (int cnt=0;cnt<size;cnt++) 
	  tmp[cnt] = (unsigned char)image[cnt];
	imagefile.write((char*)tmp, size);
	delete [] tmp;
      }
    }
  }
  imagefile.close();
  std::cout << "File " << filename << " saved. ";
  if (type) std::cout << "[" << small << "," << large << "]";
  std::cout << std::endl;
}

template<class T>
void Image<T>::StoreRGB(const char *filename) const
{
  assert(!(width%3));
  std::ofstream imagefile;
  imagefile.open(filename);
  imagefile << "P3\n";
  imagefile << width/3 << " " << height << "\n";
  imagefile << "255\n";
  int size = width * height;
  for(int cnt=0;cnt<size;cnt+=3) {
    imagefile << (int)image[cnt+0] << " ";
    imagefile << (int)image[cnt+1] << " ";
    imagefile << (int)image[cnt+2];
    if ((cnt%15)==12) imagefile << "\n";
    else imagefile << " ";
  }
  imagefile.close();
  std::cout << "File " << filename << " saved. " << std::endl;
}

template<class T>
void Image<T>::StoreYUV(const char *filename) const
{
  assert(!(width%2));
  std::ofstream imagefile;
  imagefile.open(filename);
  imagefile << "P3\n";
  imagefile << width/2 << " " << height << "\n";
  imagefile << "255\n";
  int size = width * height;
  for(int cnt=0;cnt<size;cnt+=4) {
    int y1 = (int)image[cnt+3];
    int v = (int)image[cnt+2]-128;
    int y0 = (int)image[cnt+1];
    int u = (int)image[cnt]-128;
    int r = (int)(1.0000*y0 - 0.0009*u + 1.1359*v);
    int g = (int)(1.0000*y0 - 0.3959*u - 0.5783*v);
    int b = (int)(1.0000*y0 + 2.0411*u - 0.0016*v);
    r = (r<0 ? 0 : (r>255 ? 255 : r));
    g = (g<0 ? 0 : (g>255 ? 255 : g));
    b = (b<0 ? 0 : (b>255 ? 255 : b));
    imagefile << r << " " << g << " " << b << " ";
    r = (int)(1.0000*y1 - 0.0009*u + 1.1359*v);
    g = (int)(1.0000*y1 - 0.3959*u - 0.5783*v);
    b = (int)(1.0000*y1 + 2.0411*u - 0.0016*v);
    r = (r<0 ? 0 : (r>255 ? 255 : r));
    g = (g<0 ? 0 : (g>255 ? 255 : g));
    b = (b<0 ? 0 : (b>255 ? 255 : b));
    imagefile << r << " " << g << " " << b << " ";
    if ((cnt%15)==12) imagefile << "\n";
    else imagefile << " ";
  }
  imagefile.close();
  std::cout << "File " << filename << " saved. " << std::endl;
}

template <class T>
void Image<T>::operator=(Image<T> &src)
{
  memcpy(image, src.GetData(), sizeof(T) * width * height);
}

template<class T>
Buffer<T>::Buffer(int w, int h) : width(w), height(h)
{
  int extra = 16 / sizeof(T);
  img = new T[w*h+extra];
  //std::cout << "Buffer: " << img << " " << &img[w*h+extra] << std::endl;
  //image  = (T *)((unsigned long)(img+extra) & (unsigned long)(~15));
#if __WORDSIZE == 64
    image = (T *)((uint64_t)(img+extra) & (uint64_t)(~15));
#else 
    image = (T *)((uint32_t)(img+extra) & (uint32_t)(~15));
#endif
}

template<class T>
void Buffer<T>::Clear(T val)
{
  for (int i=0;i<(width*height);i++) 
    image[i] = (T) val;
}

#endif // TPIMAGE_H
