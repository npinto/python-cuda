# coding:utf-8: © Arno Pähler, 2007-08

from ctypes import *

c_int_p   = POINTER(c_int)
c_uint_p  = POINTER(c_uint)
c_float_p = POINTER(c_float)
c_double_p = POINTER(c_double)

ch256 = c_char*256

#
# CUDA: driver_types.h
#

#enum cudaError
#{
#  cudaSuccess = 0,
#  cudaErrorMissingConfiguration,
#  cudaErrorMemoryAllocation,
#  cudaErrorInitializationError,
#  cudaErrorLaunchFailure,
#  cudaErrorPriorLaunchFailure,
#  cudaErrorLaunchTimeout,
#  cudaErrorLaunchOutOfResources,
#  cudaErrorInvalidDeviceFunction,
#  cudaErrorInvalidConfiguration,
#  cudaErrorInvalidDevice,
#  cudaErrorInvalidValue,
#  cudaErrorInvalidPitchValue,
#  cudaErrorInvalidSymbol,
#  cudaErrorMapBufferObjectFailed,
#  cudaErrorUnmapBufferObjectFailed,
#  cudaErrorInvalidHostPointer,
#  cudaErrorInvalidDevicePointer,
#  cudaErrorInvalidTexture,
#  cudaErrorInvalidTextureBinding,
#  cudaErrorInvalidChannelDescriptor,
#  cudaErrorInvalidMemcpyDirection,
#  cudaErrorAddressOfConstant,
#  cudaErrorTextureFetchFailed,
#  cudaErrorTextureNotBound,
#  cudaErrorSynchronizationError,
#  cudaErrorInvalidFilterSetting,
#  cudaErrorInvalidNormSetting,
#  cudaErrorMixedDeviceExecution,
#  cudaErrorCudartUnloading,
#  cudaErrorUnknown,
#  cudaErrorNotYetImplemented,
#  cudaErrorMemoryValueTooLarge,
#  cudaErrorInvalidResourceHandle,
#  cudaErrorNotReady,
#  cudaErrorStartupFailure = 0x7f,
#  cudaErrorApiFailureBase = 10000
#};

cudaError = c_int
(
  cudaSuccess,
  cudaErrorMissingConfiguration,
  cudaErrorMemoryAllocation,
  cudaErrorInitializationError,
  cudaErrorLaunchFailure,
  cudaErrorPriorLaunchFailure,
  cudaErrorLaunchTimeout,
  cudaErrorLaunchOutOfResources,
  cudaErrorInvalidDeviceFunction,
  cudaErrorInvalidConfiguration,
  cudaErrorInvalidDevice,
  cudaErrorInvalidValue,
  cudaErrorInvalidPitchValue,
  cudaErrorInvalidSymbol,
  cudaErrorMapBufferObjectFailed,
  cudaErrorUnmapBufferObjectFailed,
  cudaErrorInvalidHostPointer,
  cudaErrorInvalidDevicePointer,
  cudaErrorInvalidTexture,
  cudaErrorInvalidTextureBinding,
  cudaErrorInvalidChannelDescriptor,
  cudaErrorInvalidMemcpyDirection,
  cudaErrorAddressOfConstant,
  cudaErrorTextureFetchFailed,
  cudaErrorTextureNotBound,
  cudaErrorSynchronizationError,
  cudaErrorInvalidFilterSetting,
  cudaErrorInvalidNormSetting,
  cudaErrorMixedDeviceExecution,
  cudaErrorCudartUnloading,
  cudaErrorUnknown,
  cudaErrorNotYetImplemented,
  cudaErrorMemoryValueTooLarge,
  cudaErrorInvalidResourceHandle,
  cudaErrorNotReady,
  cudaErrorStartupFailure,
  cudaErrorApiFailureBase
) = range(35)+[0x7f,10000]

#enum cudaChannelFormatKind
#{
#  cudaChannelFormatKindSigned,
#  cudaChannelFormatKindUnsigned,
#  cudaChannelFormatKindFloat
#};

cudaChannelFormatKind = c_int
(
  cudaChannelFormatKindSigned,
  cudaChannelFormatKindUnsigned,
  cudaChannelFormatKindFloat
) = range(3)

#struct cudaChannelFormatDesc
#{
#  int                        x;
#  int                        y;
#  int                        z;
#  int                        w;
#  enum cudaChannelFormatKind f;
#};

class cudaChannelFormatDesc(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int),
                ("z", c_int),
                ("w", c_int),
                ("f", cudaChannelFormatKind)]
cudaChannelFormatDesc_p = POINTER(cudaChannelFormatDesc)

#struct cudaArray;
class cudaArray(Structure):
    pass
cudaArray_p = POINTER(cudaArray)

#enum cudaMemcpyKind
#{
#  cudaMemcpyHostToHost,
#  cudaMemcpyHostToDevice,
#  cudaMemcpyDeviceToHost,
#  cudaMemcpyDeviceToDevice
#};
cudaMemcpyKind = c_int
(
  cudaMemcpyHostToHost,
  cudaMemcpyHostToDevice,
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice
) = range(4)

#struct cudaPitchedPtr
#{
#  void   *ptr;
#  size_t  pitch;
#  size_t  xsize;
#  size_t  ysize;
#};
#
#struct cudaExtent
#{
#  size_t width;
#  size_t height;
#  size_t depth;
#};
#
#struct cudaPos
#{
#  size_t x;
#  size_t y;
#  size_t z;
#};
#
class cudaPitchedPtr(Structure):
    _fields_ = [("width",  c_uint),
                ("height", c_uint),
                ("depth",  c_uint)]

class cudaExtent(Structure):
    _fields_ = [("x", c_uint),
                ("y", c_uint),
                ("z", c_uint)]

class cudaPos(Structure):
    _fields_ = [("ptr",   c_void_p),
                ("pitch", c_uint),
                ("xsize", c_uint),
                ("ysize", c_uint)]

#struct cudaMemcpy3DParms
#{
#  struct cudaArray      *srcArray;
#  struct cudaPos         srcPos;
#  struct cudaPitchedPtr  srcPtr;
#
#  struct cudaArray      *dstArray;
#  struct cudaPos         dstPos;
#  struct cudaPitchedPtr  dstPtr;
#
#  struct cudaExtent      extent;
#  enum cudaMemcpyKind    kind;
#};
class cudaMemcpy3DParms(Structure):
    _fields_ = [("srcArray", cudaArray_p),
                ("srcPos",   cudaPos),
                ("srcPtr",   cudaPitchedPtr),
                ("dstArray", cudaArray_p),
                ("dstPos",   cudaPos),
                ("dstPtr",   cudaPitchedPtr),
                ("extent",   cudaExtent),
                ("kind",     cudaMemcpyKind)]

#struct cudaDeviceProp
#{
#  char   name[256];
#  size_t totalGlobalMem;
#  size_t sharedMemPerBlock;
#  int    regsPerBlock;
#  int    warpSize;
#  size_t memPitch;
#  int    maxThreadsPerBlock;
#  int    maxThreadsDim[3];
#  int    maxGridSize[3]; 
#  int    clockRate;
#  size_t totalConstMem; 
#  int    major;
#  int    minor;
#  size_t textureAlignment;
#  int    deviceOverlap;
#  int    multiProcessorCount;
#  int    __cudaReserved[40];
#};

class cudaDeviceProp(Structure):
    _fields_ = [("name",                c_char*256),
                ("totalGlobalMem",      c_uint),
                ("sharedMemPerBlock",   c_int),
                ("regsPerBlock",        c_int),
                ("warpSize",            c_int),
                ("memPitch",            c_uint),
                ("maxThreadsPerBlock",  c_uint),
                ("maxThreadsDim",       c_uint*3),
                ("maxGridSize",         c_uint*3),
                ("clockRate",           c_int),
                ("totalConstMem",       c_uint),
                ("major",               c_int),
                ("minor",               c_int),
                ("textureAlignment",    c_uint),
                ("deviceOverlap",       c_int),
                ("multiProcessorCount", c_int),
                ("__cudaReserved",      c_int*40)]

    def __repr__(self):
        '''Print structured objects'''
        res = []
        for field in self._fields_:
            res.append('%s=%s' % (field[0], repr(getattr(self, field[0]))))
        return self.__class__.__name__ + '(' + ','.join(res) + ')'

    def __str__(self):
        '''Print structured objects'''
        res = []
        for field in self._fields_:
            data = getattr(self,field[0])
            if field[0] == "maxThreadsDim":
                data = "%d %d %d" % (data[0],data[1],data[2])
            if field[0] == "maxGridSize":
                data = "%d %d %d" % (data[0],data[1],data[2])
            res.append('%-18s = %s' % (field[0], data))
        return "\n".join(res)
cudaDeviceProp_p = POINTER(cudaDeviceProp)

#typedef cuddaError cudaError_t;
#typedef int cudaStream_t;
#typedef int cudaEvent_t;
cudaError_t = cudaError
cudaStream_t = c_int
cudaEvent_t = c_int

#
# CUDA: texture_types.h
#

#enum cudaTextureAddressMode
#{
#  cudaAddressModeWrap,
#  cudaAddressModeClamp
#};
cudaTextureAddressMode = c_int
(
  cudaAddressModeWrap,
  cudaAddressModeClamp
) = range(2)

#enum cudaTextureFilterMode
#{
#  cudaFilterModePoint,
#  cudaFilterModeLinear
#};
cudaTextureFilterMode = c_int
(
  cudaFilterModePoint,
  cudaFilterModeLinear
) = range(2)

#enum cudaTextureReadMode
#{
#  cudaReadModeElementType,
#  cudaReadModeNormalizedFloat
#};
cudaTextureReadMode = c_int
(
  cudaReadModeElementType,
  cudaReadModeNormalizedFloat
) = range(2)

#struct textureReference
#{
#  int                           normalized;
#  enum cudaTextureFilterMode    filterMode;
#  enum cudaTextureAddressMode   addressMode[2];
#  struct cudaChannelFormatDesc  channelDesc;
#};
class textureReference(Structure):
    _fields_ = [("normalized",  c_int),
                ("filterMode",  cudaTextureFilterMode),
                ("addressMode", c_uint*2),
                ("channelDesc", cudaChannelFormatDesc)]

textureReference_p = POINTER(textureReference)

#template<class T, int dim = 1, enum cudaTextureReadMode = cudaReadModeElementType>
#struct texture : public textureReference
#{
#  __host__ texture(int                         norm  = 0,
#                   enum cudaTextureFilterMode  fMode = cudaFilterModePoint,
#                   enum cudaTextureAddressMode aMode = cudaAddressModeClamp)
#  {
#    normalized     = norm;
#    filterMode     = fMode;
#    addressMode[0] = aMode;
#    addressMode[1] = aMode;
#    channelDesc    = cudaCreateChannelDesc<T>();
#  }
#
#  __host__ texture(int                          norm,
#                   enum cudaTextureFilterMode   fMode,
#                   enum cudaTextureAddressMode  aMode,
#                   struct cudaChannelFormatDesc desc)
#  {
#    normalized     = norm;
#    filterMode     = fMode;
#    addressMode[0] = aMode;
#    addressMode[1] = aMode;
#    channelDesc    = desc;
#  }
#};
##    This will not work, have to think about it more
##class texture(textureReference):
##    def __init__(self,norm=0,fMode=cudaFilterModePoint,
##                aMode=cudaAddressModeClamp,desc=None):
##        self.normalized = norm
##        self.filterMode = fMode
##        self.addressMode[0] = aMode
##        self.addressMode[1] = aMode
##        self.channelDesc = cudaCreateChannelDesc()
##        self.type = None
##        self.dim = 1
##        self.cudaTextureReadMode = cudaReadModeElementType

#
# CUDA: device_types.h
#

#enum cudaRoundMode
#{
#  cudaRoundNearest,
#  cudaRoundZero,
#  cudaRoundPosInf,
#  cudaRoundMinInf
#};

cudaRoundMode = c_int
(
  cudaRoundNearest,
  cudaRoundZero,
  cudaRoundPosInf,
  cudaRoundMinInf
) = range(4)


#
# CUDA: vector_types.h
#

#struct char1
#{
#  signed char x;
#};
#
#struct uchar1 
#{
#  unsigned char x;
#};
#
class char1(Structure):
    _fields_ = [("x",   c_byte)]
class uchar1(Structure):
    _fields_ = [("x",   c_ubyte)]

#struct __builtin_align__(2) char2
#{
#  signed char x, y;
#};
#
#struct __builtin_align__(2) uchar2
#{
#  unsigned char x, y;
#};
#
class char2(Structure):
    _fields_ = [("x",   c_byte),
                ("y",   c_byte)]
class uchar2(Structure):
    _fields_ = [("x",   c_ubyte),
                ("y",   c_ubyte)]

#struct char3
#{
#  signed char x, y, z;
#};
#
#struct uchar3
#{
#  unsigned char x, y, z;
#};
#
class char3(Structure):
    _fields_ = [("x",   c_byte),
                ("y",   c_byte),
                ("z",   c_byte)]
class uchar3(Structure):
    _fields_ = [("x",   c_ubyte),
                ("y",   c_ubyte),
                ("z",   c_ubyte)]

#struct __builtin_align__(4) char4
#{
#  signed char x, y, z, w;
#};
#
#struct __builtin_align__(4) uchar4
#{
#  unsigned char x, y, z, w;
#};
#
class char4(Structure):
    _fields_ = [("x",   c_byte),
                ("y",   c_byte),
                ("z",   c_byte),
                ("w",   c_byte)]
class uchar4(Structure):
    _fields_ = [("x",   c_ubyte),
                ("y",   c_ubyte),
                ("z",   c_ubyte),
                ("w",   c_ubyte)]

#struct short1
#{
#  short x;
#};
#
#struct ushort1
#{
#  unsigned short x;
#};
#
class short1(Structure):
    _fields_ = [("x",   c_short)]
class ushort1(Structure):
    _fields_ = [("x",   c_ushort)]

#struct __builtin_align__(4) short2
#{
#  short x, y;
#};
#
#struct __builtin_align__(4) ushort2
#{
#  unsigned short x, y;
#};
#
class short2(Structure):
    _fields_ = [("x",   c_short),
                ("y",   c_short)]
class ushort2(Structure):
    _fields_ = [("x",   c_ushort),
                ("y",   c_ushort)]

#struct short3
#{
#  short x, y, z;
#};
#
#struct ushort3
#{
#  unsigned short x, y, z;
#};
#
class short3(Structure):
    _fields_ = [("x",   c_short),
                ("y",   c_short),
                ("z",   c_short)]
class ushort3(Structure):
    _fields_ = [("x",   c_ushort),
                ("y",   c_ushort),
                ("z",   c_ushort)]

#struct __builtin_align__(8) short4
#{
#  short x, y, z, w;
#};
#
#struct __builtin_align__(8) ushort4
#{
#  unsigned short x, y, z, w;
#};
#
class short4(Structure):
    _fields_ = [("x",   c_short),
                ("y",   c_short),
                ("z",   c_short),
                ("w",   c_short)]
class ushort4(Structure):
    _fields_ = [("x",   c_ushort),
                ("y",   c_ushort),
                ("z",   c_ushort),
                ("w",   c_ushort)]

#struct int1
#{
#  int x;
#};
#
#struct uint1
#{
#  unsigned int x;
#};
#
class int1(Structure):
    _fields_ = [("x",   c_int)]
class uint1(Structure):
    _fields_ = [("x",   c_uint)]

#struct __builtin_align__(8) int2
#{
#  int x, y;
#};
#
#struct __builtin_align__(8) uint2
#{
#  unsigned int x, y;
#};
#
class int2(Structure):
    _fields_ = [("x",   c_int),
                ("y",   c_int)]
class uint2(Structure):
    _fields_ = [("x",   c_uint),
                ("y",   c_uint)]

#struct int3
#{
#  int x, y, z;
#};
#
#struct uint3
#{
#  unsigned int x, y, z;
#};
#
class int3(Structure):
    _fields_ = [("x",   c_int),
                ("y",   c_int),
                ("z",   c_int)]
class uint3(Structure):
    _fields_ = [("x",   c_uint),
                ("y",   c_uint),
                ("z",   c_uint)]

#struct __builtin_align__(16) int4
#{
#  int x, y, z, w;
#};
#
#struct __builtin_align__(16) uint4
#{
#  unsigned int x, y, z, w;
#};
#
class int4(Structure):
    _fields_ = [("x",   c_int),
                ("y",   c_int),
                ("z",   c_int),
                ("w",   c_int)]
class uint4(Structure):
    _fields_ = [("x",   c_uint),
                ("y",   c_uint),
                ("z",   c_uint),
                ("w",   c_uint)]

#struct long1
#{
#  long x;
#};
#
#struct ulong1
#{
#  unsigned long x;
#};
#
class long1(Structure):
    _fields_ = [("x",   c_long)]
class ulong1(Structure):
    _fields_ = [("x",   c_ulong)]

#struct __builtin_align__(8) long2
#{
#  long x, y;
#};
#
#struct __builtin_align__(8) ulong2
#{
#  unsigned long x, y;
#};
#
class long2(Structure):
    _fields_ = [("x",   c_long),
                ("y",   c_long)]
class ulong2(Structure):
    _fields_ = [("x",   c_ulong),
                ("y",   c_ulong)]

#struct long3
#{
#  long x, y, z;
#};
#
#struct ulong3
#{
#  unsigned long x, y, z;
#};
#
class long3(Structure):
    _fields_ = [("x",   c_long),
                ("y",   c_long),
                ("z",   c_long)]
class ulong3(Structure):
    _fields_ = [("x",   c_ulong),
                ("y",   c_ulong),
                ("z",   c_ulong)]

#struct __builtin_align__(16) long4
#{
#  long x, y, z, w;
#};
#
#struct __builtin_align__(16) ulong4
#{
#  unsigned long x, y, z, w;
#};
#
class long4(Structure):
    _fields_ = [("x",   c_long),
                ("y",   c_long),
                ("z",   c_long),
                ("w",   c_long)]
class ulong4(Structure):
    _fields_ = [("x",   c_ulong),
                ("y",   c_ulong),
                ("z",   c_ulong),
                ("w",   c_ulong)]

#struct float1
#{
#  float x;
#};
#
class float1(Structure):
    _fields_ = [("x",   c_float)]

#struct __builtin_align__(8) float2
#{
#  float x, y;
#};
#
class float2(Structure):
    _fields_ = [("x",   c_float),
                ("y",   c_float)]

#struct float3
#{
#  float x, y, z;
#};
#
class float3(Structure):
    _fields_ = [("x",   c_float),
                ("y",   c_float),
                ("z",   c_float)]

#struct __builtin_align__(16) float4
#{
#  float x, y, z, w;
#};
#
class float4(Structure):
    _fields_ = [("x",   c_float),
                ("y",   c_float),
                ("z",   c_float),
                ("w",   c_float)]

#struct double1
#{
#  double x;
#};
class double1(Structure):
    _fields_ = [("x",   c_float)]

#struct __builtin_align__(16) double2
#{
#  double x, y;
#};
class double2(Structure):
    _fields_ = [("x",   c_float),
                ("y",   c_float)]

#struct dim3
#{
#    unsigned int x, y, z;
##if defined(__cplusplus)
#    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
#    dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
#    operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
##endif /* __cplusplus */
#};
#
class dim3(Structure):
    _fields_ = [("x",   c_uint),
                ("y",   c_uint),
                ("z",   c_uint)]

# for OpenGL

GLuint = c_uint
