# coding:utf-8: © Arno Pähler, 2007-08
# NP: remove absolute path in CUDART dyn lib

from cuda_defs import *

CUDART = "libcudart.so"
cudart = CDLL(CUDART)

class cudaException(Exception):
    pass

#cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchDevPtr,
#    struct cudaExtent extent);
cudaMalloc3D = cudart.cudaMalloc3D
cudaMalloc3D.restype = cudaError_t
cudaMalloc3D.argtypes = [ POINTER(cudaPitchedPtr), cudaExtent ]

#cudaError_t cudaMalloc3DArray(struct cudaArray** arrayPtr,
#    const struct cudaChannelFormatDesc* desc, struct cudaExtent extent);
cudaMalloc3DArray = cudart.cudaMalloc3DArray
cudaMalloc3DArray.restype = cudaError_t
cudaMalloc3DArray.argtypes = [ POINTER(cudaArray_p),
    POINTER(cudaChannelFormatDesc), cudaExtent ]

#cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchDevPtr,
#    int value, struct cudaExtent extent);
cudaMemset3D = cudart.cudaMemset3D
cudaMemset3D.restype = cudaError_t
cudaMemset3D.argtypes = [ POINTER(cudaPitchedPtr), c_int, cudaExtent ]

#cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
cudaMemcpy3D = cudart.cudaMemcpy3D
cudaMemcpy3D.restype = cudaError_t
cudaMemcpy3D.argtypes = [ POINTER(cudaMemcpy3DParms) ]

#cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p,
#    cudaStream_t stream);
cudaMemcpy3DAsync = cudart.cudaMemcpy3DAsync
cudaMemcpy3DAsync.restype = cudaError_t
cudaMemcpy3DAsync.argtypes = [ POINTER(cudaMemcpy3DParms), cudaStream_t ]

#cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaMalloc = cudart.cudaMalloc
cudaMalloc.restype = cudaError_t
cudaMalloc.argtypes = [ POINTER(c_void_p), c_uint ]

#cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaMallocHost = cudart.cudaMallocHost
cudaMallocHost.restype = cudaError_t
cudaMallocHost.argtypes = [ POINTER(c_void_p), c_uint ]

#cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
cudaMallocPitch = cudart.cudaMallocPitch
cudaMallocPitch.restype = cudaError_t
cudaMallocPitch.argtypes = [ POINTER(c_void_p), c_uint_p, c_uint, c_uint ]

#cudaError_t cudaMallocArray(struct cudaArray **array,
#const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(1));
cudaMallocArray = cudart.cudaMallocArray
cudaMallocArray.restype = cudaError_t
cudaMallocArray.argtypes = \
    [ POINTER(cudaArray_p), cudaChannelFormatDesc_p, c_uint, c_uint ]

#cudaError_t cudaFree(void *devPtr);
cudaFree = cudart.cudaFree
cudaFree.restype = cudaError_t
cudaFree.argtypes = [ c_void_p ]

#cudaError_t cudaFreeHost(void *ptr);
cudaFreeHost = cudart.cudaFreeHost
cudaFreeHost.restype = cudaError_t
cudaFreeHost.argtypes = [ c_void_p ]

#cudaError_t cudaFreeArray(struct cudaArray *array);
cudaFreeArray = cudart.cudaFreeArray
cudaFreeArray.restype = cudaError_t
cudaFreeArray.argtypes = [ cudaArray_p ]

#cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
#enum cudaMemcpyKind kind);
cudaMemcpy = cudart.cudaMemcpy
cudaMemcpy.restype = cudaError_t
cudaMemcpy.argtypes = [ c_void_p, c_void_p, c_uint, c_uint ]

#cudaError_t cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset,
#const void *src, size_t count, enum cudaMemcpyKind kind);
cudaMemcpyToArray = cudart.cudaMemcpyToArray
cudaMemcpyToArray.restype = cudaError_t
cudaMemcpyToArray.argtypes = [ cudaArray_p, c_uint, c_uint, c_void_p, c_uint, c_uint]

#cudaError_t cudaMemcpyFromArray(void *dst, const struct cudaArray *src,
#size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
cudaMemcpyFromArray = cudart.cudaMemcpyFromArray
cudaMemcpyFromArray.restype = cudaError_t
cudaMemcpyFromArray.argtypes = [ c_void_p, cudaArray_p,
    c_uint, c_uint, c_uint, c_uint ]

#cudaError_t cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst,
#size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc,
#size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
cudaMemcpyArrayToArray = cudart.cudaMemcpyArrayToArray
cudaMemcpyArrayToArray.restype = cudaError_t
cudaMemcpyArrayToArray.argtypes = [ cudaArray_p, c_uint, c_uint,
cudaArray_p, c_uint, c_uint, c_uint, c_uint]

#cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
#size_t width, size_t height, enum cudaMemcpyKind kind);
cudaMemcpy2D = cudart.cudaMemcpy2D
cudaMemcpy2D.restype = cudaError_t
cudaMemcpy2D.argtypes = [ c_void_p, c_uint, c_void_p, c_uint,
c_uint, c_uint, c_uint ]

#cudaError_t cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset,
#size_t hOffset, const void *src, size_t spitch, size_t width, size_t height,
#enum cudaMemcpyKind kind);
cudaMemcpy2DToArray = cudart.cudaMemcpy2DToArray
cudaMemcpy2DToArray.restype = cudaError_t
cudaMemcpy2DToArray.argtypes = [ cudaArray_p, c_uint, c_uint, 
c_void_p, c_uint, c_uint, c_uint, c_uint ]

#cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch,
#const struct cudaArray *src, size_t wOffset, size_t hOffset,
#size_t width, size_t height, enum cudaMemcpyKind kind);
cudaMemcpy2DFromArray = cudart.cudaMemcpy2DFromArray
cudaMemcpy2DFromArray.restype = cudaError_t
cudaMemcpy2DFromArray.argtypes = [ c_void_p, c_uint, cudaArray_p, c_uint, c_uint,
c_uint, c_uint, c_uint ]

#cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst,
#size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc,
#size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
cudaMemcpy2DArrayToArray = cudart.cudaMemcpy2DArrayToArray
cudaMemcpy2DArrayToArray.restype = cudaError_t
cudaMemcpy2DArrayToArray.argtypes = [ cudaArray_p, c_uint, c_uint,
cudaArray_p, c_uint, c_uint, c_uint, c_uint, c_uint ]

#cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count,
#size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));
cudaMemcpyToSymbol = cudart.cudaMemcpyToSymbol
cudaMemcpyToSymbol.restype = cudaError_t
cudaMemcpyToSymbol.argtypes = [ c_char_p, c_void_p, c_uint, c_uint, c_uint ]

#cudaError_t cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count,
#size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));
cudaMemcpyFromSymbol = cudart.cudaMemcpyFromSymbol
cudaMemcpyFromSymbol.restype = cudaError_t
cudaMemcpyFromSymbol.argtypes = [ c_void_p, c_char_p, c_uint, c_uint, c_uint ]

#cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
#enum cudaMemcpyKind kind, cudaStream_t stream);
cudaMemcpyAsync = cudart.cudaMemcpyAsync
cudaMemcpyAsync.restype = cudaError_t
cudaMemcpyAsync.argtypes = [ c_void_p, c_void_p, c_uint, c_int, c_int ]

#cudaError_t cudaMemcpyToArrayAsync(struct cudaArray *dst,
#size_t wOffset, size_t hOffset, const void *src, size_t count,
#enum cudaMemcpyKind kind, cudaStream_t stream);
cudaMemcpyToArrayAsync = cudart.cudaMemcpyToArrayAsync
cudaMemcpyToArrayAsync.restype = cudaError_t
cudaMemcpyToArrayAsync.argtypes = [ c_void_p, c_uint, c_uint,
    c_void_p, c_uint, c_int, c_int ]

#cudaError_t cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src,
#size_t wOffset, size_t hOffset, size_t count,
#enum cudaMemcpyKind kind, cudaStream_t stream);
cudaMemcpyFromArrayAsync = cudart.cudaMemcpyFromArrayAsync
cudaMemcpyFromArrayAsync.restype = cudaError_t
cudaMemcpyFromArrayAsync.argtypes = [ c_void_p, c_uint, c_uint,
    c_void_p, c_uint, c_int, c_int ]

#cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch,
#const void *src, size_t spitch, size_t width, size_t height,
#enum cudaMemcpyKind kind, cudaStream_t stream);
cudaMemcpy2DAsync = cudart.cudaMemcpy2DAsync
cudaMemcpy2DAsync.restype = cudaError_t
cudaMemcpy2DAsync.argtypes = [ c_void_p, c_uint, c_void_p, c_uint,
    c_uint, c_uint, c_int, c_int ]

#cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray *dst,
#size_t wOffset, size_t hOffset, const void *src,
#size_t spitch, size_t width, size_t height,
#enum cudaMemcpyKind kind, cudaStream_t stream);
cudaMemcpy2DToArrayAsync = cudart.cudaMemcpy2DToArrayAsync
cudaMemcpy2DToArrayAsync.restype = cudaError_t
cudaMemcpy2DToArrayAsync.argtypes = [ c_void_p, c_uint, c_uint,
    c_void_p, c_uint, c_uint, c_uint, c_uint, c_int, c_int ]

#cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
#const struct cudaArray *src, size_t wOffset, size_t hOffset,
#size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaMemcpy2DFromArrayAsync = cudart.cudaMemcpy2DFromArrayAsync
cudaMemcpy2DFromArrayAsync.restype = cudaError_t
cudaMemcpy2DFromArrayAsync.argtypes = [ c_void_p, c_uint, c_uint,
    c_void_p, c_uint, c_uint, c_uint, c_uint, c_int, c_int ]

#cudaError_t cudaMemset(void *mem, int c, size_t count);
cudaMemset = cudart.cudaMemset
cudaMemset.restype = cudaError_t
cudaMemset.argtypes = [ c_void_p, c_int, c_uint ]

#cudaError_t cudaMemset2D(void *mem, size_t pitch, int c,
#size_t width, size_t height);
cudaMemset2D = cudart.cudaMemset2D
cudaMemset2D.restype = cudaError_t
cudaMemset2D.argtypes = [ c_void_p, c_uint,  c_int, c_uint, c_uint ]

#cudaError_t cudaGetSymbolAddress(void **devPtr, const char *symbol);
cudaGetSymbolAddress = cudart.cudaGetSymbolAddress
cudaGetSymbolAddress.restype = cudaError_t
cudaGetSymbolAddress.argtypes = [ POINTER(c_void_p), c_char_p ]

#cudaError_t cudaGetSymbolSize(size_t *size, const char *symbol);
cudaGetSymbolSize = cudart.cudaGetSymbolSize
cudaGetSymbolSize.restype = cudaError_t
cudaGetSymbolSize.argtypes = [ c_uint_p, c_char_p ]

#cudaError_t cudaGetDeviceCount(int* count);
cudaGetDeviceCount = cudart.cudaGetDeviceCount
cudaGetDeviceCount.restype = cudaError_t
cudaGetDeviceCount.argtypes = [ c_int_p ]

#cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int dev);
cudaGetDeviceProperties = cudart.cudaGetDeviceProperties
cudaGetDeviceProperties.restype = cudaError_t
cudaGetDeviceProperties.argtypes = [ cudaDeviceProp_p, c_int ]

#cudaError_t cudaChooseDevice(int* dev, const struct cudaDeviceProp* prop);
cudaChooseDevice = cudart.cudaChooseDevice
cudaChooseDevice.restype = cudaError_t
cudaChooseDevice.argtypes = [ c_int_p, cudaDeviceProp_p ]

#cudaError_t cudaSetDevice(int dev);
cudaSetDevice = cudart.cudaSetDevice
cudaSetDevice.restype = cudaError_t
cudaSetDevice.argtypes = [ c_int ]

#cudaError_t cudaGetDevice(int* dev);
cudaGetDevice = cudart.cudaGetDevice
cudaGetDevice.restype = cudaError_t
cudaGetDevice.argtypes = [ c_int_p ]

#cudaError_t cudaBindTexture(size_t *offset, const struct textureReference *texref,
#const void *devPtr, const struct cudaChannelFormatDesc *desc,
#size_t size __dv(UINT_MAX));
cudaBindTexture = cudart.cudaBindTexture
cudaBindTexture.restype = cudaError_t
cudaBindTexture.argtypes = [ c_uint, textureReference_p, c_void_p,
    cudaChannelFormatDesc_p ]

#cudaError_t cudaBindTextureToArray(const struct textureReference *texref,
#const struct cudaArray *array, const struct cudaChannelFormatDesc *desc);
cudaBindTextureToArray = cudart.cudaBindTextureToArray
cudaBindTextureToArray.restype = cudaError_t
cudaBindTextureToArray.argtypes = [ textureReference_p, cudaArray_p,
cudaChannelFormatDesc_p ]

#cudaError_t cudaUnbindTexture(const struct textureReference *texref);
cudaUnbindTexture = cudart.cudaUnbindTexture
cudaUnbindTexture.restype = cudaError_t
cudaUnbindTexture.argtypes = [ textureReference_p ]

#cudaError_t cudaGetTextureAlignmentOffset(size_t *offset,
#const struct textureReference *texref);
cudaGetTextureAlignmentOffset = cudart.cudaGetTextureAlignmentOffset
cudaGetTextureAlignmentOffset.restype = cudaError_t
cudaGetTextureAlignmentOffset.argtypes = [ c_uint_p, textureReference_p ]

#cudaError_t cudaGetTextureReference(const struct textureReference **texref,
#const char *symbol);
cudaGetTextureReference = cudart.cudaGetTextureReference
cudaGetTextureReference.restype = cudaError_t
cudaGetTextureReference.argtypes = [ textureReference_p, c_char_p ]

#cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc,
#const struct cudaArray *array);
cudaGetChannelDesc = cudart.cudaGetChannelDesc
cudaGetChannelDesc.restype = cudaError_t
cudaGetChannelDesc.argtypes = [ cudaChannelFormatDesc_p, cudaArray_p ]

# struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
#enum cudaChannelFormatKind f);
cudaCreateChannelDesc = cudart.cudaCreateChannelDesc
cudaCreateChannelDesc.restype = cudaChannelFormatDesc
cudaCreateChannelDesc.argtypes = [ c_int, c_int, c_int, textureReference_p, c_uint ]

#cudaError_t cudaGetLastError(void);
cudaGetLastError = cudart.cudaGetLastError
cudaGetLastError.restype = cudaError_t
cudaGetLastError.argtypes = None

#const char* cudaGetErrorString(cudaError_t error);
cudaGetErrorString = cudart.cudaGetErrorString
cudaGetErrorString.restype = c_char_p
cudaGetErrorString.argtypes = [ c_uint ]

#cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
#size_t sharedMem __dv(0), int tokens __dv(0));
cudaConfigureCall = cudart.cudaConfigureCall
cudaConfigureCall.restype = cudaError_t
cudaConfigureCall.argtypes = [ dim3, dim3, c_uint, c_int ]

#cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset);
cudaSetupArgument = cudart.cudaSetupArgument
cudaSetupArgument.restype = cudaError_t
cudaSetupArgument.argtypes = [ c_void_p, c_uint, c_uint ]

#cudaError_t cudaLaunch(const char *symbol);
cudaLaunch = cudart.cudaLaunch
cudaLaunch.restype = cudaError_t
cudaLaunch.argtypes = [ c_char_p ]

#cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaStreamCreate = cudart.cudaStreamCreate
cudaStreamCreate.restype = cudaError_t
cudaStreamCreate.argtypes = [ POINTER(cudaStream_t) ]

#cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaStreamDestroy = cudart.cudaStreamDestroy
cudaStreamDestroy.restype = cudaError_t
cudaStreamDestroy.argtypes = [ cudaStream_t ]

#cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaStreamSynchronize = cudart.cudaStreamSynchronize
cudaStreamSynchronize.restype = cudaError_t
cudaStreamSynchronize.argtypes = [ cudaStream_t ]

#cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaStreamQuery = cudart.cudaStreamQuery
cudaStreamQuery.restype = cudaError_t
cudaStreamQuery.argtypes = [ cudaStream_t ]


#cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaEventCreate = cudart.cudaEventCreate
cudaEventCreate.restype = cudaError_t
cudaEventCreate.argtypes = [ POINTER(cudaEvent_t) ]

#cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaEventRecord = cudart.cudaEventRecord
cudaEventRecord.restype = cudaError_t
cudaEventRecord.argtypes = [ cudaEvent_t, cudaStream_t ]

#cudaError_t cudaEventQuery(cudaEvent_t event);
cudaEventQuery = cudart.cudaEventQuery
cudaEventQuery.restype = cudaError_t
cudaEventQuery.argtypes = [ cudaEvent_t ]

#cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaEventSynchronize = cudart.cudaEventSynchronize
cudaEventSynchronize.restype = cudaError_t
cudaEventSynchronize.argtypes = [ cudaEvent_t ]

#cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaEventDestroy = cudart.cudaEventDestroy
cudaEventDestroy.restype = cudaError_t
cudaEventDestroy.argtypes = [ cudaEvent_t ]

#cudaError_t cudaEventElapsedTime(float *ms,
#cudaEvent_t start, cudaEvent_t end);
cudaEventElapsedTime = cudart.cudaEventElapsedTime
cudaEventElapsedTime.restype = cudaError_t
cudaEventElapsedTime.argtypes = [ c_float_p, cudaEvent_t, cudaEvent_t ]

#cudaError_t cudaSetDoubleForDevice(double *d);
cudaSetDoubleForDevice = cudart.cudaSetDoubleForDevice
cudaSetDoubleForDevice.restype = cudaError_t
cudaSetDoubleForDevice.argtypes = [ c_double_p ]

#cudaError_t cudaSetDoubleForHost(double *d);
cudaSetDoubleForHost = cudart.cudaSetDoubleForHost
cudaSetDoubleForHost.restype = cudaError_t
cudaSetDoubleForHost.argtypes = [ c_double_p ]

#cudaError_t cudaThreadExit(void);
cudaThreadExit = cudart.cudaThreadExit
cudaThreadExit.restype = cudaError_t
cudaThreadExit.argtypes = None

#cudaError_t cudaThreadSynchronize(void);
cudaThreadSynchronize = cudart.cudaThreadSynchronize
cudaThreadSynchronize.restype = cudaError_t
cudaThreadSynchronize.argtypes = None

##  extras
def configure(fn,g,b,s=0,t=0):
    def fnx(g,b,s,t):
        cudaConfigureCall(g,b,s,t)
        return fn
    return fnx

# cuda_gl_interop.h

#cudaError_t cudaGLRegisterBufferObject(GLuint bufObj);
cudaGLRegisterBufferObject = cudart.cudaGLRegisterBufferObject
cudaGLRegisterBufferObject.restype = cudaError_t
cudaGLRegisterBufferObject.argtypes = [ GLuint ]

#cudaError_t cudaGLMapBufferObject(void **devPtr, GLuint bufObj);
cudaGLMapBufferObject = cudart.cudaGLMapBufferObject
cudaGLMapBufferObject.restype = cudaError_t
cudaGLMapBufferObject.argtypes = [ POINTER(c_void_p), GLuint ]

#cudaError_t cudaGLUnmapBufferObject(GLuint bufObj);
cudaGLUnmapBufferObject = cudart.cudaGLUnmapBufferObject
cudaGLUnmapBufferObject.restype = cudaError_t
cudaGLUnmapBufferObject.argtypes = [ GLuint ]

#cudaError_t cudaGLUnregisterBufferObject(GLuint bufObj);
cudaGLUnregisterBufferObject = cudart.cudaGLUnregisterBufferObject
cudaGLUnregisterBufferObject.restype = cudaError_t
cudaGLUnregisterBufferObject.argtypes = [ GLuint ]
