from cu_defs import *
from utils import libutils

# CUDa file: cuda.h

cu = libutils.get_cuda(RTLD_GLOBAL)
#CU = "/usr/lib/libcuda.so"
#cu = CDLL(CU,mode=RTLD_GLOBAL)

class cuException(Exception):
    pass

#    /*********************************
#     ** Initialization
#     *********************************/
#    CUresult cuInit(unsigned int Flags);
cuInit = cu.cuInit
cuInit.restype = CUresult
cuInit.argtypes = [ c_uint ]

#
#    /************************************
#     **
#     **    Device management
#     **
#     ***********************************/
#
#    CUresult cuDeviceGet(CUdevice *device, int ordinal);
#    CUresult cuDeviceGetCount(int *count);
#    CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
#    CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
#    CUresult cuDeviceTotalMem(unsigned int *bytes, CUdevice dev);
#    CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev);
#    CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
cuDeviceGet = cu.cuDeviceGet
cuDeviceGet.restype = CUresult
cuDeviceGet.argtypes = [ POINTER(CUdevice), c_int ]

cuDeviceGetCount = cu.cuDeviceGetCount
cuDeviceGetCount.restype = CUresult
cuDeviceGetCount.argtypes = [ c_int_p ]

cuDeviceGetName = cu.cuDeviceGetName
cuDeviceGetName.restype = CUresult
cuDeviceGetName.argtypes = [ c_char_p, c_int, CUdevice ]

cuDeviceComputeCapability = cu.cuDeviceComputeCapability
cuDeviceComputeCapability.restype = CUresult
cuDeviceComputeCapability.argtypes = [ c_int_p, c_int_p, CUdevice ]

cuDeviceTotalMem = cu.cuDeviceTotalMem
cuDeviceTotalMem.restype = CUresult
cuDeviceTotalMem.argtypes = [ c_uint_p, CUdevice ]

cuDeviceGetProperties = cu.cuDeviceGetProperties
cuDeviceGetProperties.restype = CUresult
cuDeviceGetProperties.argtypes = [ POINTER(CUdevprop), CUdevice ]

cuDeviceGetAttribute = cu.cuDeviceGetAttribute
cuDeviceGetAttribute.restype = CUresult
cuDeviceGetAttribute.argtypes = [ c_int_p, CUdevice_attribute, CUdevice ]

#
#    /************************************
#     **
#     **    Context management
#     **
#     ***********************************/
#
#    CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev );
#    CUresult cuCtxDestroy( CUcontext ctx );
#    CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags);
#    CUresult cuCtxDetach(CUcontext ctx);
#    CUresult cuCtxPushCurrent( CUcontext ctx );
#    CUresult cuCtxPopCurrent( CUcontext *pctx );
#    CUresult cuCtxGetDevice(CUdevice *device);
#    CUresult cuCtxSynchronize(void);
cuCtxCreate = cu.cuCtxCreate
cuCtxCreate.restype = CUresult
cuCtxCreate.argtypes = [ POINTER(CUcontext), c_uint, CUdevice ]

cuCtxDestroy = cu.cuCtxDestroy
cuCtxDestroy.restype = CUresult
cuCtxDestroy.argtypes = [ POINTER(CUcontext) ]

cuCtxAttach = cu.cuCtxAttach
cuCtxAttach.restype = CUresult
cuCtxAttach.argtypes = [ POINTER(CUcontext), c_uint ]

cuCtxDetach = cu.cuCtxDetach
cuCtxDetach.restype = CUresult
cuCtxDetach.argtypes = [ CUcontext ]

cuCtxPushCurrent = cu.cuCtxPushCurrent
cuCtxPushCurrent.restype = CUresult
cuCtxPushCurrent.argtypes = [ CUcontext ]

cuCtxPopCurrent = cu.cuCtxPopCurrent
cuCtxPopCurrent.restype = CUresult
cuCtxPopCurrent.argtypes = [ POINTER(CUcontext) ]

# this is not in libcuda.so (and not in libcudart.so either!)
cuCtxGetDevice = cu.cuCtxGetDevice
cuCtxGetDevice.restype = CUresult
cuCtxGetDevice.argtypes = [ POINTER(CUdevice) ]

cuCtxSynchronize = cu.cuCtxSynchronize
cuCtxSynchronize.restype = CUresult
cuCtxSynchronize.argtypes = None

#    /************************************
#     **
#     **    Module management
#     **
#     ***********************************/
#
#    CUresult cuModuleLoad(CUmodule *module, const char *fname);
#    CUresult cuModuleLoadData(CUmodule *module, const void *image);
#    CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin);
#    CUresult cuModuleUnload(CUmodule hmod);
#    CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
#    CUresult cuModuleGetGlobal(CUdeviceptr *dptr, unsigned int *bytes,
#       CUmodule hmod, const char *name);
#    CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name);
cuModuleLoad = cu.cuModuleLoad
cuModuleLoad.restype = CUresult
cuModuleLoad.argtypes = [ POINTER(CUmodule), c_char_p ]

cuModuleLoadData = cu.cuModuleLoadData
cuModuleLoadData.restype = CUresult
cuModuleLoadData.argtypes = [ POINTER(CUmodule), c_void_p ]

cuModuleLoadFatBinary = cu.cuModuleLoadFatBinary
cuModuleLoadFatBinary.restype = CUresult
cuModuleLoadFatBinary.argtypes = [ POINTER(CUmodule), c_void_p ]

cuModuleUnload = cu.cuModuleUnload
cuModuleUnload.restype = CUresult
cuModuleUnload.argtypes = [ CUmodule ]

cuModuleGetFunction = cu.cuModuleGetFunction
cuModuleGetFunction.restype = CUresult
cuModuleGetFunction.argtypes = [ POINTER(CUfunction), CUmodule, c_char_p ]

cuModuleGetGlobal = cu.cuModuleGetGlobal
cuModuleGetGlobal.restype = CUresult
cuModuleGetGlobal.argtypes = [ POINTER(CUdeviceptr), c_uint, CUmodule, c_char_p ]

cuModuleGetTexRef = cu.cuModuleGetTexRef
cuModuleGetTexRef.restype = CUresult
cuModuleGetTexRef.argtypes = [ POINTER(CUtexref), CUmodule, c_char_p ]

#    /************************************
#     **
#     **    Memory management
#     **
#     ***********************************/
#
#    CUresult cuMemGetInfo(unsigned int *free, unsigned int *total);
#
#    CUresult cuMemAlloc( CUdeviceptr *dptr, unsigned int bytesize);
#    CUresult cuMemAllocPitch( CUdeviceptr *dptr,
#                              unsigned int *pPitch,
#                              unsigned int WidthInBytes,
#                              unsigned int Height,
#                              // size of biggest r/w to be performed by kernels
#                              // on this memory : 4, 8 or 16 bytes
#                              unsigned int ElementSizeBytes
#                                     );
#    CUresult cuMemFree(CUdeviceptr dptr);
#    CUresult cuMemGetAddressRange( CUdeviceptr *pbase, unsigned int *psize,
#       CUdeviceptr dptr );
#
#    CUresult cuMemAllocHost(void **pp, unsigned int bytesize);
#    CUresult cuMemFreeHost(void *p);
cuMemGetInfo = cu.cuMemGetInfo
cuMemGetInfo.restype = CUresult
cuMemGetInfo.argtypes = [ c_uint_p, c_uint_p ]

cuMemAlloc = cu.cuMemAlloc
cuMemAlloc.restype = CUresult
cuMemAlloc.argtypes = [ POINTER(CUdeviceptr), c_uint ]

cuMemAllocPitch = cu.cuMemAllocPitch
cuMemAllocPitch.restype = CUresult
cuMemAllocPitch.argtypes = [ POINTER(CUdeviceptr), c_uint_p, c_uint, c_uint, c_uint ]

cuMemFree = cu.cuMemFree
cuMemFree.restype = CUresult
cuMemFree.argtypes = [ CUdeviceptr ]

cuMemGetAddressRange = cu.cuMemGetAddressRange
cuMemGetAddressRange.restype = CUresult
cuMemGetAddressRange.argtypes = [ POINTER(CUdeviceptr), c_uint_p, CUdeviceptr ]

cuMemAllocHost = cu.cuMemAllocHost
cuMemAllocHost.restype = CUresult
cuMemAllocHost.argtypes = [ POINTER(c_void_p), c_uint ]

cuMemFreeHost = cu.cuMemFreeHost
cuMemFreeHost.restype = CUresult
cuMemFreeHost.argtypes = [ c_void_p ]

#    /************************************
#     **
#     **    Synchronous Memcpy
#     **
#     ***********************************/
#
#    // 1D functions
#    CUresult cuMemcpyHtoD (CUdeviceptr dstDevice, const void *srcHost,
#       unsigned int ByteCount );
#    CUresult cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice,
#       unsigned int ByteCount );
#    CUresult cuMemcpyDtoD (CUdeviceptr dstDevice, CUdeviceptr srcDevice,
#       unsigned int ByteCount );
#    CUresult cuMemcpyDtoA ( CUarray dstArray, unsigned int dstIndex,
#       CUdeviceptr srcDevice, unsigned int ByteCount );
#    CUresult cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray hSrc,
#       unsigned int SrcIndex, unsigned int ByteCount );
#    CUresult cuMemcpyHtoA( CUarray dstArray, unsigned int dstIndex,
#       const void *pSrc, unsigned int ByteCount );
#    CUresult cuMemcpyAtoH( void *dstHost, CUarray srcArray,
#       unsigned int srcIndex, unsigned int ByteCount );
#    CUresult cuMemcpyAtoA( CUarray dstArray, unsigned int dstIndex,
#       CUarray srcArray, unsigned int srcIndex, unsigned int ByteCount );
cuMemcpyHtoD = cu.cuMemcpyHtoD
cuMemcpyHtoD.restype = CUresult
cuMemcpyHtoD.argtypes = [ CUdeviceptr, c_void_p, c_uint ]

cuMemcpyDtoH = cu.cuMemcpyDtoH
cuMemcpyDtoH.restype = CUresult
cuMemcpyDtoH.argtypes = [ c_void_p, CUdeviceptr, c_uint ]

cuMemcpyDtoD = cu.cuMemcpyDtoD
cuMemcpyDtoD.restype = CUresult
cuMemcpyDtoD.argtypes = [ CUdeviceptr, CUdeviceptr, c_uint ]

cuMemcpyDtoA = cu.cuMemcpyDtoA
cuMemcpyDtoA.restype = CUresult
cuMemcpyDtoA.argtypes = [ CUarray, c_uint, CUdeviceptr, c_uint ]

cuMemcpyAtoD = cu.cuMemcpyAtoD
cuMemcpyAtoD.restype = CUresult
cuMemcpyAtoD.argtypes = [ CUdeviceptr, CUarray, c_uint, c_uint ]

cuMemcpyHtoA = cu.cuMemcpyHtoA
cuMemcpyHtoA.restype = CUresult
cuMemcpyHtoA.argtypes = [ CUarray, c_uint, c_void_p, c_uint ]

cuMemcpyAtoH = cu.cuMemcpyAtoH
cuMemcpyAtoH.restype = CUresult
cuMemcpyAtoH.argtypes = [ c_void_p, CUarray, c_uint, c_uint ]

cuMemcpyAtoA = cu.cuMemcpyAtoA
cuMemcpyAtoA.restype = CUresult
cuMemcpyAtoA.argtypes = [ CUarray, c_uint, CUarray, c_uint, c_uint ]

#    // 2D memcpy
#    CUresult cuMemcpy2D( const CUDA_MEMCPY2D *pCopy );
#    CUresult cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy );
cuMemcpy2D = cu.cuMemcpy2D
cuMemcpy2D.restype = CUresult
cuMemcpy2D.argtypes = [ POINTER(CUDA_MEMCPY2D) ]

cuMemcpy2DUnaligned = cu.cuMemcpy2DUnaligned
cuMemcpy2DUnaligned.restype = CUresult
cuMemcpy2DUnaligned.argtypes = [ POINTER(CUDA_MEMCPY2D) ]

#    // 3D memcpy
#        CUresult cuMemcpy3D( const CUDA_MEMCPY3D *pCopy );
cuMemcpy3D = cu.cuMemcpy3D
cuMemcpy3D.restype = CUresult
cuMemcpy3D.argtypes = [ POINTER(CUDA_MEMCPY3D) ]

#    /************************************
#     **
#     **    Asynchronous Memcpy
#     ...
#     **
#     ***********************************/
#
#    // 1D functions
#        // system <-> device memory
#        CUresult cuMemcpyHtoDAsync (CUdeviceptr dstDevice, 
#            const void *srcHost, unsigned int ByteCount, CUstream hStream );
#        CUresult cuMemcpyDtoHAsync (void *dstHost, 
#            CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
#
#        // system <-> array memory
#        CUresult cuMemcpyHtoAAsync( CUarray dstArray, unsigned int dstIndex, 
#            const void *pSrc, unsigned int ByteCount, CUstream hStream );
#        CUresult cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, unsigned int srcIndex, 
#            unsigned int ByteCount, CUstream hStream );
#
#        // 2D memcpy
#        CUresult cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, CUstream hStream );
cuMemcpyHtoDAsync = cu.cuMemcpyHtoDAsync
cuMemcpyHtoDAsync.restype = CUresult
cuMemcpyHtoDAsync.argtypes = [ CUdeviceptr, c_void_p, c_uint, CUstream ]

cuMemcpyDtoHAsync = cu.cuMemcpyDtoHAsync
cuMemcpyDtoHAsync.restype = CUresult
cuMemcpyDtoHAsync.argtypes = [ c_void_p, CUdeviceptr, c_uint, CUstream ]

cuMemcpyHtoAAsync = cu.cuMemcpyHtoAAsync
cuMemcpyHtoDAsync.restype = CUresult
cuMemcpyHtoDAsync.argtypes = [ CUarray, c_uint, c_void_p, c_uint, CUstream ]

cuMemcpyAtoHAsync = cu.cuMemcpyAtoHAsync
cuMemcpyHtoDAsync.restype = CUresult
cuMemcpyHtoDAsync.argtypes = [ c_void_p, CUarray, c_uint, c_uint, CUstream ]

cuMemcpy2DAsync = cu.cuMemcpy2DAsync
cuMemcpy2DAsync.restype = CUresult
cuMemcpy2DAsync.argtypes = [ POINTER(CUDA_MEMCPY2D), CUstream ]
#        // 3D memcpy
#        CUresult cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, CUstream hStream );
cuMemcpy3DAsync = cu.cuMemcpy3DAsync
cuMemcpy3DAsync.restype = CUresult
cuMemcpy3DAsync.argtypes = [ POINTER(CUDA_MEMCPY2D), CUstream ]

#    /************************************
#     **
#     **    Memset
#     **
#     ***********************************/
#    CUresult cuMemsetD8( CUdeviceptr dstDevice, unsigned char uc, unsigned int N );
#    CUresult cuMemsetD16( CUdeviceptr dstDevice, unsigned short us, unsigned int N );
#    CUresult cuMemsetD32( CUdeviceptr dstDevice, unsigned int ui, unsigned int N );
#
#    CUresult cuMemsetD2D8( CUdeviceptr dstDevice, unsigned int dstPitch,
#       unsigned char uc, unsigned int Width, unsigned int Height );
#    CUresult cuMemsetD2D16( CUdeviceptr dstDevice, unsigned int dstPitch,
#       unsigned short us, unsigned int Width, unsigned int Height );
#    CUresult cuMemsetD2D32( CUdeviceptr dstDevice, unsigned int dstPitch,
#       unsigned int ui, unsigned int Width, unsigned int Height );
cuMemsetD8 = cu.cuMemsetD8
cuMemsetD8.restype = CUresult
cuMemsetD8.argtypes = [ CUdeviceptr, c_ubyte, c_uint ]

cuMemsetD16 = cu.cuMemsetD16
cuMemsetD16.restype = CUresult
cuMemsetD16.argtypes = [ CUdeviceptr, c_ushort, c_uint ]

cuMemsetD32 = cu.cuMemsetD32
cuMemsetD32.restype = CUresult
cuMemsetD32.argtypes = [ CUdeviceptr, c_uint, c_uint ]

cuMemsetD2D8 = cu.cuMemsetD2D8
cuMemsetD2D8.restype = CUresult
cuMemsetD2D8.argtypes = [ CUdeviceptr, c_uint, c_ubyte, c_uint, c_uint ]

cuMemsetD2D16 = cu.cuMemsetD2D16
cuMemsetD2D16.restype = CUresult
cuMemsetD2D16.argtypes = [ CUdeviceptr, c_uint, c_ushort, c_uint, c_uint ]

cuMemsetD2D32 = cu.cuMemsetD2D32
cuMemsetD2D32.restype = CUresult
cuMemsetD2D32.argtypes = [ CUdeviceptr, c_uint, c_uint, c_uint, c_uint ]

#    /************************************
#     **
#     **    Function management
#     **
#     ***********************************/
#
#    CUresult cuFuncSetBlockShape (CUfunction hfunc, int x, int y, int z);
#    CUresult cuFuncSetSharedSize (CUfunction hfunc, unsigned int bytes);
cuFuncSetBlockShape = cu.cuFuncSetBlockShape
cuFuncSetBlockShape.restype = CUresult
cuFuncSetBlockShape.argtypes = [ CUfunction, c_int, c_int, c_int ]

cuFuncSetSharedSize = cu.cuFuncSetSharedSize
cuFuncSetSharedSize.restype = CUresult
cuFuncSetSharedSize.argtypes = [ CUfunction, c_uint ]

#    /************************************
#     **
#     **    Array management
#     **
#     ***********************************/
#
#    CUresult cuArrayCreate( CUarray *pHandle,
#       const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
#    CUresult cuArrayGetDescriptor( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor,
#       CUarray hArray );
#    CUresult cuArrayDestroy( CUarray hArray );

#    CUresult cuArray3DCreate( CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray );
#    CUresult cuArray3DGetDescriptor( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
cuArrayCreate = cu.cuArrayCreate
cuArrayCreate.restype = CUresult
cuArrayCreate.argtypes = [ POINTER(CUarray), POINTER(CUDA_ARRAY_DESCRIPTOR) ]

cuArrayGetDescriptor = cu.cuArrayGetDescriptor
cuArrayGetDescriptor.restype = CUresult
cuArrayGetDescriptor.argtypes = [ POINTER(CUDA_ARRAY_DESCRIPTOR), CUarray ]

cuArrayDestroy = cu.cuArrayDestroy
cuArrayDestroy.restype = CUresult
cuArrayDestroy.argtypes = [ CUarray ]

cuArray3DCreate = cu.cuArray3DCreate
cuArray3DCreate.restype = CUresult
cuArray3DCreate.argtypes = [ POINTER(CUarray), POINTER(CUDA_ARRAY3D_DESCRIPTOR) ]

cuArray3DGetDescriptor = cu.cuArray3DGetDescriptor
cuArray3DGetDescriptor.restype = CUresult
cuArray3DGetDescriptor.argtypes = [ POINTER(CUDA_ARRAY3D_DESCRIPTOR), CUarray ]

#    /************************************
#     **
#     **    Texture reference management
#     **
#     ***********************************/
#    CUresult cuTexRefCreate( CUtexref *pTexRef );
#    CUresult cuTexRefDestroy( CUtexref hTexRef );
#
#    CUresult cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, unsigned int Flags );
#    CUresult cuTexRefSetAddress( unsigned int *ByteOffset, CUtexref hTexRef,
#       CUdeviceptr dptr, int bytes );
#    CUresult cuTexRefSetFormat( CUtexref hTexRef, CUarray_format fmt,
#       int NumPackedComponents );
#
#    CUresult cuTexRefSetAddressMode( CUtexref hTexRef, int dim, CUaddress_mode am );
#    CUresult cuTexRefSetFilterMode( CUtexref hTexRef, CUfilter_mode fm );
#    CUresult cuTexRefSetFlags( CUtexref hTexRef, unsigned int Flags );
cuTexRefCreate = cu.cuTexRefCreate
cuTexRefCreate.restype = CUresult
cuTexRefCreate.argtypes = [ POINTER(CUtexref) ]

cuTexRefDestroy = cu.cuTexRefDestroy
cuTexRefDestroy.restype = CUresult
cuTexRefDestroy.argtypes = [ CUtexref ]

cuTexRefSetArray = cu.cuTexRefSetArray
cuTexRefSetArray.restype = CUresult
cuTexRefSetArray.argtypes = [ CUtexref, CUarray, c_uint ]

cuTexRefSetAddress = cu.cuTexRefSetAddress
cuTexRefSetAddress.restype = CUresult
cuTexRefSetAddress.argtypes = [ c_uint_p, CUtexref, CUdeviceptr, c_uint ]

cuTexRefSetFormat = cu.cuTexRefSetFormat
cuTexRefSetFormat.restype = CUresult
cuTexRefSetFormat.argtypes = [ CUtexref, CUarray_format, c_int ]

cuTexRefSetAddressMode = cu.cuTexRefSetAddressMode
cuTexRefSetAddressMode.restype = CUresult
cuTexRefSetAddressMode.argtypes = [ CUtexref, c_int, CUaddress_mode ]

cuTexRefSetFilterMode = cu.cuTexRefSetFilterMode
cuTexRefSetFilterMode.restype = CUresult
cuTexRefSetFilterMode.argtypes = [ CUtexref, CUfilter_mode ]

cuTexRefSetFlags = cu.cuTexRefSetFlags
cuTexRefSetFlags.restype = CUresult
cuTexRefSetFlags.argtypes = [ CUtexref, c_uint ]

#    CUresult cuTexRefGetAddress( CUdeviceptr *pdptr, CUtexref hTexRef );
#    CUresult cuTexRefGetArray( CUarray *phArray, CUtexref hTexRef );
#    CUresult cuTexRefGetAddressMode( CUaddress_mode *pam, CUtexref hTexRef, int dim );
#    CUresult cuTexRefGetFilterMode( CUfilter_mode *pfm, CUtexref hTexRef );
#    CUresult cuTexRefGetFormat( CUarray_format *pFormat, int *pNumChannels,
#       CUtexref hTexRef );
#    CUresult cuTexRefGetFlags( unsigned int *pFlags, CUtexref hTexRef );
cuTexRefGetAddress = cu.cuTexRefGetAddress
cuTexRefGetAddress.restype = CUresult
cuTexRefGetAddress.argtypes = [ POINTER(CUdeviceptr), CUtexref ]

cuTexRefGetArray = cu.cuTexRefGetArray
cuTexRefGetArray.restype = CUresult
cuTexRefGetArray.argtypes = [ POINTER(CUarray), CUtexref ]

cuTexRefGetAddressMode = cu.cuTexRefGetAddressMode
cuTexRefGetAddressMode.restype = CUresult
cuTexRefGetAddressMode.argtypes = [ POINTER(CUaddress_mode), CUtexref, c_int ]

cuTexRefGetFilterMode = cu.cuTexRefGetFilterMode
cuTexRefGetFilterMode.restype = CUresult
cuTexRefGetFilterMode.argtypes = [ POINTER(CUfilter_mode), CUtexref ]

# this is not in libcuda.so (and not in libcudart.so either!)
cuTexRefGetFormat = cu.cuTexRefGetFormat
cuTexRefGetFormat.restype = CUresult
cuTexRefGetFormat.argtypes = [ POINTER(CUarray_format), c_int_p, CUtexref ]

cuTexRefGetFlags = cu.cuTexRefGetFlags
cuTexRefGetFlags.restype = CUresult
cuTexRefGetFlags.argtypes = [ c_uint_p, CUtexref ]

#    /************************************
#     **
#     **    Parameter management
#     **
#     ***********************************/
#
#    CUresult cuParamSetSize (CUfunction hfunc, unsigned int numbytes);
#    CUresult cuParamSeti    (CUfunction hfunc, int offset, unsigned int value);
#    CUresult cuParamSetf    (CUfunction hfunc, int offset, float value);
#    CUresult cuParamSetv    (CUfunction hfunc, int offset, void * ptr,
#       unsigned int numbytes);
#    CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);
cuParamSetSize = cu.cuParamSetSize
cuParamSetSize.restype = CUresult
cuParamSetSize.argtypes = [ CUfunction, c_uint ]

cuParamSeti = cu.cuParamSeti
cuParamSeti.restype = CUresult
cuParamSeti.argtypes = [ CUfunction, c_int, c_uint ]

cuParamSetf = cu.cuParamSetf
cuParamSetf.restype = CUresult
cuParamSetf.argtypes = [ CUfunction, c_int, c_float ]

cuParamSetv = cu.cuParamSetv
cuParamSetv.restype = CUresult
cuParamSetv.argtypes = [ CUfunction, c_int, c_void_p, c_uint ]

cuParamSetTexRef = cu.cuParamSetTexRef
cuParamSetTexRef.restype = CUresult
cuParamSetTexRef.argtypes = [ CUfunction, c_int, CUtexref ]

#    /************************************
#     **
#     **    Launch functions
#     **
#     ***********************************/
#
#    CUresult cuLaunch ( CUfunction f );
#    CUresult cuLaunchGrid (CUfunction f, int grid_width, int grid_height);
#    CUresult cuLaunchGridAsync( CUfunction f, int grid_width, int grid_height,
#        CUstream hStream );
cuLaunch = cu.cuLaunch
cuLaunch.restype = CUresult
cuLaunch.argtypes = [ CUfunction ]

cuLaunchGrid = cu.cuLaunchGrid
cuLaunchGrid.restype = CUresult
cuLaunchGrid.argtypes = [ CUfunction, c_int, c_int ]

cuLaunchGridAsync = cu.cuLaunchGridAsync
cuLaunchGridAsync.restype = CUresult
cuLaunchGridAsync.argtypes = [ CUfunction, c_int, c_int, CUstream ]

#    /************************************
#     **
#     **    Events
#     **
#     ***********************************/
#    CUresult cuEventCreate( CUevent *phEvent, unsigned int Flags );
#    CUresult cuEventRecord( CUevent hEvent, CUstream hStream );
#    CUresult cuEventQuery( CUevent hEvent );
#    CUresult cuEventSynchronize( CUevent hEvent );
#    CUresult cuEventDestroy( CUevent hEvent );
#    CUresult cuEventElapsedTime( float *pMilliseconds,
#        CUevent hStart, CUevent hEnd );
cuEventCreate = cu.cuEventCreate
cuEventCreate.restype = CUresult
cuEventCreate.argtypes = [ POINTER(CUevent), c_uint ]

cuEventRecord = cu.cuEventRecord
cuEventRecord.restype = CUresult
cuEventRecord.argtypes = [ CUevent, CUstream ]

cuEventQuery = cu.cuEventQuery
cuEventQuery.restype = CUresult
cuEventQuery.argtypes = [ CUevent ]

cuEventSynchronize = cu.cuEventSynchronize
cuEventSynchronize.restype = CUresult
cuEventSynchronize.argtypes = [ CUevent ]

cuEventDestroy = cu.cuEventDestroy
cuEventDestroy.restype = CUresult
cuEventDestroy.argtypes = [ CUevent ]

cuEventElapsedTime = cu.cuEventElapsedTime
cuEventElapsedTime.restype = CUresult
cuEventElapsedTime.argtypes = [ c_float_p, CUevent, CUevent ]

#    /************************************
#     **
#     **    Streams
#     **
#     ***********************************/
#    CUresult cuStreamCreate( CUstream *phStream, unsigned int Flags );
#    CUresult cuStreamQuery( CUstream hStream );
#    CUresult cuStreamSynchronize( CUstream hStream );
#    CUresult cuStreamDestroy( CUstream hStream );
cuStreamCreate = cu.cuStreamCreate
cuStreamCreate.restype = CUresult
cuStreamCreate.argtypes = [ POINTER(CUstream), c_uint ]

cuStreamQuery = cu.cuStreamQuery
cuStreamQuery.restype = CUresult
cuStreamQuery.argtypes = [ CUstream ]

cuStreamSynchronize = cu.cuStreamSynchronize
cuStreamSynchronize.restype = CUresult
cuStreamSynchronize.argtypes = [ CUstream ]

cuStreamDestroy = cu.cuStreamDestroy
cuStreamDestroy.restype = CUresult
cuStreamDestroy.argtypes = [ CUstream ]

# cudaGL.h

#CUresult cuGLInit(void);
#CUresult cuGLRegisterBufferObject( GLuint bufferobj );
#CUresult cuGLMapBufferObject( CUdeviceptr *dptr,
#unsigned int *size,  GLuint bufferobj ); 
#CUresult cuGLUnmapBufferObject( GLuint bufferobj );
#CUresult cuGLUnregisterBufferObject( GLuint bufferobj );
cuGLInit = cu.cuGLInit
cuGLInit.restype = CUresult
cuGLInit.argtypes = [ ]

cuGLRegisterBufferObject = cu.cuGLRegisterBufferObject
cuGLRegisterBufferObject.restype = CUresult
cuGLRegisterBufferObject.argtypes = [ GLuint ]

cuGLMapBufferObject = cu.cuGLMapBufferObject
cuGLMapBufferObject.restype = CUresult
cuGLMapBufferObject.argtypes = [ POINTER(CUdeviceptr), c_uint, GLuint ]

cuGLUnmapBufferObject = cu.cuGLUnmapBufferObject
cuGLUnmapBufferObject.restype = CUresult
cuGLUnmapBufferObject.argtypes = [ GLuint ]

cuGLUnregisterBufferObject = cu.cuGLUnregisterBufferObject
cuGLUnregisterBufferObject.restype = CUresult
cuGLUnregisterBufferObject.argtypes = [ GLuint ]
