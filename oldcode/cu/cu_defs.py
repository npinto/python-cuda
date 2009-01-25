from ctypes import *
# coding:utf-8: © Arno Pähler, 2007-08

# CUDa file: cuda.h

c_int_p   = POINTER(c_int)
c_uint_p  = POINTER(c_uint)
c_float_p = POINTER(c_float)

#/* CUDA API version number 2.0 */
##define CUDA_VERSION 2000
CUDA_VERSION = 2000

#    typedef unsigned int CUdeviceptr;
#
#    typedef int CUdevice;
#    typedef struct CUctx_st *CUcontext;
#    typedef struct CUmod_st *CUmodule;
#    typedef struct CUfunc_st *CUfunction;
#    typedef struct CUarray_st *CUarray;
#    typedef struct CUtrigger_st *CUtrigger;
#    typedef struct CUevent_st *CUevent;
#    typedef struct CUtexref_st *CUtexref;
#    typedef struct CUevent_st *CUevent;
#    typedef struct CUstream_st *CUstream;
CUdeviceptr = c_uint

CUdevice = c_int
class _CUcontext(Structure):
    _fields_ = []
CUcontext = POINTER(_CUcontext)
class _CUmodule(Structure):
    _fields_ = []
CUmodule = POINTER(_CUmodule)
class _CUfunction(Structure):
    _fields_ = []
CUfunction = POINTER(_CUfunction)
class _CUarray(Structure):
    _fields_ = []
CUarray = POINTER(_CUarray)
class _CUtrigger(Structure):
    _fields_ = []
CUtrigger = POINTER(_CUtrigger)
class _CUevent(Structure):
    _fields_ = []
CUevent = POINTER(_CUevent)
class _CUtexref(Structure):
    _fields_ = []
CUtexref = POINTER(_CUtexref)
class _CUevent(Structure):
    _fields_ = []
CUevent = POINTER(_CUevent)
class _CUstream(Structure):
    _fields_ = []
CUstream = POINTER(_CUstream)

#//
#// context creation flags
#//
#typedef enum CUctx_flags_enum {
#    CU_CTX_SCHED_AUTO  = 0,
#    CU_CTX_SCHED_SPIN  = 1,
#    CU_CTX_SCHED_YIELD = 2,
#    CU_CTX_SCHED_MASK  = 0x3,
#    CU_CTX_FLAGS_MASK  = CU_CTX_SCHED_MASK
#} CUctx_flags;
#//
#// context creation flags
#//
CUctx_flags = c_int
CU_CTX_SCHED_AUTO  = 0,
CU_CTX_SCHED_SPIN  = 1,
CU_CTX_SCHED_YIELD = 2,
CU_CTX_SCHED_MASK  = 0x3,
CU_CTX_FLAGS_MASK  = CU_CTX_SCHED_MASK

#//
#// array formats
#//
#typedef enum CUarray_format_enum {
#    CU_AD_FORMAT_UNSIGNED_INT8  = 0x01,
#    CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
#    CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
#    CU_AD_FORMAT_SIGNED_INT8    = 0x08,
#    CU_AD_FORMAT_SIGNED_INT16   = 0x09,
#    CU_AD_FORMAT_SIGNED_INT32   = 0x0a,
#    CU_AD_FORMAT_HALF           = 0x10,
#    CU_AD_FORMAT_FLOAT          = 0x20
#} CUarray_format;
CUarray_format = c_int
CU_AD_FORMAT_UNSIGNED_INT8  = 0x01
CU_AD_FORMAT_UNSIGNED_INT16 = 0x02
CU_AD_FORMAT_UNSIGNED_INT32 = 0x03
CU_AD_FORMAT_SIGNED_INT8    = 0x08
CU_AD_FORMAT_SIGNED_INT16   = 0x09
CU_AD_FORMAT_SIGNED_INT32   = 0x0a
CU_AD_FORMAT_HALF           = 0x10
CU_AD_FORMAT_FLOAT          = 0x20

#//
#// Texture reference addressing modes
#//
#typedef enum CUaddress_mode_enum {
#    CU_TR_ADDRESS_MODE_WRAP = 0,
#    CU_TR_ADDRESS_MODE_CLAMP = 1,
#    CU_TR_ADDRESS_MODE_MIRROR = 2,
#} CUaddress_mode;
CUaddress_mode= c_int
CU_TR_ADDRESS_MODE_WRAP   = 0
CU_TR_ADDRESS_MODE_CLAMP  = 1
CU_TR_ADDRESS_MODE_MIRROR = 2

#//
#// Texture reference filtering modes
#//
#typedef enum CUfilter_mode_enum {
#    CU_TR_FILTER_MODE_POINT = 0,
#    CU_TR_FILTER_MODE_LINEAR = 1
#} CUfilter_mode;
CUfilter_mode = c_int
CU_TR_FILTER_MODE_POINT  = 0
CU_TR_FILTER_MODE_LINEAR = 1

#//
#// Device properties
#//
#typedef enum CUdevice_attribute_enum {
#    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
#    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
#    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
#    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
#    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
#    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
#    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
#    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
#    CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,      // Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
#    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
#    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
#    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
#    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
#    CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,         // Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
#    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
#    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
#
#    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
#    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
#} CUdevice_attribute;
CUdevice_attribute = c_int
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8
CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9
CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10
CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12
CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14

CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16

#//
#// Legacy device properties
#//
#typedef struct CUdevprop_st {
#    int maxThreadsPerBlock;
#    int maxThreadsDim[3];
#    int maxGridSize[3];
#    int sharedMemPerBlock;
#    int totalConstantMemory;
#    int SIMDWidth;
#    int memPitch;
#    int regsPerBlock;
#    int clockRate;
#    int textureAlign;
#} CUdevprop;
class CUdevprop(Structure):
    _fields_ = [("maxThreadsPerBlock",  c_int),
                ("maxThreadsDim",       c_int*3),
                ("maxGridSize",         c_int*3),
                ("sharedMemPerBlock",   c_int),
                ("totalConstantMemory", c_int),
                ("SIMDWidth",           c_int),
                ("memPitch",            c_int),
                ("regsPerBlock",        c_int),
                ("clockRate",           c_int),
                ("textureAlign",        c_int)]

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
            res.append('%-19s = %s' % (field[0], data))
        return "\n".join(res)

#//
#// Memory types
#//
#typedef enum CUmemorytype_enum {
#    CU_MEMORYTYPE_HOST = 0x01,
#    CU_MEMORYTYPE_DEVICE = 0x02,
#    CU_MEMORYTYPE_ARRAY = 0x03
#} CUmemorytype;
CUmemorytype = c_int
CU_MEMORYTYPE_HOST   = 0x01
CU_MEMORYTYPE_DEVICE = 0x02
CU_MEMORYTYPE_ARRAY  = 0x03

#/************************************
# **
# **    Error codes
# **
# ***********************************/
#
#typedef enum cudaError_enum {
#
#    CUDA_SUCCESS                    = 0,
#    CUDA_ERROR_INVALID_VALUE        = 1,
#    CUDA_ERROR_OUT_OF_MEMORY        = 2,
#    CUDA_ERROR_NOT_INITIALIZED      = 3,
#
#    CUDA_ERROR_NO_DEVICE            = 100,
#    CUDA_ERROR_INVALID_DEVICE       = 101,
#
#    CUDA_ERROR_INVALID_IMAGE        = 200,
#    CUDA_ERROR_INVALID_CONTEXT      = 201,
#    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
#    CUDA_ERROR_MAP_FAILED           = 205,
#    CUDA_ERROR_UNMAP_FAILED         = 206,
#    CUDA_ERROR_ARRAY_IS_MAPPED      = 207,
#    CUDA_ERROR_ALREADY_MAPPED       = 208,
#    CUDA_ERROR_NO_BINARY_FOR_GPU    = 209,
#    CUDA_ERROR_ALREADY_ACQUIRED     = 210,
#    CUDA_ERROR_NOT_MAPPED           = 211,
#
#    CUDA_ERROR_INVALID_SOURCE       = 300,
#    CUDA_ERROR_FILE_NOT_FOUND       = 301,
#
#    CUDA_ERROR_INVALID_HANDLE       = 400,
#
#    CUDA_ERROR_NOT_FOUND            = 500,
#
#    CUDA_ERROR_LAUNCH_FAILED        = 700,
#    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
#    CUDA_ERROR_LAUNCH_TIMEOUT       = 702,
#    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
#
#    CUDA_ERROR_UNKNOWN              = 999
#} CUresult;
CUresult = c_int

CUDA_SUCCESS                    = 0
CUDA_ERROR_INVALID_VALUE        = 1
CUDA_ERROR_OUT_OF_MEMORY        = 2
CUDA_ERROR_NOT_INITIALIZED      = 3

CUDA_ERROR_NO_DEVICE            = 100
CUDA_ERROR_INVALID_DEVICE       = 101

CUDA_ERROR_INVALID_IMAGE        = 200
CUDA_ERROR_INVALID_CONTEXT      = 201
CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202
CUDA_ERROR_MAP_FAILED           = 205
CUDA_ERROR_UNMAP_FAILED         = 206
CUDA_ERROR_ARRAY_IS_MAPPED      = 207
CUDA_ERROR_ALREADY_MAPPED       = 208
CUDA_ERROR_NO_BINARY_FOR_GPU    = 209
CUDA_ERROR_ALREADY_ACQUIRED     = 210
CUDA_ERROR_NOT_MAPPED           = 211

CUDA_ERROR_INVALID_SOURCE       = 300
CUDA_ERROR_FILE_NOT_FOUND       = 301

CUDA_ERROR_INVALID_HANDLE       = 400

CUDA_ERROR_NOT_FOUND            = 500

CUDA_ERROR_LAUNCH_FAILED        = 700
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
CUDA_ERROR_LAUNCH_TIMEOUT       = 702
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703

CUDA_ERROR_UNKNOWN              = 999

#    // 2D memcpy
#
#        typedef struct CUDA_MEMCPY2D_st {
#
#            unsigned int srcXInBytes, srcY;
#            CUmemorytype srcMemoryType;
#                const void *srcHost;
#                CUdeviceptr srcDevice;
#                CUarray srcArray;
#                unsigned int srcPitch; // ignored when src is array
#
#            unsigned int dstXInBytes, dstY;
#            CUmemorytype dstMemoryType;
#                void *dstHost;
#                CUdeviceptr dstDevice;
#                CUarray dstArray;
#                unsigned int dstPitch; // ignored when dst is array
#
#            unsigned int WidthInBytes;
#            unsigned int Height;
#        } CUDA_MEMCPY2D;
class CUDA_MEMCPY2D(Structure):
    _fields_ = [("srcXInBytes",     c_uint),
                ("srcY",            c_uint),
                ("srcMemoryType",   CUmemorytype),
                ("srcHost",         c_void_p),
                ("srcDevice",       CUdeviceptr),
                ("srcArray",        CUarray),
                ("srcPitch",        c_uint),
                ("dstY",            c_uint),
                ("dstMemoryType",   CUmemorytype),
                ("dstHost",         c_void_p),
                ("dstDevice",       CUdeviceptr),
                ("dstArray",        CUarray),
                ("dstPitch",        c_uint),
                ("WidthInBytes",    c_uint),
                ("Height",          c_uint)]

#    // 3D memcpy
#
#        typedef struct CUDA_MEMCPY3D_st {
#
#            unsigned int srcXInBytes, srcY, srcZ;
#            unsigned int srcLOD;
#            CUmemorytype srcMemoryType;
#                const void *srcHost;
#                CUdeviceptr srcDevice;
#                CUarray srcArray;
#                void *reserved0;        // must be NULL
#                unsigned int srcPitch;  // ignored when src is array
#                unsigned int srcHeight; // ignored when src is array; may be 0 if Depth==1
#
#            unsigned int dstXInBytes, dstY, dstZ;
#            unsigned int dstLOD;
#            CUmemorytype dstMemoryType;
#                void *dstHost;
#                CUdeviceptr dstDevice;
#                CUarray dstArray;
#                void *reserved1;        // must be NULL
#                unsigned int dstPitch;  // ignored when dst is array
#                unsigned int dstHeight; // ignored when dst is array; may be 0 if Depth==1
#
#            unsigned int WidthInBytes;
#            unsigned int Height;
#            unsigned int Depth;
#        } CUDA_MEMCPY3D;
class CUDA_MEMCPY3D(Structure):
    _fields_ = [("srcXInBytes",     c_uint),
                ("srcY",            c_uint),
                ("srcZ",            c_uint),
                ("srcMemoryType",   CUmemorytype),
                ("srcHost",         c_void_p),
                ("srcDevice",       CUdeviceptr),
                ("srcArray",        CUarray),
                ("reserved0",       c_void_p),
                ("srcPitch",        c_uint),
                ("srcHeight",       c_uint),
                ("dstY",            c_uint),
                ("dstZ",            c_uint),
                ("dstLOD",          c_uint),
                ("dstMemoryType",   CUmemorytype),
                ("dstHost",         c_void_p),
                ("dstDevice",       CUdeviceptr),
                ("dstArray",        CUarray),
                ("reserved1",       c_void_p),
                ("dstPitch",        c_uint),
                ("dstHeight",       c_uint),
                ("WidthInBytes",    c_uint),
                ("Height",          c_uint),
                ("Depth",           c_uint)]

#    typedef struct
#    {
#        //
#        // dimensions
#        //
#            unsigned int Width;
#            unsigned int Height;
#
#        //
#        // format
#        //
#            CUarray_format Format;
#
#            // channels per array element
#            unsigned int NumChannels;
#    } CUDA_ARRAY_DESCRIPTOR;
#    typedef struct

#    {
#        //
#        // dimensions
#        //
#            unsigned int Width;
#            unsigned int Height;
#            unsigned int Depth;
#        //
#        // format
#        //
#            CUarray_format Format;
#
#            // channels per array element
#            unsigned int NumChannels;
#        //
#        // flags
#        //
#            unsigned int Flags;
#
#    } CUDA_ARRAY3D_DESCRIPTOR;

class CUDA_ARRAY_DESCRIPTOR(Structure):
    _fields_ = [("Width",       c_uint),
                ("Height",      c_uint),
                ("Format",      CUarray_format),
                ("NumChannels", c_uint)]

class CUDA_ARRAY3D_DESCRIPTOR(Structure):
    _fields_ = [("Width",       c_uint),
                ("Height",      c_uint),
                ("Depth",       c_uint),
                ("Format",      CUarray_format),
                ("NumChannels", c_uint),
                ("Flags",       c_uint)]

#        // override the texref format with a format inferred from the array
#        #define CU_TRSA_OVERRIDE_FORMAT 0x01
CU_TRSA_OVERRIDE_FORMAT = 0x01
#        // read the texture as integers rather than promoting the values
#        // to floats in the range [0,1]
#        #define CU_TRSF_READ_AS_INTEGER         0x01
#
#        // use normalized texture coordinates in the range [0,1) instead of [0,dim)
#        #define CU_TRSF_NORMALIZED_COORDINATES  0x02
CU_TRSF_READ_AS_INTEGER        = 0x01
CU_TRSF_NORMALIZED_COORDINATES = 0x02

#        // for texture references loaded into the module,
#        // use default texunit from texture reference
#        #define CU_PARAM_TR_DEFAULT -1
CU_PARAM_TR_DEFAULT = -1

# for OpenGL

GLuint = c_uint
