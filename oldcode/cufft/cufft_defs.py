# coding:utf-8: © Arno Pähler, 2007-08

from ctypes import *

#// CUFFT API function return values 
#typedef enum cufftResult_t {
#    CUFFT_SUCCESS        = 0x0,
#    CUFFT_INVALID_PLAN   = 0x1,
#    CUFFT_ALLOC_FAILED   = 0x2,
#    CUFFT_INVALID_TYPE   = 0x3,
#    CUFFT_INVALID_VALUE  = 0x4,
#    CUFFT_INTERNAL_ERROR = 0x5,
#    CUFFT_EXEC_FAILED    = 0x6,
#    CUFFT_SETUP_FAILED   = 0x7,
#    CUFFT_INVALID_SIZE   = 0x8
#} cufftResult;

cufftResult = c_int

CUFFT_SUCCESS        = 0x0
CUFFT_INVALID_PLAN   = 0x1
CUFFT_ALLOC_FAILED   = 0x2
CUFFT_INVALID_TYPE   = 0x3
CUFFT_INVALID_VALUE  = 0x4
CUFFT_INTERNAL_ERROR = 0x5
CUFFT_EXEC_FAILED    = 0x6
CUFFT_SETUP_FAILED   = 0x7
CUFFT_INVALID_SIZE   = 0x8

#// CUFFT defines and supports the following data types
#
#// cufftHandle is a handle type used to store and access CUFFT plans.
#typedef unsigned int cufftHandle;
#
#// cufftReal is a single-precision, floating-point real data type.
#typedef float cufftReal;
#
#// cufftComplex is a single-precision, floating-point complex data type that 
#// consists of interleaved real and imaginary components.
#typedef float cufftComplex[2];

cufftHandle  = c_uint
cufftReal    = c_float
cufftComplex = (c_float*2)

cufftHandle_p = POINTER(cufftHandle)

#// CUFFT transform directions 
##define CUFFT_FORWARD -1 // Forward FFT
##define CUFFT_INVERSE  1 // Inverse FFT

CUFFT_FORWARD = -1  ## Forward FFT
CUFFT_INVERSE =  1  ## Inverse FFT

#// CUFFT supports the following transform types 
#typedef enum cufftType_t {
#    CUFFT_R2C = 0x2a, // Real to Complex (interleaved)
#    CUFFT_C2R = 0x2c, // Complex (interleaved) to Real
#    CUFFT_C2C = 0x29  // Complex to Complex, interleaved
#} cufftType;

cufftType = c_int

CUFFT_R2C = 0x2a  ## Real to Complex (interleaved)
CUFFT_C2R = 0x2c  ## Complex (interleaved) to Real
CUFFT_C2C = 0x29  ## Complex to Complex, interleaved
