#!/usr/bin/python
# -*- coding: utf-8 -*- 

# XXX
# --
# Code forked from python-cuda-2.0_42 © Arno Pähler, 2007-08
# -- 

from cufft_defs import *
from cuda.utils import libutils

cufft = libutils.get_lib("cufft", RTLD_GLOBAL)

#cufftResult CUFFTAPI cufftPlan1d(cufftHandle *plan, 
#                                 int nx, 
#                                 cufftType type, 
#                                 int batch);
cufftPlan1d = cufft.cufftPlan1d
cufftPlan1d.restype = cufftResult
cufftPlan1d.argtypes = [ cufftHandle_p,
                        c_int, cufftType, c_int ]

#cufftResult CUFFTAPI cufftPlan2d(cufftHandle *plan, 
#                                 int nx, int ny,
#                                 cufftType type);
cufftPlan2d = cufft.cufftPlan2d
cufftPlan2d.restype = cufftResult
cufftPlan2d.argtypes = [ cufftHandle_p,
                        c_int, c_int, cufftType ]

#cufftResult CUFFTAPI cufftPlan3d(cufftHandle *plan, 
#                                 int nx, int ny, int nz, 
#                                 cufftType type);
cufftPlan3d = cufft.cufftPlan3d
cufftPlan3d.restype = cufftResult
cufftPlan3d.argtypes = [ cufftHandle_p,
                        c_int, c_int, c_int, cufftType ]

#cufftResult CUFFTAPI cufftDestroy(cufftHandle plan);
cufftDestroy = cufft.cufftDestroy
cufftDestroy.restype = cufftResult
cufftDestroy.argtypes = [ cufftHandle ]

#cufftResult CUFFTAPI cufftExecC2C(cufftHandle plan, 
#                                  cufftComplex *idata,
#                                  cufftComplex *odata,
#                                  int direction);
cufftExecC2C = cufft.cufftExecC2C
cufftExecC2C.restype = cufftResult
cufftExecC2C.argtypes = [ cufftHandle, c_uint, c_uint, c_int ]

#cufftResult CUFFTAPI cufftExecR2C(cufftHandle plan, 
#                                  cufftReal *idata,
#                                  cufftComplex *odata);
cufftExecR2C = cufft.cufftExecR2C
cufftExecR2C.restype = cufftResult
cufftExecR2C.argtypes = [ cufftHandle, c_uint, c_uint ]

#cufftResult CUFFTAPI cufftExecC2R(cufftHandle plan, 
#                                  cufftComplex *idata,
#                                  cufftReal *odata);
cufftExecC2R = cufft.cufftExecC2R
cufftExecC2R.restype = cufftResult
cufftExecC2R.argtypes = [ cufftHandle, c_uint, c_uint ]
