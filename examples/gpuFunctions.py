# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *

cvp = c_void_p
_cf = c_float
_ci = c_int

lib = CDLL("./libgpuFunctions.so")

#__global__ void gpuGFLOPS()
gpuGFLOPS = lib.__device_stub_gpuGFLOPS
gpuGFLOPS.restype = None
gpuGFLOPS.argtypes = [ ]

#__global__ void gpuBLSC(
#float *d_Calls, float *d_Puts,
#float *d_S, float *d_X, float *d_T,
#float R, float V, int OptN)
gpuBLSC = lib.__device_stub_gpuBLSC
gpuBLSC.restype = None
gpuBLSC.argtypes = [ cvp, cvp, cvp, cvp, cvp,
    _cf, _cf, _ci ]

#__global__ void gpuPOLY5(
#float *d_In1, float *d_Out1, int size )
gpuPOLY5 = lib.__device_stub_gpuPOLY5
gpuPOLY5.restype = None
gpuPOLY5.argtypes = [ cvp, cvp, _ci ]

#__global__ void gpuPOLY10(
#float *d_In1, float *d_Out1, int size )
gpuPOLY10 = lib.__device_stub_gpuPOLY10
gpuPOLY10.restype = None
gpuPOLY10.argtypes = [ cvp, cvp, _ci ]

#__global__ void gpuPOLY20(
#float *d_In1, float *d_Out1, int size )
gpuPOLY20 = lib.__device_stub_gpuPOLY20
gpuPOLY20.restype = None
gpuPOLY20.argtypes = [ cvp, cvp, _ci ]

#__global__ void gpuPOLY40(
#float *d_In1, float *d_Out1, int size )
gpuPOLY40 = lib.__device_stub_gpuPOLY40
gpuPOLY40.restype = None
gpuPOLY40.argtypes = [ cvp, cvp, _ci ]

#__global__ void gpuSAXPY(
#float Factor, float *d_In1, float *d_In2, int size )
gpuSAXPY = lib.__device_stub_gpuSAXPY
gpuSAXPY.restype = None
gpuSAXPY.argtypes = [ _cf, cvp, cvp, _ci ]

#__global__ void gpuVADD(
#float *d_In1, float *d_In2, int size )
gpuVADD = lib.__device_stub_gpuVADD
gpuVADD.restype = None
gpuVADD.argtypes = [ cvp, cvp, _ci ]

#__global__ void gpuSGEMM(
#float* C, float* A, float* B, int wA, int wB )
gpuSGEMM = lib.__device_stub_gpuSGEMM
gpuSGEMM.restype = None
gpuSGEMM.argtypes = [ cvp, cvp, cvp, _ci, _ci ]

#__global__ void gpuTRIG(
#float *d_Out1, float *d_Out2, float *d_In1, int size )
gpuTRIG = lib.__device_stub_gpuTRIG
gpuTRIG.restype = None
gpuTRIG.argtypes = [ cvp, cvp, cvp, _ci ]

#__global__ void gpuScale(
#float *d_Out1, _F *d_In1, _F scale, int size )
gpuScale = lib.__device_stub_gpuScale
gpuScale.restype = None
gpuScale.argtypes = [ cvp, cvp, _cf, _ci ]

#// for streams example
#__global__ void init_array(
#int *g_data, int *factor){ 
init_array = lib.__device_stub_init_array
init_array.restype = None
init_array.argtypes = [ c_int, c_int ]
