# coding:utf-8: © Arno Pähler, 2007-08
# NP: remove absolute paths (argnnnnn!)

from ctypes import c_int,c_void_p
from ctypes import CDLL,POINTER,RTLD_GLOBAL

##  This version is for 32-bit floats

_ci  = c_int
_cip = POINTER(c_int)
_cvp = c_void_p

cc = CDLL("libsfftw.so",mode=RTLD_GLOBAL)
cr = CDLL("libsrfftw.so")

fftw_plan = _ci

##extern [r]fftwnd_plan fftwnd_create_plan(
##    int rank, const int *n,
##    fftw_direction dir, int flags);
##
##  all plans have the same signature

CreatePlan_c = cc.fftwnd_create_plan
CreatePlan_c.restype = fftw_plan
CreatePlan_c.argtypes = [ _ci, _cip, _ci, _ci ]

CreatePlan_r = cr.rfftwnd_create_plan
CreatePlan_r.restype = fftw_plan
CreatePlan_r.argtypes = [ _ci, _cip, _ci, _ci ]

##extern void [r]fftwnd_destroy_plan(rfftwnd_plan plan);
DestroyPlan_c = cc.fftwnd_destroy_plan
DestroyPlan_c.restype = None
DestroyPlan_c.argtypes = [ _ci ]

DestroyPlan_r = cr.rfftwnd_destroy_plan
DestroyPlan_r.restype = None
DestroyPlan_r.argtypes = [ _ci ]

##extern void fftwnd_one(fftwnd_plan p,
##            fftw_complex *in, fftw_complex *out);
Execute_c2c = cc.fftwnd_one
Execute_c2c.restype = None
Execute_c2c.argtypes = [ _ci, _cvp, _cvp ]

##extern void rfftwnd_one_real_to_complex(rfftwnd_plan p,
##                  fftw_real *in, fftw_complex *out);
##extern void rfftwnd_one_complex_to_real(rfftwnd_plan p,
##                  fftw_complex *in, fftw_real *out);
Execute_r2c = cr.rfftwnd_one_real_to_complex
Execute_r2c.restype = None
Execute_r2c.argtypes = [ _ci, _cvp, _cvp ]

Execute_c2r = cr.rfftwnd_one_complex_to_real
Execute_c2r.restype = None
Execute_c2r.argtypes = [ _ci, _cvp, _cvp ]
