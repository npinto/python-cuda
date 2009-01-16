# coding:utf-8: © Arno Pähler, 2007-08

from ctypes import c_double,c_float,c_int

#cannot mix single and double in one file
#segmentation fault on conflicting loads
#of both single and double versions of fft
#import dfft,sfft

import sfft
from cpuFunctions import scale

_cd  = c_double
_cf  = c_float
_ci  = c_int

##  constants
FORWARD  = -1  ## Forward FFT
BACKWARD =  1  ## Inverse FFT

ESTIMATE = 0  ## for plans
MEASURE  = 1  ## for plans

OUT_OF_PLACE =   0
IN_PLACE     =   8
USE_WISDOM   =  16
THREADSAFE   = 128

R2C = FORWARD
C2R = BACKWARD

fftw_plan = _ci

x_cache = {}

def getType(item):
    """gets the type of ctypes item"""
    itype = item._type_._type_
    if itype == 'd':
#        return _cd,dfft
        return None,None
    elif itype == 'f':
        return _cf,sfft
    else:
        return None,None

def rcfft(r,dims):
    global x_cache
    if not isinstance(dims,tuple):
        dims = tuple(dims)
    dimx = list(dims)
    dimx[-1] = 2*(dimx[-1]/2+1)
    size = reduce(lambda x,y:x*y,dimx)
    ndim = len(dims)
    x_type,x_fft = getType(r)
    if x_type is None:
        return None
    c = (x_type*size)()
    try:
        wsave = x_cache[('rc',dims)]
    except KeyError:
        xdim = (_ci*ndim)(*dims)
        wsave = x_fft.CreatePlan_r(ndim,xdim,
                R2C,ESTIMATE)
        x_cache[('rc',dims)] = wsave
    x_fft.Execute_r2c(wsave,r,c)
    return c

def crfft(c,dims):
    global x_cache
    if not isinstance(dims,tuple):
        dims = tuple(dims)
    dims = list(dims)
    dims = tuple(dims)
    size = reduce(lambda x,y:x*y,dims)
    ndim = len(dims)
    x_type,x_fft = getType(c)
    if x_type is None:
        return None
    r = (x_type*size)()
    try:
        wsave = x_cache[('cr',dims)]
    except KeyError:
        xdim = (_ci*ndim)(*dims)
        wsave = x_fft.CreatePlan_r(ndim,xdim,
                C2R,ESTIMATE)
        x_cache[('cr',dims)] = wsave
    x_fft.Execute_c2r(wsave,c,r)
    sc = 1./float(size)
    scale(r,sc)
    return r

def ccfft(c,dims):
    global x_cache
    if not isinstance(dims,tuple):
        dims = tuple(dims)
    size = reduce(lambda x,y:x*y,dims)
    ndim = len(dims)
    x_type,x_fft = getType(c)
    if x_type is None:
        return None
    z = (x_type*(size*2))()
    try:
        wsave = x_cache[('cc',dims)]
    except KeyError:
        xdim = (_ci*ndim)(*dims)
        wsave = x_fft.CreatePlan_c(ndim,xdim,
                FORWARD,ESTIMATE)
        x_cache[('cc',dims)] = wsave
    x_fft.Execute_c2c(wsave,c,z)
    return z

def icfft(z,dims):
    global x_cache
    if not isinstance(dims,tuple):
        dims = tuple(dims)
    size = reduce(lambda x,y:x*y,dims)
    ndim = len(dims)
    x_type,x_fft = getType(z)
    if x_type is None:
        return None
    c = (x_type*(size*2))()
    try:
        wsave = x_cache[('ic',dims)]
    except KeyError:
        xdim = (_ci*ndim)(*dims)
        wsave = x_fft.CreatePlan_c(ndim,xdim,
                BACKWARD,ESTIMATE)
        x_cache[('ic',dims)] = wsave
    x_fft.Execute_c2c(wsave,z,c)
    sc = 1./float(size)
    scale(c,sc)
    return c

#shortcuts

fft  = ccfft
ifft = icfft
