# coding:utf-8: © Arno Pähler, 2007-08

from cuda.cufft_api import *
from cuda.cuda_utils import *

def makePlan(dims,kind):
    """
    dims : tuple of array dimensions (1-3 els.)
    kind : type of transform desired
           returns plan to be used by transform
    """
    spln = "cufftPlan%dd(*args)"
    ndim = len(dims)
    plan = cufftHandle()
    args = (byref(plan),)+dims+(kind,)
    if ndim == 1:
        args = args+(1,)
    eval(spln % ndim)
    return plan

def rcfft(plan,r,SC=None,c=None):
    """
    plan : plan created by fftw
    r    : real array to be transformed
    SC   : size of output array
    c    : complex array, result of transform
    """
    if c is None:
        if SC is None:
            cufftDestroy(plan)
            raise ValueError("array size missing")
        c = getMemory(SC)
    cufftExecR2C(plan,r,c)
    return c

def crfft(plan,c,SC=None,r=None):
    """
    plan : plan created by fftw
    c    : complex array to be transformed
    SC   : size of output array
    r    : real array, result of transform
    """
    if r is None:
        if SC is None:
            cufftDestroy(plan)
            raise ValueError("array size missing")
        r = getMemory(SC)
    cufftExecC2R(plan,c,r)
    return r

def ccfft(plan,c,SC=None,z=None):
    """
    plan : plan created by fftw
    c    : complex array to be transformed
    SC   : size of output array
    z    : complex array, result of transform
    """
    if z is None:
        if SC is None:
            cufftDestroy(plan)
            raise ValueError("array size missing")
        z = getMemory(SC)
    cufftExecC2C(plan,c,z,CUFFT_FORWARD)
    return z

def icfft(plan,z,SC=None,c=None):
    """
    plan : plan created by fftw
    z    : complex array to be transformed
    SC   : size of output array
    c    : complex array, result of transform
    """
    if c is None:
        if SC is None:
            cufftDestroy(plan)
            raise ValueError("array size missing")
        c = getMemory(SC)
    cufftExecC2C(plan,z,c,CUFFT_INVERSE)
    return c

#shortcuts

fft  = ccfft
ifft = icfft
