# coding:utf-8: © Arno Pähler, 2007-08

from ctypes import *

mm = CDLL("./_cpuFunctions.so")

_cvp = c_void_p
_cfl = c_float
_cui = c_uint

#
# Utility functions
#

mm.rdtsc.restype = c_uint64;
mm.rdtsc.argtypes = None

def rdtsc():
    return mm.rdtsc()

ReadTimestampCounter = rdtsc

mm.microtime.restype = c_long;
mm.microtime.argtypes = None

def microtime():
    return mm.microtime()

mm.scale.restype = None
mm.scale.argtypes = [ _cvp, _cfl, _cui ]

def scale(a,s,n=None):
    if n is None:
        n = len(a)
    mm.scale(a,s,n)

mm.l1norm.restype = _cfl
mm.l1norm.argtypes = [ _cvp, _cvp, _cui ]

def l1norm(a,b,n=None):
    if n is None:
        n = len(a)
    return mm.l1norm(a,b,n)

mm.arrayInit.restype = None
mm.arrayInit.argtypes = [ _cvp, _cui ]

def arrayInit(a,n=None):
    if n is None:
        n = len(a)
    mm.arrayInit(a,n)

vectorInit = arrayInit

mm.fixedInit.restype = None
mm.fixedInit.argtypes = [ _cvp, _cui ]

def fixedInit(a,n=None):
    if n is None:
        n = len(a)
    mm.fixedInit(a,n)

mm.randInit.restype = None
mm.randInit.argtypes = [ _cvp, _cui, _cfl, _cfl ]

def randInit(a,l,h,n=None):
    if n is None:
        n = len(a)
    mm.randInit(a,n,l,h)

mm.setZero.restype = None
mm.setZero.argtypes = [ _cvp, _cui ]

def setZero(a,n=None):
    if n is None:
        n = len(a)
    mm.setZero(a,n)

mm.checkError.restype = None
mm.checkError.argtypes = [ _cvp, _cvp, _cui, _cvp, _cvp ]

def checkError(a,b,n=None):
    if n is None:
        n = len(a)
    err = c_float()
    mxe = c_float()
    mm.checkError(a,b,n,byref(err),byref(mxe))
    return err.value,mxe.value

mm.checkTrig.restype = None
mm.checkTrig.argtypes = [ _cvp, _cvp, _cvp, _cvp, _cui ]

def checkTrig(a,b,n=None):
    if n is None:
        n = len(a)
    e = c_float()
    m = c_float()
    mm.checkTrig(byref(e),byref(m),a,b,n)
    return e.value,m.value

#
# Math functions
#

mm.gflops.restype = None
mm.gflops.argtypes = [ ]

def cpuGFLOPS():
    mm.gflops()

mm.blsc.restype = None
mm.blsc.argtypes = [ _cvp, _cvp, _cvp,
                   _cvp, _cvp, _cfl, _cfl, _cui ]

def cpuBLSC(h_C,h_P,h_S,h_X,h_T,R,V,size):
    mm.blsc(h_C,h_P,h_S,h_X,h_T,R,V,size)

mm.poly5.restype = None
mm.poly5.argtypes = [ _cvp, _cvp, _cui ]

mm.poly10.restype = None
mm.poly10.argtypes = [ _cvp, _cvp, _cui ]

mm.poly20.restype = None
mm.poly20.argtypes = [ _cvp, _cvp, _cui ]

mm.poly40.restype = None
mm.poly40.argtypes = [ _cvp, _cvp, _cui ]

def cpuPOLY5(x,y,n=None):
    if n is None:
        n = len(x)
    mm.poly5(x,y,n)

def cpuPOLY10(x,y,n=None):
    if n is None:
        n = len(x)
    mm.poly10(x,y,n)

def cpuPOLY20(x,y,n=None):
    if n is None:
        n = len(x)
    mm.poly20(x,y,n)

def cpuPOLY40(x,y,n=None):
    if n is None:
        n = len(x)
    mm.poly40(x,y,n)

mm.saxpy.restype = None
mm.saxpy.argtypes = [ _cfl, _cvp, _cvp, _cui ]

def cpuSAXPY(a,x,y,n=None):
    if n is None:
        n = len(x)
    mm.saxpy(a,x,y,n)

mm.vadd.restype = None
mm.vadd.argtypes = [ _cvp, _cvp, _cui ]

def cpuVADD(x,y,n=None):
    if n is None:
        n = len(x)
    mm.vadd(x,y,n)

mm.sdot.restype = c_float
mm.sdot.argtypes = [ _cvp, _cvp, _cui ]

def cpuSDOT(x,y,n=None):
    if n is None:
        n = len(x)
    return mm.sdot(x,y,n)

mm.sgemm.restype = None
mm.sgemm.argtypes = [
    _cvp, _cvp, _cvp,
    _cui, _cui, _cui ]

def cpuSGEMM(C,A,B,m,k,n):
    mm.sgemm(C,A,B,m,k,n)

mm.trig.restype = None
mm.trig.argtypes = [ _cvp, _cvp, _cvp, _cui ]

def cpuTRIG(a,x,y,n=None):
    if n is None:
        n = len(a)
    mm.trig(a,x,y,n)
