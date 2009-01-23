# -*- coding: utf-8 -*-
"""Array-like objects for CUDA."""

#----------------------------------------------------------------------------
# Copyright (c) 2007, Tech-X Corporation
#----------------------------------------------------------------------------



#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

from cuda.cuda import *
from cuda.cublas import *
import numpy
import ctypes
from ctypes import c_void_p

# cuda <-> dtype conversion
cudaDtypes = {'float32': ctypes.c_float,
              'int32': ctypes.c_int,
              'complex64': ctypes.c_float*2,
             }

class CudaArray(object):

    def __init__(self, size, dtype):
        self.size = size
        self.dtype = numpy.dtype(dtype)
        self.ctype = self._convertType(self.dtype)
        self.nbytes = self.size*ctypes.sizeof(self.ctype)
        self.allocated = False
        self.data = None

    def __del__(self):
        self.free()

    def _convertType(self, dtype):
        ct = cudaDtypes.get(dtype.name, None)
        if ct is None:
            raise Exception("Unsupported dtype")
        return ct

    def alloc(self):
        #self.data = cudart.malloc(self.nbytes, self.ctype)
        self.data = c_void_p()
        cudaMalloc(byref(self.data), self.nbytes)
        self.allocated = True

    def free(self):
        if self.allocated:
            cudaFree(self.data)
            self.data = None
            self.allocated = False

    def toArray(self, a=None):
        if not self.allocated:
            raise Exception("Must first allocate")
        if a is None:
            a = numpy.empty(self.size, dtype=self.dtype)
        else:
            # Check that the given array is appropriate.
            if a.size != self.size:
                raise ValueError("need an array of size %s; got %s" % (self.size, a.size))
            if a.dtype.name != self.dtype.name:
                # XXX: compare dtypes directly? issubdtype?
                raise ValueError("need an array of dtype %r; got %r" % (self.dtype, a.dtype))

        cudaMemcpy(a.ctypes.data, self.data, self.nbytes, cudaMemcpyDeviceToHost)
        return a

    def setWithArray(self, a):
        if not self.allocated:
            raise Exception("Must first allocate")
        a = numpy.ascontiguousarray(a, dtype=None)
        assert a.size == self.size, "size must be the same"
        assert a.dtype == self.dtype, "dtype must be the same"
        cudaMemcpy(self.data, a.ctypes.data, self.nbytes, cudaMemcpyHostToDevice)

class RawCudaArray(CudaArray):
    def __init__(self, size, dtype):
        CudaArray.__init__(self, size, dtype)
        self.alloc()

class CudaArrayFromArray(CudaArray):
    def __init__(self, a, dtype=None):
        a = numpy.ascontiguousarray(a, dtype=dtype)
        CudaArray.__init__(self, a.size, a.dtype)
        self.alloc()
        self.setWithArray(a)


class CublasArray(CudaArrayFromArray):
    def __init__(self, numpy_array):
        CudaArrayFromArray.__init__(self, numpy_array, numpy_array.dtype)

    def alloc(self):
        self.data = c_void_p()
        cublasAlloc(self.size ,sizeof(c_float), byref(self.data))
        self.allocated = True

