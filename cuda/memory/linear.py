#!/usr/bin/env python
"""Array-like objects for CUDA."""

from cuda.cuda import *
from cuda.cublas import *
import numpy
import ctypes
from ctypes import *

# cuda <-> dtype conversion
cudaDtypes = {'float32': ctypes.c_float,
              'int32': ctypes.c_int,
              'complex64': ctypes.c_float*2,
             }

class Linear(object):

    ref = property(fget=lambda self: self._get_ref())

    def __init__(self, shape=None, dtype='float32', order=None):
        self.shape = shape
        self.size = numpy.prod(shape)
        self.dtype = numpy.dtype(dtype)
        self.order = order
        self.ctype = self._convert_type(self.dtype)
        self.nbytes = self.size*ctypes.sizeof(self.ctype)
        self.allocated = False
        self.data = None
        self._alloc()

    def __del__(self):
        self._free()

    def _convert_type(self, dtype):
        ct = cudaDtypes.get(dtype.name, None)
        if ct is None:
            raise TypeError("Unsupported dtype")
        return ct

    def _get_ref(self):
        return cast(self.data,POINTER(self._convert_type(self.dtype)))

    def _alloc(self):
        self.data = c_void_p()
        cudaMalloc(byref(self.data), self.nbytes)
        self.allocated = True

    def _free(self):
        if self.allocated:
            cudaFree(self.data)
            self.data = None
            self.allocated = False

    def to_numpy(self, a=None):
        if not self.allocated:
            raise Exception("Must first allocate")
        if a is None:
            a = numpy.empty(self.shape, dtype=self.dtype, order=self.order)
        else:
            # Check that the given array is appropriate.
            if a.size != self.size:
                raise ValueError("need an array of size %s; got %s" % (self.size, a.size))
            if a.dtype.name != self.dtype.name:
                # XXX: compare dtypes directly? issubdtype?
                raise ValueError("need an array of dtype %r; got %r" % (self.dtype, a.dtype))
        cudaMemcpy(a.ctypes.data, self.ref, self.nbytes, cudaMemcpyDeviceToHost)
        a = a.reshape(self.shape, order=self.order)
        return a

    def from_numpy(self, a):
        if not self.allocated:
            raise Exception("Must first allocate")
        assert a.size == self.size, "size must be the same"
        assert a.dtype == self.dtype, "dtype must be the same"
        a = numpy.ascontiguousarray(a,dtype=None)
        if self.order == 'F':
            a = numpy.asfortranarray(a)
        cudaMemcpy(self.data, a.ctypes.data, self.nbytes, cudaMemcpyHostToDevice)
        return self
