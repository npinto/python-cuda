import ctypes
from cuda.cuda import *



class KernelGetter(object):
    """ Wraps a ctypes CDLL instance for accessing CUDA kernels.

    Example
    -------
    from ctypes import cdll
    mykernels = KernelGetter(cdll.LoadLibrary('libmykernels.so'))
    mykernels.FastKernel(grid, block)(x, y)
    # Equivalent CUDA call:
    #   FastKernel<<<grid, block>>>(x, y)
    """

    def __init__(self, dll):
        raise NotImplementedError
#         self.dll = dll

#     def __getattr__(self, name):
#         mangled_name = '__device_stub_%s' % name
#         try:
#             funcptr = getattr(self.dll, mangled_name)
#         except AttributeError:
#             raise AttributeError("could not find kernel named %r in %r" % (name, self.dll))

#         # Return a factory function that will create the Kernel object.
#         factory = lambda *args, **kwds: Kernel(funcptr, *args, **kwds)

#         return factory


# class Kernel(object):
#     """ Configure a CUDA kernel.
#     """

#     def __init__(self, funcptr, gridDim, blockDim, sharedMem=0, tokens=0):
#         # The function pointer to the kernel.
#         self.funcptr = funcptr

#         # The configuration parameters for the call. These are the arguments
#         # inside the <<<>>> brackets in CUDA.
#         self.gridDim = gridDim
#         self.blockDim = blockDim
#         self.sharedMem = sharedMem
#         self.tokens = tokens

#     # Delegate .restype and .argtypes attribute access to the underlying
#     # function pointer.
#     def _get_restype(self):
#         return self.funcptr.restype
#     def _set_restype(self, val):
#         self.funcptr.restype = val
#     restype = property(_get_restype, _set_restype)

#     def _get_argtypes(self):
#         return self.funcptr.argtypes
#     def _set_argtypes(self, val):
#         self.funcptr.argtypes = val
#     argtypes = property(_get_argtypes, _set_argtypes)


#     def __call__(self, *args):
#         """ Call the kernel as configured.
#         """
#         cudart.cudaConfigureCall(self.gridDim, self.blockDim, self.sharedMem, self.tokens)
#         self.funcptr(*args)
#         # Check to make sure we didn't get an error.
#         err = cudart.getLastError()
#         cudart._checkCudaStatus(err)
