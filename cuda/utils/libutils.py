#!/usr/bin/env python
from ctypes import CDLL
import platform

OSNAME = platform.system()

def get_lib(name, cdll_opts = None):
    libname = None
    if OSNAME == "Linux": 
        libname = "lib" + name + ".so"
    elif OSNAME == "Darwin": 
        libname = "lib" + name + ".dylib"
    elif OSNAME == "Windows": 
        import _winreg as wreg
        reg = wreg.ConnectRegistry(None, wreg.HKEY_LOCAL_MACHINE)
        key = wreg.OpenKey(reg, r"SOFTWARE\NVIDIA Corporation\Installed Products\NVIDIA CUDA")
        cuda_bin = os.path.join(wreg.QueryValueEx(key, "InstallDir")[0],"bin")
        libname = os.path.join(cuda_bin, "%.dll" % name)
        if name == "cuda":
            libname = "nvcuda.dll"
    if cdll_opts:
        lib = CDLL(libname, cdll_opts)
    else: 
        lib = CDLL(libname)
    return lib

if __name__ == "__main__":
    try:
        print "Loading libcuda..."
        get_lib("cuda")
        print "Test PASSED"
    except:
        print "Test FAILED"
