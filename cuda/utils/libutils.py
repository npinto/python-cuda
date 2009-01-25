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
        libname = None
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
