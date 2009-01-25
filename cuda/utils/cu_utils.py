#!/usr/bin/env python
from cuda.cu import * 
from ctypes import *

class GPUException(Exception):
    pass

class cu_CUDA(object):
    usedDevices = 0
    def __init__(self):
        flags = 0 # see manual
        self.device = None
        self.context = None
        self.module = None
        self.deviceID  = -1
        cuInit(flags)
        device_count = c_int()
        cuDeviceGetCount(byref(device_count))
        if cu_CUDA.usedDevices >= device_count.value:
            print "No more uninitialized devices available"
            return
        self.device = CUdevice()
        self.context = CUcontext()
        self.modules = list()
        self.functions = dict()
        self.deviceID = cu_CUDA.usedDevices
        cuDeviceGet(byref(self.device),self.deviceID)
        cu_CUDA.usedDevices += 1
        status = cuCtxCreate(byref(self.context),0,self.device)
        if status != CUDA_SUCCESS:
            cuCtxDetach(self.context)
            raise GPUException(
                "Failed to create CUDA context")
        self.getInfo()

    def getSourceModule(self,name=None):
        if name is None:
            name = "gpuFunctions.cubin"
        module   = CUmodule()
        status = cuModuleLoad(byref(module),name)
        if status != CUDA_SUCCESS:
            print "File not found: %s" % name
        self.modules.append(module)
        return module

    def getFunction(self,name):
        missing = True
        function = CUfunction()
        for module in self.modules:
            status = cuModuleGetFunction(function,module,name)
            if status != CUDA_SUCCESS:
                continue
            else:
                self.functions[name] = function
                missing = False
                break
        if missing:
            print "Function not found: %s" % name
            return None
        return function

    def getInfo(self):
        device = self.device
        info = dict()
        count = c_int()
        cuDeviceGetCount(byref(count))
        info["count"] = count.value
        name = (c_char*256)()
        cuDeviceGetName(name,256,device)
        info["name"] = name.value
        memsize = c_uint()
        cuDeviceTotalMem(byref(memsize),device)
        info["memory"] = memsize.value
        free,total = c_uint(),c_uint()
        cuMemGetInfo(byref(free),byref(total))
        info["free"] = free.value
        major,minor = c_int(),c_int()
        cuDeviceComputeCapability(byref(major),byref(minor),device)
        info["capability"] = (major.value,minor.value)
        props = CUdevprop()
        cuDeviceGetProperties(byref(props),device)
        info["properties"] = props
        self.info = info

    def __str__(self):
        s = ["Device Info:\n"]
        i = self.info
        s.append("%-19s = %d" % (
            "number of devices",i["count"]))
        s.append("%-19s = %d" % (
            "current device ID",self.deviceID))
        s.append("%-19s = %s" % (
            "device name =",i["name"]))
        s.append("%-19s = %.f MB" % (
            "memory size",i["memory"]/1024.**2))
        s.append("%-19s = %.f MB" % (
            "memory free",i["free"]/1024.**2))
        s.append("%-19s = %.f MHz" % (
            "clock rate",i["properties"].clockRate/1000.))
        s.append("%-19s = %d" % (
            "major",i["capability"][0]))
        s.append("%-19s = %d" % (
            "minor",i["capability"][1]))
        s.append(21*"-")
        s.append(str(i["properties"]))
        return "\n".join(s)

def getMemory(d,dtype=c_int):
    gmem =  CUdeviceptr()
    if isinstance(d,(int,long)):
        size = d*sizeof(dtype)
        status = cuMemAlloc(byref(gmem),size)
        if status != CUDA_SUCCESS:
            raise GPUException(
            "Failed to allocate memory")
        cuMemsetD8(gmem,0,size)
    else:
        size = len(d)*sizeof(d._type_)
        status = cuMemAlloc(byref(gmem),size)
        if status != CUDA_SUCCESS:
            raise GPUException(
            "Failed to allocate memory")
        cuMemcpyHtoD(gmem,d,size)
    return gmem.value

def devMemToTex(module,name,data,size):
    tex = CUtexref()
    nul = c_uint()
    status = cuModuleGetTexRef(byref(tex),module,name)
    if status != CUDA_SUCCESS:
        raise GPUException(
        "No such texture: %s" % name)
    cuTexRefSetAddress(byref(nul),tex,data,size)
    return tex
