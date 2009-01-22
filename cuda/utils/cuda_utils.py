# coding:utf-8: © Arno Pähler, 2007-08

from cuda.cuda.cuda_api import *
from cuda.cuda.cuda_defs import *

class GPUException(Exception):
    pass

class cuda_CUDA(object):
    usedDevices = 0
    def __init__(self):
        flags = 0 # see manual
        self.deviceID  = -1
        device_count = c_int()
        cudaGetDeviceCount(byref(device_count))
        if cuda_CUDA.usedDevices >= device_count.value:
            print "No more uninitialized devices available"
            return
        self.functions = dict()
        self.deviceID = cuda_CUDA.usedDevices
        cudaSetDevice(self.deviceID)
        cuda_CUDA.usedDevices += 1
        self.getInfo()

    def getInfo(self):
        device = self.deviceID
        info = dict()
        count = c_int()
        cudaGetDeviceCount(byref(count))
        info["count"] = count.value
        props = cudaDeviceProp()
        cudaGetDeviceProperties(byref(props),device)
        info["properties"] = props
        info["name"] = props.name
        info["memory"] = props.totalGlobalMem
        info["capability"] = props.major,props.minor
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
    gmem =  c_void_p()
    if isinstance(d,(int,long)):
        size = d*sizeof(dtype)
        status = cudaMalloc(byref(gmem),size)
        if status != cudaSuccess:
            raise GPUException(
            "Failed to allocate memory")
        cudaMemset(gmem,0,size)
    else:
        size = len(d)*sizeof(d._type_)
        status = cudaMalloc(byref(gmem),size)
        if status != cudaSuccess:
            raise GPUException(
            "Failed to allocate memory")
        cudaMemcpy(gmem,d,size,cudaMemcpyHostToDevice)
    return gmem.value

def mallocHost(n,dtype=c_int,pageLocked=True):
    if pageLocked:
        p = c_void_p()
        cudaMallocHost(byref(p),n*sizeof(dtype))
        r = (dtype*n).from_address(p.value)
    else:
        r = (dtype*n)()
    return r

### XXX
#def devMemToTex(name,data,size):
#    tex = textureReference_p()
#    nul = c_uint()
#    status = cudaGetTextureReference(byref(tex),name)
#    if status != cudaSuccess:
#        raise GPUException(
#        "No such texture: %s" % name)
#        T O   B E   D O N E
#    x,y,z,w = ....
#    f = cudaChannelFormatKind
#    desc = cudaCreatChannelDesc(x,y,z,f)
#    offset = c_int()
#    cudaBindTexture(byref(offset),tex,data,desc,size)
#    return tex # return offset,tex

### XXX
#void setupTexture(int x, int y) {
#    // Wrap mode appears to be the new default
#    texref.filterMode = cudaFilterModeLinear;
#    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
#
#    cudaMallocArray(&array, &desc, y, x);
#    CUT_CHECK_ERROR("cudaMalloc failed");
#}
#
#void bindTexture(void) {
#    cudaBindTextureToArray(texref, array);
#    CUT_CHECK_ERROR("cudaBindTexture failed");
#}
#
#void unbindTexture(void) {
#    cudaUnbindTexture(texref);
#    CUT_CHECK_ERROR("cudaUnbindTexture failed");
#}
#    
#void updateTexture(cData *data, size_t wib, size_t h, size_t pitch) {
#    cudaMemcpy2DToArray(array, 0, 0, data, pitch, wib, h, cudaMemcpyDeviceToDevice);
#    CUT_CHECK_ERROR("cudaMemcpy failed"); 
#}
#
#void deleteTexture(void) {
#    cudaFreeArray(array);
#    CUT_CHECK_ERROR("cudaFreeArray failed");
#}
