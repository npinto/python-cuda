#!/usr/bin/env python
import cuda.cuda as cuda
from ctypes import byref, c_int
import logging
log = logging.getLogger('python-cuda')

CUDART_VERSION = 2010

def cutilSafeCall(retval):
    if retval != 0:
        log.error( 'error! %s' % retval)

def get_device_count():
    device_count = c_int()
    cutilSafeCall(cuda.cudaGetDeviceCount(byref(device_count)));
    return device_count.value

def has_cuda_device():
    dev_count = get_device_count()
    if dev_count > 0:
        log.debug("Found %d gpu devices" % dev_count)
    else:
        log.debug("There is no device supporting CUDA")
        return False

    cuda_enabled = False

    for dev in range(0, dev_count):
        dev_prop = cuda.cudaDeviceProp()
        retval = cuda.cudaGetDeviceProperties(byref(dev_prop), dev)
        if dev_prop.major == 9999 and dev_prop.minor == 9999:
            log.debug( "Device %s does not support cuda." % dev)
            continue
        cuda_enabled = True
        break

    if not cuda_enabled:
        log.debug("There is no device supporting CUDA")
    return cuda_enabled

def needs_emulation():
    return has_cuda_device()

def get_devices(): 
    dev_count = get_device_count()
    if dev_count > 0:
        log.debug("found %d gpu devices" % dev_count)
    else:
        log.debug("there is no device supporting cuda")

    for dev in range(0, dev_count):
        dev_prop = cuda.cudaDeviceProp()
        retval = cuda.cudaGetDeviceProperties(byref(dev_prop), dev)
        if retval == 3:
            log.debug( "there is no device supporting cuda")
            break
        elif dev == 0: 
            if dev_prop.major == 9999 and dev_prop.minor == 9999:
                log.debug( "there is no device supporting cuda.")
            elif dev_count == 1:
                log.debug( "there is 1 device supporting cuda")
            else:
                log.debug( "there are %d devices supporting cuda" % dev_count)

        log.debug('Device %d: "%s"' % (dev, dev_prop.name))
        log.debug("Major revision number:                         %d" % dev_prop.major)
        log.debug("Minor revision number:                         %d" % dev_prop.minor)
        log.debug("Total amount of global memory:                 %u bytes" % dev_prop.totalGlobalMem)

        if CUDART_VERSION >= 2000:
            log.debug("Number of multiprocessors:                     %d", dev_prop.multiProcessorCount);
            log.debug("Number of cores:                               %d", 8 * dev_prop.multiProcessorCount);

        log.debug( "Total amount of constant memory:               %u bytes" % dev_prop.totalConstMem)
        log.debug( "Total amount of shared memory per block:       %u bytes" % dev_prop.sharedMemPerBlock)
        log.debug( "Total number of registers available per block: %d" % dev_prop.regsPerBlock)
        log.debug( "Warp size:                                     %d" % dev_prop.warpSize)
        log.debug( "Maximum number of threads per block:           %d" % dev_prop.maxThreadsPerBlock)
        log.debug( "Maximum sizes of each dimension of a block:    %d x %d x %d" % (dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]))
        log.debug( "Maximum sizes of each dimension of a grid:     %d x %d x %d" % (dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]))
        log.debug( "Maximum memory pitch:                          %u bytes" % dev_prop.memPitch)
        log.debug( "Texture alignment:                             %u bytes" % dev_prop.textureAlignment)
        log.debug( "Clock rate:                                    %.2f GHz" % (dev_prop.clockRate * (1e-6)))

        if CUDART_VERSION >= 2000:
            log.debug("Concurrent copy and execution:                 %s" % bool(dev_prop.deviceOverlap))
