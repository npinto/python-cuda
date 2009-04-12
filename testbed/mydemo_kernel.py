import atexit
import ctypes

import numpy as np
import os, sys

from IPython.Shell import IPShellEmbed
ipshell = IPShellEmbed(argv=[])


from cuda.memory import Linear #CudaArrayFromArray#CudaArray
from kget import KernelGetter
from cuda.cuda import dim3

#from pystream import cudaarray, cudart, kernels


BLOCK_SIZE = 16
WA = (3 * BLOCK_SIZE) # Matrix A width
HA = (5 * BLOCK_SIZE) # Matrix A height
WB = (8 * BLOCK_SIZE) # Matrix B width
HB = WA  # Matrix B height
WC = WB  # Matrix C width 
HC = HA  # Matrix C height


assert os.system("nvcc -Xcompiler='-fPIC' -c -o matrixMul_kernels.cu_o matrixMul_kernel.cu") == 0
assert os.system("g++ -shared -L/opt/cuda/lib -lcudart -lcuda -o libmatrixMul.so matrixMul_kernels.cu_o") == 0

print 'Loading kernel'
dll = ctypes.cdll.LoadLibrary("./libmatrixMul.so")

# Register the finalizer. This seems to get rid of the segfault-on-exit that
# I've been seeing.
#atexit.register(dll._fini)

matrixMul = KernelGetter(dll)

nA = np.random.random(size=(HA, WA)).astype(np.float32)
nB = np.random.random(size=(HB, WB)).astype(np.float32)
nC = np.zeros((HC,WC)).astype(np.float32)

print 'Allocating arrays'
dA = Linear(nA.shape).from_numpy(nA)
dB = Linear(nB.shape).from_numpy(nB)
dC = Linear(nC.shape)

#threads = (BLOCK_SIZE, BLOCK_SIZE)
#grid = 

print 'Calling kernel'
grid = dim3(WC // BLOCK_SIZE, HC // BLOCK_SIZE, 1)
block = dim3(BLOCK_SIZE, BLOCK_SIZE, 1)
Mul = matrixMul.matrixMul(grid, block)
Mul(dC.ref, dA.ref, dB.ref, WA, WB)

print 'Collecting results'
nC = dC.to_numpy()
nC.reshape((HC, WC))

print 'Freeing data'
dA._free()
dB._free()
dC._free()

print 'Calculating error'
print
goldC = np.dot(nA, nB)
err = nC - goldC
print 'L2 err: %r' % np.linalg.norm(err, 2)
print 'L1 err: %r' % np.linalg.norm(err, 1)
print 'Linf err: %r' % np.linalg.norm(err, np.inf)
print 'Lfro err: %r' % np.linalg.norm(err, 'fro')
