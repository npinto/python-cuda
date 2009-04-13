import numpy as np

from cuda.memory import Linear
from cuda.kernel.kernelfactoryrt import SourceModule
from cuda.cuda import dim3

#from IPython.Shell import IPShellEmbed
#ipshell = IPShellEmbed(argv=[])


BLOCK_SIZE = 16
# Matrix A width
WA = (3 * BLOCK_SIZE)
# Matrix A height
HA = (5 * BLOCK_SIZE)
# Matrix B width
WB = (8 * BLOCK_SIZE)
# Matrix B height
HB = WA
# Matrix C width 
WC = WB
# Matrix C height
HC = HA

matrixMul = SourceModule(open('matrix_mul_kernel.cu','r').read())

nA = np.random.random(size=(HA, WA)).astype(np.float32)
nB = np.random.random(size=(HB, WB)).astype(np.float32)

print 'Allocating arrays'
dA = Linear(nA.shape).from_numpy(nA)
dB = Linear(nB.shape).from_numpy(nB)
dC = Linear((HC,WC))

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
