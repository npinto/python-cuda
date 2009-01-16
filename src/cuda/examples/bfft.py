#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08

from math import log

import sys,time
from ctypes import c_float

import xfft
from cpuFunctions import arrayInit,checkError

try:
    kr = int(sys.argv[1])
    dims = tuple([int(x) for x in sys.argv[2].split(",")])
except IndexError:
    sys.exit()

doComplex = False
if kr < 0:
    kr = -kr
    doComplex = True

size = reduce(lambda x,y:x*y,dims)
if doComplex:
    r = (c_float*(size*2))()
else:
    r = (c_float*size)()
arrayInit(r)

sz = 1.e6/float(size)

fftw_start = time.clock()
wall_start = time.time()


xr = float(.5 )/float(kr)

if doComplex:
    text = "complex"
    rcfftx = xfft.ccfft
    crfftx = xfft.icfft
else:
    text = "   real"
    rcfftx = xfft.rcfft
    crfftx = xfft.crfft
for k in range(0,kr):
    c = rcfftx(r,dims)
    z = crfftx(c,dims)

fftw_end = time.clock()
wall_end = time.time()

dif = fftw_end - fftw_start
wif = wall_end - wall_start
print "\nfft elapsed real time     : %8.3f seconds" % wif
print "%d-D %s-to-complex fft: %8.3f seconds" % (len(dims),text,dif*xr)

flops = 2.*5.e-9*log(size)*size*kr/log(2.)
print "Performance               : %8.3f GFlops" % (flops/wif)
dif = dif * xr * sz
print "%d-D %s-to-complex fft: %8.3f µs/point\n" % (len(dims),text,dif)

rz = 1./size
err,mxe = checkError(r,z)
print "avg and max error         : %8.1e %8.1e" %  (err,mxe)
