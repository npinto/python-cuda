#!/usr/bin/env python
import sys
import numpy.fft
from numpy.random import randn
from numpy import allclose
import cuda.sugar.fft
        
def main():
    print "-"*55
    print "--                                                   --"
    print "--  python-cuda versions of numpy.fft.{fftn,ifftn}   --"
    print "--                                                   --"
    print "-"*55
    print ">>> Creating host signal..."

    try:
        size = int(sys.argv[1])
    except Exception,e:
        size = 10

    print ">>> Signal Size = %s" % size

    numpy_array = randn(size)
    numpy_array -= numpy_array.mean()
    numpy_array /= numpy_array.std()

    print ">>> Computing ffts on GPU (CUDA) ..."

    print "[*] Forward fft on gpu ..."
    fft_res = cuda.sugar.fft.fftn(numpy_array)

    print "[*] Inverse fft on gpu ..."
    ifft_res = cuda.sugar.fft.ifftn(fft_res) 

    print ">>> Computing references on CPU (numpy) ..."

    print "[*] Forward fft on cpu ..."
    forward_ref = numpy.fft.fftn(numpy_array)

    print "[*] Inverse fft on cpu ..."
    inverse_ref = numpy.fft.ifftn(forward_ref)
    
    print "l2norm fft: ", numpy.linalg.norm(fft_res - forward_ref)

    print "l2norm ifft: ", numpy.linalg.norm(ifft_res - inverse_ref)

if __name__ == "__main__":
    main()
