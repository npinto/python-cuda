#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08

import sys
from utilities import System

cmd = "cu_fft.py %d,%d,%d %s"
scm = ""

if len(sys.argv) > 1:
    scm = sys.argv[1]

vz = (64,128,256)
if "x" not in scm:
    vz = (64,128,256,512)

for nx in (128,256):
    for ny in (128,256):
        for nz in vz:
            s,o,e = System(cmd % (nx,ny,nz,scm))
            print o[-1]
        print
    print "---\n"
