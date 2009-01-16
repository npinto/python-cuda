#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *
from numpy import *
from numpy.random import rand
from ctypes_array import convert

ao = addressof
def fa(a,o,n,dtype=None):
    if dtype is None:
        t = a.__class__._type_
    else:
        t = dtype
    s = sizeof(t)
    return (t*n).from_address(ao(a)+o*s)
    
class x_float(c_float):
    pass

b = convert(rand(10))
a = (x_float*10)(*b)
z = (c_float*10)(*b)

try:
    u = (c_float*2).from_address(ao(a[6]))
    su = sizeof(u._type_)
    print "0x%8.8x" % ao(u)
    print "%10.7f %10.7f" % (u[0],u[1])
except TypeError:
    print "x_float does not work"

try:
    v = fa(a,6,2,c_float)
    sv = sizeof(v._type_)
except TypeError:
    print "x_float does not work"

try:
    w = fa(z,6,2)
    sw = sizeof(w._type_)
except TypeError:
    print "c_float does not work"

sz = sizeof(z._type_)
for i in range(len(v)):
    print "0x%8.8x 0x%8.8x 0x%8.8x 0x%8.8x" % (
    ao(a[i+6]),ao(v)+i*sv,ao(z)+(i+6)*sz,ao(w)+i*sw)
    print "%10.7f %10.7f %10.7f % 10.7f" % (a[i+6].value,v[i],z[i+6],w[i])
