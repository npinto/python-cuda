#!/bin/env python

from ctypes import *
from ctypes_array import *
from numpy import *
from numpy.random import rand

def pif(a):
    ai = a.__array_interface__
    b = ai["strides"] is not None
    return " ".join([
        "%-5s:" % str(b),
        ":",className(a),
        str(ai["strides"]),
        str(ai["shape"]),])

ac = rand(4,8)
af = array(ac,order="F")#ac.T

def norm2(a):
    return round(sqrt(sum(a*a)),3)

print "\nOriginal arrays ac, af = ac.T"
print "hasStrides(ac):",pif(ac)
print "hasStrides(af):",pif(af)
print "type ac,af",type(ac),type(af)

print "\nConvert to ctypes: ac => bc, af => bf"
bc = convert(ac); print "hasStrides(bc):",pif(bc)
bf = convert(af); print "hasStrides(bf):",pif(bf)
print "\ntype bc,bf",type(bc),type(bf)

print "\nCan combine numpy arrays and ctypes objects with array interface"
delta = (ac-bc).flatten()
print "L2-norm ac-bc",norm2(delta)
delta = (af-bf).flatten()
print "L2-norm af-bf",norm2(delta)

print "\nConvert to numpy: bc => cc, bf => cf, bf => CF (Fortran)"
cc = convert(bc);print "hasStrides(cc):",pif(cc)
cf = convert(bf);print "hasStrides(cf):",pif(cf)
CF = convert(bf,order="F"); print "hasStrides(CF):",pif(CF)
print "\ntype cc,cf,CF",type(cc),type(cf),type(CF)

delta = (af-CF).flatten()
print "L2-norm af-CF",norm2(delta)

print "\nConvert to ctypes ac => dc, af => df (dc,df = 1D)"
dc = (eval(typeName(bc))*ac.size)()
df = (eval(typeName(bf))*af.size)()
convert(ac,None,None,dc)
convert(af,None,None,df)
print "type dc,df",type(dc),type(df)

delta = (ac-dc).flatten()
print "L2-norm ac-dc",norm2(delta)
delta = (af-df).flatten()
print "L2-norm af-df",norm2(delta)
delta = af.flatten()-ac.flatten()
print "\ncomparing flattened ac,af"
print "L2-norm af-ac",norm2(delta)
set_printoptions(precision=3)
print "\nac[:3],af[:3]"
print ac.flatten()[:3]
print af.flatten()[:3]
print "\ndc[:3],df[:3]"
print "[%6.3f %6.3f %6.3f]" % tuple(dc[:3])
print "[%6.3f %6.3f %6.3f]" % tuple(df[:3])

print "\nConvert to numpy: dc => ec, df => ef"
ec = convert(dc,(4,8),"C")
ef = convert(df,(4,8),"F")
print "1D dc,df ctypes objects=> 2D numpy arrays ec,ef"
print ; print "ac, ec"
print pif(ac)
print pif(ec)
print ; print "af, ef"
print pif(af)
print pif(ef)
print "\nL2-norm ac-ec, af-ef, ec-ef"
print norm2(ac-ec)
print norm2(af-ef)
print norm2(ec-ef)
print "\nL2-norm ac-ef, af-ec (flattened)"
print norm2(ac.flatten()-ef.flatten())
print norm2(af.flatten()-ec.flatten())


