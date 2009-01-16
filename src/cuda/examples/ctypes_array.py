# coding:utf-8: © Arno Pähler, 2007-08
# important details are © Thomas Heller
from sys import byteorder
from ctypes import *
from numpy.core.multiarray import array as multi_array
from numpy import isfortran

__all__ = ["convert","className","typeName"]

################################################################
# some shortcut utilities

# use this as eval(PRODUCT,whatever)
PRODUCT = "reduce(lambda x,y:x*y,%s,1)"

def className(o):
    o = type(o)
    c = o.__class__.__name__
    # if gone too far, back up one level
    if c == "type":
        c = o.__name__
    return c

def typeName(o):
    if isSimpleType(o):
        return o.__class__.__name__
    while not isSimpleType(o):
        try:
            o = o._type_
        except AttributeError:
            o = type(o)
            break
    return o.__name__

def isCtypesFamily(o):
    c_n = "ArrayType","PointerType","SimpleType"
    cn = o.__class__.__name__
    isObject = cn[:2] == "c_" or cn[:5] =="LP_c_"
    isType = cn in c_n
    return isObject or isType

def isNumpyArray(o):
    return className(o) == "ndarray"

def isArrayType(o):
    return className(o) == "ArrayType"

def isSimpleType(o):
    return className(o) == "SimpleType"

if byteorder == "little":
    T = "<"
else:
    T = ">"

c_Dict = {
    "c_byte"      : "%si1" % T,
    "c_short"     : "%si2" % T,
    "c_int"       : "%si4" % T,
    "c_long"      : "%si4" % T,
    "c_longlong"  : "%si8" % T,
    "c_ubyte"     : "%su1" % T,
    "c_ushort"    : "%su2" % T,
    "c_uint"      : "%su4" % T,
    "c_ulong"     : "%su4" % T,
    "c_ulonglong" : "%su8" % T,
    "c_float"     : "%sf4" % T,
    "c_double"    : "%sf8" % T,}

n_Dict = dict(
    [(v,eval(k)) for k,v in c_Dict.items()])

################################################################
# public functions

def convert(obj,dims=None,order="C",out=None):
    """Converts ctypes array to numpy array and vice versa
    convert determines the input type (ctypes or numpy)
    internally and returns an object of the opposite type.

    NOTE: do NOT do the following:
    n1 = numpy_array; c1 = convert(n1); n1 = convert(n1)
                                        ^^           ^^
    nasty things will happen! it is ok to do
    n1 = numpy_array; c1 = convert(n1); n2 = convert(n1)
                                        ^^           ^^
    A 1D ctypes array ca can be converted to a (m,n,k,...)
    numpy array na in C order with convert(ca,(m,n,k,...),"C")
    and to na in F order with convert(ca,(m,n,k,...),"F").
    (m,n,k,...) is reversed internally and the order of
    matrix-matrix or matrix-vector nultiplication must be
    inverted, when comparing with C oder results.

    This code is based on similar code posted by Thomas Heller"""

    # for obj in simple c_types (e.g.c_float(1.))
    if isSimpleType(obj):
        """convert simple ctype to numpy array"""
        obj = obj.value,
        return multi_array(obj,copy=False)

    # for obj in scalars(e.g. 1.), lists and tuples
    if not (isCtypesFamily(obj) or isNumpyArray(obj)):
        """convert Python scalar, list or tuple to numpy array"""
        obj = tuple(obj)
        return multi_array(obj,copy=False)

    # numpy ==> ctypes
    if isNumpyArray(obj):
        """convert numpy array to ctypes array"""
        do_copy = False

        # if obj is C order and return object should be
        # Fortran order, transpose obj for Fortran order
        if not isfortran(obj) and order == "F":
            obj = obj.T

        ai = obj.__array_interface__
        if ai["strides"]:
            pass
            # do something sensible
#            obj = obj.T
#            ai = obj.__array_interface__

        addr,readonly = ai["data"]
        if readonly: # make a copy
            do_copy = True

        ## code below should consider strides
        i_size = obj.itemsize
        if out is None:
#            print "SIZE",eval(PRODUCT % "obj.shape")
            t = n_Dict[ai["typestr"]]
            for dim in ai["shape"][::-1]:
                t = t*dim
            if do_copy:
                out = t()
                memmove(out,addr,obj.size*i_size)
            else:
                out = t.from_address(addr)
            out.__array_interface__ = ai
            out.__keep = ai
            return out
        else:
            size1 = obj.size
            size2 = len(out)
            size  = min(size1,size2)*i_size
            memmove(out,addr,size)
            out.__array_interface__ = ai
            out.__keep = ai
            return out
    # ctypes ==> numpy
    else:
        """convert ctypes array to numpy array"""
        typestr = c_Dict[typeName(obj)]
        strides = None
        if dims is None:
            shape = []
            o = obj
            while isArrayType(o):
                shape.append(o._length_)
                o = o._type_
            shape = tuple(shape)
        else:
            shape = tuple(dims)
            p = sizeof(eval(typeName(obj)))
            products = [p]
            for d in dims[:-1]:
                p *= d
                products.append(p)
            if order == "F":
                strides = tuple(products)

        ao = addressof(obj)
        ai = \
            {
            'descr'  : [('',typestr)],
            '__ref'  : ao,
            'strides': strides,
            'shape'  : shape,
            'version': 3,
            'typestr': typestr,
            'data'   : (ao,False)
            }
        obj.__array_interface__ = ai

        return multi_array(obj,copy=False)
