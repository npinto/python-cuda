# coding:utf-8: © Arno Pähler, 2007-08

from ctypes import CDLL
from math import *

vml = CDLL("./_vector.so")

vcos = vml.vsCos
vcos.restype = None

vsin = vml.vsSin
vsin.restype = None

vsincos = vml.vsSinCos
vsincos.restype = None

vexp = vml.vsExp
vexp.restype = None

vlog = vml.vsLn
vlog.restype = None

vlog10 = vml.vsLog10
vlog10.restype = None

vsqrt = vml.vsSqrt
vsqrt.restype = None

def cpuTRIG(h_Y,h_Z,h_X):
    size = len(h_X)
    if False:
        vcos(size,h_X,h_Y)
        vsin(size,h_X,h_Z)
    else: # about 20% faster
        vsincos(size,h_X,h_Z,h_Y)

##////////////////////////////////////////////////////////////////////////////
## Shared CPU/GPU functions, performing calculations for single option by 
## Black-Scholes formula.
##////////////////////////////////////////////////////////////////////////////
A1 =  0.319381530
A2 = -0.356563782
A3 =  1.781477937
A4 = -1.821255978
A5 = 1.3302744290
RSQRT2PI = 0.3989422804

##Polynomial approximation of cumulative normal distribution function
def CND(d):
    K = 1.0 / (1.0 + 0.2316419 * abs(d))

    cnd = RSQRT2PI * exp(-0.5 * d * d) * \
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))

    if d > 0:
        cnd = 1.0 - cnd

    return cnd

## Calculate Black-Scholes formula for both calls and puts
##    S, ##Stock price
##    X, ##Option strike
##    T, ##Option years
##    R, ##Riskless rate
##    V  ##Volatility rate
def BlackScholesBody(
    S, X, T, R, V ):
    sqrtT = sqrt(T)
    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT

    CNDD1 = CND(d1)
    CNDD2 = CND(d2)

    ##Calculate Call and Put simultaneously
    expRT = exp(- R * T)
    CallResult = S * CNDD1 - X * expRT * CNDD2
    PutResult  = X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1)

    return CallResult,PutResult

def cpuBLSC(h_C,h_P,h_S,h_X,h_T,R,V,size):
    for i in range(size):
        h_C[i],h_P[i] = BlackScholesBody(h_S[i],h_X[i],h_T[i],R,V)
