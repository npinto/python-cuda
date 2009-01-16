#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08

import numpy as nc
import psyco
import os,sys
psyco.full()

doplot = True
usepng = True

if doplot:
    import matplotlib
    matplotlib.use("Agg")
from pylab import *


ifile = sys.argv[1]

if usepng:
    ofile = "%s.png" % ifile
else:
    ofile = "%s.ps" % ifile

f1 = open(ifile).read().splitlines()

deltay = 1000
title1 = "POLY10"
title2 = "Flops"

l = len(f1)
dn,d1,d2,d3 = [],[],[],[]
for f in f1:
    vx,j1,j2,c,g = f.split()
    dn.append(int(vx))
    d1.append(int(float(c)+.5))
    d2.append(int(float(g)+.5))

dn = nc.asarray(dn)
d1 = nc.asarray(d1)
d2 = nc.asarray(d2)

xn = dn.min()
xx = dn.max()+1
dy = max(d1.max(),d2.max())
dy = ((dy+deltay-1)/deltay)*deltay

x_ticks = [n for n in arange(xn,xx)]
x_label = ["%d" % n for n in x_ticks]
y_ticks = [i for i in range(0,dy+deltay,deltay)]
y_label = ["%d" % n for n in y_ticks]

rc("font",family="sans-serif",weight="medium",size=12)
rc("lines",linewidth=1)
if doplot and not usepng:
    rc("figure",facecolor=(.8,.8,.7),dpi=300)
else:
    rc("figure",facecolor=(.8,.8,.7),dpi=100)
if usepng:
    rc("savefig",facecolor=(.8,.8,.7),dpi=100,orientation="landscape")
else:
    rc("savefig",facecolor=(.8,.8,.7),dpi=300,orientation="landscape")
rc("axes",facecolor=(.85,.85,1.))
rc("text",usetex=True)

figure(1)

line1 = plot(dn,[y for y in d1],"r",linewidth=1)
line2 = plot(dn,[y for y in d2],"g",linewidth=1)

title("CUDA MegaFlops CPU+GPU %s vs Vector Length" % title1)

legend ((line1,line2),("CPU","GPU"),loc=4)

xticks(x_ticks,x_label)
#xlabel(r"{$Loop Length$}",fontsize=18)
xlabel(r"{$log_{2}(Vector Length)$}",fontsize=18)
yticks(y_ticks,y_label)
ylabel(r"$Mega%s$" % title2,fontsize=18)

grid(True)

if doplot:
    if not usepng:
        savefig(ofile,dpi=300,orientation="landscape")
    else:
        savefig(ofile,dpi=100,orientation="landscape")
show()
if doplot and False:
    os.system("ps2pdf %s" % ofile)
    os.system("rm %s" % ofile)
