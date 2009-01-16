# coding:utf-8: © Arno Pähler, 2007-08

from bz2 import BZ2File
from gzip import GzipFile
from collections import defaultdict
from ctypes import memmove
from os.path import splitext
from subprocess import Popen,PIPE
from sys import platform
from time import ctime,time

try:
    from numpy import empty,float32
    def c2n_array(a,m,n=1):
        ##  scipy needs array in Fortran order
        aa = empty((m,n),dtype=float32,order='F')
        memmove(aa.ctypes.data,a,4*(m*n))
        return aa
    def n2c_array(a,aa):
        ##  aa should already have been allocated!
        memmove(aa,a.ctypes.data,4*a.size)
        return aa
except ImportError:
    def c2n_array(a,m,n=1):
        return a
    def n2c_array(a,aa):
        aa = a
        return aa

from cpuFunctions import ReadTimestampCounter

class Timer(object):

    fakeClock = 1.e-9/2.8 # denominator: clock in GHz

    def __init__(self,startNow=False,useTSC=False):
        self.useTSC = useTSC
        self.wall = 0. ## later
        self.cpu  = 0. ## later
        self.running = False
        self.torg = 0.
        self.time = 0.
        self.counters = []
        self.freqs = None

        if useTSC and platform == "linux2":
            self.getFrequency(0)
        elif useTSC and platform == "win32":
            self.rtfreq = self.fakeClock
        elif useTSC:
            self.rtfreq = self.fakeClock

        if startNow:
            self.start()

    def __str__(self):
        pf = dict((
                  ("linux2","Linux"),
                  ("win32","Windows")))
        s =["System: %s" % pf[platform]]
        useTSC = self.useTSC
        s.append("UseTSC: %s" % bool(useTSC))
        if useTSC:
            s.append("Cores : %d" % len(self.freqs))
            clock = 1.e-9/self.rtfreq
            s.append("Clock : %.3f GHz" % clock)
        return "\n".join(s)

    def getFrequency(self,core=0):
        if self.freqs is None:
            cpuinfo = "/proc/cpuinfo"
            freqs = []
            self.core = core
            for line in open(cpuinfo):
                if line.startswith("cpu MHz"):
                    freq = float(line.split(":")[1])
                    freqs.append(1.e-6/freq)
            self.freqs = freqs
            self.rtfreq = freqs[core]
        else:
            self.rtfreq = freqs[core]
        return self.rtfreq

    def getTime(self):
        if self.useTSC:
            t = ReadTimestampCounter()*rfreq
        else:
            t = time()
        return t-self.torg

    def start(self):
        if self.useTSC and self.freqs is None:
            self.getFrequency()
        self.running = True
        self.torg = self.getTime()
        self.time = 0.
        self.counters = []

    def split(self):
        t = self.time
        if self.running:
            t = self.getTime()
        self.time = t
        self.counters.append(t)
        return t

    def read(self,all = True):
        o = self.torg
        t = self.time
        if self.running:
            t = self.getTime()
        self.time = t
        if all:
            return o,t,self.counters
        else:
            return o,t

    def reset(self):
        self.start()

    def stop(self):
        if self.running:
            self.running = False
            t = self.getTime()
            self.time = t
            self.torg = 0.
            return t

BSZ = 1024

## system execution of 'command' with imput 'input' to 'command'
def System(command,input = ""):
    """system execution of 'command' with input 'input' to 'command'
    Returns tuple (status,output to stdout,output to stderr)
    with outputs split on newlines and returned as lists."""
    if platform == "win32":
        run = Popen(command,shell = True,bufsize = BSZ,
              stdin = PIPE,stdout = PIPE,stderr = PIPE)
    else:
        run = Popen(command,shell = True,bufsize = BSZ,
              stdin = PIPE,stdout = PIPE,stderr = PIPE,
              close_fds = True)
    if input != "":
        run.stdin.write(input+"\n")
    runOutput = run.stdout.read().splitlines()
    runErrors = run.stderr.read().splitlines()
    status = run.wait()
    return status,runOutput,runErrors

## allow to open ordinary, gzipped, bzipped files
def xOpen(name,mode = "r"):
    """open file depending on extension (.gz,.bz2) so that ordinary
    as well as compressed files can be opened with the same syntax.
    Returns a Python file object."""
    extension = splitext(name)[-1]
    if extension == ".gz":
      file = GzipFile(name,mode)
    elif extension == ".bz2":
        file = BZ2File(name,mode)
    else:
       file = open(name,mode)
    return file

## print timing info: t0 = start time, t1 = final time
def printTiming(t0,t1):
    """ Given start time t0 and end time t1 in seconds,
    as returned by time.time(),
    print a nice representation like

    Started  : Mon Jan  7 23:18:21 2008
    Finished : Mon Jan  7 23:18:22 2008
    Elapsed  : 0.9 (00:00:00.9)

    Elapsed time as given in seconds and as (hh:mm:ss.s)"""

    dt = t1-t0
    dh = int(dt/3600.)
    du = dt-float(3600*dh)
    dm = int(du/60.)
    ds = du-float(60*dm)

    print "\nStarted  : %s" % ctime(t0)
    print "Finished : %s" % ctime(t1)
    print "Elapsed  : %.1f (%2.2d:%2.2d:%04.1f)\n" % (dt,dh,dm,ds)

## invert a dictionary swapping keys and values
def invertDict(org_dict):
    """Invert a dictionary of (key,val) and returns a dictionary (val,key).
    Fails if val is mutable, i.e. ALL vals must be immutable, e.g. strings."""
    invdict = defaultdict(list)
    for k in org_dict:
        v = org_dict[k]
        invdict[v].append(k)
    return invdict
