#!/usr/bin/env python
from os import waitpid
from os.path import basename
from subprocess import Popen, PIPE
from optparse import OptionParser

usage = "usage: %prog [options]"
parser = OptionParser(usage)
parser.add_option("-H","--header-file",
                  dest="HEADER_FILE",
                  help="full path to header file (e.g /usr/local/cuda/include/cuda.h)",
                  default = None)

parser.add_option("-l","--library-file",
                  dest="LIBRARY",
                  help="full path to library file (e.g. /usr/local/cuda/lib/libcuda.so) ",
                  default = None)

parser.add_option("-x","--xml-output-file",
                  dest="XML_FILE",
                  help="file to store xml output to (e.g. /my/project/xml/cuda.xml)",
                  default = None)

parser.add_option("-p","--python-output-file",
                  dest="PYTHON_FILE",
                  help="file to store python bindings to (e.g. /my/project/cuda.py)",
                  default = None)
(options,args) = parser.parse_args()

if options.XML_FILE is None:
    if options.HEADER_FILE is not None:
        options.XML_FILE = basename(options.HEADER_FILE) + '.xml'
    else:
        options.XML_FILE = "output.xml"

if options.PYTHON_FILE is None:
    if options.LIBRARY is not None:
        options.PYTHON_FILE = basename(options.LIBRARY) + '.py'
    else:
        options.PYTHON_FILE = "output.py"

config = {'HEADER_FILE':options.HEADER_FILE, 'LIBRARY': options.LIBRARY, 'XML_FILE': options.XML_FILE, 'PYTHON_FILE': options.PYTHON_FILE}

p = Popen('python -m ctypeslib.h2xml %(HEADER_FILE)s -o %(XML_FILE)s' % config, shell=True)
sts = waitpid(p.pid, 0)

p = Popen('python -m ctypeslib.xml2py %(XML_FILE)s -c -d -v -l %(LIBRARY)s' % config, shell=True, stdin=PIPE, stdout=PIPE, close_fds=True)
bindings = p.stdout.readlines()

file = open(config['PYTHON_FILE'], 'w')
file.writelines(bindings)
file.close()
