#!/usr/bin/env python

import re
from sys import argv,exit
from os import waitpid
from os.path import basename,dirname
from subprocess import Popen, PIPE
from optparse import OptionParser

def fix_cdll_imports(bindings, lib):
    cdll_regex = "_libraries['%s'] = CDLL('%s')" % (lib,lib)
    new_bindings = []
    libname = basename(lib).split('.')[0].replace('lib','')
    for line in bindings:
        if line.rfind(cdll_regex) != -1:
            new_bindings.append(line.replace(cdll_regex, "_libraries['%s'] = get_lib('%s')" % (libname,libname)))
        else:
            new_bindings.append(line.replace(lib,libname))
    new_bindings.insert(3,'from cuda.utils import get_lib\n')
    new_bindings.insert(4,'c_longdouble = c_double\n')
    return new_bindings

def main(args=None):
    """ Autogenerates ctype'd python version of a shared library. 
        Takes an array of arguments in the same format as sys.argv

        -H header_file
        -l library_file
        -I include_dir
        -x xml_output_file
        -p python_output-file
    """  
    if args is None:
        args = argv

    usage = "usage: %prog -H header_file -l library_file [options]"
    parser = OptionParser(usage)
    parser.add_option("-H","--header-file",
                      dest="HEADER_FILE",
                      help="(required)full path to header file (e.g /usr/local/cuda/include/cuda.h)",
                      default = None)

    parser.add_option("-l","--library-file",
                      action="append",
                      dest="LIBRARIES",
                      help="(required)full path to library file (e.g. /usr/local/cuda/lib/libcuda.so) ",
                      default = [])

    parser.add_option("-I","--include-dir",
                      action="append",
                      dest="INCLUDE_DIRS",
                      help="paths to find dependency headers (e.g /usr/local/cuda/include/)",
                      default = [])

    parser.add_option("-x","--xml-output-file",
                      dest="XML_FILE",
                      help="file to store xml output to (e.g. /my/project/xml/cuda.xml)",
                      default = None)

    parser.add_option("-p","--python-output-file",
                      dest="PYTHON_FILE",
                      help="file to store python bindings to (e.g. /my/project/cuda.py)",
                      default = None)

    (options,args) = parser.parse_args(args[1:])
    
    if options.HEADER_FILE is None or options.LIBRARIES is None: 
        print 'ERROR: You must supply both the header file (-H) and the library file (-l)'
        exit()

    if options.XML_FILE is None:
        if options.HEADER_FILE is not None:
            options.XML_FILE = basename(options.HEADER_FILE) + '.xml'
        else:
            options.XML_FILE = "output.xml"

    if options.PYTHON_FILE is None:
        if options.LIBRARIES is not None:
            options.PYTHON_FILE = basename(options.LIBRARIES) + '.py'
        else:
            options.PYTHON_FILE = "output.py"

    if len(options.INCLUDE_DIRS) == 0:
        options.INCLUDE_DIRS = dirname(options.HEADER_FILE)
    else:
        options.INCLUDE_DIRS = ""
        for inc in options.INCLUDE_DIRS:
            options.INCLUDE_DIRS += ' -I %s ' % inc
    
    library_options = []
    for lib in options.LIBRARIES:
        library_options.append('-l %s' % lib)

    config = {'HEADER_FILE':options.HEADER_FILE, 
              'XML_FILE': options.XML_FILE, 
              'PYTHON_FILE': options.PYTHON_FILE, 
              'INCLUDE': options.INCLUDE_DIRS}

    h2xmlcmd = 'python -m ctypeslib.h2xml %(HEADER_FILE)s' + \
        ' -I %(INCLUDE)s -o %(XML_FILE)s'
    h2xmlcmd = h2xmlcmd % config
    print "h2xmlcmd =", h2xmlcmd
    p = Popen(h2xmlcmd, shell=True)
    sts = waitpid(p.pid, 0)

    # XXX: hack to remove Converter tags from the xml file
    # before Justin's patch gets accepted in ctypeslib
    lines = [line for line in open(config['XML_FILE']).readlines() 
             if not line.startswith('  <Converter id="_')]
    open(config['XML_FILE'], 'w+').writelines(lines)
    
    xml2pycmd = ('python -m ctypeslib.xml2py %(XML_FILE)s -c -d -v ' \
                     % config) + ' '.join(library_options)
    print "xml2pycmd = %s" % xml2pycmd
    p = Popen(xml2pycmd, shell=True, stdin=PIPE, stdout=PIPE, close_fds=True)
    bindings = p.stdout.readlines()

    for lib in options.LIBRARIES:
        bindings = fix_cdll_imports(bindings, lib)

    file = open(config['PYTHON_FILE'], 'w')
    file.writelines(bindings)
    file.close()

if __name__ == "__main__":
    main()
