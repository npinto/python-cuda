#!/usr/bin/env python

import re
import os, sys
from elementtree import ElementTree
from optparse import OptionParser
from subprocess import Popen, PIPE

def fix_cdll_imports(bindings, lib):
    cdll_regex = "_libraries['%s'] = CDLL('%s')" % (lib,lib)
    new_bindings = []
    libname = os.path.basename(lib).split('.')[0].replace('lib','')
    for line in bindings:
        if line.rfind(cdll_regex) != -1:
            new_bindings += [line.replace(cdll_regex, "_libraries['%s'] = get_lib('%s')" % (libname,libname))]
        else:
            new_bindings += [line.replace(lib,libname)]
    new_bindings.insert(3,'from cuda.utils import get_lib\n')
    new_bindings.insert(4,'c_longdouble = c_double\n')
    return new_bindings

def clean_only_headers(xml_filename, only_headers):
    """ 
    removes definitions that are not defined in one of only_headers files

    fd - file descriptor for xml file
    only_headers - list of header files to check definitions against

    """
    gcc_xml = ElementTree.parse(xml_filename)
    root = gcc_xml.getroot()
    fields = root.getchildren()[:]
    files = root.findall('File')
    oheaders = {}

    for file in files:
        filename = file.get('name')
        if only_headers.has_key(filename):
            oheaders[file.get('id')] = file.get('name')
        else:
            for ohead in only_headers.keys():
                if re.compile(ohead,re.IGNORECASE).search(filename):
                    oheaders[file.get('id')] = file.get('name')

    print oheaders
        
    for field in fields:
        file = field.get('file')
        if file is not None:
            if not oheaders.has_key(file):
                root.remove(field)
    gcc_xml.write(xml_filename)

def main(args=None):
    """ Autogenerates ctype'd python version of a shared library. 
        Takes an array of arguments in the same format as sys.argv

        -H header_file
        -l library_file
        -I include_dir
        -o only_headers
        -x xml_output_file
        -p python_output-file
    """  
    if args is None:
        args = sys.argv

    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("-H","--header",
                      action="append",
                      dest="HEADERS",
                      help="[REQUIRED] full path to header (e.g /path/to/foo.h)",
                      default = [])

    parser.add_option("-l","--library",
                      action="append",
                      dest="LIBRARIES",
                      help="[REQUIRED] full path to library (e.g /path/to/libfoo.so)",
                      default = [])

    parser.add_option("-I","--include-dir",
                      action="append",
                      dest="INCLUDE_DIRS",
                      help="[REQUIRED] paths to find dependency headers (e.g /usr/local/cuda/include/)",
                      default = [])

    parser.add_option("-o","--only-headers",
                      action="append",
                      dest="ONLY_HEADERS",
                      help="[OPTIONAL] include only definitions found in specified headers (e.g /usr/include/example.h)",
                      default = [])

    parser.add_option("-x","--xml-output-file",
                      dest="XML_FILE",
                      help="file to store xml output to (e.g. /my/project/xml/cuda.xml)",
                      default = "output.xml")

    parser.add_option("-p","--python-output-file",
                      dest="PYTHON_FILE",
                      help="file to store python bindings to (e.g. /my/project/cuda.py)",
                      default = "output.py")

    options, args = parser.parse_args()
        
    if len(options.HEADERS) == 0 or \
            len(options.LIBRARIES) == 0 or \
            len(options.INCLUDE_DIRS) == 0:
        raise ValueError, "You must supply the header files (-H), the library files (-l) and the include dirs (-I)"

    headers = " ".join(options.HEADERS)
    libraries = "-l " + " -l ".join(options.LIBRARIES)
    include_dirs = "-I " + " -I ".join(options.INCLUDE_DIRS)
    xml_file = options.XML_FILE
    python_file = options.PYTHON_FILE

    # -- h2xml
    h2xmlcmd_l = ["python -m ctypeslib.h2xml"]
    h2xmlcmd_l += [headers]
    h2xmlcmd_l += [include_dirs]
    h2xmlcmd_l += ["-o %s" % xml_file]
    h2xmlcmd = " ".join(h2xmlcmd_l)
    print "h2xmlcmd =", h2xmlcmd
    assert os.system(h2xmlcmd) == 0

    # XXX: hack to remove Converter tags from the xml file
    lines = [line for line in open(xml_file).readlines() 
             if not line.startswith('  <Converter id="_')]

    fd = open(xml_file, 'w+')
    fd.writelines(lines)
    fd.close()

    if len(options.ONLY_HEADERS) != 0:
        only_headers = {}.fromkeys(options.ONLY_HEADERS, 1)
        clean_only_headers(xml_file, only_headers)

    # -- xml2py
    xml2pycmd_l = ["python -m ctypeslib.xml2py"]
    xml2pycmd_l += [xml_file]
    xml2pycmd_l += ["-c -d -v"]
    xml2pycmd_l += [libraries]
    xml2pycmd = " ".join(xml2pycmd_l)
    print "xml2pycmd =", xml2pycmd
    p = Popen(xml2pycmd, shell=True, stdin=PIPE, stdout=PIPE, close_fds=True)
    bindings = p.stdout.readlines()

    for lib in options.LIBRARIES:
        bindings = fix_cdll_imports(bindings, lib)

    open(python_file, 'w').writelines(bindings)

if __name__ == "__main__":
    main()
