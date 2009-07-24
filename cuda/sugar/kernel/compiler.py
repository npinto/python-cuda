from ctypes import cdll
import commands
from subprocess import Popen, PIPE
from cuda.utils import memoize

class CompileError(Exception):
    pass

@memoize
def get_nvcc_version(nvcc):
    try:
        return Popen([nvcc, "--version"], stdout=PIPE).communicate()[0]
    except OSError, e:
        raise OSError, "%s was not found (is it on the PATH?) [%s]" % (
                nvcc, str(e))

def compile_plain(source, options, keep, nvcc, cache_dir):
    from os.path import join
    from platform import architecture

    if architecture()[0] == "64bit":
        options.insert(0,"-Xcompiler='-fPIC'")

    if cache_dir:
        try:
            import hashlib
            checksum = hashlib.md5()
        except ImportError:
            # for Python << 2.5
            import md5
            checksum = md5.new()

        checksum.update(source)
        for option in options: 
            checksum.update(option)
        checksum.update(get_nvcc_version(nvcc))

        cache_file = checksum.hexdigest()
        cache_path = join(cache_dir, cache_file + ".so")

        try:
            #return open(cache_path, "r").read()
            return cdll.LoadLibrary(cache_path)
        except:
            pass

    from tempfile import mkdtemp
    file_dir = mkdtemp()
    file_root = "kernel"

    cu_file_name = file_root + ".cu"
    cu_file_path = join(file_dir, cu_file_name)

    options.append("-o")
    options.append("%s.so" % join(file_dir,file_root))

    outf = open(cu_file_path, "w")
    outf.write(str(source))
    outf.close()

    if keep:
        options = options[:]
        options.append("--keep")

        print "*** compiler output in %s" % file_dir

    #from pytools.prefork import call
    try:

        print "Compiling kernel using the following options: " 
        print ' '.join([nvcc, "--shared"] + options + [cu_file_name])

        process = Popen([nvcc, "--shared"] + options + [cu_file_path], stdout=PIPE, cwd=file_dir)
        output = process.communicate()[0]
        result = process.returncode

        if output:
            print 'Compiler output below:'
            print output

    except OSError, e:
        raise OSError, "%s was not found (is it on the PATH?) [%s]" % (
                nvcc, str(e))

    if result != 0:
        raise CompileError, "nvcc compilation of %s failed" % cu_file_path

    kdll = open(join(file_dir, file_root + ".so"), "r").read()

    if cache_dir:
        outf = open(cache_path, "w")
        outf.write(kdll)
        outf.close()

    if not keep:
        from os import listdir, unlink, rmdir
        for name in listdir(file_dir):
            unlink(join(file_dir, name))
        rmdir(file_dir)

    kdll = cdll.LoadLibrary(cache_path)

    return kdll

def compile(source, nvcc="nvcc", options=[], keep=False,
        no_extern_c=False, arch=None, code=None, cache_dir=None,
        include_dirs=[]):

    if not no_extern_c:
        source = 'extern "C" {\n%s\n}\n' % source

    options = options[:]
    if arch is None:
        try:
            # todo replace this with python-cuda equivalent
            #from pycuda.driver import Context 
            #arch = "sm_%d%d" % Context.get_device().compute_capability()
            arch = None
        except RuntimeError:
            pass

    if cache_dir is None:
        from os.path import expanduser, join, exists
        import os
        try:
            getattr( os , 'getuid' )	
        except:
            def getuid():
                return os.getenv('USERNAME')
            os.getuid = getuid

        from tempfile import gettempdir
        cache_dir = join(gettempdir(), 
                "python-cuda-compiler-cache-v1-uid%s" % os.getuid())

        if not exists(cache_dir):
            from os import mkdir
            mkdir(cache_dir)

    if arch is not None:
        options.extend(["-arch", arch])

    if code is not None:
        options.extend(["-code", code])

    include_dirs = include_dirs[:]

    for i in include_dirs:
        options.append("-I"+i)

    return compile_plain(source, options, keep, nvcc, cache_dir)
