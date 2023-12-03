import functools
import operator
import os
import subprocess
from collections import defaultdict
import sys
import numpy as np

def _dummyimport():
    import Cython

try:
    from .cythonproduct import create_product

except Exception as e:
    cstring = r"""# distutils: language=c
# distutils: extra_compile_args=/openmp /O2
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3
# cython: initializedcheck=False

from cython.parallel cimport prange
cimport cython
import numpy as np
cimport numpy as np
import cython
ctypedef fused real:
    cython.char
    cython.schar
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.size_t
    cython.Py_ssize_t
    cython.float
    cython.double
    cython.longdouble
    cython.floatcomplex
    cython.doublecomplex
    cython.longdoublecomplex
    cython.Py_hash_t
    cython.Py_UCS4
cpdef void create_product(cython.Py_ssize_t liste, cython.Py_ssize_t[:] indlist, real[:] ctypesframedataresults, real[:] ear):
    cdef cython.Py_ssize_t geht,bleibt,zahlx,q,geht1,zahl
    cdef cython.Py_ssize_t indlistlen =len(indlist)
    for q in prange(liste,nogil=True):
        geht = q
        bleibt = 0
        for zahlx in range(indlistlen):
            zahl=indlist[zahlx]
            geht1 = geht // zahl
            bleibt = geht % zahl
            geht = geht1
            ctypesframedataresults[q*indlistlen+zahlx] = ear[bleibt]
"""
    pyxfile = f"cythonproduct.pyx"
    pyxfilesetup = f"cythonproductmapcompiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
            """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'cythonproduct', 'sources': ['cythonproduct.pyx'], 'include_dirs': [\'"""
            + numpyincludefolder
            + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='cythonproduct',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    try:
        from .cythonproduct import create_product

    except Exception as fe:
        sys.stderr.write(f'{fe}')
        sys.stderr.flush()



def get_pointer_array(original):
    r"""
    Obtain a flat pointer array from a NumPy array.

    Parameters:
    -----------
    original : numpy.ndarray
        The original NumPy array for which a flat pointer array is needed.

    Returns:
    --------
    numpy.ndarray
        Flat pointer array obtained from the original array.

    Notes:
    ------
    The flat pointer array is obtained using the ctypes module, allowing direct
    access to the underlying memory without the need for additional copying. This
    results in memory savings, especially when dealing with large arrays, as it
    avoids unnecessary duplication of data. And we can use some tricks to pass any
    np.ndarray of any shape to a compiled Python function

    Example:
    --------
    >>> original_array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> pointer_array = get_pointer_array(original_array)
    >>> print(pointer_array)
    array([1, 2, 3, 4, 5, 6])
    """
    dty = np.ctypeslib.as_ctypes_type(original.dtype)

    b = original.ctypes.data
    buff = (dty * original.size).from_address(b)

    aflat = np.frombuffer(buff, dtype=original.dtype)
    return aflat

def  cartesian_product(*args,outputdtype=None):
    r"""
    Generate the Cartesian product of input iterables.

    Parameters:
    -----------
    *args : list of iterable
        Input iterables for which the Cartesian product will be computed.
    outputdtype (optional) :
        dtype of output array
    Returns:
    --------
    numpy.ndarray
        2D array containing the Cartesian product of the input iterables.

    Notes:
    ------
    This function efficiently computes the Cartesian product of the input iterables
    using Cython implementation. It outperforms the equivalent functionality provided
    by itertools.product, and returns a NumPy array (not a list of tuples like itertools.product).

    Examples:
    ---------
        # Mem usage 2 GB
        # Out[4]:
        # array([[0, 0, 0, ..., 0, 0, 0],
        #        [1, 0, 0, ..., 0, 0, 0],
        #        [2, 0, 0, ..., 0, 0, 0],
        #        ...,
        #        [5, 7, 7, ..., 7, 7, 7],
        #        [6, 7, 7, ..., 7, 7, 7],
        #        [7, 7, 7, ..., 7, 7, 7]], dtype=uint8)
        # %timeit dataresults=cartesian_product(*args2)
        # 2.65 s ± 163 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # dataresults.shape
        # Out[6]: (134217728, 9)


        # itertools.product
        # Mem usage 16 GB

        # import itertools
        # %timeit (list(itertools.product(*args2)))
        # 11.5 s ± 203 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


        # --------------------------------------------------------------------------
        # Mem usage 1.2 GB
        # args = [[411, 231.33, 4342, 12341, 1.142, 1.33, 13],
        #         [34, 231.33, 4132, 1231],
        #          [14, 44, 23454.1, .1, 23, 1],
        #          [9, 12, 1, 3, 32, 23, 21, 31],
        #          [1114, 44, 23454.1, .1, 23, 1],
        #         ]+[list(range(6)),list(range(3)),list(range(3)),list(range(3))
        #     ,list(range(3))]+[list(range(6))]
        # %timeit dataresults=cartesian_product(*args)
        # 621 ms ± 46.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


        # Mem usage 4 GB
        # import itertools
        # %timeit (list(itertools.product(*args)))
        # 2.13 s ± 26.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """
    lookup1 = {}
    lookup2 = defaultdict(list)
    i = 0
    for arg in args:
        for keyf in arg:
            lookup2[keyf].append(i)
            lookup1[i] = keyf
            i += 1

    indlist = np.array([len(x) for x in args],dtype=np.int64)
    listexx = int(np.product(indlist))
    dty = np.array(list(lookup2.keys()))
    rightshape = np.array(functools.reduce(operator.add, list(lookup2.values())))
    if not outputdtype:
        bestdtype = (dtypecheck(dty, filterna=False, float2int=False, ))
    else:
        bestdtype=outputdtype
    ear = np.zeros(rightshape.shape, dtype=bestdtype.dtype)
    for k, item in lookup1.items():
        ear[k] = item
    dataresults = np.zeros((listexx, len(indlist)), dtype=ear.dtype)
    if not dataresults.flags['C_CONTIGUOUS']:
        dataresults = np.ascontiguousarray(dataresults)
    ctypesframedataresults = get_pointer_array(dataresults)
    create_product(listexx, indlist, ctypesframedataresults, ear)
    return dataresults


def dtypecheck(array, filterna=True, float2int=True, show_exceptions=True, dtypes=(np.uint8,
                                                                                   np.int8,
                                                                                   np.uint16,
                                                                                   np.int16,
                                                                                   np.uint32,
                                                                                   np.int32,
                                                                                   np.uint64,
                                                                                   np.int64,
                                                                                   np.uintp,
                                                                                   np.intp,
                                                                                   np.float16,
                                                                                   np.float32,
                                                                                   np.float64,
                                                                                   'M',
                                                                                   'm',
                                                                                   'O',
                                                                                   'P',
                                                                                   'S',
                                                                                   'U',
                                                                                   'V',
                                                                                   'p',
                                                                                   's',
                                                                                   np.complex64,
                                                                                   np.complex128,
                                                                                   np.datetime64,

                                                                                   np.timedelta64,
                                                                                   np.void, bool,
                                                                                   object
                                                                                   )):
    try:
        arr = array.copy()
        try:
            hasdot = '.' in str(arr.ravel()[0])
        except Exception as fe:
            if show_exceptions:
                sys.stderr.write(f'{fe}\n')
                sys.stderr.flush()
                hasdot = False
        if filterna:
            try:
                arr = arr[~np.isnan(arr)]
            except Exception as ca:
                if show_exceptions:
                    sys.stderr.write(f'{ca}\n')
                    sys.stderr.flush()

        for dty in dtypes:
            try:
                if arr.ndim > 2:
                    littletest = arr[0].astype(dty)
                    _ = np.all(arr == littletest)
                nd = arr.astype(dty)
                if np.all(arr == nd):
                    if not float2int:
                        if hasdot:
                            if '.' in str(arr.ravel()[0]) and '.' not in str(nd.ravel()[0]):
                                continue

                    return array.astype(dty)
            except Exception as fe:
                if show_exceptions:
                    sys.stderr.write(f'{fe}\n')
                    sys.stderr.flush()
        return array
    except Exception as fe:
        if show_exceptions:
            sys.stderr.write(f'{fe}\n')
            sys.stderr.flush()
    return array