#slightly modified version of https://github.com/numba/numba/issues/3086
from numba.extending import get_cython_function_address
from numba import vectorize, njit
import ctypes
import numpy as np

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

gammaln_addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
gammaln_functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = gammaln_functype(gammaln_addr)

@njit
def numba_gammaln(x):
  return gammaln_float64(x)

import scipy.special.cython_special as cysp

# for item in cysp.__pyx_capi__:
#   print(item)

hyp2f1_addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1hyp2f1")
hyp2f1_functype = ctypes.CFUNCTYPE(_dble, _dble, _dble, _dble, _dble)
hyp2f1_float64 = hyp2f1_functype(hyp2f1_addr)

@vectorize('float64(float64, float64, float64, float64)')
def numba_hyp2f1(a, b, c, z):
  return hyp2f1_float64(a, b, c, z)



betainc_addr = get_cython_function_address("scipy.special.cython_special", "betainc")
betainc_functype = ctypes.CFUNCTYPE(_dble, _dble, _dble, _dble)
betainc_float64 = betainc_functype(betainc_addr)

@vectorize('float64(float64, float64, float64)')
def numba_betainc(a, b, y):
  return betainc_float64(a, b, y)