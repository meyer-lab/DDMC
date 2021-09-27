# utils.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from libc.math cimport log as clog
from libc.math cimport log2 as clog2
from libc.math cimport exp as cexp
from libc.math cimport floor
from libc.math cimport fabs
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport isnan

from scipy.linalg.cython_blas cimport dgemm


cimport cython
import numpy
cimport numpy

import numbers


numpy.import_array()

cdef extern from "numpy/ndarraytypes.h":
	void PyArray_ENABLEFLAGS(numpy.ndarray X, int flags)

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463
DEF GAMMA = 0.577215664901532860606512090


cdef ndarray_wrap_cpointer(void* data, numpy.npy_intp n):
	cdef numpy.ndarray[numpy.float64_t, ndim=1] X = numpy.PyArray_SimpleNewFromData(1, &n, numpy.NPY_FLOAT64, data)
	return X
	

cdef void mdot(double* X, double* Y, double* A, int m, int n, int k) nogil:
	cdef double alpha = 1
	cdef double beta = 0
	dgemm('N', 'N', &n, &m, &k, &alpha, Y, &n, X, &k, &beta, A, &n)


cpdef numpy.ndarray _convert( data ):
	if type(data) is numpy.ndarray:
		return data
	if type(data) is int:
		return numpy.array( [data] )
	if type(data) is float:
		return numpy.array( [data] )
	if type(data) is list:
		return numpy.array( data )

# Useful speed optimized functions
cdef double _log(double x) nogil:
	'''
	A wrapper for the c log function, by returning negative infinity if the
	input is 0.
	'''
	return clog(x) if x > 0 else NEGINF

cdef double pair_lse(double x, double y) nogil:
	'''
	Perform log-sum-exp on a pair of numbers in log space..  This is calculated
	as z = log( e**x + e**y ). However, this causes underflow sometimes
	when x or y are too negative. A simplification of this is thus
	z = x + log( e**(y-x) + 1 ), where x is the greater number. If either of
	the inputs are infinity, return infinity, and if either of the inputs
	are negative infinity, then simply return the other input.
	'''

	if x == INF or y == INF:
		return INF
	if x == NEGINF:
		return y
	if y == NEGINF:
		return x
	if x > y:
		return x + clog(cexp(y-x) + 1)
	return y + clog(cexp(x-y) + 1)


def _check_input(X, keymap=None):
	"""Check the input to make sure that it is a properly formatted array."""

	cdef numpy.ndarray X_ndarray

	try:
		X_ndarray = numpy.array(X, dtype='float64', ndmin=2, order='C')
	except:
		if not isinstance(X, (numpy.ndarray, list, tuple)):
			X_ndarray = numpy.array(keymap[0][X], dtype='float64',
			                        ndmin=2, order='C')
		else:
			X = numpy.array(X)
			X_ndarray = numpy.empty(X.shape, dtype='float64', order='C')

			if X.ndim == 1:
				for i in range(X.shape[0]):
					X_ndarray[i] = keymap[0][X[i]]
				X_ndarray = X_ndarray.reshape(-1, 1)
			else:
				for i in range(X.shape[0]):
					for j in range(X.shape[1]):
						X_ndarray[i, j] = keymap[j][X[i, j]]


	return X_ndarray

def weight_set(items, weights):
	"""Converts both items and weights to appropriate numpy arrays.

	Convert the items into a numpy array with 64-bit floats, and the weight
	array to the same. If no weights are passed in, then return a numpy array
	with uniform weights.
	"""

	items = numpy.array(items, dtype=numpy.float64)
	if weights is None: # Weight everything 1 if no weights specified
		weights = numpy.ones(items.shape[0], dtype=numpy.float64)
	else: # Force whatever we have to be a Numpy array
		weights = numpy.asarray(weights, dtype=numpy.float64)

	return items, weights

def _check_nan(X):
	"""Checks to see if a value is nan, either as a float or a string."""
	if isinstance(X, (str, unicode, numpy.string_)):
		return X == 'nan'
	if isinstance(X, (float, numpy.float32, numpy.float64)):
		return isnan(X)
	return X is None

def check_random_state(seed):
	"""Turn seed into a np.random.RandomState instance.

	This function will check to see whether the input seed is a valid seed
	for generating random numbers. This is a slightly modified version of
	the code from sklearn.utils.validation.

	Parameters
	----------
	seed : None | int | instance of RandomState
		If seed is None, return the RandomState singleton used by np.random.
		If seed is an int, return a new RandomState instance seeded with seed.
		If seed is already a RandomState instance, return it.
		Otherwise raise ValueError.
	"""

	if seed is None or seed is numpy.random:
		return numpy.random.mtrand._rand
	if isinstance(seed, (numbers.Integral, numpy.integer)):
		return numpy.random.RandomState(seed)
	if isinstance(seed, numpy.random.RandomState):
		return seed
	raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
					 ' instance' % seed)
