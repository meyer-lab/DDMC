# base.pxd
# Contact: Jacob Schreiber (jmschreiber91@gmail.com)

cimport numpy

cdef class Model(object):
	cdef public str name
	cdef public int d
	cdef public bint frozen
	cdef public str model

	cdef void _log_probability(self, double* symbol, double* log_probability, int n) nogil
	cdef double _vl_log_probability(self, double* symbol, int n) nogil
	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil
