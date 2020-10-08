
import numpy
cimport numpy

from pomegranate.distributions cimport Distribution

cdef class NormalDistribution(Distribution):
	cdef double* logWeights
	cdef double* weightsIn
	cdef double* seqWeight
