#!python
#cython: boundscheck=False
#cython: cdivision=True

import numpy
import scipy.stats as sp

from ..utils cimport isnan


cdef class SeqDistribution(Distribution):
	""" The generic distribution class. """

	def __init__(self, seqWeight, N, frozen=False):
		self.frozen = frozen
		self.summaries = None
		self.weightsIn = sp.beta.rvs(a=10, b=10, size=len(info["Sequence"]))
        self.logWeights = numpy.log(self.weights)

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.
			else:
				log_probability[i] = self.SeqWeight * self.logWeights[int(X[i])]

    def summarize(self, _, w):
        self.weights = w

    def clear_summaries(self):
        """ Clear the summary statistics stored in the object. Not needed here. """
        return
