from __future__ import (division)

from pomegranate import (Distribution,
						 NormalDistribution,
						 LogNormalDistribution,
						 ExponentialDistribution,
						 IndependentComponentsDistribution,
						 MultivariateGaussianDistribution,
						 from_json)

from nose.tools import with_setup
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_less_equal
from nose.tools import assert_true
from nose.tools import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
import pickle
import numpy

nan = numpy.nan
inf = float("inf")

def setup():
	pass


def teardown():
	pass


def discrete_equality(x, y, z=8):
	'''
	Test to see if two discrete distributions are equal to each other to
	z decimal points.
	'''

	xd, yd = x.parameters[0], y.parameters[0]
	for key, value in xd.items():
		if round(yd[key], z) != round(value, z):
			return False
	return True


def test_distributions_normal_initialization():
	d = NormalDistribution(5, 2)
	assert_equal(d.name, "NormalDistribution")
	assert_array_equal(d.parameters, [5, 2])
	assert_array_equal(d.summaries, [0, 0, 0])


def test_distributions_normal_blank():
	d = NormalDistribution.blank()
	assert_equal(d.name, "NormalDistribution")
	assert_array_equal(d.parameters, [0, 1])
	assert_array_equal(d.summaries, [0, 0, 0])


def test_distributions_normal_initialization_error():
	assert_raises(TypeError, NormalDistribution, 5)
	assert_raises(TypeError, NormalDistribution, [5, 1])
	assert_raises(TypeError, NormalDistribution, 5, 1, 4, 7, 3)


def test_distributions_normal_log_probability():
	d = NormalDistribution(5, 2)
	e = NormalDistribution(5., 2.)

	assert_almost_equal(d.log_probability(5), -1.61208571)
	assert_equal(d.log_probability(5), e.log_probability(5))
	assert_equal(d.log_probability(5), d.log_probability(5.))

	assert_almost_equal(d.log_probability(0), -4.737085713764219)
	assert_equal(d.log_probability(0), e.log_probability(0.))


def test_distributions_normal_nan_log_probability():
	d = NormalDistribution(5, 2)

	assert_equal(d.log_probability(nan), 0)
	assert_array_almost_equal(d.log_probability([nan, 5]), [0, -1.61208571])


def test_distributions_normal_underflow_log_probability():
	d = NormalDistribution(5, 1e-10)
	assert_almost_equal(d.log_probability(1e100), -4.9999999999999987e+219, delta=6.270570637641398e+203)


def test_distributions_normal_probability():
	d = NormalDistribution(5, 2)
	e = NormalDistribution(5., 2.)

	assert_almost_equal(d.probability(5), 0.19947114)
	assert_equal(d.probability(5), e.probability(5))
	assert_equal(d.probability(5), d.probability(5.))

	assert_almost_equal(d.probability(0), 0.0087641502)
	assert_equal(d.probability(0), e.probability(0.))


def test_distributions_normal_nan_probability():
	d = NormalDistribution(5, 2)

	assert_equal(d.probability(nan), 1)
	assert_array_almost_equal(d.probability([nan, 5]), [1, 0.199471])


def test_distributions_normal_underflow_probability():
	d = NormalDistribution(5, 1e-10)
	assert_almost_equal(d.probability(1e100), 0.0)


def test_distributions_normal_fit():
	d = NormalDistribution(5, 2)
	e = NormalDistribution(5, 2)

	d.fit([5, 4, 5, 4, 6, 5, 6, 5, 4, 6, 5, 4])

	assert_array_almost_equal(d.parameters, [4.9167, 0.7592], 4)
	assert_not_equal(d.log_probability(4), e.log_probability(4))
	assert_almost_equal(d.log_probability(4), -1.3723678499651766)
	assert_almost_equal(d.log_probability(18), -149.13140399454429)
	assert_almost_equal(d.log_probability(1e8), -8674697942168743.0, -4)
	assert_array_equal(d.summaries, [0, 0, 0])


def test_distributions_normal_nan_fit():
	d = NormalDistribution(5, 2)
	e = NormalDistribution(5, 2)

	d.fit([5, 4, nan, 5, 4, nan, 6, 5, 6, nan, nan, 5, 4, 6, nan, 5, 4, nan])

	assert_array_almost_equal(d.parameters, [4.9167, 0.7592], 4)
	assert_not_equal(d.log_probability(4), e.log_probability(4))
	assert_almost_equal(d.log_probability(4), -1.3723678499651766)
	assert_almost_equal(d.log_probability(18), -149.13140399454429)
	assert_almost_equal(d.log_probability(1e8), -8674697942168743.0, -4)
	assert_array_equal(d.summaries, [0, 0, 0])


def test_distributions_normal_exclusive_nan_fit():
	d = NormalDistribution(5, 2)
	e = NormalDistribution(5, 2)

	d.fit([nan, nan, nan, nan, nan])

	assert_array_equal(d.parameters, [5, 2])
	assert_almost_equal(d.log_probability(4), e.log_probability(4.))
	assert_array_equal(d.summaries, [0, 0, 0])


def test_distributions_normal_weighted_fit():
	d = NormalDistribution(5, 2)

	d.fit([0, 2, 3, 2, 100], weights=[0, 5, 2, 3, 200])
	assert_array_almost_equal(d.parameters, [95.3429, 20.8276], 4)
	assert_almost_equal(d.log_probability(50), -6.32501194)
	assert_array_equal(d.summaries, [0, 0, 0])


def test_distributions_normal_inertia_fit():
	d = NormalDistribution(5, 2)

	d.fit([0, 5, 3, 5, 7, 3, 4, 5, 2], inertia=0.5)

	assert_array_almost_equal(d.parameters, [4.3889, 1.9655], 4)
	assert_array_equal(d.summaries, [0, 0, 0])


def test_distributions_normal_fit_ooc():
	d = NormalDistribution(5, 2)
	d.summarize([0, 2], weights=[0, 5])
	d.summarize([3, 2], weights=[2, 3])
	d.summarize([100], weights=[200])

	assert_array_equal(d.summaries, [2.100000e+02, 2.002200e+04, 2.000050e+06])

	d.from_summaries()

	assert_array_equal(d.summaries, [0, 0, 0])
	assert_array_almost_equal(d.parameters, [95.3429, 20.8276], 4)


def test_distributions_normal_freeze_fit():
	d = NormalDistribution(5, 2)
	d.freeze()
	d.fit([0, 1, 1, 2, 3, 2, 1, 2, 2])

	assert_array_almost_equal(d.parameters, [5, 2])
	assert_array_equal(d.summaries, [0, 0, 0])


def test_distributions_normal_freeze_thaw_fit():
	d = NormalDistribution(5, 2)
	d.freeze()
	d.thaw()

	d.fit([5, 4, 5, 4, 6, 5, 6, 5, 4, 6, 5, 4])
	assert_array_almost_equal(d.parameters, [4.9166, 0.7592], 4)


def test_distributions_normal_from_samples():
	d = NormalDistribution.from_samples([5, 2, 4, 6, 8, 3, 6, 8, 3])

	assert_array_almost_equal(d.parameters, [5.0, 2.05480466])
	assert_array_equal(d.summaries, [0, 0, 0])


def test_distributions_normal_nan_from_samples():
	d = NormalDistribution.from_samples([5, nan, 2, nan, 4, 6, nan, 8, 3, nan, 6, nan, 8, 3])

	assert_array_almost_equal(d.parameters, [5.0, 2.05480466])
	assert_array_equal(d.summaries, [0, 0, 0])

def test_distributions_normal_pickle_serialization():
	d = NormalDistribution(5, 2)

	e = pickle.loads(pickle.dumps(d))
	assert_equal(e.name, "NormalDistribution")
	assert_array_equal(e.parameters, [5, 2])
	assert_array_equal(e.summaries, [0, 0, 0])


def test_distributions_normal_json_serialization():
	d = NormalDistribution(5, 2)

	e = Distribution.from_json(d.to_json())
	assert_equal(e.name, "NormalDistribution")
	assert_array_equal(e.parameters, [5, 2])
	assert_array_equal(e.summaries, [0, 0, 0])

def test_distributions_normal_robust_json_serialization():
	d = NormalDistribution(5, 2)

	e = from_json(d.to_json())
	assert_equal(e.name, "NormalDistribution")
	assert_array_equal(e.parameters, [5, 2])
	assert_array_equal(e.summaries, [0, 0, 0])

def test_distributions_normal_random_sample():
	d = NormalDistribution(0, 1)

	x = numpy.array([ 0.44122749, -0.33087015,  2.43077119, -0.25209213,  
		0.10960984])

	assert_array_almost_equal(d.sample(5, random_state=5), x)
	assert_raises(AssertionError, assert_array_almost_equal, d.sample(5), x)

@with_setup(setup, teardown)
def test_lognormal():
	d = LogNormalDistribution(5, 2)
	assert_equal(round(d.log_probability(5), 4), -4.6585)

	d.fit([5.1, 5.03, 4.98, 5.05, 4.91, 5.2, 5.1, 5., 4.8, 5.21])
	assert_equal(round(d.parameters[0], 4), 1.6167)
	assert_equal(round(d.parameters[1], 4), 0.0237)

	d.summarize([5.1, 5.03, 4.98, 5.05])
	d.summarize([4.91, 5.2, 5.1])
	d.summarize([5., 4.8, 5.21])
	d.from_summaries()

	assert_equal(round(d.parameters[0], 4), 1.6167)
	assert_equal(round(d.parameters[1], 4), 0.0237)

	e = Distribution.from_json(d.to_json())
	assert_equal(e.name, "LogNormalDistribution")
	assert_equal(round(e.parameters[0], 4), 1.6167)
	assert_equal(round(e.parameters[1], 4), 0.0237)

	f = pickle.loads(pickle.dumps(e))
	assert_equal(f.name, "LogNormalDistribution")
	assert_equal(round(f.parameters[0], 4), 1.6167)
	assert_equal(round(f.parameters[1], 4), 0.0237)


def test_distributions_lognormal_random_sample():
	d = LogNormalDistribution(0, 1)

	x = numpy.array([1.55461432,  0.71829843, 11.36764528,  0.77717313,  
		1.11584263])

	assert_array_almost_equal(d.sample(5, random_state=5), x)
	assert_raises(AssertionError, assert_array_almost_equal, d.sample(5), x)


@with_setup(setup, teardown)
def test_exponential():
	d = ExponentialDistribution(3)
	assert_equal(round(d.log_probability(8), 4), -22.9014)

	d.fit([2.7, 2.9, 3.8, 1.9, 2.7, 1.6, 1.3, 1.0, 1.9])
	assert_equal(round(d.parameters[0], 4), 0.4545)

	d = ExponentialDistribution(4)
	assert_not_equal(round(d.log_probability(8), 4), -22.9014)

	d.summarize([2.7, 2.9, 3.8])
	d.summarize([1.9, 2.7, 1.6])
	d.summarize([1.3, 1.0, 1.9])
	d.from_summaries()

	assert_equal(round(d.parameters[0], 4), 0.4545)

	e = Distribution.from_json(d.to_json())
	assert_equal(e.name, "ExponentialDistribution")
	assert_equal(round(e.parameters[0], 4), 0.4545)

	f = pickle.loads(pickle.dumps(e))
	assert_equal(f.name, "ExponentialDistribution")
	assert_equal(round(f.parameters[0], 4), 0.4545)


def test_distributions_exponential_random_sample():
	d = ExponentialDistribution(7)

	x = numpy.array([0.03586, 0.292267, 0.033083, 0.358359, 0.095748])

	assert_array_almost_equal(d.sample(5, random_state=5), x)
	assert_raises(AssertionError, assert_array_almost_equal, d.sample(5), x)


def test_multivariate_log_probability():
	X = numpy.random.randn(100, 5)

	d = MultivariateGaussianDistribution.from_samples(X)
	logp = d.log_probability(X)
	for i in range(100):
		assert_almost_equal(d.log_probability(X[i]), logp[i])

	d = IndependentComponentsDistribution.from_samples(X, distributions=NormalDistribution)
	logp = d.log_probability(X)
	for i in range(100):
		assert_almost_equal(d.log_probability(X[i]), logp[i])


def test_distributions_mgd_random_sample():
	mu = numpy.array([5., 1.])
	cov = numpy.eye(2)
	d = MultivariateGaussianDistribution(mu, cov)

	x = numpy.array([[5.441227, 0.66913 ],
			         [7.430771, 0.747908],
			         [5.10961 , 2.582481]])

	assert_array_almost_equal(d.sample(3, random_state=5), x)
	assert_raises(AssertionError, assert_array_almost_equal, d.sample(5), x)
