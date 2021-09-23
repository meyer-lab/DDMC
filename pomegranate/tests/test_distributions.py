from __future__ import (division)

from pomegranate import (Distribution,
						 UniformDistribution,
						 NormalDistribution,
						 DiscreteDistribution,
						 LogNormalDistribution,
						 ExponentialDistribution,
						 IndependentComponentsDistribution,
						 MultivariateGaussianDistribution,
						 BernoulliDistribution,
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


def test_distributions_uniform_initialization():
	d = UniformDistribution(0, 10)
	assert_equal(d.name, "UniformDistribution")
	assert_array_equal(d.parameters, [0, 10])
	assert_array_equal(d.summaries, [inf, -inf, 0])


def test_distributions_uniform_blank():
	d = UniformDistribution.blank()
	assert_equal(d.name, "UniformDistribution")
	assert_array_equal(d.parameters, [0, 0])
	assert_array_equal(d.summaries, [inf, -inf, 0])


def test_distributions_uniform_initialization_error():
	assert_raises(TypeError, UniformDistribution, 0)
	assert_raises(TypeError, UniformDistribution, [0, 10])
	assert_raises(TypeError, UniformDistribution, 0, 10, 4, 7, 3)


def test_distributions_uniform_log_probability():
	d = UniformDistribution(0, 10)
	e = UniformDistribution(0., 10.)

	assert_almost_equal(d.log_probability(5), -2.302585092)
	assert_equal(d.log_probability(5), e.log_probability(5))
	assert_equal(d.log_probability(5), d.log_probability(5.))

	assert_almost_equal(d.log_probability(0), -2.302585092)
	assert_equal(d.log_probability(0), e.log_probability(0.))

	assert_equal(d.log_probability(-1), -inf)
	assert_equal(d.log_probability(11), -inf)


def test_distributions_uniform_nan_log_probability():
	d = UniformDistribution(0, 10)

	assert_equal(d.log_probability(nan), 0)
	assert_array_almost_equal(d.log_probability([nan, 5]), [0, -2.302585092])


def test_distributions_uniform_underflow_log_probability():
	d = UniformDistribution(0, 10)
	assert_equal(d.log_probability(1e100), float("-inf"))


def test_distributions_uniform_probability():
	d = UniformDistribution(0, 10)
	e = UniformDistribution(0., 10.)

	assert_almost_equal(d.probability(5), 0.0999999999)
	assert_equal(d.probability(5), e.probability(5))
	assert_equal(d.probability(5), d.probability(5.))

	assert_almost_equal(d.probability(0), 0.0999999999)
	assert_equal(d.probability(0), e.probability(0.))

	assert_equal(d.probability(-1), 0)
	assert_equal(d.probability(11), 0)


def test_distributions_uniform_nan_probability():
	d = UniformDistribution(0, 10)

	assert_equal(d.probability(nan), 1)
	assert_array_almost_equal(d.probability([nan, 5]), [1, 0.0999999999])


def test_distributions_uniform_underflow_probability():
	d = UniformDistribution(0, 10)
	assert_almost_equal(d.probability(1e100), 0.0)


def test_distributions_uniform_fit():
	d = UniformDistribution(5, 2)
	e = UniformDistribution(5, 2)

	d.fit([5, 4, 5, 4, 6, 5, 6, 5, 4, 6, 5, 4])

	assert_array_equal(d.parameters, [4, 6])
	assert_not_equal(d.log_probability(4), e.log_probability(4))
	assert_almost_equal(d.log_probability(4), -0.69314718055994529)
	assert_equal(d.log_probability(18), -inf)
	assert_equal(d.log_probability(1e8), -inf)
	assert_array_equal(d.summaries, [inf, -inf, 0])


def test_distributions_uniform_nan_fit():
	d = UniformDistribution(5, 2)
	e = UniformDistribution(5, 2)

	d.fit([5, 4, nan, 5, 4, nan, 6, 5, 6, nan, nan, 5, 4, 6, nan, 5, 4, nan])

	assert_array_equal(d.parameters, [4, 6])
	assert_not_equal(d.log_probability(4), e.log_probability(4))
	assert_almost_equal(d.log_probability(4), -0.69314718055994529)
	assert_equal(d.log_probability(18), -inf)
	assert_equal(d.log_probability(1e8), -inf)
	assert_array_equal(d.summaries, [inf, -inf, 0])


def test_distributions_uniform_exclusive_nan_fit():
	d = UniformDistribution(0, 10)
	e = UniformDistribution(0, 10)

	d.fit([nan, nan, nan, nan, nan])

	assert_array_equal(d.parameters, [0, 10])
	assert_almost_equal(d.log_probability(4), e.log_probability(4.))
	assert_array_equal(d.summaries, [inf, -inf, 0])


def test_distributions_uniform_weighted_fit():
	d = UniformDistribution(0, 10)

	d.fit([0, 2, 3, 2, 100], weights=[0, 5, 2, 3, 200])
	assert_array_equal(d.parameters, [2, 100])
	assert_almost_equal(d.log_probability(50), -4.58496747867)
	assert_array_equal(d.summaries, [inf, -inf, 0])


def test_distributions_uniform_inertia_fit():
	d = UniformDistribution(0, 10)

	d.fit([0, 5, 3, 5, 7, 3, 4, 5, 2], inertia=0.5)

	assert_array_equal(d.parameters, [0, 8.5])
	assert_array_equal(d.summaries, [inf, -inf, 0])


def test_distributions_uniform_fit_ooc():
	d = UniformDistribution(0, 10)
	d.summarize([0, 2], weights=[0, 5])
	d.summarize([3, 2], weights=[2, 3])
	d.summarize([100], weights=[200])

	assert_array_equal(d.summaries, [2, 100, 210])

	d.from_summaries()

	assert_array_equal(d.summaries, [inf, -inf, 0])
	assert_array_equal(d.parameters, [2, 100])


def test_distributions_uniform_freeze_fit():
	d = UniformDistribution(0, 10)
	d.freeze()
	d.fit([0, 1, 1, 2, 3, 2, 1, 2, 2])

	assert_array_almost_equal(d.parameters, [0, 10])
	assert_array_equal(d.summaries, [inf, -inf, 0])


def test_distributions_uniform_freeze_thaw_fit():
	d = UniformDistribution(0, 10)
	d.freeze()
	d.thaw()

	d.fit([5, 4, 5, 4, 6, 5, 6, 5, 4, 6, 5, 4])
	assert_array_equal(d.parameters, [4, 6])


def test_distributions_uniform_from_samples():
	d = UniformDistribution.from_samples([5, 2, 4, 6, 8, 3, 6, 8, 3])

	assert_array_equal(d.parameters, [2, 8])
	assert_array_equal(d.summaries, [inf, -inf, 0])


def test_distributions_uniform_nan_from_samples():
	d = UniformDistribution.from_samples([5, nan, 2, nan, 4, 6, nan, 8, 3, nan, 6, nan, 8, 3])

	assert_array_equal(d.parameters, [2, 8])
	assert_array_equal(d.summaries, [inf, -inf, 0])

def test_distributions_uniform_pickle_serialization():
	d = UniformDistribution(0, 10)

	e = pickle.loads(pickle.dumps(d))
	assert_equal(e.name, "UniformDistribution")
	assert_array_equal(e.parameters, [0, 10])
	assert_array_equal(d.summaries, [inf, -inf, 0])


def test_distributions_uniform_json_serialization():
	d = UniformDistribution(0, 10)

	e = Distribution.from_json(d.to_json())
	assert_equal(e.name, "UniformDistribution")
	assert_array_equal(e.parameters, [0, 10])
	assert_array_equal(d.summaries, [inf, -inf, 0])

def test_distributions_uniform_robust_json_serialization():
	d = UniformDistribution(0, 10)

	e = from_json(d.to_json())
	assert_equal(e.name, "UniformDistribution")
	assert_array_equal(e.parameters, [0, 10])
	assert_array_equal(d.summaries, [inf, -inf, 0])

def test_distributions_uniform_random_sample():
	d = UniformDistribution(0, 10)

	x = numpy.array([2.21993171, 8.70732306, 2.06719155, 9.18610908, 
		4.88411189])

	assert_array_almost_equal(d.sample(5, random_state=5), x)
	assert_raises(AssertionError, assert_array_almost_equal, d.sample(5), x)

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
def test_distributions_discrete():
	d = DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25})

	assert_equal(d.log_probability('C'), -1.3862943611198906)
	assert_equal(d.log_probability('A'), d.log_probability('C'))
	assert_equal(d.log_probability('G'), d.log_probability('T'))
	assert_equal(d.log_probability('a'), float('-inf'))

	seq = "ACGTACGTTGCATGCACGCGCTCTCGCGC"
	d.fit(list(seq))

	assert_equal(d.log_probability('C'), -0.9694005571881036)
	assert_equal(d.log_probability('A'), -1.9810014688665833)
	assert_equal(d.log_probability('T'), -1.575536360758419)

	seq = "ACGTGTG"
	d.fit(list(seq), weights=[0., 1., 2., 3., 4., 5., 6.])

	assert_equal(d.log_probability('A'), float('-inf'))
	assert_equal(d.log_probability('C'), -3.044522437723423)
	assert_equal(d.log_probability('G'), -0.5596157879354228)

	d.summarize(list("ACG"), weights=[0., 1., 2.])
	d.summarize(list("TGT"), weights=[3., 4., 5.])
	d.summarize(list("G"), weights=[6.])
	d.from_summaries()

	assert_equal(d.log_probability('A'), float('-inf'))
	assert_equal(round(d.log_probability('C'), 4), -3.0445)
	assert_equal(round(d.log_probability('G'), 4), -0.5596)

	d = DiscreteDistribution({'A': 0.0, 'B': 1.0})
	d.summarize(list("ABABABAB"))
	d.summarize(list("ABAB"))
	d.summarize(list("BABABABABABABABABA"))
	d.from_summaries(inertia=0.75)
	assert_equal(d.parameters[0], {'A': 0.125, 'B': 0.875})

	d = DiscreteDistribution({'A': 0.0, 'B': 1.0})
	d.summarize(list("ABABABAB"))
	d.summarize(list("ABAB"))
	d.summarize(list("BABABABABABABABABA"))
	d.from_summaries(inertia=0.5)
	assert_equal(d.parameters[0], {'A': 0.25, 'B': 0.75})

	d.freeze()
	d.fit(list('ABAABBAAAAAAAAAAAAAAAAAA'))
	assert_equal(d.parameters[0], {'A': 0.25, 'B': 0.75})

	d = DiscreteDistribution.from_samples(['A', 'B', 'A', 'A'])
	assert_equal(d.parameters[0], {'A': 0.75, 'B': 0.25})

	# Test vector input instead of flat array.
	d = DiscreteDistribution.from_samples(numpy.array(['A', 'B', 'A', 'A']).reshape(-1,1))
	assert_equal(d.parameters[0], {'A': 0.75, 'B': 0.25})

	d = DiscreteDistribution.from_samples(['A', 'B', 'A', 'A'], pseudocount=0.5)
	assert_equal(d.parameters[0], {'A': 0.70, 'B': 0.30})

	d = DiscreteDistribution.from_samples(['A', 'B', 'A', 'A'], pseudocount=6)
	assert_equal(d.parameters[0], {'A': 0.5625, 'B': 0.4375})

	e = Distribution.from_json(d.to_json())
	assert_equal(e.name, "DiscreteDistribution")
	assert_equal(e.parameters[0], {'A': 0.5625, 'B': 0.4375})

	f = pickle.loads(pickle.dumps(e))
	assert_equal(f.name, "DiscreteDistribution")
	assert_equal(f.parameters[0], {'A': 0.5625, 'B': 0.4375})


def test_discrete_robust_json_serialization():
	d = DiscreteDistribution.from_samples(['A', 'B', 'A', 'A'], pseudocount=6)

	e = from_json(d.to_json())
	assert_equal(e.name, "DiscreteDistribution")
	assert_equal(e.parameters[0], {'A': 0.5625, 'B': 0.4375})

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


@with_setup(setup, teardown)
def test_bernoulli():
	d = BernoulliDistribution(0.6)
	assert_equal(d.probability(0), 0.4)
	assert_equal(d.probability(1), 0.6)
	assert_equal(d.parameters[0], 1-d.probability(0))
	assert_equal(d.parameters[0], d.probability(1))

	d.fit([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	assert_not_equal(d.probability(1), 1.0)
	assert_equal(d.probability(0), 1.0)

	a = [0.0, 0.0, 0.0]
	b = [1.0, 1.0, 1.0]
	c = [1.0, 1.0, 1.0]
	d.summarize(a)
	d.from_summaries()
	assert_equal(d.probability(0), 1)
	assert_equal(d.probability(1), 0)

	d.summarize(a)
	d.summarize(b)
	d.from_summaries()
	assert_equal(d.probability(0), 0.5)
	assert_equal(d.probability(1), 0.5)
	assert_equal(d.parameters[0], d.probability(0))
	assert_equal(d.parameters[0], d.probability(1))

	d.summarize(a)
	d.summarize(b)
	d.summarize(c)
	d.from_summaries()
	assert_equal(round(d.probability(0), 4), 0.3333)
	assert_equal(round(d.probability(1), 4), 0.6667)
	assert_equal(d.parameters[0], d.probability(1))

	d = BernoulliDistribution.from_samples([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
	assert_equal(round(d.probability(0), 4), 0.8333)
	assert_equal(round(d.probability(1), 4), 0.1667)
	assert_almost_equal(d.parameters[0], d.probability(1))

	e = Distribution.from_json(d.to_json())
	assert_equal(e.name, "BernoulliDistribution")
	assert_equal(round(e.parameters[0], 4), 0.1667)

	f = pickle.loads(pickle.dumps(e))
	assert_equal(f.name, "BernoulliDistribution")
	assert_equal(round(f.parameters[0], 4), 0.1667)

def test_distributions_uniform_kernel_random_sample():
	d = BernoulliDistribution(0.2)

	x = numpy.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
			0, 0, 0, 0, 0])

	assert_array_equal(d.sample(20, random_state=5), x)
	assert_raises(AssertionError, assert_array_equal, d.sample(20), x)

@with_setup(setup, teardown)
def test_independent():
	d = IndependentComponentsDistribution(
		[NormalDistribution(5, 2), ExponentialDistribution(2)])

	assert_equal(round(d.log_probability((4, 1)), 4), -3.0439)
	assert_equal(round(d.log_probability((100, 0.001)), 4), -1129.0459)

	d = IndependentComponentsDistribution([NormalDistribution(5, 2),
										   ExponentialDistribution(2)],
										  weights=[18., 1.])

	assert_equal(round(d.log_probability((4, 1)), 4), -32.5744)
	assert_equal(round(d.log_probability((100, 0.001)), 4), -20334.5764)

	d.fit([(5, 1), (5.2, 1.7), (4.7, 1.9), (4.9, 2.4), (4.5, 1.2)])

	assert_equal(round(d.parameters[0][0].parameters[0], 4), 4.86)
	assert_equal(round(d.parameters[0][0].parameters[1], 4), 0.2417)
	assert_equal(round(d.parameters[0][1].parameters[0], 4), 0.6098)

	d = IndependentComponentsDistribution([NormalDistribution(5, 2),
										   UniformDistribution(0, 10)])
	d.fit([(0, 0), (5, 0), (3, 0), (5, -5), (7, 0),
		   (3, 0), (4, 0), (5, 0), (2, 20)], inertia=0.5)

	assert_equal(round(d.parameters[0][0].parameters[0], 4), 4.3889)
	assert_equal(round(d.parameters[0][0].parameters[1], 4), 1.9655)

	assert_equal(d.parameters[0][1].parameters[0], -2.5)
	assert_equal(d.parameters[0][1].parameters[1], 15)

	d.fit([(0, 0), (5, 0), (3, 0), (5, -5), (7, 0),
		   (3, 0), (4, 0), (5, 0), (2, 20)], inertia=0.75)

	assert_not_equal(round(d.parameters[0][0].parameters[0], 4), 4.3889)
	assert_not_equal(round(d.parameters[0][0].parameters[1], 4), 1.9655)

	assert_not_equal(d.parameters[0][1].parameters[0], -2.5)
	assert_not_equal(d.parameters[0][1].parameters[1], 15)

	d = IndependentComponentsDistribution([NormalDistribution(5, 2),
										   UniformDistribution(0, 10)])

	d.summarize([(0, 0), (5, 0), (3, 0)])
	d.summarize([(5, -5), (7, 0)])
	d.summarize([(3, 0), (4, 0), (5, 0), (2, 20)])
	d.from_summaries(inertia=0.5)

	assert_equal(round(d.parameters[0][0].parameters[0], 4), 4.3889)
	assert_equal(round(d.parameters[0][0].parameters[1], 4), 1.9655)

	assert_equal(d.parameters[0][1].parameters[0], -2.5)
	assert_equal(d.parameters[0][1].parameters[1], 15)

	d.freeze()
	d.fit([(1, 7), (7, 2), (2, 4), (2, 4), (1, 4)])

	assert_equal(round(d.parameters[0][0].parameters[0], 4), 4.3889)
	assert_equal(round(d.parameters[0][0].parameters[1], 4), 1.9655)

	assert_equal(d.parameters[0][1].parameters[0], -2.5)
	assert_equal(d.parameters[0][1].parameters[1], 15)

	e = Distribution.from_json(d.to_json())
	assert_equal(e.name, "IndependentComponentsDistribution")

	assert_equal(round(e.parameters[0][0].parameters[0], 4), 4.3889)
	assert_equal(round(e.parameters[0][0].parameters[1], 4), 1.9655)

	assert_equal(e.parameters[0][1].parameters[0], -2.5)
	assert_equal(e.parameters[0][1].parameters[1], 15)

	f = pickle.loads(pickle.dumps(e))
	assert_equal(e.name, "IndependentComponentsDistribution")

	assert_equal(round(f.parameters[0][0].parameters[0], 4), 4.3889)
	assert_equal(round(f.parameters[0][0].parameters[1], 4), 1.9655)

	assert_equal(f.parameters[0][1].parameters[0], -2.5)
	assert_equal(f.parameters[0][1].parameters[1], 15)

	X = numpy.array([[0.5, 0.2, 0.7],
		          [0.3, 0.1, 0.9],
		          [0.4, 0.3, 0.8],
		          [0.3, 0.3, 0.9],
		          [0.3, 0.2, 0.6],
		          [0.5, 0.2, 0.8]])

	d = IndependentComponentsDistribution.from_samples(X,
		distributions=NormalDistribution)
	assert_almost_equal(d.parameters[0][0].parameters[0], 0.38333, 4)
	assert_almost_equal(d.parameters[0][0].parameters[1], 0.08975, 4)
	assert_almost_equal(d.parameters[0][1].parameters[0], 0.21666, 4)
	assert_almost_equal(d.parameters[0][1].parameters[1], 0.06872, 4)
	assert_almost_equal(d.parameters[0][2].parameters[0], 0.78333, 4)
	assert_almost_equal(d.parameters[0][2].parameters[1], 0.10672, 4)

	d = IndependentComponentsDistribution.from_samples(X,
		distributions=ExponentialDistribution)
	assert_almost_equal(d.parameters[0][0].parameters[0], 2.6087, 4)
	assert_almost_equal(d.parameters[0][1].parameters[0], 4.6154, 4)
	assert_almost_equal(d.parameters[0][2].parameters[0], 1.2766, 4)

	d = IndependentComponentsDistribution.from_samples(X,
		distributions=[NormalDistribution, NormalDistribution, NormalDistribution])
	assert_almost_equal(d.parameters[0][0].parameters[0], 0.38333, 4)
	assert_almost_equal(d.parameters[0][0].parameters[1], 0.08975, 4)
	assert_almost_equal(d.parameters[0][1].parameters[0], 0.21666, 4)
	assert_almost_equal(d.parameters[0][1].parameters[1], 0.06872, 4)
	assert_almost_equal(d.parameters[0][2].parameters[0], 0.78333, 4)
	assert_almost_equal(d.parameters[0][2].parameters[1], 0.10672, 4)

	d = IndependentComponentsDistribution.from_samples(X,
		distributions=[NormalDistribution, LogNormalDistribution, ExponentialDistribution])
	assert_almost_equal(d.parameters[0][0].parameters[0], 0.38333, 4)
	assert_almost_equal(d.parameters[0][0].parameters[1], 0.08975, 4)
	assert_almost_equal(d.parameters[0][1].parameters[0], -1.5898, 4)
	assert_almost_equal(d.parameters[0][1].parameters[1], 0.36673, 4)
	assert_almost_equal(d.parameters[0][2].parameters[0], 1.27660, 4)


def test_distributions_independent_random_sample():
	d = IndependentComponentsDistribution([NormalDistribution(5, 2),
										   UniformDistribution(0, 10),
										   ExponentialDistribution(7),
										   LogNormalDistribution(0, 0.4)])

	x = numpy.array([[5.882455, 2.219932, 0.03586 , 1.193024],
					 [4.33826 , 8.707323, 0.292267, 0.876036],
					 [9.861542, 2.067192, 0.033083, 2.644041]])

	assert_array_almost_equal(d.sample(3, random_state=5), x)
	assert_raises(AssertionError, assert_array_almost_equal, d.sample(5), x)


def test_univariate_log_probability():
	distributions = [UniformDistribution, NormalDistribution, ExponentialDistribution, LogNormalDistribution]
	X = numpy.abs(numpy.random.randn(100))

	for distribution in distributions:
		d = distribution.from_samples(X)
		logp = d.log_probability(X)

		for i in range(100):
			assert_almost_equal(d.log_probability(X[i]), logp[i])

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


def test_distributions_independent_random_sample():
	d = IndependentComponentsDistribution([NormalDistribution(5, 2),
										   UniformDistribution(0, 10),
										   ExponentialDistribution(7),
										   LogNormalDistribution(0, 0.4)])

	x = numpy.array([[5.882455, 2.219932, 0.03586 , 1.193024],
					 [4.33826 , 8.707323, 0.292267, 0.876036],
					 [9.861542, 2.067192, 0.033083, 2.644041]])

	assert_array_almost_equal(d.sample(3, random_state=5), x)
	assert_raises(AssertionError, assert_array_almost_equal, d.sample(5), x)
