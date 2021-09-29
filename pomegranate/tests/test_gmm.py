from pomegranate import *
from pomegranate.io import DataGenerator

from nose.tools import with_setup
from nose.tools import assert_true
from nose.tools import assert_equal
from nose.tools import assert_greater
from nose.tools import assert_raises
from nose.tools import assert_not_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

import pandas
import random
import pickle
import numpy as np

np.random.seed(0)
random.seed(0)

nan = numpy.nan


def setup_nothing():
    pass


def setup_multivariate_gaussian():
    """
    Set up a five component Gaussian mixture model, where each component
    is a multivariate Gaussian distribution.
    """

    global gmm

    mu = np.arange(5)
    cov = np.eye(5)

    mgs = [MultivariateGaussianDistribution(mu * i, cov) for i in range(5)]
    gmm = GeneralMixtureModel(mgs)


def setup_univariate_gaussian():
    """
    Set up a three component univariate Gaussian model.
    """

    global gmm
    gmm = GeneralMixtureModel([NormalDistribution(i * 3, 1) for i in range(3)])


def teardown():
    """
    Teardown the model, so delete it.
    """

    pass


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_log_probability():
    X = numpy.array([[1.1, 2.7, 3.0, 4.8, 6.2],
                     [1.8, 2.1, 3.1, 5.2, 6.5],
                     [0.9, 2.2, 3.2, 5.0, 5.8],
                     [1.0, 2.1, 3.5, 4.3, 5.2],
                     [1.2, 2.9, 3.1, 4.2, 5.5],
                     [1.8, 1.9, 3.0, 4.9, 5.7],
                     [1.2, 3.1, 2.9, 4.2, 5.9],
                     [1.0, 2.9, 3.9, 4.1, 6.0]])

    logp_t = [-9.8405678, -9.67171158, -9.71615297, -9.89404726,
              -10.93812212, -11.06611533, -11.31473392, -10.79220257]
    logp = gmm.log_probability(X)

    assert_array_almost_equal(logp, logp_t)


@with_setup(setup_univariate_gaussian, teardown)
def test_gmm_univariate_gaussian_log_probability():
    X = np.array([[1.1], [2.7], [3.0], [4.8], [6.2]])
    logp = [-2.35925975, -2.03120691, -1.99557605, -2.39638244, -2.03147258]
    assert_array_almost_equal(gmm.log_probability(X), logp)
    assert_array_almost_equal(gmm.log_probability(X, n_jobs=2), logp)
    assert_array_almost_equal(gmm.log_probability(X, batch_size=2), logp)

    X = np.array([[1.8], [2.1], [3.1], [5.2], [6.5]])
    logp = [-2.39618117, -2.26893273, -1.9995911, -2.22202965, -2.14007514]
    assert_array_almost_equal(gmm.log_probability(X), logp)
    assert_array_almost_equal(gmm.log_probability(X, n_jobs=2), logp)
    assert_array_almost_equal(gmm.log_probability(X, batch_size=2), logp)

    X = np.array([[0.9], [2.2], [3.2], [5.0], [5.8]])
    logp = [-2.26957032, -2.22113386, -2.01155305, -2.31613252, -2.01751101]
    assert_array_almost_equal(gmm.log_probability(X), logp)
    assert_array_almost_equal(gmm.log_probability(X, n_jobs=2), logp)
    assert_array_almost_equal(gmm.log_probability(X, batch_size=2), logp)

    X = np.array([[1.0], [2.1], [3.5], [4.3], [5.2]])
    logp = [-2.31613252, -2.26893273, -2.09160506, -2.42491769, -2.22202965]
    assert_array_almost_equal(gmm.log_probability(X), logp)
    assert_array_almost_equal(gmm.log_probability(X, n_jobs=2), logp)
    assert_array_almost_equal(gmm.log_probability(X, batch_size=2), logp)

    X = np.array([[1.2], [2.9], [3.1], [4.2], [5.5]])
    logp = [-2.39638244, -1.9995911, -1.9995911, -2.39618117, -2.09396318]
    assert_array_almost_equal(gmm.log_probability(X), logp)
    assert_array_almost_equal(gmm.log_probability(X, n_jobs=2), logp)
    assert_array_almost_equal(gmm.log_probability(X, batch_size=2), logp)

    X = np.array([[1.8], [1.9], [3.0], [4.9], [5.7]])
    logp = [-2.39618117, -2.35895351, -1.99557605, -2.35925975, -2.03559364]
    assert_array_almost_equal(gmm.log_probability(X), logp)
    assert_array_almost_equal(gmm.log_probability(X, n_jobs=2), logp)
    assert_array_almost_equal(gmm.log_probability(X, batch_size=2), logp)

    X = np.array([[1.2], [3.1], [2.9], [4.2], [5.9]])
    logp = [-2.39638244, -1.9995911, -1.9995911, -2.39618117, -2.00766654]
    assert_array_almost_equal(gmm.log_probability(X), logp)
    assert_array_almost_equal(gmm.log_probability(X, n_jobs=2), logp)
    assert_array_almost_equal(gmm.log_probability(X, batch_size=2), logp)

    X = np.array([[1.0], [2.9], [3.9], [4.1], [6.0]])
    logp = [-2.31613252, -1.9995911, -2.26893273, -2.35895351, -2.00650306]
    assert_array_almost_equal(gmm.log_probability(X), logp)
    assert_array_almost_equal(gmm.log_probability(X, n_jobs=2), logp)
    assert_array_almost_equal(gmm.log_probability(X, batch_size=2), logp)


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_json():
    gmm_2 = GeneralMixtureModel.from_json(gmm.to_json())

    X = np.array([[1.1, 2.7, 3.0, 4.8, 6.2]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -9.8406, 4)

    X = np.array([[1.8, 2.1, 3.1, 5.2, 6.5]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -9.6717, 4)

    X = np.array([[0.9, 2.2, 3.2, 5.0, 5.8]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -9.7162, 4)

    X = np.array([[1.0, 2.1, 3.5, 4.3, 5.2]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -9.894, 4)

    X = np.array([[1.2, 2.9, 3.1, 4.2, 5.5]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -10.9381, 4)

    X = np.array([[1.8, 1.9, 3.0, 4.9, 5.7]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -11.0661, 4)

    X = np.array([[1.2, 3.1, 2.9, 4.2, 5.9]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -11.3147, 4)

    X = np.array([[1.0, 2.9, 3.9, 4.1, 6.0]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -10.7922, 4)


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_robust_from_json():
    gmm_2 = from_json(gmm.to_json())

    X = np.array([[1.1, 2.7, 3.0, 4.8, 6.2]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -9.8406, 4)

    X = np.array([[1.8, 2.1, 3.1, 5.2, 6.5]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -9.6717, 4)

    X = np.array([[0.9, 2.2, 3.2, 5.0, 5.8]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -9.7162, 4)

    X = np.array([[1.0, 2.1, 3.5, 4.3, 5.2]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -9.894, 4)

    X = np.array([[1.2, 2.9, 3.1, 4.2, 5.5]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -10.9381, 4)

    X = np.array([[1.8, 1.9, 3.0, 4.9, 5.7]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -11.0661, 4)

    X = np.array([[1.2, 3.1, 2.9, 4.2, 5.9]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -11.3147, 4)

    X = np.array([[1.0, 2.9, 3.9, 4.1, 6.0]])
    assert_almost_equal(gmm_2.log_probability(X).sum(), -10.7922, 4)


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_predict_log_proba():
    posterior = np.array([[-2.10001234e+01, -1.23402948e-04, -9.00012340e+00, -4.80001234e+01, -1.17000123e+02],
                          [-2.30009115e+01, -9.11466556e-04, -7.00091147e+00, -4.40009115e+01, -1.11000911e+02]])

    X = np.array([[2., 5., 7., 3., 2.],
                  [1., 2., 5., 2., 5.]])

    assert_almost_equal(gmm.predict_log_proba(X), posterior, 4)
    assert_almost_equal(numpy.exp(gmm.predict_log_proba(X)), gmm.predict_proba(X), 4)


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_predict():
    X = np.array([[2., 5., 7., 3., 2.],
                  [1., 2., 5., 2., 5.],
                  [2., 1., 8., 2., 1.],
                  [4., 3., 8., 1., 2.]])

    assert_almost_equal(gmm.predict(X), gmm.predict_proba(X).argmax(axis=1))
    assert_almost_equal(gmm.predict(X, n_jobs=2), gmm.predict_proba(X).argmax(axis=1))
    assert_almost_equal(gmm.predict(X), gmm.predict_proba(X, n_jobs=2).argmax(axis=1))
    assert_almost_equal(gmm.predict(X, batch_size=2), gmm.predict_proba(X).argmax(axis=1))
    assert_almost_equal(gmm.predict(X), gmm.predict_proba(X, batch_size=3).argmax(axis=1))


def test_gmm_multivariate_gaussian_fit():
    d1 = MultivariateGaussianDistribution([0, 0], [[1, 0], [0, 1]])
    d2 = MultivariateGaussianDistribution([2, 2], [[1, 0], [0, 1]])
    gmm = GeneralMixtureModel([d1, d2])

    X = np.array([[0.1, 0.7],
                  [1.8, 2.1],
                  [-0.9, -1.2],
                  [-0.0, 0.2],
                  [1.4, 2.9],
                  [1.8, 2.5],
                  [1.4, 3.1],
                  [1.0, 1.0]])

    _, history = gmm.fit(X, return_history=True)
    total_improvement = history.total_improvement[-1]

    assert_almost_equal(total_improvement, 15.242416, 4)


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_fit_iterations():
    numpy.random.seed(0)
    X = numpy.concatenate([numpy.random.normal(i, 1, size=(100, 5)) for i in range(2)])

    mu, cov = numpy.ones(5), numpy.eye(5)
    d = [MultivariateGaussianDistribution(mu * i, cov) for i in range(2)]
    gmm = GeneralMixtureModel(d)
    gmm2 = gmm.copy()
    gmm3 = gmm.copy()

    gmm.fit(X)
    gmm2.fit(X, max_iterations=1)
    gmm3.fit(X, max_iterations=1)

    logp1 = gmm.log_probability(X).sum()
    logp2 = gmm2.log_probability(X).sum()
    logp3 = gmm3.log_probability(X).sum()

    assert_greater(logp1, logp2)
    assert_equal(logp2, logp3)


def test_gmm_initialization():
    assert_raises(ValueError, GeneralMixtureModel, [])

    assert_raises(TypeError, GeneralMixtureModel, [NormalDistribution(5, 2), MultivariateGaussianDistribution([5, 2], [[1, 0], [0, 1]])])
    assert_raises(TypeError, GeneralMixtureModel, [NormalDistribution(5, 2), NormalDistribution])

    X = numpy.concatenate((numpy.random.randn(300, 5) + 0.5, numpy.random.randn(200, 5)))

    MGD = MultivariateGaussianDistribution

    gmm1 = GeneralMixtureModel.from_samples(MGD, 2, X, init='first-k')
    gmm2 = GeneralMixtureModel.from_samples(MGD, 2, X, init='first-k', max_iterations=1)
    assert_greater(gmm1.log_probability(X).sum(), gmm2.log_probability(X).sum())

    assert_equal(gmm1.d, 5)
    assert_equal(gmm2.d, 5)


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_pickling():
    gmm2 = pickle.loads(pickle.dumps(gmm))

    for d in gmm2.distributions:
        assert_true(isinstance(d, MultivariateGaussianDistribution))

    assert_true(isinstance(gmm2, GeneralMixtureModel))
    assert_array_almost_equal(gmm.weights, gmm2.weights)


def test_gmm_multivariate_gaussian_ooc():
    X = numpy.concatenate([numpy.random.randn(1000, 3) + i for i in range(3)])

    gmm = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                           3, X, init='first-k', max_iterations=5)
    gmm2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                            3, X, init='first-k', max_iterations=5, batch_size=3000)
    gmm3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                            3, X, init='first-k', max_iterations=5, batch_size=500, batches_per_epoch=6)
    gmm4 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                            3, X, init='first-k', max_iterations=5, batch_size=500, batches_per_epoch=2)

    assert_almost_equal(gmm.log_probability(X).sum(), gmm2.log_probability(X).sum())
    assert_almost_equal(gmm.log_probability(X).sum(), gmm3.log_probability(X).sum(), -2)
    assert_not_equal(gmm.log_probability(X).sum(), gmm4.log_probability(X).sum())


def test_gmm_multivariate_gaussian_minibatch():
    X = numpy.concatenate([numpy.random.randn(1000, 3) + i for i in range(3)])

    gmm = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                           3, X, init='first-k', max_iterations=5)
    gmm2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                            3, X, init='first-k', max_iterations=5, batch_size=500, batches_per_epoch=1)
    gmm3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                            3, X, init='first-k', max_iterations=5, batch_size=500, batches_per_epoch=6)
    gmm4 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                            3, X, init='first-k', max_iterations=5, batch_size=3000, batches_per_epoch=1)

    assert_not_equal(gmm.log_probability(X).sum(), gmm2.log_probability(X).sum())
    assert_not_equal(gmm2.log_probability(X).sum(), gmm3.log_probability(X).sum())
    assert_raises(AssertionError, assert_array_almost_equal, gmm3.log_probability(X),
                  gmm.log_probability(X))

    assert_array_equal(gmm.log_probability(X), gmm4.log_probability(X))


def test_gmm_multivariate_gaussian_nan_from_samples():
    numpy.random.seed(1)
    X = numpy.concatenate([numpy.random.normal(0, 1, size=(300, 3)),
                           numpy.random.normal(8, 1, size=(300, 3))])
    numpy.random.shuffle(X)
    idxs = numpy.random.choice(numpy.arange(1800), replace=False, size=500)
    i, j = idxs // 3, idxs % 3

    X_nan = X.copy()
    X_nan[i, j] = numpy.nan

    mu1t = [-0.036813615311095164, 0.05802948506749107, 0.09725454186262805]
    cov1t = [[1.02529437, -0.11391075, 0.03146951],
             [-0.11391075, 1.03553592, -0.07852064],
             [0.03146951, -0.07852064, 0.83874547]]

    mu2t = [8.088079704231793, 7.927924504375215, 8.000474719123183]
    cov2t = [[0.95559825, -0.02582016, 0.07491681],
             [-0.02582016, 0.99427793, 0.03304442],
             [0.07491681, 0.03304442, 1.15403456]]

    for init in 'first-k', 'random', 'kmeans++':
        model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2,
                                                 X_nan, init=init, n_init=1)

        mu1 = model.distributions[0].parameters[0]
        cov1 = model.distributions[0].parameters[1]

        mu2 = model.distributions[1].parameters[0]
        cov2 = model.distributions[1].parameters[1]

        assert_array_almost_equal(mu1, mu1t)
        assert_array_almost_equal(mu2, mu2t)
        assert_array_almost_equal(cov1, cov1t)
        assert_array_almost_equal(cov2, cov2t)


def test_gmm_multivariate_gaussian_nan_fit():
    mu1, mu2, mu3 = numpy.zeros(3), numpy.zeros(3) + 3, numpy.zeros(3) + 5
    cov1, cov2, cov3 = numpy.eye(3), numpy.eye(3) * 2, numpy.eye(3) * 0.5

    d1 = MultivariateGaussianDistribution(mu1, cov1)
    d2 = MultivariateGaussianDistribution(mu2, cov2)
    d3 = MultivariateGaussianDistribution(mu3, cov3)
    model = GeneralMixtureModel([d1, d2, d3])

    numpy.random.seed(1)
    X = numpy.concatenate([numpy.random.normal(0, 1, size=(300, 3)),
                           numpy.random.normal(2.5, 1, size=(300, 3)),
                           numpy.random.normal(6, 1, size=(300, 3))])
    numpy.random.shuffle(X)
    idxs = numpy.random.choice(numpy.arange(2700), replace=False, size=1000)
    i, j = idxs // 3, idxs % 3

    model.fit(X)

    mu1t = [-0.003165176330948316, 0.07462401273020161, 0.04001352280548061]
    cov1t = [[0.98556769, -0.10062447, 0.08213565],
             [-0.10062447, 1.06955989, -0.03085883],
             [0.08213565, -0.03085883, 0.89728992]]

    mu2t = [2.601485766170187, 2.48231424824341, 2.52771758325412]
    cov2t = [[0.94263451, -0.00361101, -0.02668448],
             [-0.00361101, 1.06339061, -0.00408865],
             [-0.02668448, -0.00408865, 1.14789789]]

    mu3t = [5.950490843670593, 5.9572969419328725, 6.025950220056731]
    cov3t = [[1.03991941, -0.0232587, -0.02457755],
             [-0.0232587, 1.01047466, -0.04948464],
             [-0.02457755, -0.04948464, 0.85671553]]

    mu1 = model.distributions[0].parameters[0]
    cov1 = model.distributions[0].parameters[1]

    mu2 = model.distributions[1].parameters[0]
    cov2 = model.distributions[1].parameters[1]

    mu3 = model.distributions[2].parameters[0]
    cov3 = model.distributions[2].parameters[1]

    assert_array_almost_equal(mu1, mu1t)
    assert_array_almost_equal(mu2, mu2t)
    assert_array_almost_equal(mu3, mu3t)
    assert_array_almost_equal(cov1, cov1t)
    assert_array_almost_equal(cov2, cov2t)
    assert_array_almost_equal(cov3, cov3t)


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_nan_log_probability():
    numpy.random.seed(1)

    X = numpy.concatenate([numpy.random.normal(0, 1, size=(5, 5)),
                           numpy.random.normal(2.5, 1, size=(5, 5)),
                           numpy.random.normal(6, 1, size=(5, 5))])
    numpy.random.shuffle(X)
    idxs = numpy.random.choice(numpy.arange(75), replace=False, size=35)
    i, j = idxs // 5, idxs % 5
    X[i, j] = numpy.nan

    logp_t = [-7.73923053e+00, -7.81725880e+00, -4.55182482e+00, -2.45359578e+01,
              -8.01289941e+00, -1.08630517e+01, -3.10356303e+00, -3.06193233e+01,
              -5.46424483e+00, -1.84952128e+01, -3.15420910e+01, -1.01635415e+01,
              1.66533454e-16, -2.70671185e+00, -5.69860159e+00]

    logp = gmm.log_probability(X)
    logp2 = gmm.log_probability(X, n_jobs=2)
    logp4 = gmm.log_probability(X, n_jobs=4)

    assert_array_almost_equal(logp, logp_t)
    assert_array_almost_equal(logp2, logp_t)
    assert_array_almost_equal(logp4, logp_t)


def test_gmm_multivariate_gaussian_nan_fit_predict():
    X = numpy.concatenate([numpy.random.normal(0, 1, size=(300, 5)),
                           numpy.random.normal(8, 1, size=(300, 5))])
    numpy.random.shuffle(X)
    idxs = numpy.random.choice(numpy.arange(3000), replace=False, size=900)
    i, j = idxs // 5, idxs % 5

    X_nan = X.copy()
    X_nan[i, j] = numpy.nan

    for init in 'first-k', 'random', 'kmeans++':
        model = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2,
                                                 X_nan, init=init, n_init=1)

        for d in model.distributions:
            assert_equal(numpy.isnan(d.parameters[0]).sum(), 0)
            assert_equal(numpy.isnan(d.parameters[1]).sum(), 0)

        y_hat = model.predict(X)
        assert_equal(y_hat.sum(), 300)


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_nan_predict():
    numpy.random.seed(0)
    X = numpy.concatenate([numpy.random.normal(0, 1, size=(5, 5)),
                           numpy.random.normal(2, 1, size=(5, 5))])
    numpy.random.shuffle(X)
    idxs = numpy.random.choice(numpy.arange(50), replace=False, size=25)
    i, j = idxs // 5, idxs % 5
    X[i, j] = numpy.nan

    y_hat = gmm.predict(X)
    y = [0, 1, 0, 0, 0, 0, 1, 0, 1, 1]
    assert_array_equal(y, y_hat)


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_gaussian_nan_predict_proba():
    numpy.random.seed(0)
    X = numpy.concatenate([numpy.random.normal(0, 1, size=(5, 5)),
                           numpy.random.normal(2, 1, size=(5, 5))])
    numpy.random.shuffle(X)
    idxs = numpy.random.choice(numpy.arange(50), replace=False, size=25)
    i, j = idxs // 5, idxs % 5
    X[i, j] = numpy.nan

    y_hat = gmm.predict_proba(X)
    y = [[9.90174494e-01, 9.82550625e-03, 2.20378830e-10, 1.11726590e-23, 1.28030945e-42],
         [2.53691739e-01, 7.46308013e-01, 2.47068952e-07, 9.20463413e-21, 3.85907464e-41],
         [9.97082301e-01, 2.91769906e-03, 1.18573581e-16, 6.69226809e-41, 5.24561818e-76],
         [9.94236788e-01, 5.76321240e-03, 7.55111628e-11, 2.23629671e-24, 1.49699183e-43],
         [9.56664759e-01, 4.33351517e-02, 8.91201785e-08, 8.32083585e-18, 3.52706158e-32],
         [9.98269488e-01, 1.73051193e-03, 3.37590087e-13, 7.41127732e-30, 1.83098477e-53],
         [1.75007605e-01, 8.24992395e-01, 3.63922167e-13, 1.50221681e-38, 5.80259513e-77],
         [8.76384248e-01, 1.23615751e-01, 7.21849843e-10, 1.74507352e-25, 1.74652336e-48],
         [2.12508922e-03, 9.45913785e-01, 5.19607732e-02, 3.52248634e-07, 2.94694950e-16],
         [8.63688760e-03, 9.91106040e-01, 2.57071963e-04, 1.50716583e-13, 1.99728068e-28]]

    assert_array_almost_equal(y, y_hat, 2)


def test_gmm_multivariate_gaussian_ooc_nan_from_samples():
    numpy.random.seed(2)
    X = numpy.concatenate([numpy.random.normal(i * 3, 0.5, size=(200, 3)) for i in range(2)])
    numpy.random.shuffle(X)

    idxs = numpy.random.choice(numpy.arange(1200), replace=False, size=100)
    i, j = idxs // 3, idxs % 3
    X[i, j] = numpy.nan

    model1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                              2, X, init='first-k', batch_size=None, max_iterations=3)
    model2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                              2, X, init='first-k', batch_size=400, max_iterations=3)
    model3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                              2, X, init='first-k', batch_size=100, max_iterations=3)

    cov1 = model1.distributions[0].parameters[1]
    cov2 = model2.distributions[0].parameters[1]
    cov3 = model3.distributions[0].parameters[1]

    assert_array_almost_equal(cov1, cov2)
    assert_array_almost_equal(cov1, cov3)


def test_gmm_multivariate_gaussian_ooc_nan_fit():
    X = numpy.concatenate([numpy.random.normal(i * 3, 0.5, size=(100, 3)) for i in range(2)])
    numpy.random.shuffle(X)
    idxs = numpy.random.choice(numpy.arange(600), replace=False, size=100)
    i, j = idxs // 3, idxs % 3
    X[i, j] = numpy.nan

    mus = [numpy.ones(3) * i * 3 for i in range(2)]
    covs = [numpy.eye(3) for i in range(2)]

    distributions = [MultivariateGaussianDistribution(mu, cov) for mu, cov in zip(mus, covs)]
    model1 = GeneralMixtureModel(distributions)
    model1.fit(X, max_iterations=3)

    distributions = [MultivariateGaussianDistribution(mu, cov) for mu, cov in zip(mus, covs)]
    model2 = GeneralMixtureModel(distributions)
    model2.fit(X, batch_size=10, max_iterations=3)

    distributions = [MultivariateGaussianDistribution(mu, cov) for mu, cov in zip(mus, covs)]
    model3 = GeneralMixtureModel(distributions)
    model3.fit(X, batch_size=1, max_iterations=3)

    cov1 = model1.distributions[0].parameters[1]
    cov2 = model2.distributions[0].parameters[1]
    cov3 = model3.distributions[0].parameters[1]

    assert_array_almost_equal(cov1, cov2)
    assert_array_almost_equal(cov1, cov3)


def test_gmm_multivariate_gaussian_minibatch_nan_from_samples():
    X = numpy.concatenate([numpy.random.normal(i * 2, 1, size=(100, 3)) for i in range(2)])
    numpy.random.shuffle(X)
    idxs = numpy.random.choice(numpy.arange(600), replace=False, size=100)
    i, j = idxs // 3, idxs % 3
    X[i, j] = numpy.nan

    model1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                              2, X, init='first-k', batch_size=None, max_iterations=5)
    model2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                              2, X, init='first-k', batch_size=200, max_iterations=5)
    model3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                              2, X, init='first-k', batch_size=50, batches_per_epoch=4,
                                              max_iterations=5)
    model4 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                              2, X, init='first-k', batch_size=50, batches_per_epoch=1,
                                              max_iterations=5)

    cov1 = model1.distributions[0].parameters[1]
    cov2 = model2.distributions[0].parameters[1]
    cov3 = model3.distributions[0].parameters[1]
    cov4 = model4.distributions[0].parameters[1]

    assert_array_almost_equal(cov1, cov2)
    assert_raises(AssertionError, assert_array_almost_equal, cov1, cov3)
    assert_raises(AssertionError, assert_array_almost_equal, cov1, cov4)


def test_gmm_multivariate_gaussian_minibatch_nan_fit():
    X = numpy.concatenate([numpy.random.normal(i * 3, 0.5, size=(100, 3)) for i in range(2)])
    numpy.random.shuffle(X)
    idxs = numpy.random.choice(numpy.arange(600), replace=False, size=100)
    i, j = idxs // 3, idxs % 3
    X[i, j] = numpy.nan

    mus = [numpy.ones(3) * i * 3 for i in range(2)]
    covs = [numpy.eye(3) for i in range(2)]

    distributions = [MultivariateGaussianDistribution(mu, cov) for mu, cov in zip(mus, covs)]
    model1 = GeneralMixtureModel(distributions)
    model1.fit(X, batch_size=None, max_iterations=3)

    distributions = [MultivariateGaussianDistribution(mu, cov) for mu, cov in zip(mus, covs)]
    model2 = GeneralMixtureModel(distributions)
    model2.fit(X, batch_size=10, max_iterations=3)

    distributions = [MultivariateGaussianDistribution(mu, cov) for mu, cov in zip(mus, covs)]
    model3 = GeneralMixtureModel(distributions)
    model3.fit(X, batch_size=10, batches_per_epoch=5, max_iterations=3)

    cov1 = model1.distributions[0].parameters[1]
    cov2 = model2.distributions[0].parameters[1]
    cov3 = model3.distributions[0].parameters[1]

    assert_array_almost_equal(cov1, cov2)
    assert_raises(AssertionError, assert_array_equal, cov1, cov3)


@with_setup(setup_multivariate_gaussian, teardown)
def test_gmm_multivariate_mixed_random_sample():
    x = numpy.array([[-0.937128, 3.919795, 7.066424, 11.901844, 16.691532],
                     [1.875753, 0.915462, 2.591164, 3.173006, 4.656439],
                     [0.307347, 3.411364, 7.450107, 11.520762, 16.511734]])

    assert_array_almost_equal(gmm.sample(3, random_state=5), x)
    assert_raises(AssertionError, assert_array_almost_equal, gmm.sample(3), x)


def test_io_fit():
    X = numpy.random.randn(100, 5) + 0.5
    weights = numpy.abs(numpy.random.randn(100))
    data_generator = DataGenerator(X, weights)

    mu1 = numpy.array([0, 0, 0, 0, 0])
    mu2 = numpy.array([1, 1, 1, 1, 1])
    cov = numpy.eye(5)

    d1 = MultivariateGaussianDistribution(mu1, cov)
    d2 = MultivariateGaussianDistribution(mu2, cov)
    gmm1 = GeneralMixtureModel([d1, d2])
    gmm1.fit(X, weights, max_iterations=5)

    d1 = MultivariateGaussianDistribution(mu1, cov)
    d2 = MultivariateGaussianDistribution(mu2, cov)
    gmm2 = GeneralMixtureModel([d1, d2])
    gmm2.fit(data_generator, max_iterations=5)

    logp1 = gmm1.log_probability(X)
    logp2 = gmm2.log_probability(X)

    assert_array_almost_equal(logp1, logp2)


def test_io_from_samples_gmm():
    X = numpy.random.randn(100, 5) + 0.5
    weights = numpy.abs(numpy.random.randn(100))
    data_generator = DataGenerator(X, weights)

    d = MultivariateGaussianDistribution
    gmm1 = GeneralMixtureModel.from_samples(d, n_components=2, X=X,
                                            weights=weights, max_iterations=5, init='first-k')
    gmm2 = GeneralMixtureModel.from_samples(d, n_components=2,
                                            X=data_generator, max_iterations=5, init='first-k')

    logp1 = gmm1.log_probability(X)
    logp2 = gmm2.log_probability(X)

    assert_array_almost_equal(logp1, logp2)
