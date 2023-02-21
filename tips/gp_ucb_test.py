import numpy as np

from gp_ucb import GaussianProcessRegressor

TRAIN_X = np.array(
    [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7],
        [0.4, 0.5, 0.6, 0.7, 0.8],
        [0.5, 0.6, 0.7, 0.8, 0.9],
        [0.6, 0.7, 0.8, 0.9, 1.0],
    ]
)
TRAIN_Y = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
TEST_X = np.array(
    [
        [0.11, 0.19, 0.32, 0.38, 0.495],
    ]
)


def test_cholesky_solve():
    gp = GaussianProcessRegressor()
    gp.train(TRAIN_X, TRAIN_Y)
    # L^T * L * alpha = y  is the equation that we are solving for
    # in doing the cholesky decomposition during training.
    actual = gp.L_.dot(gp.L_.T.dot(gp.alpha_))
    expected = gp.y_train_
    np.testing.assert_allclose(actual, expected)


def test_train_and_predict():
    gp = GaussianProcessRegressor()
    mean, cov = gp.train(TRAIN_X, TRAIN_Y).predict(TEST_X)
    assert abs(mean[0] - 0.3) < 1e-3
    assert abs(cov[0][0]) < 1e-2
