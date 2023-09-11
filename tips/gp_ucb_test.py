import numpy as np

from tips.gp_ucb import GaussianProcessRegressor, GpUcb

TRAIN_X = np.array(
    [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7],
        [0.4, 0.5, 0.6, 0.7, 0.8],
        [0.5, 0.6, 0.7, 0.8, 0.9],
        [0.6, 0.7, 0.8, 0.9, 1.0],
    ],
)
TRAIN_Y = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
TEST_X = np.array(
    [
        [0.11, 0.19, 0.32, 0.38, 0.495],
    ],
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


def test_sin_plus_cos():
    # optimum on 0, pi is at x=pi/2, y=0
    def f(data):
        x, y = data
        return np.sin(x) + np.cos(y)

    feature_1 = np.linspace(0, np.pi, 5)
    feature_2 = np.linspace(0, np.pi, 5)
    gp_ucb = GpUcb(input_spaces=[feature_1, feature_2])

    # note how it doesn't need to sample the whole space
    # to get a good estimate for the max
    for i in range(10):
        example = gp_ucb.suggest()
        print(example)
        output = f(example)
        gp_ucb.update(output)

    # estimated_mean_of_max = gp_ucb.get_mean((feature_1[2], feature_2[0]))
    estimates = gp_ucb.unwind_estimates()
    mean_of_max, stdev_of_max = estimates[(feature_1[2], feature_2[0])]
    assert abs(mean_of_max - 2) < 1e-02
    assert abs(stdev_of_max) < 1e-04

    print("estimated max input")
    print(max(estimates, key=lambda x: estimates[x][0]))
