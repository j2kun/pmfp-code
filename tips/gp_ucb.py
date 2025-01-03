"""An implementation of Gaussian Process UCB.

The Gaussain Process Regressor was adapted from the sci-kit learn implementation.

https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b7e21201cfffb118934999025fd50cca/sklearn/gaussian_process/_gpr.py
"""

import itertools

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.spatial.distance import cdist


class GaussianProcessRegressor:
    def kernel(self, X, Y=None):
        dists = cdist(X, X if Y is None else Y, metric="sqeuclidean")
        return np.exp(-0.5 * dists)

    def train(self, X, y):
        """Fit data to a standard Gaussian Process regression model."""
        # Normalize training signals
        self._y_train_mean = np.mean(y, axis=0)
        self._y_train_std = np.std(y, axis=0)
        if self._y_train_std == 0:
            self._y_train_std = 1
        y = (y - self._y_train_mean) / self._y_train_std

        self.X_train_ = X
        self.y_train_ = y

        # Precompute quantities required for predictions which are independent
        # of actual query points
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K = self.kernel(self.X_train_)
        # added small noise to (a) ensure kernel matrix is positive definite,
        # and (b) to model gaussian noise in the observations.
        K[np.diag_indices_from(K)] += 1e-10

        try:
            self.L_ = cholesky(K, lower=True)
        except np.linalg.LinAlgError as exc:  # pragma: no cover
            exc.args = (
                "Cholseky factorization failed, kernel might "
                "not be positive semidefinite"
            ) + exc.args
            raise

        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)
        return self

    def predict(self, X):
        """Predict using the Gaussian process regression model."""
        # Alg 2.1, page 19, line 4 -> f*_bar = K(X_test, X_train) . alpha
        K_trans = self.kernel(X, self.X_train_)
        y_mean = K_trans @ self.alpha_

        # undo normalisation
        y_mean = self._y_train_std * y_mean + self._y_train_mean

        # Alg 2.1, page 19, line 5 -> v = L \ K(X_test, X_train)^T
        V = solve_triangular(self.L_, K_trans.T, lower=True, check_finite=False)

        # Alg 2.1, page 19, line 6 -> K(X_test, X_test) - v^T. v
        y_cov = self.kernel(X) - V.T @ V

        # undo normalisation
        y_cov = np.outer(y_cov, self._y_train_std**2).reshape(*y_cov.shape, -1)

        # if y_cov has shape (n_samples, n_samples, 1), reshape to
        # (n_samples, n_samples)
        if y_cov.shape[2] == 1:
            y_cov = np.squeeze(y_cov, axis=2)

        return y_mean, y_cov


class GpUcb:
    def __init__(self, input_spaces, beta=10):
        """Initalize the GP UCB algorithm."""
        self.beta = beta

        # Flatten the input space so it can be indexed by a single integer.
        # A smarter approach would use an implicit indexing scheme.
        self.features = input_spaces
        self.input_space = list(itertools.product(*input_spaces))

        # Initial mean/stdev values are chosen arbitrarily here,
        # but in Vizier, input parameters have a variety of allowed
        # configurations that impact the initial values. See
        # https://github.com/google/vizier/blob/3e2581814f219a29c2e540c3df8d8a5c911d55ce/vizier/_src/pyvizier/shared/parameter_config.py#L238
        self.mean = np.array([0] * len(self.input_space))
        self.stdev = np.array([0.5] * len(self.input_space))

        self.obs_inputs = []
        self.obs_outputs = []

    def suggest(self):
        index = np.argmax(self.mean + self.stdev * self.beta**0.5)
        suggestion = self.input_space[index]
        self.obs_inputs.append(suggestion)
        return suggestion

    def update(self, observed_output):
        self.obs_outputs.append(observed_output)
        gp = GaussianProcessRegressor()
        gp.train(self.obs_inputs, self.obs_outputs)
        self.mean, cov = gp.predict(self.input_space)
        self.stdev = cov.diagonal() ** 0.5

    def unwind_estimates(self):
        output = dict()
        indices = itertools.product(*[range(len(x)) for x in self.features])
        for flat_index, index_tuple in enumerate(indices):
            input = tuple(f[i] for (f, i) in zip(self.features, index_tuple))
            output[input] = (self.mean[flat_index], self.stdev[flat_index])
        return output
