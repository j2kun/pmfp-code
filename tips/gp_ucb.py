"""An implementation of Gaussian Process UCB.

The Gaussain Process Regressor was adapted from the
sci-kit learn implementation.

https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b7e21201cfffb118934999025fd50cca/sklearn/gaussian_process/_gpr.py
"""

from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.spatial.distance import cdist
import numpy as np


class GaussianProcessRegressor:
    def kernel(self, X, Y=None):
        dists = cdist(X, X if Y is None else Y, metric="sqeuclidean")
        return np.exp(-0.5 * dists)

    def train(self, X, y):
        """Fit data to a standard Gaussian Process regression model."""
        # Normalize training signals
        self._y_train_mean = np.mean(y, axis=0)
        self._y_train_std = np.std(y, axis=0)
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
        except np.linalg.LinAlgError as exc:
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
