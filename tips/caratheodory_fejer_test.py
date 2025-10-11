import numpy as np

from tips.caratheodory_fejer import cf_approximation, chebyshev_interpolant, inf_norm


def cf_test_relu():
    def f(x):
        return np.maximum(x, 0)

    interval = (-1, 1)
    degree = 50
    interpolant = chebyshev_interpolant(f, degree, interval)
    cheb_interp_error = inf_norm(f, interpolant, interval)

    cf_approx = cf_approximation(f, degree, interval)
    cf_approx_error = inf_norm(f, cf_approx, interval)
    assert cheb_interp_error < 0.01
    assert cf_approx_error < 0.01
    assert cf_approx_error < cheb_interp_error

    relative_error_reduction = (
        abs(cf_approx_error - cheb_interp_error) / cheb_interp_error
    )
    print(f"{cheb_interp_error=}, {cf_approx_error=}")
    print(f"{relative_error_reduction=}")
    # assert 9% better than raw cheb interpolant
    assert relative_error_reduction > 0.09


def cheb_interpolate_test_relu():
    def f(x):
        return np.maximum(x, 0)

    interval = (-1, 1)
    degree = 50
    interpolant = chebyshev_interpolant(f, degree, interval)
    error = inf_norm(f, interpolant, interval)
    # actual error is about 0.00596
    assert error < 0.01


def cheb_interpolate_test_exp():
    def f(x):
        return np.exp(x)

    interval = (-1, 1)
    degree = 50
    interpolant = chebyshev_interpolant(f, degree, interval)
    error = inf_norm(f, interpolant, interval)

    # since exp is much smoother, we expect a great approximation compared
    # to relu
    assert error < 1e-14
