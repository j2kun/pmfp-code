import numpy as np
import pytest

from tips.caratheodory_fejer import cf_approximation, chebyshev_interpolant, inf_norm


def test_cf_relu():
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


def test_cheb_relu():
    def f(x):
        return np.maximum(x, 0)

    interval = (-1, 1)
    degree = 50
    interpolant = chebyshev_interpolant(f, degree, interval)
    error = inf_norm(f, interpolant, interval)
    # actual error is about 0.00596
    assert error < 0.01


SMOOTH_FNS = [np.exp, np.sin, np.cos, np.tan]


@pytest.mark.parametrize("f", SMOOTH_FNS)
def test_cf_smooth(f):
    interval = (-1, 1)
    degree = 50
    cf_approx = cf_approximation(f, degree, interval)
    cf_approx_error = inf_norm(f, cf_approx, interval)
    assert cf_approx_error < 1e-14

    interpolant = chebyshev_interpolant(f, degree, interval)
    cheb_error = inf_norm(f, interpolant, interval)
    assert cf_approx_error <= cheb_error


@pytest.mark.parametrize("f", SMOOTH_FNS)
def test_cheb_smooth(f):
    interval = (-1, 1)
    degree = 50
    interpolant = chebyshev_interpolant(f, degree, interval)
    error = inf_norm(f, interpolant, interval)
    assert error < 1e-14


INTERVALS = [
    (-2, 2),  # exp
    (-5, 5),  # sin
    (-3, 4),  # cos
    (-1.3, 0.95),  # tan, can't get too close to +/-pi/2 = 1.5707
]


@pytest.mark.parametrize("f, interval", zip(SMOOTH_FNS, INTERVALS))
def test_cf_smooth_larger_interval(f, interval):
    degree = 50
    cf_approx = cf_approximation(f, degree, interval)
    cf_approx_error = inf_norm(f, cf_approx, interval)
    assert cf_approx_error < 1e-14
