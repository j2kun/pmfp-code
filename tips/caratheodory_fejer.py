import functools
from typing import Callable, Optional

import numpy as np
import scipy

Interval = tuple[float, float]
Function = Callable[[np.ndarray | float], np.ndarray | float]


def inf_norm(f1: Function, f2: Function, interval: Interval, num: int = 10001) -> float:
    """Estimate the max error (infinity norm) between f1 and f2 on the given
    interval."""
    a, b = interval
    x = np.linspace(a, b, num=10001)
    return np.abs(f1(x) - f2(x)).max()


def get_chebyshev_2nd_kind_points(n: int) -> np.ndarray:
    """Generate Chebyshev points of the second kind."""
    m = n - 1
    return np.sin(np.pi * (np.arange(-m, m + 1, 2)) / (2 * m))  # using sin for symmetry


def gen_chebyshev_polynomials(n: int) -> list[np.polynomial.Polynomial]:
    """Generate Chebyshev polynomials in terms of the standard monomial basis."""
    res = []
    res.append(np.polynomial.Polynomial([1]))
    res.append(np.polynomial.Polynomial([0, 1]))
    x = res[1]
    for _ in range(2, n):
        res.append(2 * x * res[-1] - res[-2])
    return res


def _standard_chop(coeffs: np.ndarray, tol: float = 1e-16) -> int:
    """Chops coeffs at a point beyond which it is smaller than tol^(2/3)."""
    # Ported from https://github.com/chebfun/chebfun/blob/master/standardChop.m
    # chops COEFFS at a point beyond which it is smaller than tol^(2/3).
    # coeffs will never be chopped unless it is of length at least 17 and falls at
    # least below TOL^(1/3). It will always be chopped if it has a long enough
    # final segment below TOL, and the final entry COEFFS(CUTOFF) will never
    # be smaller than TOL^(7/6).  All these statements are relative to
    # MAX(ABS(COEFFS)) and assume CUTOFF > 1.  These parameters result from
    # extensive experimentation involving functions such as those presented in
    # the paper cited above.  They are not derived from first principles and
    # there is no claim that they are optimal.

    # Check magnitude of TOL:
    if tol >= 1:
        return 1

    n = len(coeffs)
    if n < 17:
        return n

    # Step 1: Convert COEFFS to a new monotonically nonincreasing
    #         vector ENVELOPE normalized to begin with the value 1.

    b = np.abs(coeffs)
    envelope = np.maximum.accumulate(b[::-1])[::-1]
    if envelope[0] == 0:
        return 1
    envelope = envelope / envelope[0]

    # Step 2: Scan ENVELOPE for a value PLATEAUPOINT, the first point J-1, if any,
    # that is followed by a plateau.  A plateau is a stretch of coefficients
    # ENVELOPE(J),...,ENVELOPE(J2), J2 = round(1.25*J+5) <= N, with the property
    # that ENVELOPE(J2)/ENVELOPE(J) > R.  The number R ranges from R = 0 if
    # ENVELOPE(J) = TOL up to R = 1 if ENVELOPE(J) = TOL^(2/3).  Thus a potential
    # plateau whose starting value is ENVELOPE(J) ~ TOL^(2/3) has to be perfectly
    # flat to count, whereas with ENVELOPE(J) ~ TOL it doesn't have to be flat at
    # all.  If a plateau point is found, then we know we are going to chop the
    # vector, but the precise chopping point CUTOFF still remains to be determined
    # in Step 3.

    plateau_point = None
    for j in range(2, n + 1):
        j2 = round(1.25 * j + 5)
        if j2 > n:
            # there is no plateau: exit
            return len(coeffs)
        e1 = envelope[j - 1]
        e2 = envelope[j2 - 2]
        r = 3 * (1 - np.log(e1) / np.log(tol))
        if (e1 == 0) or (e2 / e1 > r):
            # a plateau has been found: go to Step 3
            plateau_point = j - 2
            break

    # Step 3: fix CUTOFF at a point where ENVELOPE, plus a linear function
    # included to bias the result towards the left end, is minimal.
    #
    # Some explanation is needed here.  One might imagine that if a plateau is
    # found, then one should simply set CUTOFF = PLATEAUPOINT and be done, without
    # the need for a Step 3. However, sometimes CUTOFF should be smaller or larger
    # than PLATEAUPOINT, and that is what Step 3 achieves.
    #
    # CUTOFF should be smaller than PLATEAUPOINT if the last few coefficients made
    # negligible improvement but just managed to bring the vector ENVELOPE below the
    # level TOL^(2/3), above which no plateau will ever be detected.  This part of
    # the code is important for avoiding situations where a coefficient vector is
    # chopped at a point that looks "obviously wrong" with PLOTCOEFFS.
    #
    # CUTOFF should be larger than PLATEAUPOINT if, although a plateau has been
    # found, one can nevertheless reduce the amplitude of the coefficients a good
    # deal further by taking more of them.  This will happen most often when a
    # plateau is detected at an amplitude close to TOL, because in this case, the
    # "plateau" need not be very flat.  This part of the code is important to
    # getting an extra digit or two beyond the minimal prescribed accuracy when it
    # is easy to do so.

    assert plateau_point is not None
    if envelope[plateau_point] == 0:
        return plateau_point

    j3 = np.sum(envelope >= tol ** (7 / 6))
    if j3 < j2:
        j2 = j3 + 1
        envelope[j2] = tol ** (7 / 6)
    cc = np.log10(envelope[:j2])
    cc = cc + np.linspace(0, (-1 / 3) * np.log10(tol), j2)
    d = np.argmin(cc)
    cutoff = max(int(d), 1)

    return cutoff


def _vals2coeffs(values: np.ndarray) -> np.ndarray:
    n = len(values)
    if n <= 1:
        return values
    is_even = np.max(np.abs(values - np.flip(values))) == 0
    is_odd = np.max(np.abs(values + np.flip(values))) == 0
    tmp = np.concatenate((values[:0:-1], values[: n - 1]))
    # only real
    coeffs = np.real(np.fft.ifft(tmp))

    # Truncate:
    coeffs = coeffs[:n]

    # Scale the interior coefficients:
    coeffs[1 : n - 1] = 2 * coeffs[1 : n - 1]

    # adjust coefficients for symmetry
    if is_even:
        coeffs[1::2] = 0
    if is_odd:
        coeffs[0::2] = 0
    return coeffs


class ChebFun:
    """A function represented as a linear combination of Chebyshev polynomials."""

    def __init__(self, coefs: np.ndarray, interval: Interval = (-1, 1)):
        self._coefs = coefs
        self._interval = interval

    def __call__(self, x: np.ndarray):  # x in [-1, 1]
        a, b = self._interval
        mid_point = (a + b) / 2
        half_len = (b - a) / 2
        scaled_x = (x - mid_point) / half_len
        return self._eval_on_standard_interval(scaled_x)

    def _eval_on_standard_interval(self, x: np.ndarray):
        """Evaluates the function at values x in [-1, 1].

        Uses the Clenshaw Algorithm, see
        https://en.wikipedia.org/wiki/Clenshaw_algorithm#Special_case_for_Chebyshev_series
        """
        assert x.min() >= -1 and x.max() <= 1
        b1, b2 = np.zeros_like(x), np.zeros_like(x)
        for c in self._coefs[:0:-1]:
            b0 = c + 2 * x * b1 - b2
            b1, b2 = b0, b1
        p = self._coefs[0] + x * b1 - b2
        return p

    def to_poly(self) -> np.polynomial.Polynomial:
        """Convert to the standard monomial basis."""
        p = chebyshev_coeffs_to_poly(self._coefs)
        interval = self._interval
        a, b = interval
        needs_scaling = a != -1 or b != 1
        if needs_scaling:
            mid_point = (a + b) / 2
            half_len = (b - a) / 2
            y = np.polynomial.Polynomial(
                [-mid_point / half_len, 1 / half_len],
                domain=interval,
                window=interval,
            )  # for scaling back to original interval
            p = p(y)
        return p

    def __repr__(self):
        return f"ChebFun(coefs={self._coefs}, interval={self._interval})"


def chebyshev_coeffs_to_poly(coeffs: np.ndarray) -> np.polynomial.Polynomial:
    cheb_pols = gen_chebyshev_polynomials(len(coeffs))
    return functools.reduce(
        lambda acc, v: acc + v,
        coeffs * cheb_pols,
        np.polynomial.Polynomial([0]),
    )


def get_chebcoefs(f, deg: int) -> np.ndarray:
    xs = get_chebyshev_2nd_kind_points(deg + 1)
    ys = f(xs)
    return _vals2coeffs(ys)


def get_cheb_fun_coefs(f, tol=1e-16, max_deg: int = 129) -> np.ndarray:
    """Returns coefficients of the Chebyshev polynomial approximation of f.

    Automatically chooses the degree of the polynomial, to ensure that the approximation
    is not more than max_deg or have have approximation error less than tol.
    """
    deg = 17
    while deg <= max_deg:
        coefs = get_chebcoefs(f, deg)
        cutoff = _standard_chop(coefs, tol)
        if cutoff < deg:
            coefs = coefs[:cutoff]
            return coefs
        deg = 2 * deg - 1
    return coefs


def cheb_interpolant_on_standard_inverval(
    f: Function,
    degree: Optional[int] = None,
) -> ChebFun:
    if degree is not None:
        return ChebFun(get_chebcoefs(f, degree))

    return ChebFun(get_cheb_fun_coefs(f))


def chebyshev_interpolant(
    f: Function,
    degree: Optional[int] = None,
    interval: Interval = (-1, 1),
) -> ChebFun:
    a, b = interval
    needs_scaling = a != -1 or b != 1
    mid_point = (a + b) / 2
    half_len = (b - a) / 2
    if needs_scaling:
        # scale to [-1, 1]
        def f_scaled(x):
            return f(mid_point + half_len * x)

    else:
        f_scaled = f
    interpolant = cheb_interpolant_on_standard_inverval(f_scaled, degree=degree)
    if needs_scaling:
        interpolant._interval = interval
    return interpolant


def _cf_approximation_on_standard_interval(f, degree: int) -> ChebFun:
    coefs = get_cheb_fun_coefs(f)
    cheb_degree = len(coefs) - 1
    if cheb_degree <= degree:
        return ChebFun(coefs)
    a = coefs[degree + 1 :]
    H = scipy.linalg.hankel(a)
    vals, vecs = scipy.linalg.eigh(H)
    vals = np.real(vals)
    i = np.argmax(np.abs(vals))
    v = vecs[:, i]
    v1 = v[0]
    vv = v[1:]
    b = a
    t = cheb_degree - degree - 1
    for k in range(degree, -degree - 1, -1):
        z = -(b[:t] * vv).sum() / v1
        b = np.insert(b, 0, z)
    bb = b[degree : 2 * degree + 1]
    bb[1:] += b[degree - 1 :: -1]
    pk = coefs[: degree + 1] - bb
    return ChebFun(pk)


def cf_approximation(f: Function, degree: int, interval: Interval = (-1, 1)) -> ChebFun:
    a, b = interval
    needs_scaling = a != -1 or b != 1
    mid_point = (a + b) / 2
    half_len = (b - a) / 2
    if needs_scaling:

        def f_scaled(x):
            return f(mid_point + half_len * x)  # scale to [-1, 1]

    else:
        f_scaled = f
    approximant = _cf_approximation_on_standard_interval(f_scaled, degree)
    if needs_scaling:
        approximant._interval = interval
    return approximant
