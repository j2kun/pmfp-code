from collections import Counter
import random
from typing import Hashable
from typing import Iterable

LARGE_RANDOM_PRIME = 50075507486621659027


def trailing_zeros(n, z):
    """Return the count of trailing 0's in the binary representation of an int.

    If the integer is zero, return z as a default. This assumes the input is truncated
    from a larger non-zero number, so z describes the length of the truncation.
    """
    if n == 0:
        return z
    c = 0
    while (n >> c) & 1 == 0:
        c += 1
    return c


def random_hash(modulus):
    """Return a random instance of a family of 2-universal hashes."""
    a, b = random.randint(0, modulus - 1), random.randint(0, modulus - 1)

    def f(x):
        return (a * x + b) % modulus

    return f


class LogLog:
    def __init__(self, log_bucket_count: int = 14, estimator="improved"):
        estimator_dict = {
            "geometric": self.geometric_mean_estimate,
            "harmonic": self.harmonic_mean_estimate,
            "improved": self.improved_raw_estimate,
        }
        self.estimator_names = list(estimator_dict.keys())
        self.estimator = estimator_dict[estimator]

        self.log_bucket_count = log_bucket_count
        self.buckets = [0] * (1 << log_bucket_count)
        self.m = len(self.buckets)
        self.non_index_bits = 64 - log_bucket_count
        self.hash = random_hash(LARGE_RANDOM_PRIME)

    def add(self, value: Hashable) -> None:
        hashed = self.hash(hash(value)) & ((1 << 64) - 1)
        # unpack hashed value as (s, index), where index is the last
        # log_bucket_count bits of the hash.
        index = hashed & (len(self.buckets) - 1)
        s = hashed >> self.log_bucket_count

        # most of the published literature uses "the position of the first 1
        # bit", rather than the number of trailing zeros. Choosing the latter
        # has the effect of dividing the estimate by 2; To follow the
        # literature, we use the position of the first 1 bit by adding 1 to the
        # number of trailing zeros.
        zeros = 1 + trailing_zeros(s, z=self.non_index_bits)
        self.buckets[index] = max(self.buckets[index], zeros)

    def add_all(self, data: Iterable[Hashable]) -> None:
        for item in data:
            self.add(item)

    def cardinality(self) -> float:
        return self.estimator()

    def geometric_mean_estimate(self):
        """Geometric mean of {m 2^b[0], ..., m 2^b[m-1]} with a bias correction.

        Simplified formula computing:

            (m 2^b[0] * ... * m 2^b[m-1])^(1/m)

        The bias correction factor approximates e^(−γ) * √2 / 2,
        where e is the Euler–Mascheroni constant γ ~ 0.5772156649...
        """
        return 0.39701181 * self.m * 2 ** (float(sum(self.buckets)) / self.m)

    def harmonic_mean_estimate(self):
        """Harmonic mean of {m 2^b[0], ..., m 2^b[m-1]} with a bias correction.

        Simplified formula computing:
            m / ((1 / (m 2^b[0])) + ... + (1 / (m 2^b[m-1])))

        The correction factor is an approximation of 1 / (2 ln 2).
        """
        return 0.72134752 * self.m**2 / sum(2 ** (-b) for b in self.buckets)

    def improved_raw_estimate(self):
        """A modified harmonic mean, cf.

        https://arxiv.org/abs/1702.01284.
        """
        q = self.non_index_bits

        # Here we're counting the frequency of each run-length of trailing zeros,
        # and as such the values in the counter are between 0 and q+1, inclusive.
        # q+1 is the "default" when the entire register value is zero.
        register_counts = Counter(self.buckets)
        C = [register_counts[i] for i in range(0, q + 2)]

        def sigma(x):
            # The paper from the docstring remarks that 26 iterations is enough
            return x + sum(x ** (2**k) * 2 ** (k - 1) for k in range(1, 30))

        def tau(x):
            # The paper from the docstring remarks that 22 iterations is enough
            return (
                1
                - x
                - sum(2 ** (-k) * (1 - x ** (2 ** (-k))) ** 2 for k in range(1, 30))
            ) / 3

        denom = sum(2 ** (-k) * C[k] for k in range(1, q + 1))
        denom += self.m * sigma(C[0] / self.m)
        denom += self.m * tau(1 - C[q + 1] / self.m) * 2 ** (-q)
        return 0.72134752 * self.m**2 / denom


def cardinality(
    data: Iterable[Hashable],
    log_bucket_count: int = 14,
    estimator="improved",
) -> float:
    """Estimate the number of unique items in the input dataset, up to 2^64.

    Arguments:
    - data: an iterable of arbitrary hashable data elements.
    - log_bucket_count: a value p such that the algorithm uses 2^p buckets,
      each of which holds a 6 bit integer.
    """
    hll = LogLog(log_bucket_count, estimator=estimator)
    hll.add_all(data)
    return hll.cardinality()


if __name__ == "__main__":

    def random_data(n=100000):
        return [random.random() for i in range(n)]

    n = 100000
    for _ in range(10):
        result = cardinality(random_data(n=n)) / n
        print(f"Error: {(1 - result)*100:.2f}%")
