from hashlib import md5
from typing import Hashable
from typing import Iterable


LARGE_RANDOM_PRIME = 50075507486621659027


def trailing_zeros(n, z):
    '''Return the count of trailing 0's in the binary representation of an int.

    If the integer is zero, return z as a default. This assumes the input is
    truncated from a larger non-zero number, so z describes the length of the
    truncation.
    '''
    if n == 0:
        return z
    c = 0
    while (n >> c) & 1 == 0:
        c += 1
    return c


def random_hash(modulus):
    '''Return a random instance of a family of 2-universal hashes.'''
    a, b = random.randint(0, modulus - 1), random.randint(0, modulus - 1)

    def f(x):
        return (a * x + b) % modulus

    return f


def geometric_mean_estimate(buckets):
    '''Geometric mean of {m 2^b[0], ..., m 2^b[m-1]} with a bias correction

    Simplified formula computing: 

        (m 2^b[0] * ... * m 2^b[m-1])^(1/m)

    The bias correction factor approximates e^(−γ) * √2 / 2,
    where e is the Euler–Mascheroni constant γ ~ 0.5772156649...
    '''
    m = len(buckets)
    return 0.39701181 * m * 2**(float(sum(buckets)) / m)


def harmonic_mean_estimate(buckets):
    '''Harmonic mean of {m 2^b[0], ..., m 2^b[m-1]} with a bias correction
     
     Simplified formula computing:
         m / ((1 / (m 2^b[0])) + ... + (1 / (m 2^b[m-1])))
     
     The correction factor is an approximation of 1 / (2 ln 2).
     '''
    m = len(buckets)
    return 0.72134752 * m**2 / sum(2**(-b) for b in buckets)


def cardinality(data: Iterable[Hashable], log_bucket_count: int = 14) -> int:
    '''Estimate the number of unique items in the input dataset, up to 2^64.

    Arguments:
    - data: an iterable of arbitrary hashable data elements.
    - log_bucket_count: a value p such that the algorithm uses 2^p buckets, 
      each of which holds a 6 bit integer. 
    '''
    buckets = [0] * (1 << log_bucket_count)
    non_index_bits = 64 - log_bucket_count
    rhash = random_hash(LARGE_RANDOM_PRIME)

    for value in data:
        hashed = rhash(hash(value)) & ((1 << 64) - 1)
        # unpack hashed value as (s, index), where index is the last
        # log_bucket_count bits of the hash.
        index = hashed & (len(buckets) - 1)
        s = hashed >> log_bucket_count

        # most of the published literature uses "the position of the first 1
        # bit", rather than the number of trailing zeros. Choosing the latter
        # has the effect of dividing the estimate by 2; To follow the
        # literature, we use the position of the first 1 bit by adding 1 to the
        # number of trailing zeros.
        zeros = 1 + trailing_zeros(s, z=non_index_bits)
        buckets[index] = max(buckets[index], zeros)

    return harmonic_mean_estimate(buckets)


if __name__ == "__main__":
    import random
    import cProfile

    def random_data(n=100000):
        return [random.random() for i in range(n)]

    n = 100000
    for i in range(10):
      result = cardinality(random_data(n=n)) / n
      print(f"Error: {(1 - result)*100:.2f}%")
