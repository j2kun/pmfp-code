from typing import Callable
from typing import Hashable
import math
import random


LARGE_RANDOM_PRIME = 175927138426321871


def random_hash(modulus):
    '''Return a random instance of a family of pairwise-independent hashes.'''
    a, b = random.randint(0, modulus - 1), random.randint(0, modulus - 1)

    def f(x):
        return (a * x + b) % modulus
    return f


class CountSketch:
    def __init__(self, accuracy: float = 0.001, confidence: float = 1e-05):
        '''Create an empty sketch with the given accuracy parameters.

        Arguments:
        - accuracy: the desired additive error of count estimates as a
          fraction of the total number of values processed
        - confidence: 1-confidence is the desired probability that the accuracy
          is achieved.
        '''
        self.accuracy = accuracy
        self.width = math.ceil(math.e / accuracy)
        self.hash_count = math.ceil(math.log(1 / confidence))
        self.sketch = [[0 for _ in range(self.width)] for _ in range(self.hash_count)]
        self.hashes = [random_hash(LARGE_RANDOM_PRIME) for _ in range(self.hash_count)]

    def increment(self, value: Hashable) -> None:
        for i, h in enumerate(self.hashes):
            self.sketch[i][self.hashes[i](hash(value)) % self.width] += 1

    def count(self, value: Hashable) -> int:
        return min([
            self.sketch[i][self.hashes[i](hash(value)) % self.width]
            for i, h in enumerate(self.hashes)
        ])

    def __len__(self):
        '''Return the total number of increments applied.'''
        return sum(self.sketch[0])

    def additive_error_guarantee(self):
        return math.ceil(len(self) * self.accuracy)


if __name__ == "__main__":
    sketch = CountSketch()
    for i in range(10):
        sketch.increment(12)
    for i in range(10000):
        sketch.increment(i)

    # perfect estimate would be 10
    print(sketch.count(12))

    # actual estimate is more like 35
    print(len(sketch))

    # guarantee is "overestimated by at most 70"
    print(sketch.additive_error_guarantee())
