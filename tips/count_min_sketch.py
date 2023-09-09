import heapq
import math
import random
from typing import Hashable, Iterable, List, Set, Tuple, TypeVar

LARGE_RANDOM_PRIME = 175927138426321871


def random_hash(modulus):
    """Return a random instance of a family of 2-universal hashes."""
    a, b = random.randint(0, modulus - 1), random.randint(0, modulus - 1)

    def f(x):
        return (a * x + b) % modulus

    return f


class CountMinSketch:
    def __init__(self, accuracy: float = 0.001, confidence: float = 1e-05):
        """Create an empty sketch with the given accuracy parameters.

        Arguments:
        - accuracy: the desired additive error of count estimates as a
          fraction of the total number of values processed
        - confidence: 1-confidence is the desired probability that the accuracy
          is achieved.
        """
        self.accuracy = accuracy
        self.width = math.ceil(math.e / accuracy)
        self.hash_count = math.ceil(math.log(1 / confidence))
        self.sketch = [[0 for _ in range(self.width)] for _ in range(self.hash_count)]
        self.hashes = [random_hash(LARGE_RANDOM_PRIME) for _ in range(self.hash_count)]

    def increment(self, value: Hashable) -> None:
        for i, h in enumerate(self.hashes):
            self.sketch[i][self.hashes[i](hash(value)) % self.width] += 1

    def count(self, value: Hashable) -> int:
        return min(
            [
                self.sketch[i][self.hashes[i](hash(value)) % self.width]
                for i, h in enumerate(self.hashes)
            ],
        )

    def __len__(self):
        """Return the total number of increments applied."""
        return sum(self.sketch[0])

    def additive_error_guarantee(self):
        return math.ceil(len(self) * self.accuracy)


T = TypeVar("T")


def heavy_hitters(data: Iterable[T], k: int) -> Set[T]:
    """Find the data entries that occur at least len(values) / k times.

    This computes an approximate "heavy hitters" function, whereby it guarantees to
    return all data entries that occur at least len(values) / k times, but it may return
    some values that occur at least len(values) / (2k) times. It is guaranteed with high
    probability not to return any values that occur fewer than len(values) / (2k) times.

    In exchange for this approximation, this method uses only O(k) space, and O(log(k))
    time per data entry.
    """
    heap: List[Tuple[int, T]] = []
    candidates: Set[T] = set()
    count_processed = 0
    sketch = CountMinSketch(accuracy=1 / (2 * k), confidence=1e-10)

    def remove_heap_items_below(value):
        while heap:
            min_count, min_value = heapq.heappop(heap)
            if min_count >= count_processed / k:
                heapq.heappush(heap, (min_count, min_value))
                return
            candidates.remove(min_value)

    for value in data:
        count_processed += 1
        sketch.increment(value)
        est_count = sketch.count(value)

        if est_count >= count_processed / k:
            if value in candidates:
                # This is actually O(k) time because Python's heapq doesn't
                # support deleting an arbitrary element from the heap in log
                # time. Implement a better heap to get the faster runtime.
                heap = [(c, x) for (c, x) in heap if x != value]
                heapq.heapify(heap)

            if heap:
                remove_heap_items_below(count_processed / k)
            candidates.add(value)
            heapq.heappush(heap, (est_count, value))

    return {x for (c, x) in heap}


if __name__ == "__main__":
    from collections import Counter

    random.seed(1)

    n = 1000
    k = 10

    # min output value has frequency n / 2k = 10
    # guarantees all entries with at least n / k = 20 freq

    values = [-1] * (n // k) + [-2] * (n // (2 * k) + 1) + [-3] * (n // (2 * k) + 5)

    remaining_space = n - len(values)
    threshold = n // (2 * k) - 1
    just_under_heavy = list(range(remaining_space // threshold)) * threshold
    values = values + just_under_heavy
    print(Counter(values))
    random.shuffle(values)

    # with this seed, it always chooses -1 (most frequent) but only chooses -2
    # even though -3 is more frequent (though not above the threshold for
    # guarantee)
    print(heavy_hitters(values, k))
