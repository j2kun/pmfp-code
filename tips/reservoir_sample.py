import math
import random
from typing import Iterable, List, Tuple, TypeVar

T = TypeVar("T")


def sample_without_replacement(sample_size: int, stream: Iterable[T]) -> List[T]:
    """Sample without replacement from a stream.

    The classical, but not optimally efficient reservoir sampling algorithm.
    """
    chosen = []
    for i, element in enumerate(stream):
        if i < sample_size:
            chosen.append(element)
        else:
            j = random.randint(0, i)
            if j < sample_size:
                chosen[j] = element

    return chosen


def algorithm_L(sample_size: int, stream: Iterable[T]) -> List[T]:
    """Sample without replacement from a stream.

    From Li 1994, "Reservoir-sampling algorithms of time complexity O(n(1 + log(N/n)))"

    This algorithm requires the
    """
    stream_iter = iter(stream)
    chosen = [next(stream_iter) for _ in range(sample_size)]
    W = math.exp(math.log(random.random()) / sample_size)

    while True:
        try:
            jump_ahead = math.floor(math.log(random.random()) / math.log(1 - W))
            # Python does not support "jumping ahead" in a generator efficiently
            # so this simulates that operation.
            for _ in range(jump_ahead):
                next(stream_iter)

            chosen[random.randint(0, sample_size - 1)] = next(stream_iter)
            W = W * math.exp(math.log(random.random()) / sample_size)
        except StopIteration:
            return chosen


def weighted_sample_with_replacement(
    sample_size: int,
    stream: Iterable[Tuple[T, float]],
) -> List[T]:
    """Sample with replacement from a stream according to item weights.

    This version demonstrates two variants: sampling with replacement instead
    of without replacement, and sampling with probability proportional to
    weights attached to (or quickly computable from) each stream item.

    Instead of replacing a random element, we allow each element to be independently
    replaced.
    """
    chosen: List[T] = []
    cumulative_weight = 0.0
    for element, weight in stream:
        cumulative_weight += weight
        if cumulative_weight == 0:
            continue

        if not chosen:
            chosen = [element for _ in range(sample_size)]
        else:
            for i in range(sample_size):
                choice = random.random()
                if choice < weight / cumulative_weight:
                    chosen[i] = element

    return chosen
