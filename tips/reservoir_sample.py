import math
import random

from typing import Iterable
from typing import List
from typing import TypeVar

T = TypeVar('T')


def reservoir_sample(sample_size: int, stream: Iterable[T]) -> List[T]:
    '''Sample without replacement from a stream.'''
    chosen = []
    for i, element in enumerate(stream):
        if i < sample_size:
            chosen.append(element)
        else:
            j = random.randint(0, i)
            if j < sample_size:
                chosen[j] = element

    return chosen


def reservoir_sample_L(sample_size: int, stream: Iterable[T]) -> List[T]:
    '''Sample without replacement from a stream.

    From Li 1994, "Reservoir-sampling algorithms of time complexity O(n(1 + log(N/n)))"
    '''
    stream_iter = iter(stream)
    chosen = [next(stream_iter) for _ in range(sample_size)]
    W = math.exp(math.log(random.random()) / sample_size)

    while True:
        try:
            jump_ahead = math.floor(math.log(random.random())/math.log(1-W))
            for _ in range(jump_ahead):
                next(stream_iter)

            chosen[random.randint(0, sample_size-1)] = next(stream_iter)
            W = W * math.exp(math.log(random.random()) / sample_size)
        except StopIteration:
            return chosen
