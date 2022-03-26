'''An implementation of a randomized response protocol.'''

from typing import List
from typing import Tuple
import random


def respond_privately(true_answer: bool) -> bool:
    '''Respond to a survey with plausible deniability about your answer.'''
    be_honest = random.random() < 0.5
    random_answer = random.random() < 0.5
    return true_answer if be_honest else random_answer


def aggregate_responses(responses: List[bool]) -> Tuple[float, float]:
    '''Return the estimated fraction of survey respondents that have a truthful
    "Yes" answer to the survey question.

    If p is the true fraction of respondents with a "Yes" answer, then in
    expectation we will see (1/4)(1-p) + (3/4)p = (1/4) + p/2 "Yes" responses.
    This function solves for p.
    '''
    yes_response_count = sum(responses)
    n = len(responses)
    mean = 2 * yes_response_count / n - 0.5
    # Use n-1 when estimating variance, as per Bessel's correction.
    # The second term represents the additional variance introduced
    # by the private mechanism.
    variance = mean * (1 - mean) / (n - 1) + 1 / (n - 1)
    return (mean, variance)
