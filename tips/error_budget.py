from dataclasses import dataclass
from datetime import timedelta
from typing import List
from typing import Optional


# assume minute-aligned samples
TimeSeries = List[int]


@dataclass
class SloMetric:
    violated: bool = False
    burn_rate: float = 0.0
    time_until_exhausted: Optional[timedelta] = None


def error_budget(
        requests: TimeSeries,
        errors: TimeSeries,
        budget: float,
        window_minutes: int) -> SloMetric:
    '''Return a guess of the estimated time until an error budget is exhausted.

    Assumes time series samples are minute-aligned, and that the requests and
    errors time series are aligned with each other.

    Arguments:
      - requests: cumulative requests since the start of the measurement period
      - errors: cumulative errors since the start of the measurement period
      - budget: the error budget, as a fraction of errors / requests
      - window: how far to look backwards to estimate error and request rates
    '''
    def budget_at(index):
        return int(budget * requests[index]) - errors[index]

    remaining_budget = budget_at(-1)
    violated = False
    if remaining_budget <= 0:
        violated = True

    # Estimate the first derivative of "remaining budget" curve
    prev_index = max(0, len(requests)-1 - window_minutes)
    estimated_budget_growth = (
        (remaining_budget - budget_at(prev_index)) 
        / window_minutes
    )

    if abs(estimated_budget_growth) < 1e-06:
        return SloMetric(violated=violated, burn_rate=0.0)

    # Solve budget_at(t + N) == 0 for N
    burn_minutes_remaining = remaining_budget / -estimated_budget_growth

    return SloMetric(
        violated=violated,
        # note burn rate is opposite of "budget growth"
        burn_rate=-estimated_budget_growth,
        time_until_exhausted=timedelta(minutes=burn_minutes_remaining),
    )


if __name__ == "__main__":
    from itertools import accumulate
    samples = 1000
    requests = list(accumulate([1000 for i in range(samples)]))
    errors = list(accumulate([i for i in range(samples)]))

    budget = 0.35
    print('measurement_index, violated, burn_rate, est_time_remaining')
    for index in range(5, 1000, 20):
        result = error_budget(
            requests[:index],
            errors[:index],
            budget,
            window_minutes=10,
        )
        s = f'{index}, {result.violated}, {result.burn_rate:.2f},'
        if result.burn_rate < 0 or result.violated:
            print(s)
        else:
            print(f'{s} {result.time_until_exhausted}')
