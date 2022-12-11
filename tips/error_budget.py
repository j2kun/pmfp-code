from dataclasses import dataclass
from datetime import timedelta
from typing import List
from typing import Optional


# assume minute-aligned samples
TimeSeries = List[int]


@dataclass
class SloMetric:
    violated: bool = False
    budget_growth_rate: float = 0.0
    time_until_exhausted: Optional[timedelta] = None


def error_budget_remaining(
    requests: TimeSeries, errors: TimeSeries, budget: float, window_minutes: int
) -> SloMetric:
    """Return a guess of the estimated time until an error budget is exhausted.

    Assumes time series samples are minute-aligned, and that the requests and
    errors time series are aligned with each other.

    Arguments:
      - requests: cumulative requests since the start of the measurement period
      - errors: cumulative errors since the start of the measurement period
      - budget: the error budget, as a fraction of errors / requests
      - window: how far to look backwards to estimate error and request rates
    """

    def budget_at(index):
        return int(budget * requests[index]) - errors[index]

    violated = budget_at(-1) <= 0

    # Estimate the first derivative of "remaining budget" curve
    prev_index = max(0, len(requests) - 1 - window_minutes)
    estimated_budget_growth_rate = (
        budget_at(-1) - budget_at(prev_index)
    ) / window_minutes

    if abs(estimated_budget_growth_rate) < 1e-06:
        return SloMetric(violated=violated, budget_growth_rate=0.0)

    burn_minutes_remaining = budget_at(-1) / -estimated_budget_growth_rate
    return SloMetric(
        violated=violated,
        budget_growth_rate=estimated_budget_growth_rate,
        time_until_exhausted=timedelta(minutes=burn_minutes_remaining),
    )


if __name__ == "__main__":
    from datetime import datetime
    from itertools import accumulate
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    import random

    random.seed(123)
    samples = 1000
    requests = list(accumulate([1000 for i in range(samples)]))
    errors = list(accumulate([random.randint(0, 2 * i) for i in range(samples)]))

    budget = 0.35
    print("measurement_index, violated, budget_growth_rate, est_time_remaining")
    for index in range(5, 1000, 20):
        result = error_budget_remaining(
            requests[:index],
            errors[:index],
            budget,
            window_minutes=10,
        )
        s = f"{index}, {result.violated}, {result.budget_growth_rate:.2f},"
        if result.budget_growth_rate > 0 or result.violated:
            print(s)
        else:
            print(f"{s} {result.time_until_exhausted}")

    # draw the estimated budget curve starting from index 600
    index = 600
    result = error_budget_remaining(
        requests[:index],
        errors[:index],
        budget,
        window_minutes=10,
    )

    d = [datetime.now() + timedelta(minutes=i) for i in range(samples)]
    values = [budget * requests[i] - errors[i] for i in range(samples)]
    violation_time = d[index - 1] + (
        result.time_until_exhausted or timedelta(minutes=0)
    )

    plt.plot(d[:index], values[:index], linewidth=3, label=r"error budget")

    xaxis_sample_width = dates.date2num(d[1]) - dates.date2num(d[0])
    plt.axline(
        (dates.date2num(d[index - 1]), values[index - 1]),
        slope=result.budget_growth_rate / xaxis_sample_width,
        linewidth=1,
        color="black",
        linestyle="--",
        label="burn rate",
    )
    plt.axvline(
        violation_time,
        linewidth=1,
        color="red",
        linestyle=":",
        label="estimated violation",
    )
    plt.xlim(d[0], d[-200])
    plt.xlabel("time")

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()

    ax = plt.gca()
    ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    ax.set_ylim(bottom=0)

    plt.gcf().autofmt_xdate()
    plt.show()
