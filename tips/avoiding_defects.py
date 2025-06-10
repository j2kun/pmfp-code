import random
from dataclasses import dataclass

import scipy.stats


@dataclass
class SingleSamplingPlan:
    # The number of items to sample from the lot
    sample_size: int

    # The maximum number of defective items in the sample before the lot is
    # rejected
    acceptance_number: int


def hypergeom(
    sample_size: int,
    acceptance_number: int,
    population_size: int,
    defective_count: int,
) -> float:
    """Calculate the hypergeometric function as used by the single sampling plan."""
    return scipy.stats.hypergeom.cdf(
        k=acceptance_number,  # accept if we find <= c defects in the sample
        M=population_size,  # population size
        n=sample_size,  # sample size
        N=defective_count,  # number of defects in the population
    )


def find_plan_hypergeom(
    producer_risk_point: tuple[float, float],
    consumer_risk_point: tuple[float, float],
    population_size: int,
) -> SingleSamplingPlan:
    """Find the best single sampling plan.

    Args:
        producer_risk_point: (d, p) implying lots with defect rate <= d are
            accepted with probability >= p
        consumer_risk_point: (d, p) implying lots with defect rate >= d are
            accepted with probability <= p
        population_size: Size of the lot

    Returns: a tuple containing the best sample size and number of defects allowed.
    """
    assert 0 < producer_risk_point[0] < 1
    assert 0 < producer_risk_point[1] < 1
    assert 0 < consumer_risk_point[0] < 1
    assert 0 < consumer_risk_point[1] < 1

    defects_accepted = 0
    sample_size = defects_accepted + 1

    while True:
        prob_accept_bad_lot = hypergeom(
            sample_size=sample_size,
            acceptance_number=defects_accepted,
            population_size=population_size,
            defective_count=int(consumer_risk_point[0] * population_size),
        )
        prob_accept_good_lot = hypergeom(
            sample_size=sample_size,
            acceptance_number=defects_accepted,
            population_size=population_size,
            defective_count=int(producer_risk_point[0] * population_size),
        )

        if prob_accept_bad_lot > consumer_risk_point[1]:
            sample_size += 1  # Increase sample size to be more stringent
        elif prob_accept_good_lot < producer_risk_point[1]:
            defects_accepted += 1  # Allow one more defect to be more lenient
        else:
            # If both checks pass, the plan is good
            break

    return SingleSamplingPlan(sample_size, defects_accepted)


def simulate_single_sampling_plan(
    population_size: int,
    actual_defective: int,
    plan: tuple[int, int],
    seed: int = 0,
    rounds: int = 1000,
):
    items = list(range(population_size))
    random.seed(seed)

    accepted_count = 0
    for i in range(rounds):
        defective_items = random.sample(items, actual_defective)
        sample = random.sample(items, plan[0])
        accepted = len(set(sample) & set(defective_items)) <= plan[1]
        accepted_count += int(accepted)

    return accepted_count / rounds


if __name__ == "__main__":
    sample_size = 50
    acceptance_number = 10
    population_size = 1000
    defective_count = list(range(500))

    # Calculate the Operating Characteristic (OC) probability for a
    # single-stage hypergeometric sampling plan. The graph depicts the
    # probability of accepting the lot given the sampling plan.
    p_accept = [
        hypergeom(sample_size, acceptance_number, population_size, defects)
        for defects in defective_count
    ]

    import matplotlib.pyplot as plt

    plt.plot(
        defective_count,
        p_accept,
        linewidth=2,
    )
    plt.xlabel("actual defect count")
    plt.ylabel("acceptance probability")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
