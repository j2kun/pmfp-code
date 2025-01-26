from typing import Iterable, Union

import scipy.stats


def calculate_oc_hypergeom(
    sample_size: int,
    acceptance_number: int,
    population_size: int,
    defective_count: Union[int, Iterable[int]],
) -> Union[float, list[float]]:
    """Calculate the Operating Characteristic (OC) probability for a single-stage
    hypergeometric sampling plan.

    Returns float or list of floats computing the probability of accepting the lot given
    the sampling plan.
    """
    if not hasattr(defective_count, "__iter__"):
        defective_count = [defective_count]

    p_accept = [
        scipy.stats.hypergeom.cdf(
            k=acceptance_number,  # accept if we find <= c defects in the sample
            M=population_size,  # population size
            n=sample_size,  # sample size
            N=defects,  # number of defects in the population
        )
        for defects in defective_count
    ]

    # If input was single number, return single number
    if len(p_accept) == 1:
        return p_accept[0]
    return p_accept


def hypergeom(
    sample_size: int,
    acceptance_number: int,
    population_size: int,
    defective_count: int,
) -> float:
    """Calculate the hypergeometric function as used by the single sampling scheme."""
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
) -> tuple[int, int]:
    """Find the best single-scheme sampling plan.

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

    return (sample_size, defects_accepted)


def single_sampling_scheme():
    pass
