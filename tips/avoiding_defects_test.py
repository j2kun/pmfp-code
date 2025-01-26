import hypothesis.strategies as st
from hypothesis import example, given, settings

from tips.avoiding_defects import find_plan_hypergeom, simulate_single_sampling_scheme


def test_simple():
    producer_risk_point = (0.05, 0.95)
    consumer_risk_point = (0.15, 0.20)
    population_size = 500

    sample_size, acceptance_number = find_plan_hypergeom(
        producer_risk_point,
        consumer_risk_point,
        population_size,
    )

    assert sample_size == 51
    assert acceptance_number == 5


def floats(min_value, max_value):
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_infinity=False,
        allow_nan=False,
        exclude_min=False,
    )


@settings(deadline=None)
@given(
    floats(min_value=0.01, max_value=0.10),
    floats(min_value=0.80, max_value=0.99),
    floats(min_value=0.01, max_value=0.10),
    floats(min_value=0.05, max_value=0.20),
)
@example(0.05, 0.95, 0.15, 0.20)
def test_acceptance_threshold(
    producer_risk_defect_rate,
    producer_risk_prob,
    consumer_risk_defect_rate_inc,
    consumer_risk_prob,
):
    # The producer's defect rate is for the prob of rejecting a good lot, while
    # the consumer's defect rate is for the prob of accepting a bad lot. In
    # order to make this consistent, the defect rate assumed by the consumer
    # must be larger than the defect rate assumed by the producer, or else the
    # scheme will trivially require 100% inspection. This materializes as the
    # hypergeometric function having invalid inputs or else producing a target
    # sample size that equals the population size.
    consumer_risk_defect_rate = (
        producer_risk_defect_rate + consumer_risk_defect_rate_inc
    )

    producer_risk_point = (producer_risk_defect_rate, producer_risk_prob)
    consumer_risk_point = (consumer_risk_defect_rate, consumer_risk_prob)
    population_size = 500

    sample_size, acceptance_number = find_plan_hypergeom(
        producer_risk_point,
        consumer_risk_point,
        population_size,
    )
    print(producer_risk_point, consumer_risk_point)
    print((population_size, sample_size, acceptance_number))

    if sample_size >= population_size:
        return  # hypothesis picked unreasonable parameters

    proportion_accepted = simulate_single_sampling_scheme(
        population_size=population_size,
        actual_defective=acceptance_number,
        plan=(sample_size, acceptance_number),
        seed=0,
        rounds=1000,
    )

    assert proportion_accepted > consumer_risk_point[1]
    assert (1 - proportion_accepted) < producer_risk_point[1]
