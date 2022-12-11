from assertpy import assert_that
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import composite
from hypothesis.strategies import decimals
import numpy

from resource_usage_attribution import attribute_resource_usage

RESOURCES = ["flour", "leather"]
SERVICES = ["miller", "leathersmith"]
CUSTOMERS = ["cake", "handbag"]


def test_empty_inputs():
    resources = []
    services = []
    customers = []

    def usageFn(x, y):
        return 1

    assert_that(
        attribute_resource_usage(resources, services, customers, usageFn)
    ).is_equal_to(dict())


def test_single_path():
    resources = ["flour"]
    services = ["miller"]
    customers = ["cake"]

    usages = {("flour", "miller"): 1, ("miller", "cake"): 1}

    def usageFn(x, y):
        return usages.get((x, y), 0)

    expected_attribution = {"flour": {"cake": 1}}

    assert_that(
        attribute_resource_usage(resources, services, customers, usageFn)
    ).is_equal_to(expected_attribution)


def test_parallel_disjoint_edges():
    usages = {
        ("flour", "miller"): 1,
        ("miller", "cake"): 1,
        ("leather", "leathersmith"): 1,
        ("leathersmith", "handbag"): 1,
    }

    def usageFn(x, y):
        return usages.get((x, y), 0)

    expected_attribution = {
        "flour": {
            "cake": 1,
            "handbag": 0,
        },
        "leather": {
            "cake": 0,
            "handbag": 1,
        },
    }

    assert_that(
        attribute_resource_usage(RESOURCES, SERVICES, CUSTOMERS, usageFn)
    ).is_equal_to(expected_attribution)


def test_err_on_unnormalized_inputs():
    usages = {
        ("flour", "miller"): 1,
        ("miller", "cake"): 1,
        ("leather", "leathersmith"): 0.9,
        ("leathersmith", "handbag"): 1,
    }

    def usageFn(x, y):
        return usages.get((x, y), 0)

    assert_that(attribute_resource_usage).raises(ValueError).when_called_with(
        RESOURCES, SERVICES, CUSTOMERS, usageFn
    )


def test_err_on_unnormalized_inputs_2():
    usages = {
        ("flour", "miller"): 1.1,
        ("miller", "cake"): 1,
        ("leather", "leathersmith"): 1,
        ("leathersmith", "handbag"): 1,
    }

    def usageFn(x, y):
        return usages.get((x, y), 0)

    assert_that(attribute_resource_usage).raises(ValueError).when_called_with(
        RESOURCES, SERVICES, CUSTOMERS, usageFn
    )


def test_equal_split_at_service():
    usages = {
        ("flour", "miller"): 0.5,
        ("leather", "miller"): 0.5,
        ("flour", "leathersmith"): 0.5,
        ("leather", "leathersmith"): 0.5,
        ("miller", "cake"): 1,
        ("leathersmith", "handbag"): 1,
    }

    def usageFn(x, y):
        return usages.get((x, y), 0)

    """
    Markov transition matrix is

    flour  leather  miller  leathersmith  cake  handbag
    0      0        0.5     0.5           0     0
    0      0        0.5     0.5           0     0
    0      0        0       0             1     0
    0      0        0       0             0     1
    0      0        0       0             1     0
    0      0        0       0             0     1
    """

    expected_attribution = {
        "flour": {
            "cake": 0.5,
            "handbag": 0.5,
        },
        "leather": {
            "cake": 0.5,
            "handbag": 0.5,
        },
    }

    assert_that(
        attribute_resource_usage(RESOURCES, SERVICES, CUSTOMERS, usageFn)
    ).is_equal_to(expected_attribution)


def test_equal_split_at_customer():
    usages = {
        ("flour", "miller"): 1,
        ("leather", "leathersmith"): 1,
        ("miller", "cake"): 0.5,
        ("miller", "handbag"): 0.5,
        ("leathersmith", "handbag"): 0.5,
        ("leathersmith", "cake"): 0.5,
    }

    def usageFn(x, y):
        return usages.get((x, y), 0)

    """
    Markov transition matrix is

    flour  leather  miller  leathersmith  cake  handbag
    0      0        1       0             0     0
    0      0        0       1             0     0
    0      0        0       0             0.5   0.5
    0      0        0       0             0.5   0.5
    0      0        0       0             1     0
    0      0        0       0             0     1
    """

    expected_attribution = {
        "flour": {
            "cake": 0.5,
            "handbag": 0.5,
        },
        "leather": {
            "cake": 0.5,
            "handbag": 0.5,
        },
    }

    assert_that(
        attribute_resource_usage(RESOURCES, SERVICES, CUSTOMERS, usageFn)
    ).is_equal_to(expected_attribution)


def test_unequal_split_at_both_service_and_customer():
    usages = {
        ("flour", "miller"): 0.6,
        ("flour", "leathersmith"): 0.4,
        ("leather", "leathersmith"): 0.9,
        ("leather", "miller"): 0.1,  # leather...cake delivery bags!
        ("miller", "cake"): 0.8,
        ("miller", "handbag"): 0.2,
        ("leathersmith", "handbag"): 0.7,
        ("leathersmith", "cake"): 0.3,  # night job delivering cakes
    }

    def usageFn(x, y):
        return usages.get((x, y), 0)

    """
    Since it's still a DAG, absorbing probabilities are the sums
    of products of probabilities along all paths.
    """

    expected_attribution = {
        "flour": {
            "cake": 0.6 * 0.8 + 0.4 * 0.3,
            "handbag": 0.6 * 0.2 + 0.4 * 0.7,
        },
        "leather": {
            "cake": 0.1 * 0.8 + 0.9 * 0.3,
            "handbag": 0.9 * 0.7 + 0.1 * 0.2,
        },
    }

    actual_attribution = attribute_resource_usage(
        RESOURCES, SERVICES, CUSTOMERS, usageFn
    )

    for resource in RESOURCES:
        for customer in CUSTOMERS:
            assert_that(actual_attribution[resource][customer]).described_as(
                "attribution[%s][%s]" % (resource, customer)
            ).is_close_to(expected_attribution[resource][customer], 1e-10)


def test_many_cycles_slight_bias():
    resources = ["R_" + str(i) for i in range(5)]
    services = ["S_" + str(i) for i in range(5)]
    customers = ["C_" + str(i) for i in range(5)]

    no_bias = 1.0 / 10
    slight_bias = 1.0 / 10 + 0.001

    def usageFn(x, y):
        if y[0] == "R" or x[0] == "C":
            return 0
        if x[0] == "R":
            return no_bias
        if x == "S_0":
            if y == "C_4":
                return slight_bias
            else:
                return (1 - slight_bias) / 9  # remaining options
        return no_bias

    actual_attribution = attribute_resource_usage(
        resources, services, customers, usageFn
    )

    for resource in resources:
        for customer in customers:
            descr = "attribution[%s][%s]" % (resource, customer)
            if customer == customers[-1]:
                assert_that(actual_attribution[resource][customer]).described_as(
                    descr
                ).is_greater_than(1.0 / 5)
            else:
                assert_that(actual_attribution[resource][customer]).described_as(
                    descr
                ).is_less_than(1.0 / 5)


DIM = 5


@composite
def decimal_as_float(draw):
    x = draw(
        decimals(
            min_value=0.1,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            places=1,
        )
    )
    return float(x)


@settings(
    deadline=10000,  # 10 seconds
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    providers_to_customers_transition=arrays(
        float, (2 * DIM, 2 * DIM), elements=decimal_as_float()
    )
)
def test_exact_solution_matches_simulated_approximation(
    providers_to_customers_transition,
):
    """
    A hypothesis test with 10 resources, 10 services, arbitrary transitions,
    and arbitrary customer usage (normalized during test setup).

    Assert that the implementation is close to the process of iteratively applying
    the transition matrix to a vector of all 1's for each resource, i.e. simulating
    the flow. To help avoid numerical error messing things up, lower bound each
    transition probability to 0.1.
    """

    def normalize_rows(array):
        """Return a new array containing the normalized rows of the input array."""
        row_sums = array.sum(axis=1, keepdims=True)
        return (
            array
            / row_sums[
                :,
            ]
        )

    providers_to_customers_transition = normalize_rows(
        providers_to_customers_transition
    )

    resources = ["R_" + str(i) for i in range(DIM)]
    services = ["S_" + str(i) for i in range(DIM)]
    customers = ["C_" + str(i) for i in range(DIM)]
    providers = resources + services
    consumers = services + customers

    def usageFn(x, y):
        if x[0] == "C" and y[0] == "C":
            return 1 if x == y else 0

        if x[0] == "C" or y[0] == "R":
            return 0

        x_id = providers.index(x)
        y_id = consumers.index(y)

        return providers_to_customers_transition[x_id, y_id]

    actual_attribution = attribute_resource_usage(
        resources, services, customers, usageFn
    )

    total_transition_matrix = numpy.zeros((3 * DIM, 3 * DIM), dtype=float)
    total_transition_matrix[: 2 * DIM, DIM:] = providers_to_customers_transition
    total_transition_matrix[2 * DIM :, 2 * DIM :] = numpy.identity(DIM)

    # computes the probability that we end in state j given we start
    # from state i and given that we took k steps to get to state j
    exponentiated_transition = numpy.linalg.matrix_power(total_transition_matrix, 1024)

    for (i, resource) in enumerate(resources):
        state = numpy.zeros(3 * DIM, dtype=float)
        state[i] = 1.0
        state = numpy.dot(state, exponentiated_transition)

        for (j, customer) in enumerate(customers):
            expected_attribution = state[2 * DIM + j]
            assert_that(actual_attribution[resource][customer]).described_as(
                "attribution[%s][%s]" % (resource, customer)
            ).is_close_to(expected_attribution, 1e-02)
