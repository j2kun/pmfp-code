from assertpy import assert_that

from resource_usage_attribution import attribute_resource_usage


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
    resources = ['flour']
    services = ['miller']
    customers = ['cake']

    usages = {('flour', 'miller'): 1, ('miller', 'cake'): 1}

    def usageFn(x, y):
        return usages.get((x, y), 0)

    expected_attribution = {'flour': {'cake': 1}}

    assert_that(
        attribute_resource_usage(resources, services, customers, usageFn)
    ).is_equal_to(expected_attribution)


def test_parallel_disjoint_edges():
    resources = ['flour', 'leather']
    services = ['miller', 'leathersmith']
    customers = ['cake', 'handbag']

    usages = {
        ('flour', 'miller'): 1,
        ('miller', 'cake'): 1,
        ('leather', 'leathersmith'): 1,
        ('leathersmith', 'handbag'): 1,
    }

    def usageFn(x, y):
        return usages.get((x, y), 0)

    expected_attribution = {
        'flour': {
            'cake': 1,
            'handbag': 0,
        },
        'leather': {
            'cake': 0,
            'handbag': 1,
        },
    }

    assert_that(
        attribute_resource_usage(resources, services, customers, usageFn)
    ).is_equal_to(expected_attribution)


def test_equal_split_at_service():
    resources = ['flour', 'leather']
    services = ['miller', 'leathersmith']
    customers = ['cake', 'handbag']

    usages = {
        ('flour', 'miller'): 0.5,
        ('leather', 'miller'): 0.5,
        ('flour', 'leathersmith'): 0.5,
        ('leather', 'leathersmith'): 0.5,
        ('miller', 'cake'): 1,
        ('leathersmith', 'handbag'): 1,
    }

    def usageFn(x, y):
        return usages.get((x, y), 0)

    '''
    Markov transition matrix is

    flour  leather  miller  leathersmith  cake  handbag
    0      0        0.5     0.5           0     0
    0      0        0.5     0.5           0     0
    0      0        0       0             1     0
    0      0        0       0             0     1
    0      0        0       0             1     0
    0      0        0       0             0     1
    '''

    expected_attribution = {
        'flour': {
            'cake': 0.5,
            'handbag': 0.5,
        },
        'leather': {
            'cake': 0.5,
            'handbag': 0.5,
        },
    }

    assert_that(
        attribute_resource_usage(resources, services, customers, usageFn)
    ).is_equal_to(expected_attribution)


def test_equal_split_at_customer():
    resources = ['flour', 'leather']
    services = ['miller', 'leathersmith']
    customers = ['cake', 'handbag']

    usages = {
        ('flour', 'miller'): 1,
        ('leather', 'leathersmith'): 1,
        ('miller', 'cake'): 0.5,
        ('miller', 'handbag'): 0.5,
        ('leathersmith', 'handbag'): 0.5,
        ('leathersmith', 'cake'): 0.5,
    }

    def usageFn(x, y):
        return usages.get((x, y), 0)

    '''
    Markov transition matrix is

    flour  leather  miller  leathersmith  cake  handbag
    0      0        1       0             0     0
    0      0        0       1             0     0
    0      0        0       0             0.5   0.5
    0      0        0       0             0.5   0.5
    0      0        0       0             1     0
    0      0        0       0             0     1
    '''

    expected_attribution = {
        'flour': {
            'cake': 0.5,
            'handbag': 0.5,
        },
        'leather': {
            'cake': 0.5,
            'handbag': 0.5,
        },
    }

    assert_that(
        attribute_resource_usage(resources, services, customers, usageFn)
    ).is_equal_to(expected_attribution)


def test_unequal_split_at_both_service_and_customer():
    resources = ['flour', 'leather']
    services = ['miller', 'leathersmith']
    customers = ['cake', 'handbag']

    usages = {
        ('flour', 'miller'): 0.6,
        ('flour', 'leathersmith'): 0.4,
        ('leather', 'leathersmith'): 0.9,
        ('leather', 'miller'): 0.1,  # leather...cake delivery bags!
        ('miller', 'cake'): 0.8,
        ('miller', 'handbag'): 0.2,
        ('leathersmith', 'handbag'): 0.7,
        ('leathersmith', 'cake'): 0.3,  # night job delivering cakes
    }

    def usageFn(x, y):
        return usages.get((x, y), 0)

    '''
    Since it's still a DAG, absorbing probabilities are the sums
    of products of probabilities along all paths.
    '''

    expected_attribution = {
        'flour': {
            'cake': 0.6 * 0.8 + 0.4 * 0.3,
            'handbag': 0.6 * 0.2 + 0.4 * 0.7,
        },
        'leather': {
            'cake': 0.1 * 0.8 + 0.9 * 0.3,
            'handbag': 0.9 * 0.7 + 0.1 * 0.2,
        },
    }

    actual_attribution = attribute_resource_usage(
        resources, services, customers, usageFn
    )

    for resource in resources:
        for customer in customers:
            assert_that(actual_attribution[resource][customer]).described_as(
                "attribution[%s][%s]" % (resource, customer)
            ).is_close_to(expected_attribution[resource][customer], 1e-10)
