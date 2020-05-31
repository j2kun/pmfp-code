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
