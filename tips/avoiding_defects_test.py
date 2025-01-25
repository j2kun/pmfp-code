from tips.avoiding_defects import single_sampling_scheme


def test_simple():
    assert single_sampling_scheme() is None
