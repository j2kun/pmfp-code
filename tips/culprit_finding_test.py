from tips.culprit_finding import find_culprits


def test_trivial():
    assert find_culprits(lambda _: True, [], (1.0,), 0.0) is None
