from dataclasses import replace

import pytest

from tips.bibd import ALL_BIBDS, BIBDParams, bibd_8_4_3, bibd_15_3_1, is_bibd


@pytest.mark.parametrize("bibd", ALL_BIBDS)
def test_is_bibd(bibd):
    assert is_bibd(bibd)


@pytest.mark.parametrize("bibd", ALL_BIBDS)
def test_dropping_a_block_breaks_bibd(bibd):
    for i in range(len(bibd)):
        dropped_one_block = bibd[:i] + bibd[i + 1 :]
        assert not is_bibd(dropped_one_block)


def test_different_block_sizes_break_bibd():
    assert not is_bibd(((1, 2), (1, 2, 3)))


def test_incorrect_pairwise_membership_counts_breaks_isbibd():
    # This trick just appends two identical block designs but has no overlap in
    # the treatments from the appended designs. This results in a design that
    # is correct except it is missing some pairwise memberships.
    assert not is_bibd(
        (
            (1, 2, 3),
            (1, 2, 4),
            (2, 3, 4),
            (1, 3, 4),
            (5, 6, 7),
            (5, 6, 8),
            (6, 7, 8),
            (5, 7, 8),
        ),
    )


def test_unequal_pairwise_memberships_breaks_isbibd():
    # similar to previous test, but a few sets are swapped
    # to give unequal membership counts
    assert not is_bibd(
        (
            (1, 2, 4),
            (2, 3, 4),
            (1, 3, 4),
            (5, 6, 8),
            (6, 7, 8),
            (5, 7, 8),
            (1, 2, 5),
            (3, 6, 7),
        ),
    )


def test_from_bibd():
    expected = BIBDParams(
        subjects=35,
        treatments=15,
        treatments_per_subject=3,
        subjects_per_treatment=7,
        subjects_per_treatment_pair=1,
    )
    actual = BIBDParams.from_bibd(bibd_15_3_1)
    assert expected == actual


def test_from_bibd_2():
    expected = BIBDParams(
        treatments=8,
        subjects=14,
        subjects_per_treatment=7,
        treatments_per_subject=4,
        subjects_per_treatment_pair=3,
    )
    actual = BIBDParams.from_bibd(bibd_8_4_3)
    assert expected == actual


@pytest.mark.parametrize("bibd", ALL_BIBDS)
def test_bibd_params_satisfy_necessary_conditions(bibd):
    assert BIBDParams.from_bibd(bibd_15_3_1).satisfies_necessary_conditions()


@pytest.mark.parametrize("bibd", ALL_BIBDS)
def test_bibd_params_tweaked_fail_conditions(bibd):
    params = BIBDParams.from_bibd(bibd)
    bad_params = [
        replace(params, subjects=params.subjects + 1),
        replace(params, subjects=params.subjects - 1),
        replace(params, treatments=params.treatments + 1),
        replace(params, treatments=params.treatments - 1),
        replace(params, treatments_per_subject=params.treatments_per_subject + 1),
        replace(params, treatments_per_subject=params.treatments_per_subject - 1),
        replace(params, subjects_per_treatment=params.subjects_per_treatment + 1),
        replace(params, subjects_per_treatment=params.subjects_per_treatment - 1),
        replace(
            params,
            subjects_per_treatment_pair=params.subjects_per_treatment_pair + 1,
        ),
        replace(
            params,
            subjects_per_treatment_pair=params.subjects_per_treatment_pair - 1,
        ),
    ]

    for params in bad_params:
        assert not params.satisfies_necessary_conditions()


def test_specific_efficiency_factor():
    params = BIBDParams.from_bibd(bibd_8_4_3)
    assert (params.efficiency_factor() - 0.857142) < 1e-05


@pytest.mark.parametrize("bibd", ALL_BIBDS)
def test_efficiency_factor_agrees_with_book(bibd):
    # In the book I use a different formula for efficiency factor to avoid
    # introducing more variables at that part of the prose.
    params = BIBDParams.from_bibd(bibd)
    book_claim = (params.r * (params.k - 1) + params.lambda_) / (params.r * params.k)
    assert (params.efficiency_factor() - book_claim) < 1e-05


@pytest.mark.parametrize("bibd", ALL_BIBDS)
def test_efficiency_factor_less_than_1(bibd):
    params = BIBDParams.from_bibd(bibd)
    assert 0 < params.efficiency_factor() < 1
