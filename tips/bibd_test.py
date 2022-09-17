from dataclasses import replace
import pytest

from bibd import ALL_BIBDS
from bibd import BIBDParams
from bibd import is_bibd
from bibd import bibd_15_3_1


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


def test_from_bibd():
    expected = BIBDParams(
        subjects=35,
        treatments=15,
        treatments_per_subject=3,
        treatment_replication=7,
        pairwise_treatment_replication=1,
    )
    actual = BIBDParams.from_bibd(bibd_15_3_1)
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
        replace(params, treatment_replication=params.treatment_replication + 1),
        replace(params, treatment_replication=params.treatment_replication - 1),
        replace(
            params,
            pairwise_treatment_replication=params.pairwise_treatment_replication + 1,
        ),
        replace(
            params,
            pairwise_treatment_replication=params.pairwise_treatment_replication - 1,
        ),
    ]

    for params in bad_params:
        assert not params.satisfies_necessary_conditions()
