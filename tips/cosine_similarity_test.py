import math

import numpy as np

from tips.cosine_similarity import cosine_similarity


def test_cosine_similarity():
    x = np.array([1, 2, 3])
    y = np.array([2, 1, 1])
    assert np.isclose(cosine_similarity(x, y), 0.763763)


def test_cosine_similarity_fully_aligned():
    x = np.array([1, 2, 3])
    assert np.isclose(cosine_similarity(x, x), 1)


def test_cosine_similarity_orthogonal_2d():
    x = np.array([1, 2])
    rot90 = np.array([[0, -1], [1, 0]])
    y = np.dot(rot90, x)
    assert np.isclose(cosine_similarity(x, y), 0)
    y = np.dot(-rot90, x)
    assert np.isclose(cosine_similarity(x, y), 0)


def test_cosine_similarity_orthogonal_3d():
    x = np.array([1, math.sqrt(2), 1])
    y = np.array([1, -math.sqrt(2), 1])
    assert np.isclose(cosine_similarity(x, y), 0)


def test_cosine_similarity_opposites():
    x = np.array([1, 2, 3])
    assert np.isclose(cosine_similarity(x, -x), -1)
