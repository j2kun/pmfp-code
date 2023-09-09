from assertpy import assert_that

import tips.five_star_ranking as five_star_ranking

TOLERANCE = 1e-10


def test_no_ratings_uses_prior():
    ratings = [0, 0]
    prior = [2, 3]

    expected_score = 3.0 / 5
    actual_score = five_star_ranking.score(ratings=ratings, rating_prior=prior)

    assert_that(expected_score).is_close_to(actual_score, TOLERANCE)


def test_thumbs_up_thumbs_down_no_prior():
    ratings = [2, 3]
    prior = [0, 0]

    expected_score = 3.0 / 5
    actual_score = five_star_ranking.score(ratings=ratings, rating_prior=prior)

    assert_that(expected_score).is_close_to(actual_score, TOLERANCE)


def test_thumbs_up_thumbs_down_with_prior():
    ratings = [2, 3]
    prior = [4, 3]

    expected_score = 6.0 / 12
    actual_score = five_star_ranking.score(ratings=ratings, rating_prior=prior)

    assert_that(expected_score).is_close_to(actual_score, TOLERANCE)


def test_thumbs_up_thumbs_down_with_prior_and_utility():
    ratings = [10, 10]
    prior = [10, 10]
    utility = [-4, 3]  # weight downvotes slightly higher than upvotes

    expected_score = (-4.0 * 20 + 3 * 20) / 40
    actual_score = five_star_ranking.score(
        ratings=ratings, rating_prior=prior, rating_utility=utility,
    )

    assert_that(expected_score).is_close_to(actual_score, TOLERANCE)


def test_five_star_rating():
    ratings = [10, 3, 4, 6, 9]
    prior = [1, 2, 3, 2, 1]
    utility = [-4, -3, 0, 2, 5]

    total = 41
    expected_score = (-44.0 + -15 + 16 + 50) / total
    actual_score = five_star_ranking.score(
        ratings=ratings, rating_prior=prior, rating_utility=utility,
    )

    assert_that(expected_score).is_close_to(actual_score, TOLERANCE)
