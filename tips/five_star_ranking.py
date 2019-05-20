from typing import List
from typing import Optional


def score(ratings: List[int],
          rating_prior: List[int],
          rating_utility: Optional[List[float]] = None) -> float:
    '''Compute the expected utility of a listing with discrete ratings.

    This score can be used as a sorting key to rank listings by the order they
    should be displayed to a user. Higher scores should be displayed first.

    Arguments:
      - ratings: A histogram of numeric ratings for the listing to be scored
      - rating_prior: A histogram of "ghost ratings" that represent the
          prior belief of quality for a listing with no ratings in the system.
      - rating_utility: A list of utility values for each rating option. If
          None, a linear utility scale is used by default.

    The input arguments must each have the same length. E.g., if the listing
    has 1-5 star ratings, each list has length 5. The ratings argument
    contains a count of ratings for each star value, the prior contains a
    similar count for the prior, and the utility contains a float denoting
    how much value each star rating counts for. For the latter, and depending
    on your application, a certain rating might act as a user "default,"
    and can be given less weight accordingly.

    Returns:
      A float score representing the expected utility of a listing with the
      given ratings.
    '''
    if not rating_utility:
        rating_utility = list(range(len(ratings)))

    total_ratings = [r + p for (r, p) in zip(ratings, rating_prior)]
    score = sum(r * u for (r, u) in zip(total_ratings, rating_utility))
    return score / sum(total_ratings)
