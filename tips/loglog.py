from typing import Hashable
from typing import Iterable



def loglog(data: Iterable[Hashable], log_bucket_count: int) -> int:
    '''Estimate the number of unique items in the input dataset.

    Arguments:
    - data: an iterable of arbitrary hashable data elements.
    - log_bucket_count: a value k such that the algorithm uses 2^k buckets, 
      each of which holds 5 bits.
    '''
    # TODO: improve 5 bits (2^27 max cardinality) to something larger?
    # what will it take to get a 64-bit hash to work here?
    pass

