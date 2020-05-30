from typing import Callable
from typing import Dict
from typing import List
from typing import TypeVar
from typing import Union
import numpy

# just for type clarity
Resource = TypeVar('Resource')
Service = TypeVar('Service')
Customer = TypeVar('Customer')

# A service can be a provider for other services
Provider = Union[Resource, Service]
Consumer = Union[Service, Customer]

UsageFn = Callable[[Provider, Consumer], float]
Attribution = Dict[Resource, Dict[Consumer, float]]


def attribute_resource_usage(resources: List[Resource],
                             services: List[Service],
                             customers: List[Customer],
                             usageFn: UsageFn) -> Attribution:
    '''Attribute the resource usage among the terminal customers of services.

    In this context a Resource is something that is provided out of thin air,
    consumes nothing, and can be consumed by either Services or Customers.
    A Service can consume resources or the output of other services, and
    whose output can be Consumed by other Services or by Customers. A
    Customer can consume resources like a Service, but produces no
    consumable output relevant to this model.

    A Provider is either a Resource or a Service (provides a consumable thing)
    and a Consumer is either a Service or a Consumer (consumes a consumable
    thing).  We require no explicit description of the quantity provided by a
    Service, so long as it is divisible among the clients of the Service.

    The usageFn describes the fraction in [0,1] of the output of each Provider
    that is consumed by each Consumer.

    Each Resource has total supply 1, which can be rescaled by the caller.

    If some Service consumes resources, but not all of its output is consumed
    by Consumers, then the remaining fraction of resources used are considered
    to be used by the Service as if it were a Customer.

    Arguments:
      - resources: a list of Resources
      - services: a list of Services
      - customers: a list of Customers
      - usageFn: a callable accepting as input a Provider (resource/service) P
          and a consuming Consumer (service/customer) U, and produces as output the
          fraction of the total output of P consumed by U. For a fixed P, the sum
          of the outputs of usageFn(P, U) over all users U must be at most 1, or
          a ValueError is raised.

    Returns:
      A dict describing the attribution of each resource among the Services and
      Customers.
    '''
    # construct the array of transitions among transient states Q
    Q = np.array([[1, 2], [3, 4]])
    R = np.array([[1, 2], [3, 4]])

    # invert N = (1-Q)^{-1}
    fundamental_matrix = np.linalg.inv(I - Q)

    # compute N * R to get absorbing probabilities
    absorbing_probabilities = numpy.dot(N, R)

    # construct output dict
    return dict()
