'''An implementation of a media mix model.'''

from typing import Dict
from typing import Iterable
from dataclasses import dataclass

import pymc3 as pm
import numpy as np


# TODO: add adstock function


def tanh_saturation(x, users_at_saturation, initial_cost_per_user):
    '''A saturation function based on the hyperbolic tangent.

    The saturation function represents the diminishing returns of an
    advertising channel.

    This is PyMC Labs's reparameterization of the original HelloFresh model, cf.
    https://web.archive.org/web/20211224060713/https://www.pymc-labs.io/blog-posts/reducing-customer-acquisition-costs-how-we-helped-optimizing-hellofreshs-marketing-budget/
    '''
    return users_at_saturation * np.tanh(x / (users_at_saturation * initial_cost_per_user))


def logistic_saturation(x, mu):
    '''A saturation function based on a logistic-like curve.

    This is HelloFresh's original model, cf.
    https://web.archive.org/web/20210123170937/https://discourse.pymc.io/t/a-bayesian-approach-to-media-mix-modeling-by-michael-johns-zhenyu-wang/6024

    With a per-channel prior on mu as pm.Gamma(alpha=3, beta=1).
    '''
    return (1 - np.exp(-mu * x)) / (1 + np.exp(-mu * x))


# TODO: Turn args into dataclasses & document their semantics
def make_model(channel_data, acquired_users):
    with pm.Model() as model:
        channel_models = []

        for (channel, weekly_spending) in channel_data.items():
            acquisition_rate = pm.HalfNormal(f'acquisition_rate_{channel}', sd=5)
            saturation_users = pm.HalfNormal(f'saturation_users_{channel}', sd=50)
            initial_cost = pm.HalfNormal(f'initial_cost_{channel}', sd=1)
            channel_models.append(
                acquisition_rate * tanh_saturation(
                    weekly_spending, saturation_users, initial_cost
                )
            )

        # The control coefficient is a stand-in for uncontrollable factors such
        # as overall trends in interest/disinterest in the product, or seasonal
        # factors.
        # TODO: explore adding a control component.
        # control_coeff = pm.Normal(f'control_coefs_{channel}', sd=1)

        baseline = pm.HalfNormal(f'baseline_{channel}', sd=1)
        output_noise = pm.Exponential(f'output_noise_{channel}', 10)
        new_customers = baseline + sum(channel_models)
        _ = pm.Normal(
            'likelihood', mu=new_customers, sd=output_noise, observed=acquired_users
        )

    return model


if __name__ == "__main__":
    channel_data = {
        'tv': np.array([10, 20]),
        'radio': np.array([20, 5]),
    }
    acquired_users = np.array([10, 8])
    model = make_model(channel_data, acquired_users)
    outcomes = pm.find_MAP(model=model)
    print(outcomes)
