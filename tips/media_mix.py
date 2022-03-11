'''An implementation of a media mix model.'''

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
    return users_at_saturation * pm.math.tanh(x / (users_at_saturation * initial_cost_per_user))


def logistic_saturation(x, mu):
    '''A saturation function based on a logistic-like curve.

    This is HelloFresh's original model, cf.
    https://web.archive.org/web/20210123170937/https://discourse.pymc.io/t/a-bayesian-approach-to-media-mix-modeling-by-michael-johns-zhenyu-wang/6024

    With a per-channel prior on mu as pm.Gamma(alpha=3, beta=1).
    '''
    return (1 - np.exp(-mu * x)) / (1 + np.exp(-mu * x))


# TODO: Turn args into dataclasses & document their semantics
def make_model(channel_data, sales):
    with pm.Model() as model:
        channel_models = []

        for (channel, weekly_spending) in channel_data.items():
            acquisition_rate = pm.Gamma(f'acquisition_rate_{channel}', alpha=2, beta=1)

            # in the tanh saturation function, we require the product of these
            # two terms is nonzero to avoid dividing by zero.
            saturation_users = pm.Gamma(f'saturation_users_{channel}', alpha=15, beta=0.5)
            initial_cost = pm.Gamma(f'initial_cost_{channel}', alpha=5, beta=1)
            channel_models.append(
                acquisition_rate * tanh_saturation(
                    weekly_spending, saturation_users, initial_cost
                )
            )

        # The control coefficient is a stand-in for uncontrollable factors such
        # as overall trends in interest/disinterest in the product, or seasonal
        # factors.
        # TODO: explore adding a control component.
        # control_coeff = pm.Normal('control_coefs', sd=1)

        baseline = pm.HalfNormal('baseline', sigma=1)
        output_noise = pm.HalfNormal('output_noise', sigma=1)
        new_sales = baseline + sum(channel_models)
        _ = pm.Normal(
            'likelihood', mu=new_sales, sd=output_noise, observed=sales
        )

    return model


if __name__ == "__main__":
    import arviz as az
    import matplotlib.pyplot as plt
    import csv

    data = []
    with open('data/media_mix_data.csv', 'r') as infile:
        reader = csv.reader(infile)
        is_header = True
        headers = []
        for row in reader:
            if is_header:
                headers = row
                is_header = False
            else:
                data.append(dict(zip(headers, row)))

    channel_data = {k: [float(d[k]) for d in data] for k in data[0]}
    sales = channel_data['sales']
    del channel_data['week']
    del channel_data['sales']

    model = make_model(channel_data, sales)
    with model:
        trace = pm.sample(10, return_inferencedata=False)
        summary = az.summary(trace, round_to=2)
        print(summary)

        az.plot_trace(trace)
        plt.tight_layout()
        plt.savefig('plot.pdf')

    # outcomes = pm.find_MAP(model=model)
    # for k, v in outcomes.items():
    #     print(f'{k}: {v}')
