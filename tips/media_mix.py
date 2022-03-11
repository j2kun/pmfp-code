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


def load_data(filepath: str):
    data = []
    with open(filepath, 'r') as infile:
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
    return channel_data, sales


if __name__ == "__main__":
    import arviz as az
    import matplotlib.pyplot as plt
    import csv

    # channel_data, sales = load_data('data/media_mix_data.csv')
    channel_data = {
        'tv': [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8],
        'radio': [37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 2.1, 2.6],
        'newspaper': [69.2, 45.1, 69.3, 58.5, 58.4, 75.0, 23.5, 11.6, 1.0, 21.2],
    }
    sales = [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6]
    print(channel_data, sales)

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
