'''An implementation of a media mix model.'''

import pymc3 as pm

# TODO: add adstock function


def tanh_saturation(x, revenue_at_saturation, initial_roas):
    '''A saturation function based on the hyperbolic tangent.

    The saturation function represents the diminishing returns of an
    advertising channel.

    This is PyMC Labs's reparameterization of the original HelloFresh model, cf.
    https://web.archive.org/web/20211224060713/https://www.pymc-labs.io/blog-posts/reducing-customer-acquisition-costs-how-we-helped-optimizing-hellofreshs-marketing-budget/
    '''
    return revenue_at_saturation * pm.math.tanh(initial_roas * x / revenue_at_saturation)


# TODO: Turn args into dataclasses & document their semantics
def make_model(channel_data, sales):
    with pm.Model() as model:
        channel_models = []

        for (channel, weekly_spending) in channel_data.items():
            # The coefficient determining how much revenue can be attributed to
            # this channel after the effects of saturation have been accounted for.
            return_post_reach = pm.HalfNormal(f'return_post_reach_{channel}', sigma=5)

            # The maximum amount of revenue (before scaling by return_post_reach)
            # this channel can generate in a given time period, i.e., at full saturation.
            revenue_at_saturation = pm.HalfNormal(f'revenue_at_saturation_{channel}', sigma=50)

            # The initial return on advertising spend (before scaling by
            # return_post_reach) at zero dollars spent. I.e., the slope of the
            # most linear part of the saturation curve.
            initial_roas = pm.HalfNormal(f'initial_roas_{channel}', sigma=2)
            channel_models.append(
                return_post_reach * tanh_saturation(
                    weekly_spending, revenue_at_saturation, initial_roas
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
            'likelihood', mu=new_sales, sigma=output_noise, observed=sales
        )

    return model


def load_data(filepath: str):
    import csv

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

    channel_data, sales = load_data('data/media_mix_data.csv')
    data_prefix = 20
    channel_data = {k: v[:data_prefix] for (k, v) in channel_data.items()}
    sales = sales[:data_prefix]
    print(channel_data, sales)

    model = make_model(channel_data, sales)
    print(model.check_test_point())
    with model:
        trace = pm.sample(draws=1000, chains=4, tune=1000, return_inferencedata=False)
        summary = az.summary(trace, round_to=2)
        print(summary)

        divergent = trace["diverging"]
        print("Number of Divergent %d" % divergent.nonzero()[0].size)
        divperc = divergent.nonzero()[0].size / len(trace) * 100
        print("Percentage of Divergent %.1f" % divperc)

        az.plot_trace(trace)
        plt.tight_layout()
        plt.savefig('media_mix_plot.pdf')

    # add posterior predictive

    outcomes = pm.find_MAP(model=model)
    for k, v in outcomes.items():
        print(f'{k}: {v}')
