'''A Bayesian network for a TrueSkill-like team-based skill ranking.'''

from dataclasses import dataclass
from typing import List

import pymc3 as pm


@dataclass
class SkillDistribution:
    '''A class representing the data of a skill distribution.

    Used to represent both the prior and posterior distribution of individual
    performance, in both cases a Normal distribution.
    '''
    mean: float = 1500
    stddev: float = 10


def define_model(p1_prior, p2_prior, perf_stddev):
    # Starting with Elo for simplicity, to get something working
    model = pm.Model()
    with model:
        p1_skill = pm.Normal("P1 skill", mu=p1_prior.mean, sigma=p1_prior.stddev)
        p2_skill = pm.Normal("P2 skill", mu=p2_prior.mean, sigma=p2_prior.stddev)

        p1_perf = pm.Normal("P1 performance", mu=p1_skill, sigma=perf_stddev)
        p2_perf = pm.Normal("P2 performance", mu=p2_skill, sigma=perf_stddev)

        pm.Deterministic("P1 wins", p1_perf > p2_perf)

    return model


def update_skills(game_outcome):
    pass


if __name__ == "__main__":
    p1_prior = SkillDistribution()
    p2_prior = SkillDistribution()
    model = define_model(p1_prior, p2_prior, 10)

    # requires `pip install graphviz`
    graph = pm.model_graph.model_to_graphviz(model=model)
    graph.render('team_skill_ranking_model.gv', view=True)
