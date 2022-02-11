'''A Bayesian network for a TrueSkill-like team-based skill ranking.'''
from typing import Dict
from typing import List
from typing import NewType
from dataclasses import dataclass
from statistics import mean

import numpy as np
import pymc3 as pm


@dataclass
class SkillDistribution:
    '''A class representing the data of a skill distribution.

    Used to represent both the prior and posterior distribution of individual
    performance, in both cases a Normal distribution.
    '''
    mean: float = 1500
    stddev: float = 50


@dataclass
class Team:
    members: List[int]


PlayerRatings = NewType("PlayerRatings", Dict[int, SkillDistribution])


def update_ratings(team1: Team, team2: Team, ratings:PlayerRatings) -> PlayerRatings:
    # Starting with Elo for simplicity, to get something working
    model = pm.Model()
    with model:
        p1_skill = pm.Normal("p1_skill", mu=p1_prior.mean, sigma=p1_prior.stddev)
        p2_skill = pm.Normal("p2_skill", mu=p2_prior.mean, sigma=p2_prior.stddev)

        p1_perf = pm.Normal("p1_perf", mu=p1_skill, sigma=20)
        p2_perf = pm.Normal("p2_perf", mu=p2_skill, sigma=20)

        perf_diff = p1_perf - p2_perf
        pm.Bernoulli(
            "p1_win", logit_p=from_elo_scale(perf_diff), observed=np.array(outcome)
        )

    return model


if __name__ == "__main__":
    p1_prior = SkillDistribution(mean=1564, stddev=50)
    p2_prior = SkillDistribution(mean=1442, stddev=50)
    matches = [0]
    print(f"P1: {p1_prior.mean}, P2: {p2_prior.mean}")

    for i, match in enumerate(matches):
        print(f"-------------\nRound {i}\n-------------")
        model = define_model(p1_prior, p2_prior, 10)
        trace = pm.sample(model=model)
        # outcomes = pm.find_MAP(model=model)
        # posterior mean is new prior
        p1_prior = SkillDistribution(mean=mean(trace['p1_skill']), stddev=50)
        p2_prior = SkillDistribution(mean=mean(trace['p2_skill']), stddev=50)
        print(f"P1: {p1_prior.mean}, P2: {p2_prior.mean}")
