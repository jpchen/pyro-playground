import argparse
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Marginal, Search, SVI
from pdb import set_trace as bb

"""
Bayesian Avalon Resistance
"""
rounds = {"0": 3, "1" : 4, "2": 4, "3": 5, "4": 5}


def init(num_players):
    # (role, prior_0...prior_n)
    policies = torch.randn(8, 9)
    policies[:, 0] = 1
    for i in range(3):
        # first three are bad
        for j in range(9):
            # bad knows the roles of the other players
            if j < 4:
                policies[i, j] = 0
            else:
                policies[i, j] = 1
    for i in range(3, policies.shape[0]):
        # init priors
        policies[i, 1:] = 0.5
    return policies


def move(p, i):
    return pyro.sample(i, dist.Bernoulli(p))


def model(policies):
    policies = pyro.param("policies", policies)
    for i in range(5):
        n = rounds[str(i)]
        for j in pyro.irange('data', 8, subsample_size=n):
            action = move(policies[i][j], i)
            pyro.sample('obs_'.format(i), dist.Bernoulli(policies[i, j]), obs=action)


def main(args):
    policies = init(args.num_players)
    model(policies)
    bb()
    lr = 1e-2
    inference = SVI(model, guide, Adam({"lr": lr}), 'ELBO')
    for i in range(args.num_samples):
        loss = inference.step()
        print("loss = {}".format(loss))
    print(pyro.get_param_store()._params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-samples', default=50, type=int)
    parser.add_argument('-p', '--num-players', default=8, type=int)
    args = parser.parse_args()
    main(args)

