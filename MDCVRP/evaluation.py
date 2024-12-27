import os
import time

# from TSPlIbInstance import create_data,reward
from collections import OrderedDict

import torch
from create_vrp import create_data, reward1
from VRP_Actor import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rollout(model, dataset, T=1, Sample=False):
    # Put in greedy evaluation mode!
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(bat, n_nodes * 2, Sample, T)

            cost1 = reward1(bat.x, cost.detach(), n_nodes)
        return cost1.cpu()

    totall_cost = torch.cat([eval_model_bat(bat.to(device)) for bat in dataset], 0)
    return totall_cost


def evaliuate(valid_loder, T=1, Sample=False):
    agent = Model(3, 128, 1, 16, conv_layers=4).to(device)
    agent.to(device)

    folder = "Vrp-22-GAT"
    filename = "rollout"
    filepath = os.path.join(folder, filename)
    path = os.path.join(filepath, "%s" % 22)
    if os.path.exists(path):
        path1 = os.path.join(path, "actor.pt")
        state_dict = torch.load(path1, map_location="cuda:0")
        new_state_dict = OrderedDict()
        print(1)
        for k, v in state_dict.items():
            name = k[:]  # remove `module.`
            new_state_dict[name] = v

        agent.load_state_dict(new_state_dict)

    start = time.time()
    cost = rollout(agent, valid_loder, T, Sample)
    end = time.time() - start

    cost1 = cost.mean()
    cost2 = cost.min()
    if Sample:
        print(cost1, "greedy----Gap", 100 * (cost1 - 6.1) / 6.1)
    else:
        print(cost2, "Sampleing----Gap", "---------", T, 100 * (cost2 - 6.1) / 6.1)

    return cost1, cost2


greedy = True
n_nodes = 22
batch_size = 256
n = []
tours = []
valuate_size = 10
valid_loder = create_data(n_nodes, valuate_size, batch_size=batch_size)
_, sample_cost = evaliuate(valid_loder, 1, greedy)
