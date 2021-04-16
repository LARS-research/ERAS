import torch
from collections import defaultdict
import numpy as np


class BernCorrupter:
    def __init__(self, data, n_ent, n_rel):
        self.bern_prob = self.get_bern_prob(data, n_ent, n_rel)
        self.n_ent = n_ent

    def corrupt(self, head, tail, rela, n_sample, uni=False):
        n = len(head)
        if uni:
            prob = 0.5*torch.ones((n,))
        else:
            prob = self.bern_prob[rela]
        selection = torch.bernoulli(prob).numpy().astype('bool')
        head_out = np.tile(head.cpu().numpy(), (n_sample, 1)).transpose()
        tail_out = np.tile(tail.cpu().numpy(), (n_sample, 1)).transpose()
        rela_out = rela.unsqueeze(1).expand(n, n_sample)

        ent_random = np.random.choice(self.n_ent, (n, n_sample))
        head_out[selection, :] = ent_random[selection]
        tail_out[~selection, :] = ent_random[~selection]
        return torch.from_numpy(head_out).contiguous().view(-1).cuda(), torch.from_numpy(tail_out).contiguous().view(-1).cuda(), rela_out.contiguous().view(-1)

    def get_bern_prob(self, data, n_ent, n_rel):
        head, tail, rela = data
        edges = defaultdict(lambda: defaultdict(lambda: set()))
        rev_edges = defaultdict(lambda: defaultdict(lambda: set()))
        for h, t, r in zip(head, tail, rela):
            edges[r][h].add(t)
            rev_edges[r][t].add(h)
        bern_prob = torch.zeros(n_rel)
        for k in edges.keys():
            right = sum(len(tails) for tails in edges[k].values()) / len(edges[k])
            left = sum(len(heads) for heads in rev_edges[k].values()) / len(rev_edges[k])
            bern_prob[k] = right / (right + left)
        return bern_prob


