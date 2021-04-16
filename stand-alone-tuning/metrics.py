import torch as t
import numpy as np

def mrr_mr_hitk(scores, target, k=[1,3,10]):
    _, sorted_idx = t.sort(scores, descending=True)
    find_target = sorted_idx == target
    #target_rank = t.nonzero(find_target, as_tuple=False)[0, 0] + 1
    target_rank = t.nonzero(find_target)[0, 0] + 1
    #print("target: {}      target_rank: {}      rev: {}".format(target, target_rank, 1./float(target_rank)))
    hits = np.zeros((len(k),))
    for i, r in enumerate(k):
        hits[i] = int(target_rank <= r)
    return 1. / float(target_rank), target_rank, hits  #int(target_rank <= k)
