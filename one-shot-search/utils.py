import logging
import os
import datetime
import random

from collections import defaultdict
import numpy as np
import torch
from torch.autograd import Variable



class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out


def logger_init(args):
    logging.basicConfig(level=logging.DEBUG, format='%(module)15s %(asctime)s %(message)s', datefmt='%H:%M:%S')
    if args.log_to_file:
        log_filename = os.path.join(args.log_dir, args.log_prefix+datetime.datetime.now().strftime("%m%d%H%M%S"))
        logging.getLogget().addHandler(logging.FileHandler(log_filename))


def plot_config(args):
    out_str = "\nn_batch:{}, n_oas_epoch:{}, n_stand_epoch:{}, optim:{}, lr:{} lamb:{}, decay_rate:{}, n_dim:{}, controller_optim:{}, n_controller_epoch:{}, n_derive_sample:{}\n".format(
            args.n_batch, args.n_oas_epoch, args.n_stand_epoch, args.optim, args.lr, args.lamb, args.decay_rate, args.n_dim, args.controller_optim, args.n_controller_epoch, args.n_derive_sample)
    print(out_str)
    with open(args.perf_file, 'a') as f:
        f.write(out_str)
        

def inplace_shuffle(*lists):
    idx = []
    for i in range(len(lists[0])):
        idx.append(random.randint(0, i))
    for ls in lists:
        j = idx[i]
        ls[i], ls[j] = ls[j], ls[i]

def batch_by_num(n_batch, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    for i in range(n_batch):
        start = int(n_sample * i / n_batch)
        end = int(n_sample * (i+1) / n_batch)
        ret = [ls[start:end] for ls in lists]
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    start = 0
    while(start < n_sample):
        end = min(n_sample, start + batch_size)
        ret = [ls[start:end] for ls in lists]
        start += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]
        
def gen_struct(num):
    struct = []
    for i in range(num):
        if i < 4:
            struct.append(random.randint(0,3))      #t
        else:
            struct.append(random.randint(0,3))      #h
            struct.append(random.randint(0,3))      #t
            struct.append(random.randint(-1,1))  #1
    return struct


def record(filePath, K, rewards, structs, relas, extractType="top"):
    
    rewards, structs, relas = np.asarray(rewards), np.asarray(structs), np.asarray(relas)
    
    if extractType == "top":
        indices = rewards.argsort()[-K:][::-1]
        
    #print(indices)
        
    with open(filePath, "a+") as f:
        for i in range(K):
            outstr_reward = "MRR:" + str(rewards[indices][i]) + "\n"
            outstr_struct = "struct:" + str(list(structs[indices][i].tolist())) + "\n"
            outstr_rela = "Rela:" + str(list(relas[indices][i].tolist())) + "\n"
                    
            f.write(outstr_reward)
            f.write(outstr_rela)
            f.write(outstr_struct)
            f.write("\n")
        
    return indices


    """
    elif args.dataset ==  'LDC' or '12' or '14' or '15' or 'nations' or 'umls' or 'kinship':
        args.lr = 0.4712
        args.lamb = 0.000220480328005851
        args.n_batch = 128
        args.decay_rate = 0.9903840888956048
        args.n_dim = 512
        args.epoch_per_test = 2
    """

def default_search_hyper(args):

    # hypers on learning rate, lambda, batch size, decay rate, n_dim, epoch per test
    if args.dataset == 'WN18RR':
        args.lr = 0.47102439590590006
        #args.lamb = 5.919173541218532e-05      # searched
        #args.lamb = 1.8402163609403787e-05      # SimplE
        args.lamb = 0.0002204803280058515       # ComplEx
        args.n_batch = 512
        args.decay_rate = 0.9903840888956048
        #args.n_dim = 512 #AutoSF searched
        args.n_stand_epoch = 300 #AutoSF searched
        args.n_dim = 512
        args.epoch_per_test = 20 #20
        args.n = 3
            
    elif args.dataset == 'FB15K237':
        #args.lr = 0.0885862663108572
        #args.lamb = 0.0016177695659237597
        #args.decay_rate = 0.9931763998742731
        #args.n_batch = 256
        args.lr = 0.1783468990895745
        args.lamb = 0.0025173667237246883
        args.decay_rate = 0.9915158217372417
        args.n_batch = 512
        #args.n_dim = 2048 #AutoSF searched
        args.n_stand_epoch = 500 #AutoSF searched
        args.n_dim = 512 #2048
        args.epoch_per_test = 15
        
    elif args.dataset == 'WN18':
        args.lr = 0.10926076305780041
        args.lamb = 0.0003244851835920663
        args.decay_rate = 0.9908870395744
        args.n_batch = 512 #256
        #args.n_dim = 1024 #AutoSF searched
        args.n_stand_epoch = 400 #AutoSF searched
        args.n_dim = 512
        args.epoch_per_test = 15
        args.n = 3
        
    elif args.dataset == 'FB15K':
        args.lr = 0.7040329784234945
        args.lamb = 3.49037818818688153e-5
        args.decay_rate = 0.9909065915902778
        args.n_batch = 512
        args.n_stand_epoch = 700 #AutoSF searched
        #args.n_dim = 2048 #AutoSF searched
        args.n_dim = 512
        args.epoch_per_test = 15
        
    elif args.dataset == 'YAGO':
        args.lr = 0.9513908770180219
        args.lamb = 0.00021779088577909324
        args.decay_rate = 0.9914972709145934
        args.n_batch = 512 # 2048
        #args.n_dim = 1024
        args.n_stand_epoch = 400
        args.n_dim = 512
        args.epoch_per_test = 20
    
    else:
        args.lr = 0.47102439590590006
        args.lamb = 0.0002204803280058515
        args.n_batch = 512
        args.decay_rate = 0.9903840888956048
        args.n_stand_epoch = 300 
        args.n_dim = 512
        args.epoch_per_test = 20 
    
    if args.clu == "pde":
        rela_cluster_list = {"WN18RR":[1, 1, 0, 1, 1, 0, 2, 0, 0, 1, 0],
                             "WN18":[1, 0, 2, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2],
                             "FB15K237":[3, 1, 3, 2, 3, 3, 1, 3, 2, 2, 3, 3, 1, 3, 3, 0, 0, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 2, 2, 0, 3, 2, 2, 2, 3, 3, 3, 0, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 3, 2, 0, 3, 3, 2, 3, 0, 1, 3, 3, 0, 2, 3, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 1, 3, 3, 1, 3, 3, 0, 1, 3, 2, 3, 2, 2, 3, 1, 3, 3, 2, 3, 2, 3, 2, 3, 3, 3, 0, 0, 0, 3, 2, 2, 0, 3, 0, 1, 2, 2, 3, 3, 2, 3, 0, 0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 3, 1, 3, 3, 1, 3, 3, 3, 1, 3, 2, 2, 0, 3, 3, 3, 2, 3, 3, 1, 3, 2, 2, 3, 2, 3, 2, 2, 2, 1, 1, 3, 1, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 3, 3, 3, 1, 3, 3, 3, 0, 3, 2, 0, 3, 3, 3, 3, 0, 3, 2, 2, 3, 2, 3, 3, 3, 0, 1, 3, 0, 3, 1, 2, 2, 2, 2, 3, 3, 3, 2, 3],
                             "FB15K":[0, 2, 0, 3, 0, 0, 1, 2, 0, 2, 0, 0, 0, 3, 1, 3, 1, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 1, 0, 3, 0, 3, 1, 0, 0, 0, 3, 1, 1, 3, 0, 1, 0, 3, 0, 0, 0, 0, 1, 1, 0, 1, 0, 3, 3, 3, 1, 1, 0, 2, 1, 1, 3, 1, 0, 3, 3, 0, 0, 0, 1, 1, 1, 3, 0, 0, 3, 1, 3, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 2, 0, 1, 1, 0, 0, 2, 1, 3, 1, 0, 3, 1, 0, 3, 0, 0, 1, 0, 0, 2, 3, 1, 0, 1, 1, 1, 2, 0, 2, 1, 1, 0, 1, 0, 3, 0, 1, 1, 2, 0, 0, 0, 0, 2, 1, 1, 1, 1, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 3, 1, 1, 0, 0, 1, 1, 2, 0, 1, 1, 2, 0, 0, 0, 0, 2, 1, 0, 1, 3, 3, 0, 1, 1, 2, 1, 3, 0, 2, 3, 2, 0, 0, 0, 1, 3, 0, 1, 3, 0, 3, 0, 1, 0, 3, 3, 1, 1, 1, 0, 0, 3, 3, 1, 1, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 3, 1, 1, 2, 0, 0, 3, 0, 0, 1, 2, 2, 1, 1, 2, 0, 3, 3, 3, 0, 1, 0, 0, 1, 1, 3, 0, 3, 0, 0, 3, 0, 1, 2, 1, 0, 0, 1, 1, 1, 3, 0, 1, 0, 1, 0, 1, 1, 0, 1, 3, 1, 0, 0, 1, 0, 1, 0, 3, 2, 1, 0, 0, 2, 2, 1, 1, 0, 1, 1, 0, 3, 0, 1, 1, 1, 3, 0, 3, 1, 0, 1, 1, 1, 0, 0, 1, 3, 1, 3, 2, 1, 1, 3, 3, 3, 1, 0, 1, 1, 3, 3, 1, 1, 1, 3, 1, 3, 2, 3, 1, 1, 1, 0, 1, 0, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 3, 3, 1, 0, 3, 1, 0, 3, 1, 1, 1, 0, 3, 0, 3, 0, 3, 1, 3, 3, 1, 1, 2, 0, 2, 3, 0, 3, 3, 1, 3, 2, 2, 1, 1, 3, 1, 0, 3, 1, 0, 0, 0, 3, 3, 1, 0, 1, 3, 0, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 0, 1, 1, 0, 1, 1, 1, 3, 1, 1, 1, 0, 2, 1, 1, 1, 1, 3, 0, 0, 0, 1, 0, 3, 0, 0, 1, 1, 1, 0, 1, 3, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 0, 3, 1, 3, 2, 0, 2, 3, 1, 3, 0, 3, 1, 0, 1, 3, 1, 3, 1, 1, 1, 0, 1, 3, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 3, 0, 1, 1, 1, 3, 3, 1, 1, 1, 3, 1, 1, 1, 3, 2, 3, 0, 0, 2, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 3, 1, 1, 3, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 0, 0, 1, 3, 1, 1, 3, 3, 3, 0, 1, 3, 0, 3, 3, 3, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 3, 3, 1, 1, 3, 1, 1, 3, 3, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 3, 1, 1, 0, 1, 1, 3, 1, 1, 0, 3, 1, 3, 3, 1, 3, 0, 3, 0, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 0, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 3, 1, 1, 1, 0, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 3, 1, 3, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 3, 3, 1, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 1, 3, 0, 1, 0, 3, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 0, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 3, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 3, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 3, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1],
                             "YAGO":[0 for i in range(37)]
                             }
        args.n = max(rela_cluster_list[args.dataset]) + 1 # the number of clusters
    
    elif args.clu == "scu":
        n_rels = {"WN18RR":11, "WN18":18, "FB15K237":237, "FB15K":1345, "YAGO":37, "LDC":41, "12":6, "14":6, "15":6, "nations": 55, "umls":46, "kinship":25}
        rela_cluster_list = {}
        rela_cluster_list[args.dataset] = np.random.randint(0, args.n, n_rels[args.dataset])

    return args, rela_cluster_list[args.dataset]





