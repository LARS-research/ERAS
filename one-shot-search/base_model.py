import torch
import torch.nn as nn
import numpy as np
from metrics import mrr_mr_hitk
from utils import batch_by_size, plot_config
import os
import time
from torch.optim import Adam, SGD, Adagrad
from torch.optim.lr_scheduler import ExponentialLR
from models import KGEModule
import scipy
from sklearn.cluster import KMeans

import nas.utils.tensor_utils as utils

def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]

history = []


def scale(value, last_k=10, scale_value=1):
    '''
    scale value into [-scale_value, scale_value], according last_k history
    '''
    max_reward = np.max(history[-last_k:])
    if max_reward == 0:
        return value
    return scale_value / max_reward * value

def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim

def _get_space_op(m,n):
    num_space, num_op = m*m*n, 2*m+1
    space_list = [str(i) for i in range(num_space)]
    op_list = [str(i) for i in range(num_op)]
    
    search_space = {}
    for action in space_list:
        search_space[action] = op_list
    
    return search_space, space_list

def _init_struct(sf, m=4, n=3):
    struct = []
    if sf == "DistMult":
        for i in range(n):
            sf_struct = [0 for j in range(m*m)]
            sf_struct[0], sf_struct[5], sf_struct[10], sf_struct[15] = 1, 2, 3, 4
            struct.append(sf_struct)
    return struct

class BaseModel(object):
    def __init__(self, n_ent, n_rel, args, rela_cluster, model_type):
        
        cluster_way = args.clu
        

        self.model = KGEModule(n_ent, n_rel, args, rela_cluster, model_type)
        
        if args.GPU:
            self.model.cuda()
            
        self.n_ent = n_ent
        self.n_rel = n_rel
        
        self.rela_cluster = rela_cluster
        
        self.time_tot = 0
        self.args = args
                
        self.n_dim = args.n_search_dim if model_type == "search" else args.n_stand_dim
        
        self.K = args.m
        self.n = args.n
        
        self.GPU = args.GPU
        
        self.cluster_way = cluster_way
        
        self.rela_to_dict(rela_cluster)
        

        """build controller and sub-model"""
        if model_type == "search":
            self.controller = None
            self.build_controller()
            
            #print(self.args.controller_optim)
            
            controller_lr = 3.5e-4
            controller_optimizer = _get_optimizer(self.args.controller_optim)
            self.controller_optim = controller_optimizer(self.controller.parameters(), lr=controller_lr)
            
            self.derived_raward_history = []
            self.derived_struct_history = []
            if self.cluster_way == "scu":
                self.rela_cluster_history = []
        
        
    def build_controller(self):
        
        self.search_space, self.action_list = _get_space_op(self.K, self.n)
        
        # build RNN controller
        from nas.controller import SimpleNASController
        self.controller = SimpleNASController(self.args, action_list=self.action_list,
                                              search_space=self.search_space,
                                              cuda=self.GPU)
        
        if self.GPU:
            self.controller.cuda()


    def mm_train(self, train_data, valid_data, tester_val, tester_tst, tester_trip):
        """
        Each epoch consists of two phase:
        - In the first phase, shared parameters are trained to exploration.
        - In the second phase, the controller's parameters are trained.
        """
        self.tester_val = tester_val
        self.tester_tst = tester_tst
        
        self.tester_trip = tester_trip
        
        #num_epoch = self.args.n_epoch
        derived_struct = _init_struct("DistMult", self.K, self.n)
                
            
        self.train_oas(train_data, valid_data, derived_struct)
            
        derived_struct, derived_mrr = self.derive_last()
        
        if self.cluster_way == "scu":
            return (self.derived_raward_history, self.derived_struct_history, self.rela_cluster_history), (derived_mrr, derived_struct, self.rela_cluster)
        
        elif self.cluster_way == "pde":
            return (self.derived_raward_history, self.derived_struct_history), (derived_mrr, derived_struct)
        
    
    def get_reward(self, struct_list, test=False, random=True):
        """
        Computes the reward of a single sampled model on validation data.
        """
        reward_list = []
        if random:
            randint = None
        else:
            randint = torch.randint(10000, (1,))
            
        for struct in struct_list:
            #print("x")
            
            struct = torch.LongTensor([int(item) for item in struct])
            struct = struct.view(-1, self.K*self.K)
            
            if not test:
                
                valid_mrr, valid_mr, valid_1, valid_3, valid_10 = self.tester_val(struct, test, randint)
                reward_list.append(valid_mrr)
                                
            else:
                valid_mrr, valid_mr, valid_1, valid_3, valid_10 = self.tester_val(struct, test, randint)
                test_mrr, test_mr, test_1, test_3, test_10 = self.tester_tst(struct, test, randint)
                reward_list.append(valid_mrr) # record val mrr in the file, test mrr is just for visualization
                
                
        if not test:
            return reward_list
        else:
            return reward_list, test_mrr 
    

    def train_controller(self):
        
        """
            Train controller to find better structure.
        """
        
        model = self.controller
        model.train()

        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []
        
        total_loss = 0
    
        for step in range(self.args.n_controller_epoch):
            
            #print(step)
            
            # sample struct
            structure_list, log_probs, entropies = self.controller.sample(with_details=True)
            
            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            rewards = self.get_reward(structure_list, random=False)
            rewards = np.array(rewards)
            torch.cuda.empty_cache()
            
            # discount
            discount = 1.0
            if 1 > discount > 0:
                rewards = discount(rewards, discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)
            
            
            ema_baseline_decay = 0.5
            
            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = ema_baseline_decay
                #print (decay, baseline, rewards)
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            history.append(adv)
            adv = scale(adv, scale_value=0.5)
            adv_history.extend(adv)
            
            adv = utils.get_variable(adv, self.GPU, requires_grad=False)

            # policy loss
            loss = -log_probs * adv
            
            entropy_mode = "reward"
            if entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()
            
            # update
            self.controller_optim.zero_grad()
            loss.backward()

            controller_grad_clip = 0
            if controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              controller_grad_clip)
            self.controller_optim.step()

            total_loss += utils.to_item(loss.data)

            torch.cuda.empty_cache()
            

    def derive(self, sample_num=None):
        
        derive_from_history = False
                
        if sample_num is None and derive_from_history:
            return self.derive_from_history()
        else:
            structure_list, _, entropies = self.controller.sample(sample_num, with_details=True)
                        
            rewards, test_mrr = self.get_reward(structure_list, test=True, random=True)
            rewards = torch.Tensor(rewards)
            
            best_struct = structure_list[0]                      
            best_struct = torch.LongTensor([int(item) for item in best_struct])
            best_struct = best_struct.view(-1, self.K*self.K)
            
            self.derived_raward_history.append(max(rewards))
            self.derived_struct_history.append(best_struct)
                
            return best_struct, test_mrr
    
    
    def derive_last(self):
    
        rewards_list, structs_list = [], []
        for itr in range(self.args.n_derive_sample):
            structure_list, _, entropies = self.controller.sample(1, with_details=True)
            rewards, test_mrr = self.get_reward(structure_list, test=True, random=True)
            
            struct = structure_list[0]                            
            struct = torch.LongTensor([int(item) for item in struct])
            struct = struct.view(-1, self.K*self.K)
            
            rewards_list.append(max(rewards))
            structs_list.append(struct)
        
        max_idx = rewards_list.index(max(rewards_list)) # select 
        
        return structs_list[max_idx], rewards_list[max_idx]
            
    
    def rela_to_dict(self, rela_cluster):
        self.cluster_rela_dict = dict()
        n = max(rela_cluster) + 1
        for i in range(n):
            self.cluster_rela_dict[i] = []
            
        for idx, item in enumerate(rela_cluster):
            self.cluster_rela_dict[item].append(idx)
           
        for i in range(n):
            self.cluster_rela_dict[i] = torch.LongTensor(self.cluster_rela_dict[i])
            
    
    def cluster(self):
        
        X = self.model.rel_embed.weight.data.cpu().numpy()
        kmeans = KMeans(n_clusters=self.n, random_state=0).fit(X)
        #self.rela_cluster = kmeans.labels_.tolist()
        
        return kmeans.labels_.tolist()


    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=lambda storage, location: storage.cuda()))


    

    def train_oas(self, train_data, valid_data, derived_struct):
        
        head, tail, rela = train_data
        n_train = len(head)
        
        if self.args.optim=='adam' or self.args.optim=='Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optim=='adagrad' or self.args.optim=='Adagrad':
            self.optimizer = Adagrad(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = SGD(self.model.parameters(), lr=self.args.lr)

        scheduler = ExponentialLR(self.optimizer, self.args.decay_rate)

        n_batch = self.args.n_batch
        
        for epoch in range(self.args.n_oas_epoch):
            start = time.time()
            rand_idx = torch.randperm(n_train)
            
            if self.GPU:
                head = head[rand_idx].cuda()
                tail = tail[rand_idx].cuda()
                rela = rela[rand_idx].cuda()
            else:
                head = head[rand_idx]
                tail = tail[rand_idx]
                rela = rela[rand_idx]
            
            epoch_loss = 0
            n_iters = 0

            # train model weights
            for h, t, r in batch_by_size(n_batch, head, tail, rela, n_sample=n_train):
    
                self.model.zero_grad()
    
                loss = self.model.forward(derived_struct, h, t, r, self.cluster_rela_dict)
                loss += self.args.lamb * self.model.regul
                loss.backward()
                
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
                self.prox_operator()
                
                epoch_loss += loss.data.cpu().numpy()
                n_iters += 1
                
            
            scheduler.step()
            
            if self.cluster_way == "scu":
                self.rela_cluster = self.cluster()
                self.rela_cluster_history.append(self.rela_cluster)
                self.rela_to_dict(self.rela_cluster)
    
            
            # train controller
            self.train_controller()
            
            # derive structs
            self.time_tot += time.time() - start # evaluation for the derived architecture is unnessary in searching procedure
            derived_struct, test_mrr = self.derive(sample_num = 1) 
            
            print("Epoch: %d/%d, Search Time=%.2f, Loss=%.2f, Sampled Val MRR=%.8f, Tst MRR=%.8f"%(epoch+1, self.args.n_oas_epoch, self.time_tot, epoch_loss/n_train, 
                                                                          self.derived_raward_history[-1], test_mrr))
        
    def train_stand(self, train_data, valid_data, derived_struct, rela_cluster, mrr, tester_val, tester_tst):
                
        self.tester_val, self.tester_tst = tester_val, tester_tst
        
        self.rela_to_dict(rela_cluster)
        
        self.args.perf_file = os.path.join(self.args.out_dir, self.args.dataset + '_std_' + str(self.args.m) + "_" + str(self.args.n)  + "_" + str(mrr) + '.txt')
        plot_config(self.args)
        
        head, tail, rela = train_data
        n_train = len(head)
        
        if self.args.optim=='adam' or self.args.optim=='Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optim=='adagrad' or self.args.optim=='Adagrad':
            self.optimizer = Adagrad(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = SGD(self.model.parameters(), lr=self.args.lr)
        scheduler = ExponentialLR(self.optimizer, self.args.decay_rate)

        n_batch = self.args.n_batch
                
        start = time.time()
        best_mrr = 0


        for epoch in range(self.args.n_stand_epoch):
                
                        
            #self.epoch = epoch
            rand_idx = torch.randperm(n_train)
            
            if self.GPU:
                head = head[rand_idx].cuda()
                tail = tail[rand_idx].cuda()
                rela = rela[rand_idx].cuda()
            else:
                head = head[rand_idx]
                tail = tail[rand_idx]
                rela = rela[rand_idx]
            
            epoch_loss = 0
            n_iters = 0
            #lr = scheduler.get_lr()[0]

            
            # train model weights
            for h, t, r in batch_by_size(n_batch, head, tail, rela, n_sample=n_train):
    
                self.model.zero_grad()
                
                loss = self.model.forward(derived_struct, h, t, r, self.cluster_rela_dict)
                loss += self.args.lamb * self.model.regul
                loss.backward()
                
                self.optimizer.step()
                self.prox_operator()
                
                epoch_loss += loss.data.cpu().numpy()
                n_iters += 1
                            
            scheduler.step()
                        
            print("Epoch: %d/%d, Stand Time=%.2f, Loss=%.2f"%(epoch+1, self.args.n_stand_epoch, time.time()-start, epoch_loss/n_train))
            
            if (epoch + 1) % 5 == 0:
                test, randint = True, None
                
                valid_mrr, valid_mr, valid_1, valid_3, valid_10 = self.tester_val(derived_struct, test, randint)
                test_mrr, test_mr, test_1, test_3, test_10 = self.tester_tst(derived_struct, test, randint)
                
                out_str = '%d \t %.2f \t %.2f \t %.4f  %.1f %.4f %.4f %.4f\t%.4f %.1f %.4f %.4f %.4f\n' % (epoch, time.time()-start, epoch_loss/n_train,\
                            valid_mrr, valid_mr, valid_1, valid_3, valid_10, \
                            test_mrr, test_mr, test_1, test_3, test_10)
                
                # output the best performance info
                if test_mrr > best_mrr:
                    best_mrr = test_mrr
                    best_str = out_str
                
                with open(self.args.perf_file, 'a+') as f:
                    f.write(out_str)
                    
            
        with open(self.args.perf_file, 'a+') as f:
            f.write("best performance:" + best_str + "\n")
            f.write("struct:" + str(derived_struct) + "\n")
            f.write("rela:" + str(rela_cluster) + "\n")
        
        return best_mrr
                        
    
    def prox_operator(self,):
        for n, p in self.model.named_parameters():
            if 'ent' in n:
                X = p.data.clone()
                Z = torch.norm(X, p=2, dim=1, keepdim=True)
                Z[Z<1] = 1
                X = X/Z
                p.data.copy_(X.view(self.n_ent, -1))
        return X
    
    
    def name(self, idx):
        i = idx[0]
        i_rc =  self.rela_cluster[i]
        self.r_embed[i,:,:] = self.rel_embed_2K_1[i,self.idx_list[i_rc],:] * self.model._arch_parameters[i_rc][[j for j in range(self.K*self.K)], self.idx_list[i_rc]].view(-1,1)


    def test_link(self, struct, test, randint, test_data, n_ent, heads, tails, filt=True):
        
        mrr_tot = 0.
        mr_tot = 0.
        hit_tot = np.zeros((3,))
        count = 0
        
        self.n_cluster = len(struct)
        test_batch_size = self.args.n_batch
        
        head, tail, rela = test_data
        
        if randint is None:
            rand_idx = torch.randperm(len(head))
        else:
            np.random.seed(randint)
            rand_idx = torch.LongTensor(np.random.permutation(len(head)))
        
        if self.GPU:
            head = head[rand_idx].cuda()
            tail = tail[rand_idx].cuda()
            rela = rela[rand_idx].cuda()
        else:
            head = head[rand_idx]
            tail = tail[rand_idx]
            rela = rela[rand_idx]
        
        
        for batch_h, batch_t, batch_r in batch_by_size(test_batch_size, head, tail, rela):
            
            
            if self.GPU:
                batch_h = batch_h.cuda()
                batch_t = batch_t.cuda()
                batch_r = batch_r.cuda()
            else:
                batch_h = batch_h
                batch_t = batch_t
                batch_r = batch_r
                
            h_embed = self.model.ent_embed(batch_h).view(-1, self.K, self.n_dim//self.K)
            t_embed = self.model.ent_embed(batch_t).view(-1, self.K, self.n_dim//self.K)
            
            length = self.n_dim // self.K

            # create a rela_embed with size (n_rel, 2K+1, length)
            rel_embed_pos = self.model.rel_embed.weight.view(-1, self.K, length)
            rel_embed_neg = -rel_embed_pos
            
            if self.GPU:
                rel_embed_zeros = torch.zeros(self.n_rel, 1, length).cuda()
            else:
                rel_embed_zeros = torch.zeros(self.n_rel, 1, length)
            
            self.rel_embed_2K_1 = torch.cat((rel_embed_zeros, rel_embed_pos, rel_embed_neg),1)
            

            # combine struct
            if self.GPU:
                self.r_embed = torch.zeros(self.n_rel, self.K*self.K, length).cuda()
            else:
                self.r_embed = torch.zeros(self.n_rel, self.K*self.K, length)
            
        
            for i_rc in range(self.n_cluster):
                max_idx_list = struct[i_rc]
                self.r_embed[self.cluster_rela_dict[i_rc],:,:] = self.rel_embed_2K_1[self.cluster_rela_dict[i_rc]][:,max_idx_list,:]
            
            
            self.r_embed = self.r_embed.view(-1, self.K, self.K, length)
            self.r_embed = self.r_embed[batch_r,:,:,:]

            head_scores = torch.sigmoid(self.model.test_head(self.r_embed, t_embed)).data
            tail_scores = torch.sigmoid(self.model.test_tail(h_embed, self.r_embed)).data
            

            for h, t, r, head_score, tail_score in zip(batch_h, batch_t, batch_r, head_scores, tail_scores):
                h_idx = int(h.data.cpu().numpy())
                t_idx = int(t.data.cpu().numpy())
                r_idx = int(r.data.cpu().numpy())
                if filt:            # filter
                    if tails[(h_idx,r_idx)]._nnz() > 1:
                        tmp = tail_score[t_idx].data.cpu().numpy()
                        idx = tails[(h_idx, r_idx)]._indices()
                        tail_score[idx] = 0.0
                        
                        if self.GPU:
                            tail_score[t_idx] = torch.from_numpy(tmp).cuda()
                        else:
                            tail_score[t_idx] = torch.from_numpy(tmp)
                    if heads[(t_idx, r_idx)]._nnz() > 1:
                        tmp = head_score[h_idx].data.cpu().numpy()
                        idx = heads[(t_idx, r_idx)]._indices()
                        head_score[idx] = 0.0
                        if self.GPU:
                            head_score[h_idx] = torch.from_numpy(tmp).cuda()
                        else:
                            head_score[h_idx] = torch.from_numpy(tmp)
                mrr, mr, hit = mrr_mr_hitk(tail_score, t_idx)
                mrr_tot += mrr
                mr_tot += mr
                hit_tot += hit
                
                mrr, mr, hit = mrr_mr_hitk(head_score, h_idx)
                mrr_tot += mrr
                mr_tot += mr
                hit_tot += hit
                count += 2
            
            if not test:
                break # one mini batch

        return float(mrr_tot)/count, float(mr_tot)/count, hit_tot[0]/count, hit_tot[1]/count, hit_tot[2]/count #, total_loss/n_test


    """
    def test_trip_class(self, valid_trip_pos, valid_trip_neg, test_trip_pos, test_trip_neg):
        rela_thres = {}
        valid_pos = {}
        valid_neg = {}
        interval = 0.01
        for b_h, b_t, b_r in batch_by_size(self.args.test_batch_size, *valid_trip_pos):
            h_embed = self.model.ent_embed(b_h)
            r_embed = self.model.rel_embed(b_r)
            t_embed = self.model.ent_embed(b_t)
            scores = torch.sigmoid(self.model.test_trip(h_embed, r_embed, t_embed)).cpu().data.numpy()
            for r,s in zip(b_r, scores):
                r_idx = int(r.data.cpu().numpy())
                if r_idx in valid_pos:
                    valid_pos[r_idx].append(s)
                else:
                    valid_pos[r_idx] = [s]

        for b_h, b_t, b_r in batch_by_size(self.args.test_batch_size, *valid_trip_neg):
            h_embed = self.model.ent_embed(b_h)
            r_embed = self.model.rel_embed(b_r)
            t_embed = self.model.ent_embed(b_t)
            scores = torch.sigmoid(self.model.test_trip(h_embed, r_embed, t_embed)).cpu().data.numpy()
            for r,s in zip(b_r, scores):
                r_idx = int(r.data.cpu().numpy())
                if r_idx in valid_neg:
                    valid_neg[r_idx].append(s)
                else:
                    valid_neg[r_idx] = [s]

        for r in valid_pos:
            if not (r in valid_neg):
                continue
            min_score = min(valid_pos[r] + valid_neg[r])
            max_score = max(valid_pos[r] + valid_neg[r])
            n_interval = int((max_score - min_score) / interval)
            best_Thresh = 0
            for i in range(n_interval):
                tmpThresh = min_score + i*interval
                correct = 0
                for s in valid_pos[r]:
                    if s >= tmpThresh:
                        correct += 1
                for s in valid_neg[r]:
                    if s < tmpThresh:
                        correct += 1
                tmpAcc = 1.0 * correct / (len(valid_pos[r]) + len(valid_neg[r]))
                if i==0:
                    bestThresh = tmpThresh
                    bestAcc = tmpAcc
                elif tmpAcc > bestAcc:
                    bestAcc = tmpAcc
                    bestThresh = tmpThresh
            rela_thres[r] = bestThresh

        correct = 0
        total = 0
        for b_h, b_t, b_r in batch_by_size(self.args.test_batch_size, *test_trip_pos):
            h_embed = self.model.ent_embed(b_h)
            r_embed = self.model.rel_embed(b_r)
            t_embed = self.model.ent_embed(b_t)
            scores = torch.sigmoid(self.model.test_trip(h_embed, r_embed, t_embed)).cpu().data.numpy()
            for r,s in zip(b_r, scores):
                r_idx = int(r.data.cpu().numpy())
                if not (r_idx in rela_thres):
                    continue
                total += 1
                if s >=  rela_thres[r_idx]:
                    correct += 1

        for b_h, b_t, b_r in batch_by_size(self.args.test_batch_size, *test_trip_neg):
            h_embed = self.model.ent_embed(b_h)
            r_embed = self.model.rel_embed(b_r)
            t_embed = self.model.ent_embed(b_t)
            scores = torch.sigmoid(self.model.test_trip(h_embed, r_embed, t_embed)).cpu().data.numpy()
            for r,s in zip(b_r, scores):
                r_idx = int(r.data.cpu().numpy())
                if not (r_idx in rela_thres):
                    continue
                total += 1
                if s < rela_thres[r_idx]:
                    correct += 1
        return 100*float(correct)/total
    """
                
