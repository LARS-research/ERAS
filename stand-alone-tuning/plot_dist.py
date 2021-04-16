from models import TransEModule
import torch
from read_data import DataLoader
from corrupter import BernCorrupter
import os
import argparse
import numpy as np
from utils import batch_by_size


parser = argparse.ArgumentParser(description="Parser for Knowledge Graph Embedding")
parser.add_argument('--task_dir', type=str, default='/home/yzhangee/Data/benchmarks/WN18', help='set dataset name')
parser.add_argument('--model', type=str, default='TransE', help='set model type')
parser.add_argument('--sample', type=str, default='cache', help='sampling method after pretraining')
parser.add_argument('--loss', type=str, default='pair', help='sampling method after pretraining')
parser.add_argument('--save', type=bool, default=False, help='whether save model')
parser.add_argument('--load', type=bool, default=False, help='whether load from pretrain model')
parser.add_argument('--optim', type=str, default='adam', help='optimization method')
parser.add_argument('--margin', type=float, default=2.0, help='set margin value')
parser.add_argument('--lamb', type=float, default=0.0, help='set margin value')
parser.add_argument('--hidden_dim', type=int, default=50, help='set embedding dimension')
parser.add_argument('--temp', type=float, default=1.0, help='set temporature value')
parser.add_argument('--gpu', type=str, default='0', help='set gpu #')
parser.add_argument('--p', type=int, default=1, help='set distance norm')
parser.add_argument('--lr', type=float, default=0.001, help='set learning rate')
parser.add_argument('--n_epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--pretrain', type=int, default=4000, help='number of pre-training epochs')
parser.add_argument('--s_epoch', type=int, default=3000, help='which epoch should be saved')
parser.add_argument('--n_batch', type=int, default=1000, help='number of training batches')
parser.add_argument('--n_sample', type=int, default=30, help='number of k nearest neighbors')
parser.add_argument('--n_sample_1', type=int, default=50, help='number of k nearest neighbors')
parser.add_argument('--n_sample_2', type=int, default=50, help='number of k nearest neighbors')
parser.add_argument('--epoch_per_test', type=int, default=20, help='frequency of testing')
parser.add_argument('--test_batch_size', type=int, default=50, help='size of test batch')
parser.add_argument('--filter', type=bool, default=True, help='whether do filter in testing')
parser.add_argument('--out_file', type=str, default='base.txt', help='log prefix')
parser.add_argument('--log_to_file', type=bool, default=False, help='log to file')
parser.add_argument('--log_dir', type=str, default='./log', help='log save dir')

parser.add_argument('--log_prefix', type=str, default='', help='log prefix')
args = parser.parse_args()

filename = os.path.join(args.task_dir, 'TransE.mdl')
loader = DataLoader(args.task_dir, 50)
n_ent, n_rel = loader.graph_size()
train_data = loader.load_data('train')
test_data = loader.load_data('test')
train_data = [torch.LongTensor(vec) for vec in train_data]
test_data = [torch.LongTensor(vec) for vec in test_data]
heads, tails = loader.heads_tails()

model = TransEModule(n_ent, n_rel, args)
model.cuda()
model.load_state_dict(torch.load(filename, map_location=lambda storage, location: storage.cuda()))

head, tail, rela = train_data
n_train = len(head)
n_batch = 10000

corrupter = BernCorrupter(train_data, n_ent, n_rel)

max_dist = 0.
min_dist = 111110.
head = head.cuda()
tail = tail.cuda()
rela = rela.cuda()
#p_bins = np.zeros(17)
#bins = np.zeros(17)
loss_bins = np.zeros(70)
count = 0
zeros = 0

n = 5
dist = np.zeros((n, 17))

for i in range(n):
    randint = np.random.choice(n_train)
    h = head[randint]
    t = tail[randint]
    r = rela[randint]
    for n_t in range(n_ent):
        n_t = torch.LongTensor([n_t]).cuda()
        distance = model.forward(h, n_t, r)
        d = distance.detach().cpu().numpy()
        dist[i, int(d)] += 1

    #for j in range(len(dist[i])-1):
    #    dist[i, j+1] = dist[i, j] + dist[i, j+1]

print("dddddddddd", dist/n_ent)

with open('wn_dist.txt', 'w+') as f:
    for i in range(n):
        for d in dist[i]:
            line = "%.8f "%(d/n_ent)
            f.write(line)
        f.write('\n')


#for epoch in range(args.n_epoch):
#    for h, t, r in batch_by_size(n_batch, head, tail, rela, n_sample=n_train):
#        batch_size = h.size(0)
#        #pos_dist = model.forward(h, t, r)
#        #for dist in pos_dist.detach().cpu().numpy():
#        #    p_bins[int(dist)] += 1
#        prob = corrupter.bern_prob[r]
#        selection = torch.bernoulli(prob).type(torch.ByteTensor)
#
#
#        randint = np.random.randint(low=0, high=n_ent, size=(batch_size,))
#        h_rand = torch.LongTensor(randint).cuda()
#        t_rand = torch.LongTensor(randint).cuda()
#
#        n_h = torch.LongTensor(h.cpu().numpy()).cuda()
#        n_t = torch.LongTensor(t.cpu().numpy()).cuda()
#        n_h[selection] = h_rand[selection]
#        n_t[~selection] = t_rand[~selection]
#
#
#        losses = model.pair_loss(h, t, r, n_h, n_t)
#        losses = losses.detach().cpu().numpy()
#        max_dist = max(max(losses), max_dist)
#        min_dist = min(min(losses), min_dist)
#
#        for l in losses:
#            loss_bins[int(5*l)] += 1
#            if l < 1e-9:
#                zeros += 1
#
#        #head_dist = model.forward(h_rand, t, r)
#        #tail_dist = model.forward(h, t_rand, r)
#        #max_head  = torch.max(head_dist)
#        #min_head = torch.min(head_dist)
#        #max_tail = torch.max(tail_dist)
#        #min_tail = torch.min(tail_dist)
#
#        #max_head = max_head.data.cpu().numpy()
#        #min_head = min_head.data.cpu().numpy()
#        #max_tail = max_tail.data.cpu().numpy()
#        #min_tail = min_tail.data.cpu().numpy()
#
#        #max_dist = max(max_dist, max_head, max_tail)
#        #min_dist = min(min_dist, min_head, min_tail)
#        count += batch_size
#
#        #for dist in head_dist.detach().cpu().numpy():
#        #    bins[int(dist)] += 1
#
#        #for dist in tail_dist.detach().cpu().numpy():
#        #    bins[int(dist)] += 1
#
#    print(max_dist, min_dist)
#    print(loss_bins / count, 1.*zeros/count)
#    #print(p_bins*2/count, bins/count)
#        
#with open('dist.txt', 'w+') as f:
#    for b in loss_bins:
#        line = "%.6f"%(b/count)
#        f.write(line)
#        f.write('\n')
#    f.write('\n\n')
#
#    #for b in bins:
#    #    line = "%.6f"%(b/count)
#    #    f.write(line)
#    #    f.write('\n')
#
#
