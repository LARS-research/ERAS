import torch
import itertools
import torch.nn as nn
import numpy as np
from torch.optim import Adagrad

class Regressor(nn.Module):
    def __init__(self, length=5):
        super(Regressor, self).__init__()
        self.h1_dim = 256
        self.h2_dim = 64
        self.dropout = nn.Dropout(0.0)
        self.layer1 = nn.Linear(16*6, self.h1_dim)
        self.layer2 = nn.Linear(self.h1_dim, self.h2_dim)
        self.layer3 = nn.Linear(self.h2_dim, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        n = x.size(0)
        #outs = self.dropout(outs)
        outs = self.layer1(x)
        outs = self.act(self.dropout(outs))
        outs = self.layer2(outs)
        outs = self.act(self.dropout(outs))
        outs = self.layer3(outs)
        outs = outs.squeeze()
        outs = torch.sigmoid(outs)
        return outs


def struct2matrix(struct, augment=False):
    if augment:
        matrices = []
        signs= [[1,1,1,1], [1,1,1,-1], [1,1,-1,1], [1,1,-1,-1], [1,-1,1,1], [1,-1,1,-1], [1,-1,-1,1], [1,-1,-1,-1], [-1,1,1,1], [-1,1,1,-1], [-1,1,-1,1], [-1,1,-1,-1], [-1,-1,1,1], [-1,-1,1,-1], [-1,-1,-1,1], [-1,-1,-1,-1]]
        for perm in list(itertools.permutations([0, 1, 2, 3])):
            for s in range(16):
                matrix = [0] * (16*6)
                sign = signs[s]
                base = 16*4
                for i in range(4):
                    r = perm[i]
                    h = i
                    t = struct[i]
                    matrix[h*16 + t*4 + r] = 1
                    if sign[i] == 1:
                        matrix[base+8*h+2*t+0] = 1
                    elif sign[i] == -1:
                        matrix[base+8*h+2*t+1] = 1
                matrices.append(matrix)
    else:
        matrix = [0] * (16*6)
        for i in range(4):
            r = i
            h = i
            t = struct[i]
            matrix[h*16 + t*4 + r] = 1
            matrix[4*16+8*h+2*t+0] = 1
        matrices = [matrix]
    return matrices

def struct2matrix_B5(struct):
    matrix = [0] * (16*6)
    for i in range(4):
        r = i
        h = i
        t = struct[i]
        matrix[h*16+t*4+r] = 1
        matrix[4*16+8*h+2*t+0] = 1
    r = struct[4]
    h = struct[5]
    t = struct[6]
    sign = struct[7]
    matrix[h*16+t*4+r] = 1
    if sign == 1:
        matrix[4*16+8*h+2*t+0] = 1
    else:
        matrix[4*16+8*h+2*t+1] = 1
    return matrix

    #matrix = [0] * 16
    #for i in range(4):
    #    a = struct[i]
    #    matrix[i*4+a] = i+1
    #struct = struct[4:]
    #length = len(struct) // 4
    #for i in range(length):
    #    r = struct[i*4+0] + 1
    #    h = struct[i*4+1]
    #    t = struct[i*4+2]
    #    sign = struct[i*4+3]
    #    if sign == -1:
    #        r = r+4
    #    matrix[h*4+t] = r
    #return matrix

############## read B4 ##############
in_path = 'results/WN18_64.txt'
structs = []
performs = []
with open(in_path, 'r') as f:
    lines = f.readlines()
    for l in lines:
        outs = l.strip().split()
        struct = [int(outs[0]), int(outs[1]), int(outs[2]), int(outs[3])]
        structs.append(struct)
        performs.append(float(outs[5]))

# split train and valid
mat_train = []
per_train = []
mat_valid = []
per_valid = []
mat_all = []
per_all = performs

n = len(structs)
n_train = n*4//5
for i in range(n):
    struct = structs[i]
    perform = performs[i]
    if i<n_train:
        mat = struct2matrix(struct, True)
        mat_train += mat
        per_train += [perform] * len(mat)
    else:
        mat = struct2matrix(struct, False)
        mat_valid += mat
        per_valid += [perform] * len(mat)
    mat_all += struct2matrix(struct, False)


x_all = np.array(mat_all)
x_train = np.array(mat_train)
x_valid = np.array(mat_valid)

y_all = np.array(per_all)
y_train = np.array(per_train)
y_valid = np.array(per_valid)

n_train = len(y_train)

x_all = torch.FloatTensor(x_all).cuda()
y_all = torch.FloatTensor(y_all).cuda()
x_train = torch.FloatTensor(x_train).cuda()
y_train = torch.FloatTensor(y_train).cuda()
x_valid = torch.FloatTensor(x_valid).cuda()
y_valid = torch.FloatTensor(y_valid).cuda()
print(x_train.size(), x_valid.size(), x_all.size())



############## read B5 #################
in_path = 'results/WN18_B5.txt'
structs = []
mat_B5 = []
performs = []
with open(in_path, 'r') as f:
    lines = f.readlines()
    for l in lines:
        outs = l.strip().split()
        struct = [int(o) for o in outs[0:8]]
        mat = struct2matrix_B5(struct)
        structs.append(struct)
        mat_B5.append(mat)
        performs.append(float(outs[9]))
x_B5 = torch.FloatTensor(np.array(mat_B5)).cuda()
y_B5 = torch.FloatTensor(np.array(performs)).cuda()

model = Regressor().cuda()
l1 = nn.L1Loss()

n_epoch = 1000
optim = Adagrad(model.parameters(), lr=0.01, weight_decay=0.003)
batch_size = 1000
n_batch = n_train // batch_size + 1

for epoch in range(n_epoch):
    loss_t = 0
    for i in range(n_batch):
        start = i*batch_size
        end = min(n_train, (i+1)*batch_size)
        x_batch = x_train[start:end]
        y_batch = y_train[start:end]
        model.zero_grad()
        y_p = model(x_batch)
        loss = l1(y_p, y_batch)
        loss.backward()
        optim.step()
        loss_t += loss.cpu().data.numpy()
    y_pred_v = model(x_valid)
    loss_v = l1(y_pred_v, y_valid)
    y_pred_a = model(x_all)
    loss_a = l1(y_pred_a, y_all)
    print('Epoch:{}, Loss:{}, Loss_valid:{}, Loss_all: {}'.format(epoch, loss_t/n_batch, loss_v.data.cpu().numpy(), loss_a.data.cpu().numpy()))
print(y_valid.topk(10))
print(y_pred_v.topk(10))
print(y_all.topk(20))
print(y_pred_a.topk(20))
print('\ntesting.........................B5')
y_p_B5 = model(x_B5)
loss_b5 = l1(y_p_B5, y_B5)
print(loss_b5)
print(y_p_B5)
print(y_p_B5.topk(20))
print(y_B5.topk(20))
print(y_p_B5[119], y_p_B5[312], y_p_B5[544], y_p_B5[27], y_p_B5[90], y_p_B5[163], y_p_B5[204], y_p_B5[541], y_p_B5[331], y_p_B5[497])


#top_val = y_p.topk(20)[0].cpu().data.numpy()
#top_idx = y_p.topk(20)[1].cpu().data.numpy()
#
#for i in range(len(top_idx)):
#    idx = top_idx[i]
#    print(idx, top_val[i])
#    matrixs = []
#    for a in range(4):
#        for b in range(4):
#            for c in range(4):
#                struct = structs[idx]
#                struct += [a,b,c,1]
#                matrixs.append(struct2matrix(struct))
#    x = torch.LongTensor(np.array(matrixs, dtype='int')).cuda()
#    y_pred = model(x)
#    print(y_pred)


