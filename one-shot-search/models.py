import torch
import torch.nn as nn


class KGEModule(nn.Module):
    def __init__(self, n_ent, n_rel, args, rela_cluster, model_type):
        
        
        
        super(KGEModule, self).__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        
        self.args = args
        self.n_dim = args.n_search_dim if model_type == "search" else args.n_stand_dim
        self.lamb = args.lamb

        self.ent_embed = nn.Embedding(n_ent, self.n_dim)
        self.rel_embed = nn.Embedding(n_rel, self.n_dim)
        self.init_weight()
        
        self.K = args.m
        self.GPU = args.GPU
        self.rela_cluster = rela_cluster
        self.n_cluster = args.n
        
    def init_weight(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param.data)
            
    def name(self, idx):
        i = idx[0]
        i_rc =  self.rela_cluster[i]
        self.r_embed[i,:,:] = self.rel_embed_2K_1[i,self.idx_list[i_rc],:] * self._arch_parameters[i_rc][[j for j in range(self.K*self.K)], self.idx_list[i_rc]].view(-1,1)

    def forward(self, struct, head, tail, rela, cluster_rela_dict, updateType="weights"):

        self.cluster_rela_dict = cluster_rela_dict
        
        """convert the architect into struct list"""        
        length = self.n_dim // self.K
        
        # create a rela_embed with size (n_rel, 2K+1, length)
        rel_embed_pos = self.rel_embed.weight.view(-1, self.K, length)
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
        
#        for i_rc in range(self.n_cluster):
#            x = torch.LongTensor([i_rc for i in range(len(self.cluster_rela_dict[i_rc]))])
#            max_idx_list = torch.argmax(struct[i_rc][:,:],1)
#            self.r_embed[self.cluster_rela_dict[i_rc],:,:] = self.rel_embed_2K_1[self.cluster_rela_dict[i_rc]][:,max_idx_list,:] #* struct[x][:,[i for i in range(16)],max_idx_list].view(-1,16,1)

        for i_rc in range(self.n_cluster):
            #x = torch.LongTensor([i_rc for i in range(len(self.cluster_rela_dict[i_rc]))])
            max_idx_list = struct[i_rc]
            self.r_embed[self.cluster_rela_dict[i_rc],:,:] = self.rel_embed_2K_1[self.cluster_rela_dict[i_rc]][:,max_idx_list,:]
        
        self.r_embed = self.r_embed.view(-1, self.K, self.K, length)
                
        head = head.view(-1)
        tail = tail.view(-1)
        rela = rela.view(-1)
        
        head_embed = self.ent_embed(head).view(-1, self.K, self.n_dim//self.K)
        tail_embed = self.ent_embed(tail).view(-1, self.K, self.n_dim//self.K)
        rela_embed = self.r_embed[rela,:,:,:]
        
        pos_trip = self.test_trip(head_embed, rela_embed, tail_embed)
        
        neg_tail = self.test_tail(head_embed, rela_embed)
        neg_head = self.test_head(rela_embed, tail_embed)

        max_t = torch.max(neg_tail, 1, keepdim=True)[0]
        max_h = torch.max(neg_head, 1, keepdim=True)[0]

        loss = - 2*pos_trip + max_t + torch.log(torch.sum(torch.exp(neg_tail - max_t), 1)) +\
               max_h + torch.log(torch.sum(torch.exp(neg_head - max_h), 1))
        
        self.regul = torch.sum(rela_embed**2)
        

        return torch.sum(loss) 
    
    def test_trip(self, head, rela, tail):
        vec_hr = self.get_hr(head, rela)
        scores = (vec_hr*tail).view(-1, self.n_dim)
        return torch.sum(scores, 1)


    def get_hr(self, head, rela):
        n_head, length = len(head), self.n_dim//self.K
        if self.GPU:
            vs = torch.zeros(n_head, self.K, length).cuda()
        else:
            vs = torch.zeros(n_head, self.K, length)
        for i in range(self.K):
            #vs += head*rela[:,:,i,:] #please pay attention to it
            vs[:,i,:] = torch.sum((head*rela[:,:,i,:]),1)
        return vs
    
    def get_rt(self, rela, tail):
        n_head, length = len(tail), self.n_dim//self.K
        if self.GPU:
            vs = torch.zeros(n_head, self.K, length).cuda()
        else:
            vs = torch.zeros(n_head, self.K, length)
        for i in range(self.K):
            #vs += tail*rela[:,:,i,:]
            #vs += tail*rela[:,i,:,:] #please pay attention to it
            vs[:,i,:] = torch.sum((tail*rela[:,i,:,:]) ,1)            
        return vs
        
    def test_tail(self, head, rela):
        #print(self.struct)
        vec_hr = self.get_hr(head, rela).view(-1,self.n_dim)
        tail_embed = self.ent_embed.weight
        scores = torch.mm(vec_hr, tail_embed.transpose(1,0))
        return scores
    
    def test_head(self, rela, tail):
        #print(self.struct)
        vec_rt = self.get_rt(rela, tail).view(-1,self.n_dim)
        head_embed = self.ent_embed.weight
        scores = torch.mm(vec_rt, head_embed.transpose(1,0))
        return scores
    
    
    def arch_parameters(self):
        return self._arch_parameters

    
    def _loss(self, h, t, r, updateType, cluster_rela_dict):
        return self.forward(h, t, r, cluster_rela_dict, updateType)


    




