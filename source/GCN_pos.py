from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self,atom_fea_len,nbr_fea_len):
        super(ConvLayer,self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        self.tanh_e = nn.Tanh()
        self.tanh_v = nn.Tanh()

        self.bn_v = nn.BatchNorm1d(self.atom_fea_len)

        self.phi_e = nn.Sequential(
											nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,self.atom_fea_len),
											nn.LeakyReLU(0.2),
											nn.Linear(self.atom_fea_len,self.atom_fea_len),
											nn.LeakyReLU(0.2),
											nn.Linear(self.atom_fea_len,self.atom_fea_len))

        self.phi_v = nn.Sequential(
											nn.Linear(2*self.atom_fea_len,self.atom_fea_len),
											nn.LeakyReLU(0.2),
											nn.Linear(self.atom_fea_len,self.atom_fea_len),
											nn.LeakyReLU(0.2),
											nn.Linear(self.atom_fea_len,self.atom_fea_len))

    def forward(self,atom_in_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx):
        N,M = atom_in_fea.shape
        atom_nbr_fea1 = atom_in_fea[nbr_fea_idx1,:]
        atom_nbr_fea2 = atom_in_fea[nbr_fea_idx2,:]
        nbr_num_fea = num_nbrs[nbr_fea_idx1].view(-1,1)

		#update edge
        total_nbr_fea = torch.cat([atom_nbr_fea1,atom_nbr_fea2,nbr_fea],dim=1)
        ek = self.phi_e(total_nbr_fea)
        rho_e_v = Variable(torch.zeros((N,M)).cuda()).scatter_add(0, nbr_fea_idx1.view(-1,1).repeat(1,M),ek/nbr_num_fea)
		
		#update node
        total_node_fea = torch.cat([atom_in_fea,rho_e_v],dim=1)
        vi = self.phi_v(total_node_fea)		
        vi = self.bn_v(vi)

        ek = nbr_fea + ek
        vi = atom_in_fea + vi
		
        ek_sum = Variable(torch.zeros((N,M)).cuda()).scatter_add(0,nbr_fea_idx1.view(-1,1).repeat(1,M),ek/nbr_num_fea)

        Ncrys = torch.unique(crystal_atom_idx.view(-1,1)).shape[0]
        atom_nbr_fea = torch.cat([vi,ek_sum],dim=1) 
        global_fea = Variable(torch.zeros((Ncrys,2*M)).cuda()).scatter_add(0,crystal_atom_idx.view(-1,1).repeat(1,2*M),atom_nbr_fea)

        return ek,vi,global_fea,atom_nbr_fea

class SemiFullGN(nn.Module):
    def __init__(self,orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len,n_h,n_feature):    
        super(SemiFullGN, self).__init__()
        self.node_embedding = nn.Linear(orig_atom_fea_len,atom_fea_len)
        self.edge_embedding = nn.Linear(nbr_fea_len,atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,nbr_fea_len=atom_fea_len)
																for _ in range(n_conv)])

        
        self.feature_embedding = nn.Sequential(nn.Linear(2*n_feature,512))
        self.feature_delta_embedding = nn.Sequential(nn.Linear(n_feature,128))
        self.atom_nbr_fea_embedding = nn.Sequential(nn.Linear(2*atom_fea_len,128))
        self.cell_embedding = nn.Sequential(nn.Linear(9,128))
        self.phi_pos = nn.Sequential(nn.Linear(512+128+128+128,512),
                                     nn.BatchNorm1d(512),
                                     nn.LeakyReLU(0.2))
        self.conv = nn.Sequential(nn.Conv1d(64,512,3,stride=1,padding=0),nn.BatchNorm1d(512),nn.LeakyReLU(0.2),
                                   nn.Conv1d(512,512,3,stride=1,padding=0),nn.BatchNorm1d(512),nn.LeakyReLU(0.2),
                                   nn.Conv1d(512,256,3,stride=1,padding=1),nn.LeakyReLU(0.2),
                                   nn.Conv1d(256,256,3,stride=1,padding=1),nn.LeakyReLU(0.2),
                                   nn.Conv1d(256,1,2,stride=1,padding=0))

    def forward(self,atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx,unrelaxed_feature,relaxed_feature,cell):
        atom_fea = self.node_embedding(atom_fea)
        nbr_fea = self.edge_embedding(nbr_fea)
        N,M = atom_fea.shape 

        Ncrys = torch.unique(crystal_atom_idx.view(-1,1)).shape[0]
        atom_nums_ = Variable(torch.ones((N,1)).cuda()) 
        atom_nums = Variable(torch.zeros((Ncrys,1)).cuda()).scatter_add(0,crystal_atom_idx.view(-1,1),atom_nums_)#shape = (Batch_size,1)

        N,M = atom_fea.shape ; 
        for conv_func in self.convs:
            nbr_fea,atom_fea,global_fea,atom_nbr_fea = conv_func(atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx)
        u_feature = unrelaxed_feature[crystal_atom_idx]
        r_feature = relaxed_feature[crystal_atom_idx]
        feature_delta = u_feature - r_feature
        u_cell = cell[crystal_atom_idx]
        feature = torch.cat((u_feature,r_feature),dim=-1)
        feature = self.feature_embedding(feature)
        feature_delta = self.feature_delta_embedding(feature_delta)
        atom_nbr_fea = self.atom_nbr_fea_embedding(atom_nbr_fea)
        u_cell = self.cell_embedding(u_cell)
        
        
        final_feature = torch.cat((atom_nbr_fea,u_cell,feature,feature_delta),dim=-1)
        pos_delta = self.phi_pos(final_feature)
        pos_delta = pos_delta.view(N,64,8)
        pos_delta = self.conv(pos_delta).squeeze(1)
        #print('pos_delta shape is ', pos_delta, pos_delta.shape)
        return pos_delta
