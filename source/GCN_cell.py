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

		return ek,vi,global_fea

class SemiFullGN(nn.Module):
    def __init__(self,orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len,n_h,n_feature):    
        super(SemiFullGN, self).__init__()
        self.node_embedding = nn.Linear(orig_atom_fea_len,atom_fea_len)
        self.edge_embedding = nn.Linear(nbr_fea_len,atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,nbr_fea_len=atom_fea_len)
																for _ in range(n_conv)])

        self.phi_u = nn.Sequential(
											nn.Linear(2*atom_fea_len,h_fea_len),
											nn.LeakyReLU(0.2),
											nn.Linear(h_fea_len,h_fea_len),
											nn.Tanh())

        self.conv_to_fc = nn.Linear(h_fea_len, h_fea_len) ; self.delta_embedding = nn.Linear(n_feature,h_fea_len,bias=False)
        self.conv_to_fc_lrelu = nn.LeakyReLU(0.2) ; self.z_cat = nn.Sequential(nn.Linear(h_fea_len+n_feature*2,h_fea_len), nn.LeakyReLU(0.2), nn.BatchNorm1d(h_fea_len))

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h-1)])
            self.activations = nn.ModuleList([nn.LeakyReLU(0.2) for _ in range(n_h-1)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(h_fea_len) for _ in range(n_h-1)])
        self.fc_out = nn.Sequential(nn.Linear(h_fea_len,256), nn.LeakyReLU(0.2), nn.Linear(256,256), nn.LeakyReLU(0.2), nn.Linear(256,64*3),nn.LeakyReLU(0.2))
        self.output_layer=nn.Sequential(nn.Linear(64*3,128),nn.LeakyReLU(0.2),nn.Linear(128,9))

    def forward(self,atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx,unrelaxed_feature,relaxed_feature,cell,delta):
        atom_fea = self.node_embedding(atom_fea)
        nbr_fea = self.edge_embedding(nbr_fea)
        N,M = atom_fea.shape 

        Ncrys = torch.unique(crystal_atom_idx.view(-1,1)).shape[0]
        atom_nums_ = Variable(torch.ones((N,1)).cuda()) 
        atom_nums = Variable(torch.zeros((Ncrys,1)).cuda()).scatter_add(0,crystal_atom_idx.view(-1,1),atom_nums_)#shape = (Batch_size,1)

        N,M = atom_fea.shape ; 
        for conv_func in self.convs:
            nbr_fea,atom_fea,global_fea = conv_func(atom_fea,nbr_fea,nbr_fea_idx1,nbr_fea_idx2,num_nbrs,crystal_atom_idx)

        global_fea = global_fea / atom_nums
        z = self.phi_u(global_fea)
        #print(z.shape,unrelaxed_feature.shape,relaxed_feature.shape)
        zz = torch.cat([z,unrelaxed_feature,relaxed_feature],dim=-1)
        zz = self.z_cat(zz)
        delta_ = self.delta_embedding(delta) ; zzz = z*delta_
        crys_fea = self.conv_to_fc_lrelu(self.conv_to_fc(zz))
        if hasattr(self,'fcs') and hasattr(self,'activations'):
            for fc,activation,bn in zip(self.fcs,self.activations,self.bns):
                crys_fea = activation(fc(crys_fea))
#                crys_fea = bn(crys_fea)
#                crys_fea = F.dropout(crys_fea,0.2,training=self.training)

        z_last = crys_fea
        out = self.fc_out(crys_fea)
        batch_size = out.shape[0]
        #print(cell.shape)
        out = out.view(batch_size,64,3)
        cell = cell.view(batch_size,3,3)
        out = torch.matmul(out,cell)
#        print(out.shape)
        out = out.view(batch_size,-1)
        out = self.output_layer(out)
        return out,z
