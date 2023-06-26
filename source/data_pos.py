from __future__ import print_function, division
import os
import csv
import re
import json
import functools
import random

from ase.io import read

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.core.structure import Structure
from pymatgen.io.ase import *

def get_data_loader(dataset, collate_fn=default_collate,
                              batch_size=64,num_workers=0, pin_memory=False, test=False):

    total_size = len(dataset)
    if not test:
        data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)

    else:
        data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=False,
                          num_workers=num_workers,
                          collate_fn=collate_fn, pin_memory=pin_memory)

    return data_loader

def collate_pool(dataset_list):
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx1, batch_nbr_fea_idx2 = [], [], [], []
    batch_num_nbr = [] ; batch_unrelaxed_feature = [] ; batch_cell_atoms=[]; batch_cell_crys = [];batch_relaxed_feature = []
    crystal_atom_idx= [] ;  batch_target_cell = [] ; batch_target_pos= [] ; batch_delta = []; batch_pos=[]
    batch_cif_ids = [] ; batch_dij_ = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx1, nbr_fea_idx2, num_nbr, dij_), (pos,cell_atoms,cell_crys), target_cell, target_pos,cif_id)\
        in enumerate(dataset_list):       
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        #atom & bonding feature vector
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea);batch_dij_.append(dij_)
				#graph indexing 
        tt1 = np.array(nbr_fea_idx1)+base_idx
        tt2 = np.array(nbr_fea_idx2)+base_idx

        batch_nbr_fea_idx1.append(torch.LongTensor(tt1.tolist()))
        batch_nbr_fea_idx2.append(torch.LongTensor(tt2.tolist()))
        batch_num_nbr.append(num_nbr)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
				#crystal_atom_idx.append(new_idx)
        crystal_atom_idx.append(torch.LongTensor([i]*n_i))
        batch_target_cell.append(target_cell)
        batch_target_pos.append(target_pos)
        batch_cell_atoms.append(cell_atoms); batch_cell_crys.append(cell_crys)
        batch_pos.append(pos)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx1, dim=0),torch.cat(batch_nbr_fea_idx2, dim=0),
            torch.cat(batch_num_nbr, dim=0),torch.cat(crystal_atom_idx,dim=0), torch.cat(batch_dij_,dim=0),
            torch.cat(batch_pos,dim=0), torch.cat(batch_cell_atoms,dim=0), torch.cat(batch_cell_crys)),\
            torch.stack(batch_target_cell,dim=0),torch.cat(batch_target_pos, dim=0),batch_cif_ids


class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
#        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
		def __init__(self, elem_embedding_file):
				elem_embedding = json.load(open(elem_embedding_file))
				elem_embedding = {int(key): value for key, value in elem_embedding.items()}
				atom_types = set(elem_embedding.keys())
				super(AtomCustomJSONInitializer, self).__init__(atom_types)
#				for key, _ in elem_embedding.items():
				for key in range(101):
						zz = np.zeros((101,))
						zz[key] = 1.0
						self._embedding[key] = zz.reshape(1,-1)

class CIFData(Dataset):
    def __init__(self,root_dir,root_dir_pos,root_dir_cell,csv_file,radius=4.0,dmin=0,step=0.2,random_seed=123):
        self.root_dir = root_dir ; self.root_dir_pos = root_dir_pos; self.root_dir_cell = root_dir_cell
        self.radius = radius
#        id_prop_file = os.path.join(self.root_dir, 'test.csv')
        id_prop_file = os.path.join(csv_file)
        
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        np.random.seed(random_seed)
        np.random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self,idx):
        cif_id, target = self.id_prop_data[idx] 
        name = '_'.join(cif_id.split('_')[:-1])
        with open(os.path.join(self.root_dir,cif_id+'.json')) as f:
            crystal_data = json.load(f)
        nums = crystal_data['numbers']
        key_unrelaxed = name+'_unrelaxed'
        key_relaxed = name+'_relaxed'
        key_delta = cif_id+'_delta'
#        key_delta = name+'_delta'
        key_delta_pos = name+'_delta'
        atom_fea = np.vstack([self.ari.get_atom_fea(nn) for nn in nums])
#        unrelaxed_pos = np.load(self.root_dir2+'/'+key_unrelaxed+'.npy')
#        relaxed_pos = np.load(self.root_dir2+'/'+key_relaxed+'.npy')
        pos = np.load(self.root_dir_pos+'/'+cif_id+'.npy')
        cell = np.load(self.root_dir_cell+'/'+cif_id+'.npy')
        cell_repeat = np.repeat(cell[0,0:9].reshape(1,9),len(nums),axis=0)
        #atom_fea =  np.hstack((atom_fea,pospos,cell_))
        
        index1 = np.array(crystal_data['index1'])
        nbr_fea_idx = np.array(crystal_data['index2'])
        dij = np.array(crystal_data['dij']); dij_ = torch.Tensor(dij)
        nbr_fea = self.gdf.expand(dij)
        num_nbr = np.array(crystal_data['nn_num'])
        

        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx1 = torch.LongTensor(index1)
        nbr_fea_idx2 = torch.LongTensor(nbr_fea_idx)
        num_nbr = torch.Tensor(num_nbr)
        
        pos = torch.Tensor(pos)
        cell_crys = torch.Tensor(cell)
        cell_atoms = torch.Tensor(cell_repeat)
        target_pos = torch.Tensor(np.load(self.root_dir_pos+'/'+key_delta+'.npy'))
        target_cell = torch.Tensor(np.load(self.root_dir_cell+'/'+key_delta+'.npy').reshape(9,)).float()
        #unrelaxed_feature = torch.Tensor(np.load(self.root_dir3+'/'+cif_id+'.npy')).reshape(1,400)
        #relaxed_feature = torch.Tensor(np.load(self.root_dir3+'/'+key_relaxed+'.npy')).reshape(1,400)
        #print(unrelaxed_feature.shape)
        #feature_delta = unrelaxed_feature - relaxed_feature
        #feature_shape = feature_delta.shape
        #delta_noise = torch.normal(0,1e-6,size=feature_shape)
        #feature_delta += delta_noise
        
        return (atom_fea, nbr_fea, nbr_fea_idx1, nbr_fea_idx2, num_nbr,dij_), (pos,cell_atoms,cell_crys), target_cell, target_pos,cif_id
