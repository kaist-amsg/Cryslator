import numpy as np
import argparse
import sys
import os
import shutil
import json
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
from source.GCN_pos import SemiFullGN
from source.data_pos import collate_pool, get_data_loader, CIFData
from tqdm import tqdm
from source.models_cryslator import Generator
from source.GCN import GCN

def cal_cell_volume(cell):
    batch_size = cell.shape[0]
    result  = []
    for i in range(batch_size):
        c = cell[i]
        v = np.inner(np.cross(c[0,:],c[1,:]),c[2,:])
        result.append(v)
    return np.array(result)

def load_gcn(gcn_name):
    checkpoint = torch.load(gcn_name)
    x = checkpoint['model_args']
    N_tr= x['N_tr']
    N_val = x['N_val']
    N_test = x['N_test']   
    atom_fea_len = x['atom_fea_len']
    h_fea_len = x['h_fea_len']
    n_conv = x['n_conv']
    n_h = x['n_h']
    orig_atom_fea_len = x['orig_atom_fea_len']
    nbr_fea_len = x['nbr_fea_len']
    model =GCN(orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len,n_h)
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return N_tr, N_val, N_test, orig_atom_fea_len, nbr_fea_len, model

def load_model(model_name,orig_atom_fea_len,nbr_fea_len):
    checkpoint = torch.load(model_name)
    x = checkpoint['model_args']
    N_tr= x['N_tr']
    N_val = x['N_val']
    N_test = x['N_test']    
    atom_fea_len = x['atom_fea_len']
    h_fea_len = x['h_fea_len']
    n_conv = x['n_conv']
    n_h = x['n_h']
    orig_atom_fea_len = orig_atom_fea_len+3
    nbr_fea_len = nbr_fea_len
    model = SemiFullGN(orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len,n_h,n_feature=256)
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def main():
    #taken from sys.argv
    model_folder = 'saved_models'
    best_name = model_folder+'/'+'best_pos'
    best_gcn = model_folder+'/'+'best_xmno'
    
    N_tr, N_val, N_test, orig_atom_fea_len, nbr_fea_len, gcn = load_gcn(best_gcn)
    model = load_model(best_name,orig_atom_fea_len, nbr_fea_len)

    root_dir ='../Cryslator/data/jsons_xmno_rcut6/'
    root_dir_pos ='../Cryslator/data/pos_xmno/'
    #root_dir3 ='./data/feature_xmno/'
    root_dir_cell ='../Cryslator/data/cell_xmno/'

    max_num_nbr = 8
    radius = 6
    dmin = 0
    step = 0.2
    random_seed = 1234
    batch_size = 4
    N_tot = 28579 #full data
    N_tr = int(N_tot*0.8)
    N_val = int(N_tot*0.2)
    N_test = N_tot - N_tr - N_val
    train_idx = list(range(N_tr))
    val_idx = list(range(N_tr,N_tr+N_val))
    test_idx = list(range(N_tr+N_val,N_tr+N_val+N_test))
    num_workers = 0
    pin_memory = False
    return_test = True

#var for model
    train_csv = root_dir+'/'+'id_prop_train_all.csv'
    val_csv = root_dir+'/'+'id_prop_val_all.csv'
    test_csv = root_dir+'/'+'id_prop_test_all.csv'
    train_dataset = CIFData(root_dir,root_dir_pos,root_dir_cell,train_csv,radius,dmin,step,random_seed)
    val_dataset = CIFData(root_dir,root_dir_pos,root_dir_cell,val_csv,radius,dmin,step,random_seed)
    test_dataset = CIFData(root_dir,root_dir_pos,root_dir_cell,test_csv,radius,dmin,step,random_seed)
    collate_fn = collate_pool

    train_loader = get_data_loader(train_dataset,collate_fn,batch_size,num_workers,pin_memory,False)
    val_loader = get_data_loader(val_dataset,collate_fn,batch_size,num_workers,pin_memory,True)
    test_loader= get_data_loader(test_dataset,collate_fn,batch_size,num_workers,pin_memory,True)
    generator = Generator(4,4,4).cuda()
    generator.load_state_dict(torch.load(model_folder+'/'+'best_G.pth'))
    generator.eval()
    
#    pos_save_folder  = './pos_save_xmno_notchange'
    pos_save_folder_relaxed = './pos_predicted_relaxed'
    pos_save_folder_unrelaxed = './pos_predicted_unrelaxed'
    os.makedirs(pos_save_folder_relaxed, exist_ok=True)
    os.makedirs(pos_save_folder_unrelaxed, exist_ok=True)
    mae_list = [] ; error_list = []; v_list = [] ; mae_list2 = [] ; vv_unrelaxed = [] ; vv_relaxed = []; vv_prediction = []
    mae_list_unrelaxed = [] ; mae_list_relaxed = []; error_list_unrelaxed = [] ; error_list_relaxed = [] ; mae_list_previous = []
    for i, (input,target_cell,target_pos,cif_ids) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            input_var = (Variable(input[0].cuda()),
								 Variable(input[1].cuda()),
								 input[2].cuda(),
								 input[3].cuda(),
								 input[4].cuda(),
								 input[5].cuda())
            unrelaxed_feature = gcn.Encoding(*input_var)
            unrelaxed_feature = Variable(unrelaxed_feature,volatile=True)
            h_fea_len = unrelaxed_feature.shape[-1]
            feature_length = int(h_fea_len**0.5)
            feature_delta = generator(unrelaxed_feature.reshape(-1,1,feature_length,feature_length)).reshape(-1,h_fea_len)
            translated_feature = unrelaxed_feature - feature_delta
            atoms_fea = torch.cat((input[0],input[7]),dim=-1)
            input_var2 = (Variable(atoms_fea.cuda()),
								 Variable(input[1].cuda()),
								 input[2].cuda(),
								 input[3].cuda(),
								 input[4].cuda(),
								 input[5].cuda(),
                                 unrelaxed_feature,
                                 translated_feature,
                                 input[9][:,:9].cuda()
                                 ) ; target = target_pos
            batch_size = target.shape[0]
            target_var = Variable(target.cuda(), volatile=True)
            output = model(*input_var2)
            mae_error = mae(output.data.cpu(),target)
            mae_list.append(mae_error)
#        print(input[8][:,:9], target, output)

            crystal_atom_idx = input[5].numpy()
            #unrelaxed_pos = input[0][:,-3:]
            unrelaxed_pos = input[7]
#        print(unrelaxed_pos, unrelaxed_pos.shape)
            count = 0
            idxs, nums = np.unique(crystal_atom_idx,return_counts=True)
            output = output.detach().cpu().numpy()
            for i in idxs:
                num_i = nums[i]
                pos_unrelaxed = unrelaxed_pos[count:count+num_i].numpy()
                pos_relaxed = target[count:count+num_i].numpy() + pos_unrelaxed 
                pos_prediction = output[count:count+num_i] + pos_unrelaxed
    #            pos_prediction = pos_unrelaxed #not changed
                count += num_i
                if '_unrelaxed' in cif_ids[i]:

                    name = cif_ids[i].split('_unrelaxed')[0]+'_pos.npy'
                    mae_list_unrelaxed.append(mae(torch.Tensor(pos_prediction),torch.Tensor(pos_relaxed)))
                    error_list_unrelaxed.append(np.mean(np.mean(abs((pos_relaxed - pos_unrelaxed) - (pos_prediction - pos_unrelaxed)),axis=-1),axis=-1))
                    mae_list_previous.append(mae(torch.Tensor(pos_unrelaxed),torch.Tensor(pos_relaxed)))
                    np.save(pos_save_folder_unrelaxed+'/'+name,pos_prediction)

                else:
                    name = cif_ids[i].split('_relaxed')[0]+'_pos.npy'
                    mae_list_relaxed.append(mae(torch.Tensor(pos_prediction),torch.Tensor(pos_relaxed)))
                    error_list_relaxed.append(np.mean(np.mean(abs((pos_relaxed - pos_unrelaxed) - (pos_prediction - pos_unrelaxed)),axis=-1),axis=-1))
                    np.save(pos_save_folder_relaxed+'/'+name,pos_prediction)

    print('Total MAE : ',np.mean(np.array(mae_list)))
    print('Unrelaxed MAE : ',np.mean(np.array(mae_list_unrelaxed)))
    print('Relaxed MAE : ',np.mean(np.array(mae_list_relaxed)))
    print('Previous MAE : ',np.mean(np.array(mae_list_previous)))
    error_list_unrelaxed = np.array(error_list_unrelaxed)
    error_list_relaxed = np.array(error_list_relaxed)
    print(error_list_unrelaxed.shape)
    np.save('pos_error_list_unrelaxed.npy',error_list_unrelaxed)
    np.save('pos_error_list_relaxed.npy',error_list_relaxed)

def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))

if __name__ == '__main__':
    main()
