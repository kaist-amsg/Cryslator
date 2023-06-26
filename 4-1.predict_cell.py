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
from source.GCN_cell import SemiFullGN
#from data_mod_cell import collate_pool, get_train_val_test_loader, CIFData
from source.data_cell import collate_pool, get_data_loader, CIFData
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
    orig_atom_fea_len = orig_atom_fea_len+9+3
    nbr_fea_len = nbr_fea_len
    model = SemiFullGN(orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len,n_h,n_feature=h_fea_len)
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def main():
    #taken from sys.argv
    model_folder = 'saved_models'
    best_name = model_folder+'/'+'best_cell'
    best_gcn = './saved_models/best_xmno'
    
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
    cell_save_folder_relaxed = './cell_predicted_relaxed'
    cell_save_folder_unrelaxed = './cell_predicted_unrelaxed'
#    cell_save_folder = './cell_save_xmno_notchange'
    os.makedirs(cell_save_folder_relaxed, exist_ok=True)
    os.makedirs(cell_save_folder_unrelaxed, exist_ok=True)
    mae_list = [] ; error_list = []
#    for i, (input, target, batch_cif_ids) in enumerate(tqdm(val_loader)):
    v_error_list_u, v_change0_list_u, v_error_list_r, v_change0_list_r = [],[],[],[]
    vv_unrelaxed_u, vv_relaxed_u, vv_prediction_u = [],[],[]
    vv_unrelaxed_r, vv_relaxed_r, vv_prediction_r = [],[],[]
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
            #input[7]:pos, input[8]:cell_atoms, input[9]:cell_crys
            atoms_fea = torch.cat((input[0],input[7],input[8]),dim=-1)
            input_var2 = (Variable(atoms_fea.cuda()),
								 Variable(input[1].cuda()),
								 input[2].cuda(),
								 input[3].cuda(),
								 input[4].cuda(),
								 input[5].cuda(),
                                 unrelaxed_feature,
                                 translated_feature,
                                 input[9][:,:9].cuda(),
                                 feature_delta) ; target = target_cell
            target_var = Variable(target.cuda(), volatile=True)
            batch_size = target_var.shape[0]
            output,_ = model(*input_var2)
            
            mae_error = mae(output.data.cpu(),target)
            error_list.append(abs(output.data.cpu()-target))
            mae_list.append(mae_error)
        
            cell_unrelaxed = (input[9][:,:9]).reshape(batch_size,3,3)*15.0
            cell_relaxed = (cell_unrelaxed - target.reshape(batch_size,3,3)*15.0) 
            cell_prediction = (cell_unrelaxed - output.detach().cpu().reshape(batch_size,3,3)*15.0) 
#           cell_prediction = cell_unrelaxed # not changed

            volume_unrelaxed = cal_cell_volume(cell_unrelaxed)
            volume_relaxed = cal_cell_volume(cell_relaxed)
            volume_prediction = cal_cell_volume(cell_prediction)
            volume_change0 = abs(volume_relaxed - volume_unrelaxed)
            #volume_change2 = abs(volume_prediction - volume_unrelaxed)/volume_unrelaxed
            volume_error = abs(volume_prediction - volume_relaxed)

            #vv_unrelaxed.append(volume_unrelaxed)
            #vv_relaxed.append(volume_relaxed)
            #vv_prediction.append(volume_prediction)
           


        for i in range(batch_size):
            if '_unrelaxed' in cif_ids[i]:
                #print(batch_cif_ids[i])
                name = cif_ids[i].split('_unrelaxed')[0]+'_cell.npy'
                np.save(cell_save_folder_unrelaxed+'/'+name,cell_prediction[i].reshape(3,3))
                v_error_list_u.append(np.mean(volume_error[i]))
                v_change0_list_u.append(np.mean(volume_change0[i]))
                vv_unrelaxed_u.append(volume_unrelaxed[i])
                vv_relaxed_u.append(volume_relaxed[i])
                vv_prediction_u.append(volume_prediction[i])
            else:
                name = cif_ids[i].split('_relaxed')[0]+'_cell.npy'
                np.save(cell_save_folder_relaxed+'/'+name,cell_prediction[i].reshape(3,3))
                v_error_list_r.append(np.mean(volume_error[i]))
                v_change0_list_r.append(np.mean(volume_change0[i]))
                vv_unrelaxed_r.append(volume_unrelaxed[i])
                vv_relaxed_r.append(volume_relaxed[i])
                vv_prediction_r.append(volume_prediction[i])

        

    x = torch.cat((error_list),dim=0).numpy()
    print(x.shape)
    x = np.mean(x,axis=-1)
    np.save('cell_error_list_augmentation.npy',x)

    
    #plt.show()    
    print(np.mean(np.array(mae_list)))
    print('traslated volume error (unrelaxed) is ', np.mean(np.array(v_error_list_u)))
    print('previous volume change (unrelaxed) is ', np.mean(np.array(v_change0_list_u)))
    print('traslated volume error (relaxed) is ', np.mean(np.array(v_error_list_r)))
    print('previous volume change (relaxed) is ', np.mean(np.array(v_change0_list_r)))




    #vv_unrelaxed_u = np.array(vv_unrelaxed_u)
    #vv_relaxed_u = np.array(vv_relaxed_u)
    #vv_prediction_u = np.array(vv_prediction_u)
    #vv_unrelaxed_r = np.array(vv_unrelaxed_r)
    #vv_relaxed_r = np.array(vv_relaxed_r)
    #vv_prediction_r = np.array(vv_prediction_r)
    #print(vv_unrelaxed_u.shape, vv_relaxed_u.shape, vv_prediction_u.shape)
    #print(vv_unrelaxed_r.shape, vv_relaxed_r.shape, vv_prediction_r.shape)
    #os.makedirs('volume_saved',exist_ok=True)
    #np.save('volume_save/volume_unrelaxed_u.npy',vv_unrelaxed_u)
    #np.save('volume_save/volume_relaxed_u.npy',vv_relaxed_u)
    #np.save('volume_save/volume_prediction_u.npy',vv_prediction_u)
    #np.save('volume_save/volume_unrelaxed_r.npy',vv_unrelaxed_r)
    #np.save('volume_save/volume_relaxed_r.npy',vv_relaxed_r)
    #np.save('volume_save/volume_prediction_r.npy',vv_prediction_r)
    
def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))

if __name__ == '__main__':
    main()
