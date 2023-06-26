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
from source.models_cryslator import Generator
from source.GCN_pos import SemiFullGN
from source.data_pos import collate_pool, get_data_loader, CIFData,GaussianDistance
from source.GCN import GCN

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
    
    return N_tr, N_val, N_test, model

def main():
	#taken from sys.argv
    model_folder = 'saved_models'
    chk_name = model_folder+'/'+'chk_pos'
    best_name = model_folder+'/'+'best_pos'
    save_name = model_folder+'/'+'save_pos'
    best_gcn = model_folder+'/'+'best_xmno'
    N_tr, N_val, N_test, gcn = load_gcn(best_gcn)
	#var. for dataset loader
    root_dir ='../Cryslator/data/jsons_xmno_rcut6/'
    root_dir_pos ='../Cryslator/data/pos_xmno/'
    #root_dir3 ='./data/feature_xmno/'
    root_dir_cell ='../Cryslator/data/cell_xmno/'
    max_num_nbr = 8
    radius = 6
    dmin = 0
    step = 0.2
    random_seed = 1234
    batch_size = 24
    N_tr = N_tr*2
    N_val = N_val*2
    N_test = N_test*2
    print(N_tr, N_val, N_test)

    train_idx = list(range(N_tr))
    val_idx = list(range(N_tr,N_tr+N_val))
    test_idx = list(range(N_tr+N_val,N_tr+N_val+N_test))

    num_workers = 0
    pin_memory = False
    return_test = True

	#var for model
    model_args = torch.load(best_gcn)['model_args']
    atom_fea_len = 128
    h_fea_len = model_args['h_fea_len']
    n_conv = 9
    n_h = 6
    lr_decay_rate = 0.99
    lr = 0.0003
    weight_decay = 0.0
    resume = False
    resume_path = 'ddd'
    gdf = GaussianDistance(dmin=0.0, dmax=6.0, step=0.2)
    model_args = {'radius':radius,'dmin':dmin,'step':step,'batch_size':batch_size,
							  'random_seed':random_seed,'N_tr':N_tr,'N_val':N_val,'N_test':N_test,
								'atom_fea_len':atom_fea_len,'h_fea_len':h_fea_len,
								'n_conv':n_conv,'n_h':n_h,'lr':lr,'lr_decay_rate':lr_decay_rate,'weight_decay':weight_decay}
								
	#var for training
    best_mae_error = 1e10
    start_epoch = 0
    epochs = 1000

	#setup
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

    print('# of trainset: ',len(train_loader.dataset))
    print('# of valset: ',len(val_loader.dataset))
    print('# of testset: ',len(test_loader.dataset))

    sample_data_list = [train_dataset[i] for i in sample(range(len(train_dataset)), 500)]
    _, sample_target_cell,sample_target_pos, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target_pos)

	#build model
    structures, _,_,_,_ = train_dataset[0]
    orig_atom_fea_len = structures[0].shape[-1] + 3
    nbr_fea_len = structures[1].shape[-1]
    n_feature=h_fea_len
    model = SemiFullGN(orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len,n_h,n_feature)
    model.cuda()
    netG_A2B = Generator(4,4,4).cuda()
    netG_A2B.load_state_dict(torch.load(model_folder+'/'+'best_G.pth'))
    netG_A2B.eval()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr,weight_decay=weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay_rate)
	
	
    t0 = time.time()
    for epoch in range(start_epoch,epochs):
        train(train_loader,model,gcn,netG_A2B,criterion,optimizer,epoch,normalizer)
        mae_error = validate(val_loader,model,gcn,netG_A2B,criterion,normalizer)
        scheduler.step()
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(),'best_mae_error': best_mae_error,
                         'optimizer': optimizer.state_dict(),'normalizer': normalizer.state_dict(),'model_args':model_args},is_best,chk_name,best_name)

    t1 = time.time()
    print('--------Training time in sec-------------')
    print(t1-t0)
    print('---------Best Model on Validation Set---------------')
    best_checkpoint = torch.load(best_name)
    print(best_checkpoint['best_mae_error'].cpu().numpy())
    print('---------Evaluate Model on Test Set---------------')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader,model,gcn,netG_A2B,criterion,normalizer,test=True)

def train(train_loader, model, gcn, generator, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    
    model.train()
    end = time.time()
    for i, (input,target_cell,target_pos,cif_ids) in enumerate(train_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            input_var = (Variable(input[0].cuda()),
								 Variable(input[1].cuda()),
								 input[2].cuda(),
								 input[3].cuda(),
								 input[4].cuda(),
								 input[5].cuda())
            unrelaxed_feature = gcn.Encoding(*input_var)
        
        unrelaxed_feature = Variable(unrelaxed_feature)
        h_fea_len = unrelaxed_feature.shape[-1]
        feature_length = int(h_fea_len**0.5)
        feature_delta = generator(unrelaxed_feature.reshape(-1,1,feature_length,feature_length)).reshape(-1,h_fea_len)
        
        translated_feature = unrelaxed_feature - feature_delta
        atoms_fea = torch.cat((input[0],input[7]),dim=-1) #input[7]:pos, input[8]:cell_atoms, input[9]:cell_crys
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
#								 [crys_idx.cuda(async=True) for crys_idx in input[5]])

		#normalize target
#		target_normed = normalizer.norm(target)
#		target_var = Variable(target_normed.cuda())
		
		#compute output
        output = model(*input_var2) ; target_var = Variable(target).cuda()
        loss = criterion(output, target_var) ; mae_error = mae(output.data.cpu(), target)

		#measure accuracy
#		mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

		#backward operation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
	
		#measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
          epoch, i, len(train_loader), batch_time=batch_time,
          data_time=data_time, loss=losses, mae_errors=mae_errors))

def validate(val_loader,model,gcn,generator,criterion,normalizer,test=False,save_name='test.csv'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
	#switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input,target_cell,target_pos,cif_ids) in enumerate(val_loader):
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
        
            atoms_fea = torch.cat((input[0],input[7]),dim=-1) #input[7]:pos, input[8]:cell_atoms, input[9]:cell_crys
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

            target_var = Variable(target.cuda(), volatile=True)
        #compute output
            output = model(*input_var2)
            loss = criterion(output, target_var) ; mae_error = mae(output.data.cpu(),target)

		#measure accuracy and record loss
#		mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))

		#measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Test: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
          i, len(val_loader), batch_time=batch_time, loss=losses,
          mae_errors=mae_errors)) 

    star_label = '*'

    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,mae_errors=mae_errors)) ; print(output[0:10]) ; print(target[0:10])
    return mae_errors.avg

class Normalizer(object):
	def __init__(self, tensor):
		self.mean = torch.mean(tensor)
		self.std = torch.std(tensor)

	def norm(self, tensor):
		return (tensor - self.mean) / self.std

	def denorm(self, normed_tensor):
		return normed_tensor * self.std + self.mean

	def state_dict(self):
		return {'mean': self.mean,'std': self.std}

	def load_state_dict(self, state_dict):
		self.mean = state_dict['mean']
		self.std = state_dict['std']

def mae(prediction, target):
	return torch.mean(torch.abs(target - prediction))

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self,val,n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def save_checkpoint(state,is_best,chk_name,best_name):
	torch.save(state, chk_name)
	if is_best:
		shutil.copyfile(chk_name,best_name)

if __name__ == '__main__':
	main()
