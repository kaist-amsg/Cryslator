import argparse
import sys
import os
import shutil
import json
import time
import warnings
from random import sample

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from source.utils import *
from source.models_GCN import GCN
from source.data_GCN_xmno import collate_pool, get_train_val_test_loader, CIFData,GaussianDistance
from adabelief_pytorch import AdaBelief

def main():
    model_folder = 'saved_models'
    os.makedirs(model_folder, exist_ok=True)
    chk_name = model_folder+'/'+'chk_gcn'
    best_name = model_folder+'/'+'best_xmno'
    save_name = model_folder+'/'+'save_gcn'

	#var. for dataset loader
    root_dir = '../Cryslator/data/jsons_xmno_rcut6/'
    max_num_nbr = 8
    radius = 6.0
    dmin = 0
    step = 0.2
    random_seed = 1234
    batch_size = 256
    N_tot = 28579 #full data
    N_tr = int(N_tot*0.8)
    N_val = int(N_tot*0.1)
    N_test = N_tot - N_tr - N_val
    

    train_idx = list(range(N_tr))
    val_idx = list(range(N_tr,N_tr+N_val))
    test_idx = list(range(N_tr+N_val,N_tot))
    

    num_workers = 0
    pin_memory = False
    return_test = True

	#var for model
    atom_fea_len = 64
    h_fea_len = 256
    n_conv = 5
    n_h = 3
    lr_decay_rate = 0.98
    lr = 0.001
    weight_decay = 0.0
    noise = 1e-5
    gdf = GaussianDistance(dmin=0.0, dmax=6.0, step=0.2)
    model_args = {'radius':radius,'dmin':dmin,'step':step,'batch_size':batch_size,
							  'random_seed':random_seed,'N_tr':N_tr,'N_val':N_val,'N_test':N_test,
								'atom_fea_len':atom_fea_len,'h_fea_len':h_fea_len,
								'n_conv':n_conv,'n_h':n_h,'lr':lr,'lr_decay_rate':lr_decay_rate,'weight_decay':weight_decay}
								
	#var for training
    best_mae_error = 1e10
    epochs = 500

	#setup
    dataset = CIFData(root_dir,radius,dmin,step,is_unrelaxed=False,random_seed=random_seed)
    collate_fn = collate_pool

    train_loader, val_loader, test_loader = get_train_val_test_loader(dataset,collate_fn,batch_size,
                                                          train_idx,val_idx,test_idx,num_workers,pin_memory)

    sample_target = sampling(root_dir+'/'+'id_prop_relaxed.csv')
    normalizer = Normalizer(sample_target)

	#build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = GCN(orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len,n_h)
    model.cuda()
    model_args['orig_atom_fea_len'] = orig_atom_fea_len
    model_args['nbr_fea_len'] = nbr_fea_len
    model_args['noise'] = noise

    criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(),lr,weight_decay=weight_decay)
    #optimizer = AdaBelief(model.parameters(), lr = lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay_rate)

    t0 = time.time()
    for epoch in range(epochs):
        train(train_loader,model,criterion,optimizer,epoch,normalizer,gdf,noise)
        mae_error = validate(val_loader,model,criterion,normalizer)

        scheduler.step()
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_mae_error': best_mae_error,
                         'optimizer': optimizer.state_dict(),
                         'normalizer': normalizer.state_dict(),
                         'model_args':model_args},is_best,chk_name,best_name)

    t1 = time.time()
    print('--------Training time in sec-------------')
    print(t1-t0)
    print('---------Best Model on Validation Set---------------')
    best_checkpoint = torch.load(best_name)
    print(best_checkpoint['best_mae_error'].cpu().numpy())
    print('---------Evaluate Model on Test Set---------------')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer)

def train(train_loader, model, criterion, optimizer, epoch, normalizer,gdf,e):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    
	#switch to train model
    model.train()
    end = time.time()
    for i, (input,target,_) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input1 = input[1]
        input6 = input[6]
        noise = torch.Tensor(float(e)*np.random.normal(size=input6.shape))
        
        input6 += noise
        input6 = np.array(input6)
        input1_noise = torch.Tensor(gdf.expand(input6))
        input_var = (Variable(input[0].cuda()),
								 Variable(input[1].cuda()),
								 input[2].cuda(),
								 input[3].cuda(),
								 input[4].cuda(),
								 input[5].cuda())
        input_var_noise = (Variable(input[0].cuda()),
								 Variable(input1_noise.cuda()),
								 input[2].cuda(),
								 input[3].cuda(),
								 input[4].cuda(),
								 input[5].cuda())
		#normalize target
        target_normed = normalizer.norm(target)
        target_var = Variable(target_normed.cuda())
		
		#compute output
        output = model(*input_var)
        output_noise = model(*input_var_noise)
        loss = criterion(output, target_var) + criterion(output_noise, target_var)

		#measure accuracy
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
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

def validate(test_loader,model,criterion,normalizer):
	batch_time = AverageMeter()
	losses = AverageMeter()
	mae_errors = AverageMeter()

	#switch to evaluate mode
	model.eval()
	end = time.time()
	for i, (input, target, batch_cif_ids) in enumerate(test_loader):
		input_var = (Variable(input[0].cuda(), volatile=True),
								 Variable(input[1].cuda(), volatile=True),
								 input[2].cuda(),
								 input[3].cuda(),
								 input[4].cuda(),
								 input[5].cuda())

		target_normed = normalizer.norm(target)
		target_var = Variable(target_normed.cuda(),volatile=True)

		#compute output
		output = model(*input_var)
		loss = criterion(output, target_var)

		#measure accuracy and record loss
		mae_error = mae(normalizer.denorm(output.data.cpu()), target)
		losses.update(loss.item(), target.size(0))
		mae_errors.update(mae_error, target.size(0))

		#measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		print('Test: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
          i, len(test_loader), batch_time=batch_time, loss=losses,
          mae_errors=mae_errors))
		star_label = '*'

	print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,mae_errors=mae_errors))
	return mae_errors.avg



if __name__ == '__main__':
	main()
