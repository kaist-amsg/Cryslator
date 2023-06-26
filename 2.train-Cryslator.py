import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset
from torch.autograd import Variable
from PIL import Image
import torch
from random import sample
from source.models_cryslator import Generator
from source.models_cryslator import Discriminator
from source.utils import *
#from source.data_pix2pix import *
from source.models_GCN import GCN
from source.data_GCN_xmno import collate_pool, get_train_val_test_loader, CIFData
import glob
import os
from tqdm import tqdm 

def construct_dataset(dataset, dataset_u, batch_size, train_idx, val_idx, test_idx):
    collate_fn = collate_pool
    dataset_train = Subset(dataset,train_idx)
    dataset_val = Subset(dataset,val_idx)
    dataset_test = Subset(dataset,test_idx)
    dataset_train_u = Subset(dataset_u,train_idx)
    dataset_val_u = Subset(dataset_u,val_idx)
    dataset_test_u = Subset(dataset_u,test_idx)
    train_loader = DataLoader(dataset_train, batch_size=batch_size,collate_fn=collate_fn,shuffle=False)
    train_loader_u = DataLoader(dataset_train_u, batch_size=batch_size,collate_fn=collate_fn,shuffle=False)
    val_loader = DataLoader(dataset_val, batch_size=batch_size,collate_fn=collate_fn,shuffle=False)
    val_loader_u = DataLoader(dataset_val_u, batch_size=batch_size,collate_fn=collate_fn,shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size,collate_fn=collate_fn,shuffle=False)
    test_loader_u = DataLoader(dataset_test_u, batch_size=batch_size,collate_fn=collate_fn,shuffle=False)

    return train_loader, train_loader_u, val_loader, val_loader_u, test_loader, test_loader_u

def training(epoch, G, D, gcn, loader, loader_u, criterion_mse, criterion_l1, normalizer, optimizer_G, optimizer_D):
    G.train()
    D.train()
    gcn.eval()
    l_G, l_D_A, l_D_B, l_i_A, l_i_B, l_c_A, l_c_B, l_g_B, l_g_A, l_A2B = 0,0,0,0,0,0,0,0,0,0
    real_label = Variable(torch.tensor(1.0),requires_grad = False).cuda()
    fake_label = Variable(torch.tensor(0.0),requires_grad=False).cuda()
    N_tr = 0
    for i, ((input,target,cifs),(input_u,target_u,cifs_u)) in enumerate(tqdm(zip(loader,loader_u),total=len(loader))):
        with torch.no_grad():
             input_var = (input[0].cuda(),
    					 input[1].cuda(),
    					 input[2].cuda(),
    					 input[3].cuda(),
    					 input[4].cuda(),
    					 input[5].cuda())
             input_var_u = (Variable(input_u[0].cuda()),
    					 Variable(input_u[1].cuda()),
    					 input_u[2].cuda(),
    					 input_u[3].cuda(),
    					 input_u[4].cuda(),
    					 input_u[5].cuda())
             real_A = gcn.Encoding(*input_var_u)
             real_B = gcn.Encoding(*input_var)
        
        #print(cifs[0], cifs_u[0])
        feature_length = int(real_A.shape[-1]**0.5)     
        real_A = Variable(real_A).cuda().view(-1,1,feature_length,feature_length)
        real_B = Variable(real_B).cuda().view(-1,1,feature_length,feature_length)
        diff = real_A-real_B
        diff_same = real_B-real_B
        batch_size = real_A.size(0)
        
        ###### Training Generator, G ######
        optimizer_G.zero_grad()
        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        diff_same_pred = G(real_B)
        same_B = real_B - diff_same_pred 
        loss_identity_B = criterion_l1(same_B, real_B)
         
        # GAN loss
        diff_pred = G(real_A)
        fake_B = real_A - diff_pred
        pred_fake_B = D(real_A,fake_B)
                       
        target_tensor = real_label 
        target_real = target_tensor.expand_as(pred_fake_B)
        loss_GAN_A2B = criterion_mse(pred_fake_B, target_real)
            
        loss_A2B = criterion_l1(fake_B,real_B)
        # Total loss
        loss_G = 0.1*loss_identity_B + 0.1*loss_GAN_A2B + loss_A2B
    
        l_A2B += loss_A2B.item()*batch_size
        l_i_B += loss_identity_B.item()*batch_size
        l_g_B += loss_GAN_A2B.item()*batch_size
    
        loss_G.backward()
        optimizer_G.step()
    
        ###### Training Discriminator, D ######
        optimizer_D.zero_grad()
        # Real loss
        pred_real = D(real_A,real_B)
        target_tensor = real_label
        target_real = target_tensor.expand_as(pred_real)
        loss_D_real = criterion_mse(pred_real, target_real)
        # Fake loss
        pred_fake = D(real_A,fake_B.detach())
        target_tensor = fake_label
        target_fake = target_tensor.expand_as(pred_fake)
        loss_D_fake = criterion_mse(pred_fake, target_fake)
        # Total loss
        loss_D = (loss_D_real + loss_D_fake)*0.5
        loss_D.backward()
    
        optimizer_D.step()
    
        l_G += loss_G.item()*batch_size
        l_D_B += loss_D.item()*batch_size
        N_tr += batch_size
    
    print("[Epoch : %d] [Loss G: %f] [Loss D_B : %f] " %(epoch,l_G/N_tr, l_D_B/N_tr))
    print("[Epoch : %d] [Identity B:%f][G_B:%f][A2B:%f]"%(epoch,l_i_B/N_tr,l_g_B/N_tr,l_A2B/N_tr))

def validate(epoch, G, D, gcn, loader, loader_u, criterion_mse, criterion_l1, normalizer):
    G.eval()
    D.eval()
    gcn.eval()
    l_i_A = 0; l_i_B =0; l_c_A = 0; l_c_B = 0; l_A2B=0
    e_i, e_A2B = [], []
    N_val = 0
    for i, ((input,target,cifs),(input_u,target_u,cifs_u)) in enumerate(zip(loader,loader_u)):    
        with torch.no_grad():
            input_var = (Variable(input[0].cuda()),
     							  Variable(input[1].cuda()),
     							  input[2].cuda(),
     							  input[3].cuda(),
     							  input[4].cuda(),
     							  input[5].cuda())
            input_var_u = (Variable(input_u[0].cuda()),
     								 Variable(input_u[1].cuda()),
     								 input_u[2].cuda(),
     								 input_u[3].cuda(),
     								 input_u[4].cuda(),
     								 input_u[5].cuda())
            real_A = gcn.Encoding(*input_var_u)
            real_B = gcn.Encoding(*input_var)
            feature_length = int(real_A.shape[-1]**0.5)
            target = Variable(target).cuda()

            real_A = Variable(real_A).cuda().view(-1,1,feature_length,feature_length)
            real_B = Variable(real_B).cuda().view(-1,1,feature_length,feature_length)
            batch_size = real_A.size(0)
              

            diff_same_pred = G(real_B)
            same_B = real_B - diff_same_pred
            loss_identity_B = criterion_l1(same_B, real_B)
            diff_pred = G(real_A)
            fake_B = real_A - diff_pred
                 
            output_u = gcn.Regressor(fake_B.view(batch_size,-1))
            output_r = gcn.Regressor(same_B.view(batch_size,-1))
                 
            e_i.append(torch.mean(abs(normalizer.denorm(output_r) - target)).item())
            e_A2B.append(torch.mean(abs(normalizer.denorm(output_u) - target)).item())

#             print(recovered_A.shape, real_A.shape)
            loss_A2B = criterion_l1(fake_B,real_B)
            l_A2B += loss_A2B.item()*batch_size
            l_i_B += loss_identity_B.item()*batch_size
            N_val += batch_size
    print("[Epoch : %d] [Val Identity B:%f][Val A2B:%f]"%(epoch,l_i_B/N_val,l_A2B/N_val))    
    print(sum(e_i)/len(e_i), sum(e_A2B)/len(e_A2B))
    return l_A2B/N_val     
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=10, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--load_model', type=bool, default=False)
    opt = parser.parse_args()
    print(opt)
    
    if torch.cuda.is_available() and not opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    G = Generator(4,4,4).cuda()
    D = Discriminator().cuda()
    
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)
    
    criterion_mse = torch.nn.MSELoss()
    criterion_l1 = torch.nn.L1Loss()
    
    optimizer_G = torch.optim.Adam(G.parameters(),lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr*0.03, betas=(0.5, 0.999))
    
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    
    if opt.load_model :
        netG_A2B.load_state_dict(torch.load('model_save_xmno_pix2pix_210812/netG_A2B.pth'))
        netD_B.load_state_dict(torch.load('model_save_xmno_pix2pix_210812/netD_B.pth'))
        print("Models were loaded.")
    
    model_save_folder = './saved_models/'
    if not os.path.isdir(model_save_folder):
        os.makedirs(model_save_folder)
    root_dir = '../Cryslator/data/jsons_xmno_rcut6/'
    
    N_tot = 28579 #full data
    #N_tot = len(glob.glob(root_dir+'/'+'m*.json'))
    #N_val = int(N_tot*0.1)
    best_name = model_save_folder+'/'+'best_xmno'
    checkpoint = torch.load(best_name)
    random_seed = 1234
    feature_length = 16
    model_args = checkpoint['model_args']
    N_tr = model_args['N_tr']
    N_val = model_args['N_val']
    N_test = model_args['N_test']
    print(N_tr,N_test)
    train_idx = list(range(N_tr))
    val_idx = list(range(N_tr,N_tr+N_val))
    test_idx = list(range(N_tr+N_val,N_tot))
    max_num_nbr = 8
    radius = 6.0
    dmin = 0
    step = 0.2
    atom_fea_len = model_args['atom_fea_len']
    h_fea_len = model_args['h_fea_len']
    n_conv = model_args['n_conv']
    n_h = model_args['n_h']
    orig_atom_fea_len = model_args['orig_atom_fea_len']
    nbr_fea_len = model_args['nbr_fea_len']
    dataset = CIFData(root_dir,radius,dmin,step,is_unrelaxed=False,random_seed=random_seed)
    dataset_u = CIFData(root_dir,radius,dmin,step,is_unrelaxed=True,random_seed=random_seed)
    sample_target = sampling(root_dir+'/'+'id_prop_relaxed.csv')
    normalizer = Normalizer(sample_target)
    normalizer.load_state_dict(checkpoint['normalizer'])
    train_loader, train_loader_u,val_loader, val_loader_u, test_loader, test_loader_u = construct_dataset(dataset,dataset_u,
                                                                                opt.batch_size,
                                                                                train_idx,
                                                                                val_idx,
                                                                                test_idx)
    batch_size = opt.batch_size
    best_error = 100 
    gcn = GCN(orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len,n_h)
    gcn.cuda()
    gcn.load_state_dict(checkpoint['state_dict'])
    gcn.eval()
    for epoch in range(opt.epoch, opt.n_epochs):
        training(epoch, G, D, gcn, train_loader, train_loader_u,
                 criterion_mse, criterion_l1,
                 normalizer, optimizer_G, optimizer_D)
        test_loss = validate(epoch, G, D, gcn, test_loader, test_loader_u, criterion_mse, criterion_l1, normalizer)
            
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        for param_group in optimizer_G.param_groups:
            print('Current learning rate :', param_group['lr'])
    
        if best_error >= test_loss:
            torch.save(G.state_dict(), model_save_folder+'/'+'best_G.pth')
            torch.save(D.state_dict(), model_save_folder+'/'+'best_D.pth')
            best_error = test_loss
            print('Best MAE is ', best_error)
    
    G.load_state_dict(torch.load(model_save_folder+'/'+'best_G.pth'))
    D.load_state_dict(torch.load(model_save_folder+'/'+'best_D.pth'))
    best_test_loss = validate(epoch, G, D, gcn, test_loader, test_loader_u, criterion_mse, criterion_l1, normalizer)
            
if __name__ =='__main__':
    main()