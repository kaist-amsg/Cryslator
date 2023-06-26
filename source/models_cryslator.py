import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [nn.Conv2d(in_features, in_features,kernel_size=3,stride=1,padding=1),nn.InstanceNorm2d(in_features),nn.LeakyReLU(0.2,inplace=True),
                      nn.Conv2d(in_features, in_features,kernel_size=3,stride=1,padding=1),nn.InstanceNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)
    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self,n_residual_blocks=3,n_downsampling=3,n_upsampling=3):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.n_downsampling = n_downsampling
        self.n_upsampling = n_upsampling
        # Initial Convolution block
        model =[nn.Conv2d(1,64,3), nn.InstanceNorm2d(64),nn.LeakyReLU(0.2,inplace=True)]
        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(self.n_downsampling):
            model += [nn.Conv2d(in_features, out_features,3,stride=1, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.LeakyReLU(0.2,inplace=True)]
            in_features = out_features
            out_features = in_features*2
        # Residual blocks
        for _ in range(self.n_residual_blocks):
            model += [ResidualBlock(in_features)]
        # Upsampling
        out_features = in_features//2
        for _ in range(self.n_upsampling):
            model += [nn.ConvTranspose2d(in_features, out_features,3,stride=1, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.LeakyReLU(0.2,inplace=True)]
            in_features = out_features
            out_features = in_features//2
        # Output layer
        self.output_layer = nn.Sequential(nn.ConvTranspose2d(64,32,2,stride=1,padding=0),nn.LeakyReLU(0.2,inplace=True),nn.ConvTranspose2d(32,1,2,stride=1,padding=0))
        self.model = nn.Sequential(*model)
    def forward(self, x):
        h = self.model(x)
        output = self.output_layer(h)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        model = [nn.Conv2d(2, 64, 4, stride=1, padding=1),nn.LeakyReLU(0.2, inplace=True) ]
        model += [nn.Conv2d(64, 128, 4, stride=1, padding=1),nn.InstanceNorm2d(128),nn.LeakyReLU(0.2, inplace=True) ]
        model += [nn.Conv2d(128, 256, 3, stride=1, padding=1),nn.InstanceNorm2d(256),nn.LeakyReLU(0.2, inplace=True) ]
        model += [nn.Conv2d(256, 512, 3, stride=1,padding=1),nn.InstanceNorm2d(512),nn.LeakyReLU(0.2, inplace=True) ]
        model += [nn.Conv2d(512, 1, 2, stride=1,padding=1)]

        self.model = nn.Sequential(*model)
#        self.fc = nn.Sequential(nn.Linear(225,64),nn.InstanceNorm1d(64),nn.LeakyReLU(0.2,inplace=True),nn.Linear(64,1),nn.Sigmoid())

    def forward(self,x_a,x_b):
        m = x_a.size(0)
        x = torch.cat((x_a,x_b),dim=1)
        x =  self.model(x)
        validity = x
#        x_ = x.view(m,-1)
#        print('x_ shape is ', x_.shape)
#        validity = self.fc(x_)
        # Average pooling and flatten
        return validity

class CrystalGraphConvNet(nn.Module):
    def __init__(self,atom_fea_len,h_fea_len,n_h):
        super(CrystalGraphConvNet, self).__init__()
#        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
#        self.conv_to_fc_softplus = nn.Softplus()     
#        if n_h > 1:
#            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h-1)])
#            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h-1)])
        self.fc_out = nn.Linear(h_fea_len,1)

    def forward(self,crys_fea):
#        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
#        crys_fea = self.conv_to_fc_softplus(crys_fea)
#        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
#            for fc,softplus in zip(self.fcs,self.softpluses):
#                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        return out

class Regressor(nn.Module):
    def __init__(self,atom_fea_len,n_conv,h_fea_len,n_h):
        super(Regressor, self).__init__()
        self.fc_out = nn.Linear(h_fea_len,1)
        self.conv_to_fc = nn.Linear(h_fea_len, h_fea_len)
        self.conv_to_fc_lrelu = nn.LeakyReLU(0.2)
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h-1)])
            self.activations = nn.ModuleList([nn.LeakyReLU(0.2) for _ in range(n_h-1)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(h_fea_len) for _ in range(n_h-1)])
        self.fc_out = nn.Linear(h_fea_len,1)

    def forward(self,crys_fea):
        crys_fea = self.conv_to_fc_lrelu(self.conv_to_fc(crys_fea))
        if hasattr(self,'fcs') and hasattr(self,'activations'):
            for fc,activation,bn in zip(self.fcs,self.activations,self.bns):
                 crys_fea = activation(fc(crys_fea))
#        z_last = crys_fea
        out = self.fc_out(crys_fea)
        return out

if __name__ =='__main__':
    g = Generator().cuda()
    d = Discriminator().cuda()
    x = Variable(torch.zeros((32,1,16,16))).cuda()

    x_ = g(x)
    validity = d(x_)




