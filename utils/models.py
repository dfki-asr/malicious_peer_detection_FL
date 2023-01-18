from typing import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import os

# Hard coding the value for testing purpose
flat_shape = [784]
PRINT_REQ = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
class Encoder(nn.Module):
    def __init__(self, dim_x, dim_y, dim_z):

        super().__init__()
         # Encoder layers
        self.conv1 = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=5, stride=1,padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2,padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1,padding='same')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2,padding=0)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=7, stride=1,padding='valid')
        self.lin1 = nn.Linear(in_features=80, out_features=20)
        self.lin2 = nn.Linear(in_features=80, out_features=20)

        # reparameterization
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, inputs, device=DEVICE):
        x = inputs[0].to(device)
        y = inputs[1].to(device)

        # y = F.one_hot(y, 10).to(device)
        y = y.view(-1, 10, 1, 1).to(device)

        ones = torch.ones(x.size()[0], 
                                10,
                                x.size()[2], 
                                x.size()[3], 
                                dtype=x.dtype).to(device)
        y = ones * y
        x = torch.cat((x, y), dim=1)

        print_debug(f"input shape: {x.shape}")
        x = F.relu(self.conv1(x))
        print_debug(x.shape)
        # 32, 28, 28
        x = F.pad(x, (0,3,0,3))
        print_debug(x.shape)
        # 32, 31, 31
        x = F.relu(self.conv2(x))
        print_debug(x.shape)
        # 32, 14, 14
        x = F.relu(self.conv3(x))
        print_debug(x.shape)
        # 64, 14, 14
        x = F.pad(x, (0,3,0,3))
        print_debug(x.shape)
        # 64, 17, 17
        x = F.relu(self.conv4(x))
        print_debug(x.shape)
        # 64, 7, 7
        x = F.relu(self.conv5(x))
        print_debug(x.shape)
        # 80, 1, 1
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        print_debug(f"After flatten shape: {x.shape}")
        # 80
        # print_debug(f"Concatenating {x.shape} with {y.shape}")
        # concat = torch.cat([x, y], dim=-1)
        # print_debug(f"After concatenation shape: {concat.shape}")
        # 90
        # loc=torch.zeros(mu_logvar.shape)
        # scale=torch.ones(mu_logvar.shape)
        # diagn = Independent(Normal(loc, scale), 1)
        mu = self.lin1(x)
        print_debug(f"mu shape: {mu.shape}")
        # 20
        logvar = self.lin2(x)
        print_debug(f"logvar shape: {logvar.shape}")
        z = self.reparameterize(mu, logvar)
        print_debug(f"Returning shape {z.shape}")
        return  mu, logvar, z



class Decoder(nn.Module):
    def __init__(self, dim_y, dim_z):
        super().__init__()
        self.dim_z = dim_z
        self.dim_y = dim_y
        self.deconv1 = nn.ConvTranspose2d(in_channels=30, out_channels=64, kernel_size=7, stride=1, padding=0) # valid means no pad
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1) # pad operation added in forward
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1) # pad operation added in forward
        self.deconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1,padding='same')
        

    def forward(self, inputs, device=DEVICE):
        x = inputs[0].to(device)#.unsqueeze(dim=0)
        y = inputs[1].to(device)
        print_debug(f"latent space shape: {x.shape}, labels shape: {y.shape}")
        x = torch.cat([x, y], dim=1)
        x = torch.reshape(x, (-1, self.dim_z+self.dim_y, 1, 1))
        print_debug(f"After concatenation shape: {x.shape}")
        x = F.relu(self.deconv1(x))
        print_debug(f"ConvTrans1 output shape: {x.shape}")
        x = F.relu(self.deconv2(x))
        print_debug(f"ConvTrans2 output shape: {x.shape}")
        x = F.pad(x, (0,0,0,0))
        x = F.relu(self.deconv3(x))
        print_debug(f"ConvTrans3 output shape: {x.shape}")
        x = F.relu(self.deconv4(x))
        print_debug(f"ConvTrans4 output shape: {x.shape}")
        # x = F.pad(x, (0,3,0,3))
        x = F.relu(self.deconv5(x))
        print_debug(f"ConvTrans5 output shape: {x.shape}")
        x = F.relu(self.deconv6(x))
        print_debug(f"ConvTrans6 output shape: {x.shape}")
        x = torch.sigmoid(self.conv(x))
        print_debug(f"Conv output shape: {x.shape}")
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        print_debug(f"After flatten shape: {x.shape}")
        return x



class CVAE(nn.Module):
    def __init__(self, dim_x, dim_y, dim_z):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flat_shape[0], out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=dim_y),
            nn.Softmax(dim=-1)
        )

        #Encoder 
        self.encoder = Encoder(dim_x=dim_x, dim_y=dim_y, dim_z=dim_z)

        #Decoder
        self.decoder = Decoder(dim_y=dim_y, dim_z=dim_z)

    def forward(self, inputs, device=DEVICE):
        x, y = inputs      
        x = x.to(device)
        y = F.one_hot(y, 10).to(device)  
        print_debug(f"Inputs shape: {x.shape} and labels: {y.shape}")
        c_out = self.classifier(x)
        mu, logvar, z = self.encoder((x,y))
        out = self.decoder((z, y))
        print_debug(f"decoder output shape is: {out.shape}")
        return mu, logvar, out, c_out

    def set_weights(self, weights):
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

   
def print_debug(data):
    if PRINT_REQ:
        print(data)
    else:
        pass