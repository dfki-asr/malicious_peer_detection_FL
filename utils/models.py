from typing import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from torchmetrics import StructuralSimilarityIndexMeasure as ssim

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

# Harding coding the value for testing purpose
flat_shape = []
flat_shape.append(784)
PRINT_REQ = False
cond_shape=10


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

	def forward(self, inputs, device="cpu"):
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
        

	def forward(self, inputs, device="cpu"):
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

	def forward(self, inputs, device="cpu"):
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

	def get_weights(self):
		"""Get model weights as a list of NumPy ndarrays."""
		return [val.cpu().numpy() for _, val in self.state_dict().items()]

   
def print_debug(data):
	if PRINT_REQ:
		print(data)
	else:
		pass



def loss_fn(recon, x, mu, logvar, c_out, y_onehot, device="cpu"):
	y_onehot1 = y_onehot.type(torch.FloatTensor).to(device)
	# print(c_out.shape, y_onehot.shape, c_out.dtype, y_onehot.dtype)
	classif_loss = torch.nn.BCELoss()(c_out, y_onehot1)
	BCE = F.binary_cross_entropy(recon, x, reduction='sum')        
	KLD = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
	return classif_loss+BCE+KLD, classif_loss, BCE, KLD



def accuracy_fn(y_true, y_pred):
	"""Calculates accuracy between truth labels and predictions.

	Args:
	    y_true (torch.Tensor): Truth labels for predictions.
	    y_pred (torch.Tensor): Predictions to be compared to predictions.

	Returns:
	    [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
	"""

	correct = torch.eq(y_true, y_pred).sum().item()
	acc = (correct / len(y_pred)) * 100
	return acc



def train(model, train_dataloader, epochs, device="cpu"):

	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	train_loss = 0
	classif_accuracy = 0
	for epoch in range(1, epochs + 1):
		for batch, (X, y) in enumerate(train_dataloader):
			X = X.to(device) #[64, 1, 28, 28]
			y = y.to(device)


			# 1. Forward pass
			mu, logvar, recon_batch, c_out = model((X, y))
			# print(f"---------------{torch.argmax(c_out, dim=1).shape}")
			# print(f"---------------{y.shape}")
			flat_data = X.view(-1, flat_shape[0]).to(device)                            
			y_onehot = F.one_hot(y, cond_shape).to(device)
			inp = torch.cat((flat_data, y_onehot), 1)

			# 2. Calculate loss
			loss, C_loss, BCE, KLD = loss_fn(recon_batch, flat_data, mu, logvar, c_out, y_onehot)
			train_loss += loss.item()
			classif_accuracy += accuracy_fn(y, torch.argmax(c_out, dim=1))



			# 3. Zero grad
			optimizer.zero_grad()

			# 4. Backprop
			loss.backward()

			# 5. Step
			optimizer.step()

			if batch % 10 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBCE:{:.4f}\tKLD:{:.4f}\tC_loss:{:.4f}'.format(
					epoch,
					batch * len(X),
					len(train_dataloader.dataset),
					100. * batch / len(train_dataloader),
					loss.item() / len(X), BCE.item() / len(X), KLD.item() / len(X), C_loss.item() / len(X)))
		print('====> Epoch: {} Average loss: {:.4f}\tClassifier Accuracy: {:.4f}'.format(
			epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/len(train_dataloader)))



def test(model, test_dataloader, device="cpu"):
	#Sets the module in evaluation mode
	model.eval()
	test_loss = 0
	classif_accuracy = 0
	with torch.inference_mode():
		for i, (X, y) in enumerate(test_dataloader):
			X = X.to(device)
			y = y.to(device)
			# 1. Forward pass
			mu, logvar, recon_batch, c_out = model((X, y))

			flat_data = X.view(-1, flat_shape[0]).to(device)
			y_onehot = F.one_hot(y, cond_shape).to(device)
			inp = torch.cat((flat_data, y_onehot), 1)

			# 2. Loss
			tot_loss, C_loss, BCE, KLD = loss_fn(recon_batch, flat_data, mu, logvar, c_out, y_onehot)
			test_loss += tot_loss.item()
			classif_accuracy += accuracy_fn(y, torch.argmax(c_out, dim=1))


	test_loss /= len(test_dataloader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	return test_loss, classif_accuracy/len(test_dataloader)


