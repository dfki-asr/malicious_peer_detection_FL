from collections import OrderedDict
from typing import List, Tuple
import argparse

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from utils.datasets import load_partition
from utils.models import CVAE
from utils.partition_data import Partition

import flwr as fl
import numpy as np
import os

torch.manual_seed(0)
# DEVICE='cpu'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(f"Client device: {DEVICE}")
batch_size = 64
local_epochs = 1

# Hard coding values for testing purpose
flat_shape = [784]
cond_shape=10


def train(model, train_dataloader, epochs, device=DEVICE):
    """Train the network on the training set."""
    log_img_dir = 'fl_logs/img/client_generation'
    if not(os.path.isdir(log_img_dir)):
            os.mkdir(log_img_dir)   

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        classif_accuracy = 0
        for batch, (images, labels) in enumerate(train_dataloader):
            images = images.to(device) #[64, 1, 28, 28]
            labels = labels.to(device)

            # 1. Forward pass
            mu, logvar, recon_batch, c_out = model((images, labels))
            flat_data = images.view(-1, flat_shape[0]).to(device)                            
            y_onehot = F.one_hot(labels, cond_shape).to(device)
            inp = torch.cat((flat_data, y_onehot), 1)

            # 2. Calculate loss
            loss, C_loss, BCE, KLD = loss_fn(recon_batch, flat_data, mu, logvar, c_out, y_onehot)
            train_loss += loss.item()
            classif_accuracy += accuracy_fn(labels, torch.argmax(c_out, dim=1))

            # 3. Zero grad
            optimizer.zero_grad()

            # 4. Backprop
            loss.backward()

            # 5. Step
            optimizer.step()

            if batch % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBCE:{:.4f}\tKLD:{:.4f}\tC_loss:{:.4f}'.format(
                    epoch,
                    batch * len(images),
                    len(train_dataloader.dataset),
                    100. * batch / len(train_dataloader),
                    loss.item() / len(images), BCE.item() / len(images), KLD.item() / len(images), C_loss.item() / len(images)))
        print('====> Epoch: {} Average loss: {:.4f}\tClassifier Accuracy: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/len(train_dataloader)))


    model.eval()
    for i in range(2):
        sample = torch.randn(1, 20).to(DEVICE)
        c = np.zeros(shape=(sample.shape[0],))
        label = i+1
        c[:] = label
        c = torch.FloatTensor(c)
        c = c.to(torch.int64)
        c = c.to(DEVICE)
        c = F.one_hot(c, cond_shape)
        model.eval()
        with torch.inference_mode():
            sample = model.decoder((sample, c)).to(DEVICE)
            sample = sample.reshape([1, 1, 28, 28])
            print(f'Ending round: Generating 1 and 2')
            save_image(sample, f'{log_img_dir}/client-{args.num}-label-{label}.png')


def test(model, test_dataloader, device=DEVICE):
    #Sets the module in evaluation mode
    model.eval()
    test_loss = 0
    c_test_loss = 0
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
            c_test_loss += C_loss.item()
            classif_accuracy += accuracy_fn(y, torch.argmax(c_out, dim=1))


    test_loss /= len(test_dataloader.dataset)
    c_test_loss /= len(test_dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, c_test_loss, classif_accuracy/len(test_dataloader)



def loss_fn(recon, x, mu, logvar, c_out, y_onehot, device=DEVICE):
    y_onehot1 = y_onehot.type(torch.FloatTensor).to(device)
    classif_loss = criterion(c_out, y_onehot1)
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



class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.debug = 0

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.classifier.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.classifier.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=local_epochs, device=DEVICE)
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, c_loss, accuracy = test(self.model, self.valloader, device=DEVICE)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num", type=int, required=False, default=0, help="client number"
    )
    parser.add_argument(
        "--malicious", action='store_true'
    )
    args = parser.parse_args()

    model = CVAE(dim_x=(28, 28, 1), dim_y=10, dim_z=20).to(DEVICE)
    trainloader, testloader, _ = load_partition(args.num, batch_size)


    # if args.num == 3:
    #     writer = SummaryWriter(log_dir="./fl_logs/img")
    #     imgs, labels = next(iter(trainloader))
    #     if args.malicious == True:
    #         for i in range(8):
    #             writer.add_image(f'malicious/img-{i}-label={labels[i]}', imgs[i])
    #     else:
    #         for i in range(8):
    #             writer.add_image(f'non-malicious/img-{i}-label={labels[i]}', imgs[i])

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(
            model=model,
            trainloader=trainloader,
            valloader=testloader
        )
    )