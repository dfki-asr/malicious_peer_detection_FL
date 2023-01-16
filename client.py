from collections import OrderedDict
from typing import List, Tuple
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from utils.datasets import load_partition
from utils.models import train, test, CVAE
from utils.partition_data import Partition

import flwr as fl
import numpy as np

torch.manual_seed(0)
# DEVICE='cpu'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(f"Client device: {DEVICE}")
batch_size = 64
local_epochs = 1


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

    if args.num == 3:
        writer = SummaryWriter(log_dir="./fl_logs/img")
        imgs, labels = next(iter(trainloader))
        if args.malicious == True:
            for i in range(8):
                writer.add_image(f'malicious/img-{i}-label={labels[i]}', imgs[i])
        else:
            for i in range(8):
                writer.add_image(f'non-malicious/img-{i}-label={labels[i]}', imgs[i])

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(
            model=model,
            trainloader=trainloader,
            valloader=testloader
        )
    )