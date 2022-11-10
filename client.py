from collections import OrderedDict
from typing import List, Tuple
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from utils.datasets import load_partition
from utils.models import train, test, TwoConvCNN
from utils.partition_data import Partition

import flwr as fl
import numpy as np

torch.manual_seed(0)
batch_size = 64
DEVICE = "cpu"


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
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

	model = TwoConvCNN()
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
		server_address="[::]:8080",
		client=FlowerClient(
			net=model,
			trainloader=trainloader,
			valloader=testloader
		)
	)