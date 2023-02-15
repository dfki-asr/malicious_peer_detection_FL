from collections import OrderedDict
from typing import List, Tuple
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import load_partition
from utils.models import CVAE
from utils.partition_data import Partition
from utils.attacks import sign_flipping_attack, additive_noise_attack, same_value_attack
from utils.function import train, train_label_flipping, test
import logging
import flwr as fl

torch.manual_seed(0)
# DEVICE='cpu'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(f"Client device: {DEVICE}")
logging.info(f"Client device: {DEVICE}")
batch_size = 64
logging.basicConfig(filename="log_traces/logfilename.log", level=logging.INFO)

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
        if args.attack == 'none':
            self.set_parameters(parameters)
            train(self.model, self.trainloader, config=config, device=DEVICE, args=args)

        elif args.attack == "label_flipping":
            self.set_parameters(parameters)
            train_label_flipping(self.model, self.trainloader, config=config, device=DEVICE, args=args)

        elif args.attack == "sign_flipping":
            self.set_parameters(parameters)
            train(self.model, self.trainloader, config=config, device=DEVICE, args=args)
            self.model.classifier.load_state_dict(sign_flipping_attack(self.model.classifier.state_dict()))

        elif args.attack == "additive_noise":
            self.set_parameters(parameters)
            train(self.model, self.trainloader, config=config, device=DEVICE, args=args)
            self.model.classifier.load_state_dict(additive_noise_attack(self.model.classifier.state_dict(), device=DEVICE))

        elif args.attack == "same_value":
            params_dict = zip(self.model.classifier.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.classifier.load_state_dict(same_value_attack(state_dict))

        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, c_loss, accuracy = test(self.model, self.valloader, device=DEVICE)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
		"--strategy", action='store_true', default="detection_strategy", help="Set of strategies: fedavg, detection_strategy"
	)
    parser.add_argument(
        "--attack", type=str, required=False, default="none", help="Set of attacks"
    )
    parser.add_argument(
        "--num", type=int, required=False, default=0, help="client number"
    )
    parser.add_argument(
        "--seed", type=int, required=False, default=0, help="random seed for flipping labels"
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