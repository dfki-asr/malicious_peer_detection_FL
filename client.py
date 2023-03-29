from collections import OrderedDict
from typing import List, Tuple
import argparse
import copy

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import load_partition
from utils.models import CVAE, CVAE_regression, Classifier, LogisticRegression
from utils.partition_data import Partition
from utils.attacks import sign_flipping_attack, additive_noise_attack, same_value_attack
from utils.function import train, train_standard_classifier, train_regression, train_cvae_regression, test, test_standard_classifier, test_regression, test_cvae_regression
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
        self.firstSample = True

    def get_parameters(self, config=None):
        if args.model == 'cvae' or 'cvae_regression':
            model_copy = copy.deepcopy(self.model)
            model_copy.encoder = None
            return [val.cpu().numpy() for _, val in model_copy.state_dict().items()]
        else:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        if args.model == 'cvae':
            params_dict = zip(self.model.classifier.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.classifier.load_state_dict(state_dict, strict=True)
        elif args.model == 'cvae_regression':
            params_dict = zip(self.model.linear.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.linear.load_state_dict(state_dict, strict=True)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        if args.attack == 'none':
            self.set_parameters(parameters)
            if args.model == 'cvae':
                train(self.model, self.trainloader, config=config, device=DEVICE, args=args)
            if args.model == 'cvae_regression':
                train_cvae_regression(self.model, self.trainloader, config=config, device=DEVICE, args=args)
            elif args.model == 'classifier':
                train_standard_classifier(self.model, self.trainloader, config=config, device=DEVICE, args=args)
            elif args.model == 'regression':
                train_regression(self.model, self.trainloader, config=config, device=DEVICE, args=args)

        elif args.attack == "label_flipping":
            self.set_parameters(parameters)
            if args.model == 'cvae':
                train(self.model, self.trainloader, config=config, label_flipping=True, device=DEVICE, args=args)
            if args.model == 'cvae_regression':
                train_cvae_regression(self.model, self.trainloader, config=config, label_flipping=True,device=DEVICE, args=args)
            elif args.model == 'classifier':
                train_standard_classifier(self.model, self.trainloader, config=config, label_flipping=True, device=DEVICE, args=args)
            elif args.model == 'regression':
                train_regression(self.model, self.trainloader, config=config, label_flipping=True, device=DEVICE, args=args)

        elif args.attack == "sign_flipping":
            self.set_parameters(parameters)
            if args.model == 'cvae':
                train(self.model, self.trainloader, config=config, device=DEVICE, args=args)
                self.model.classifier.load_state_dict(sign_flipping_attack(self.model.classifier.state_dict()))
            if args.model == 'cvae_regression':
                train_cvae_regression(self.model, self.trainloader, config=config, device=DEVICE, args=args)
                self.model.linear.load_state_dict(sign_flipping_attack(self.model.linear.state_dict()))
            elif args.model == 'classifier':
                train_standard_classifier(self.model, self.trainloader, config=config, device=DEVICE, args=args)
                self.model.load_state_dict(sign_flipping_attack(self.model.state_dict()))
            elif args.model == 'regression':
                train_regression(self.model, self.trainloader, config=config, device=DEVICE, args=args)
                self.model.load_state_dict(sign_flipping_attack(self.model.state_dict()))

        elif args.attack == "additive_noise":
            self.set_parameters(parameters)
            if args.model == 'cvae':
                train(self.model, self.trainloader, config=config, device=DEVICE, args=args)
                self.model.classifier.load_state_dict(additive_noise_attack(self.model.classifier.state_dict(), device=DEVICE))
            if args.model == 'cvae_regression':
                train_cvae_regression(self.model, self.trainloader, config=config, device=DEVICE, args=args)
                self.model.linear.load_state_dict(additive_noise_attack(self.model.linear.state_dict(), device=DEVICE))
            elif args.model == 'classifier':
                train_standard_classifier(self.model, self.trainloader, config=config, device=DEVICE, args=args)
                self.model.load_state_dict(additive_noise_attack(self.model.state_dict(), device=DEVICE))
            elif args.model == 'regression':
                train_regression(self.model, self.trainloader, config=config, device=DEVICE, args=args)
                self.model.load_state_dict(additive_noise_attack(self.model.state_dict(), device=DEVICE))

        elif args.attack == "same_value":
            if args.model == 'cvae':
                params_dict = zip(self.model.classifier.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
                self.model.classifier.load_state_dict(same_value_attack(state_dict))
            elif args.model == 'cvae_regression':
                params_dict = zip(self.model.linear.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
                self.model.linear.load_state_dict(same_value_attack(state_dict))
            else:
                params_dict = zip(self.model.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
                self.model.load_state_dict(same_value_attack(state_dict))

        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        if args.model == "cvae":
            loss, c_loss, accuracy = test(self.model, self.valloader, device=DEVICE)
        if args.model == "cvae_regression":
            loss, c_loss, accuracy = test_cvae_regression(self.model, self.valloader, device=DEVICE)
        elif args.model == 'classifier':
            loss, accuracy = test_standard_classifier(self.model, self.valloader, device=DEVICE)
        elif args.model == 'regression':
            loss, accuracy = test_regression(self.model, self.valloader, device=DEVICE)

        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="cvae", help="Model to train: cvae, classifier"
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
    parser.add_argument(
        "--server_address", type=str, required=False, default="127.0.0.1:8080", help="gRPC server address"
    )
    args = parser.parse_args()

    if args.model == 'cvae':
        model = CVAE(dim_x=(28, 28, 1), dim_y=10, dim_z=20).to(DEVICE)
    elif args.model == 'cvae_regression':
        model = CVAE_regression(dim_x=(28, 28, 1), dim_y=10, dim_z=20, input_size=784, num_classes=10).to(DEVICE)
    elif args.model == 'classifier':
        model = Classifier(dim_y=10).to(DEVICE)
    elif args.model == 'regression':
        model = LogisticRegression(input_size=784, num_classes=10).to(DEVICE)

    trainloader, testloader, _ = load_partition(args.num, batch_size)

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FlowerClient(
            model=model,
            trainloader=trainloader,
            valloader=testloader
        )
    )