from typing import Dict, Optional, Tuple
import argparse
import logging
import flwr as fl
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy
import os
import time
from utils.datasets import load_data
from utils.models import CVAE, CVAE_regression, Classifier, LogisticRegression
from utils.function import test, test_standard_classifier, test_regression, test_cvae_regression

from strategies.MaliciousUpdateDetectionStrategy import MaliciousUpdateDetection
from strategies.TensorboardStrategy import TensorboardStrategy
from strategies.FedMedian import FedMedian
from strategies.Krum import Krum
from strategies.Spectral import Spectral



torch.manual_seed(0)
# DEVICE='cpu'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Server device: {DEVICE}")
batch_size = 64
dataset = "mnist"


# Centralized eval function
def get_eval_fn(model):

	# Load test data
	_, testloader, num_examples = load_data(dataset, batch_size)

	# Evaluate funcion
	def evaluate(server_round, weights, conf):
		model.set_weights(weights)  # Update model with the latest parameters

		if args.model == "cvae":
			loss, c_loss, accuracy = test(model, testloader, device=DEVICE)
			return loss, {"accuracy": accuracy, "c_loss": c_loss}
		elif args.model == "cvae_regression":
			loss, c_loss, accuracy = test_cvae_regression(model, testloader, device=DEVICE)
			return loss, {"accuracy": accuracy, "c_loss": c_loss}
		elif args.model == 'classifier':
			loss, accuracy = test_standard_classifier(model, testloader, device=DEVICE)
			return loss, {"accuracy": accuracy}
		elif args.model == 'regression':
			loss, accuracy = test_regression(model, testloader, device=DEVICE)
			return loss, {"accuracy": accuracy}


	return evaluate


def fig_config(server_round: int):
	"""Return training configuration dict for each round."""
	config = {
		"batch_size": 64,
		"current_round": server_round,
		"local_epochs": args.local_epochs,
		"log_img": False,
	}
	return config


if __name__ == "__main__":
	logging.basicConfig(filename="log_traces/logfilename.log", level=logging.INFO)
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--strategy", type=str, default="detection_strategy", help="Set of strategies: fedavg, detection_strategy"
	)
	parser.add_argument(
		"--model", type=str, default="cvae", help="Model to train: cvae, classifier"
	)
	parser.add_argument(
		"--attack", type=str, default="label_flipping", help="Set of attacks"
	)
	parser.add_argument(
		"--server_address", type=str, required=False, default="127.0.0.1:8080", help="gRPC server address"
	)
	parser.add_argument(
		"--num_rounds", type=int, required=False, default=20, help="number of FL rounds"
	)
	parser.add_argument(
		"--fraction_fit", type=float, required=False, default=1, help="Fraction of clients selected on each rounds"
	)
	parser.add_argument(
		"--min_fit_clients", type=int, required=False, default=2, help="Minimum number of clients selected on each rounds"
	)
	parser.add_argument(
		"--min_available_clients", type=int, required=False, default=2, help="Minimum number of clients selected on each rounds"
	)
	parser.add_argument(
		"--server_lr", type=float, required=False, default=1, help="Server learning rate: [0, 1]"
	)
	parser.add_argument(
		"--server_momentum", type=float, required=False, default=0, help="Server momentum: [0, 1]"
	)
	parser.add_argument(
		"--local_epochs", type=int, required=False, default=1, help="Local epochs"
	)

	args = parser.parse_args()
	print(f"Running {args.strategy} for {args.attack} attack. Total number of rounds: {args.num_rounds}")
	logging.info(f"Running {args.strategy} for {args.attack} attack. Total number of rounds: {args.num_rounds}")
	# Global Model
	if args.model == 'cvae':
		model = CVAE(dim_x=(28, 28, 1), dim_y=10, dim_z=20).to(DEVICE)
	elif args.model == 'cvae_regression':
		model = CVAE_regression(dim_x=(28, 28, 1), dim_y=10, dim_z=20, input_size=784, num_classes=10).to(DEVICE)
	elif args.model == 'classifier':
		model = Classifier(dim_y=10).to(DEVICE)
	elif args.model == 'regression':
		model = LogisticRegression(input_size=784, num_classes=10).to(DEVICE)

	# SummaryWriter
	writer = SummaryWriter(log_dir=f"./fl_logs/{args.attack}-{args.strategy}", filename_suffix=f'{args.attack}-{args.strategy}')

	writer.add_scalar("hp/batch_size", batch_size)
	writer.add_scalar("hp/num_rounds", args.num_rounds)
	writer.add_scalar("hp/min_fit_clients", args.min_fit_clients)
	writer.add_scalar("hp/fraction_fit", args.fraction_fit)
	writer.add_scalar("hp/server_lr", args.server_lr)
	writer.add_scalar("hp/server_momentum", args.server_momentum)
	writer.add_scalar("hp/local_epochs", args.local_epochs)
	writer.add_text("hp/strategy", args.strategy)
	writer.add_text("hp/attack", args.attack)


	# Optimization strategy
	if args.strategy == "detection_strategy":
		if args.model == "cvae":
			strategy = MaliciousUpdateDetection(
				min_fit_clients=args.min_fit_clients,
				min_available_clients=args.min_available_clients,
				fraction_fit=args.fraction_fit,
				eval_fn=get_eval_fn(model),
				writer=writer,
				on_fit_config_fn=fig_config,
				server_lr=args.server_lr,
				server_momentum=args.server_momentum,
				model_inst=CVAE
			)
		elif args.model == "cvae_regression":
			strategy = MaliciousUpdateDetection(
				min_fit_clients=args.min_fit_clients,
				min_available_clients=args.min_available_clients,
				fraction_fit=args.fraction_fit,
				eval_fn=get_eval_fn(model),
				writer=writer,
				on_fit_config_fn=fig_config,
				server_lr=args.server_lr,
				server_momentum=args.server_momentum,
				model_inst=CVAE_regression
			)
	elif args.strategy == "fedavg":
		strategy = TensorboardStrategy(
			min_fit_clients=args.min_fit_clients,
			min_available_clients=args.min_available_clients,
			fraction_fit=args.fraction_fit,
			eval_fn=get_eval_fn(model),
			writer=writer,
			on_fit_config_fn=fig_config,
		)
	elif args.strategy == "fedmedian":
		strategy = FedMedian(
			min_fit_clients=args.min_fit_clients,
			min_available_clients=args.min_available_clients,
			fraction_fit=args.fraction_fit,
			eval_fn=get_eval_fn(model),
			writer=writer,
			on_fit_config_fn=fig_config,
		)
	elif args.strategy == "krum":
		strategy = Krum(
			min_fit_clients=args.min_fit_clients,
			min_available_clients=args.min_available_clients,
			fraction_fit=args.fraction_fit,
			eval_fn=get_eval_fn(model),
			writer=writer,
			on_fit_config_fn=fig_config,
		)
	elif args.strategy == "spectral":
		flat_model_shape = np.array([])

		state_dict = copy.deepcopy(model).cpu().state_dict()
		for key in state_dict:
			data_idx_key = np.array(state_dict[key]).flatten()
			flat_model_shape = copy.deepcopy(np.hstack((flat_model_shape, data_idx_key)))
		flat_model_shape = flat_model_shape.shape[0]

		strategy = Spectral(
			min_fit_clients=args.min_fit_clients,
			min_available_clients=args.min_available_clients,
			fraction_fit=args.fraction_fit,
			eval_fn=get_eval_fn(model),
			writer=writer,
			on_fit_config_fn=fig_config,
			flat_model_shape=flat_model_shape,
		)


	# Federation config
	config = fl.server.ServerConfig(
		num_rounds=args.num_rounds
	)
	
	fl.server.start_server(
		server_address=args.server_address,
		config=config,
		strategy=strategy
	)