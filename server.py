from typing import Dict, Optional, Tuple
import argparse

import flwr as fl

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import load_data
from utils.models import test, TwoConvCNN
from strategies.TensorboardStrategy import TensorboardStrategy


torch.manual_seed(0)
batch_size = 64
num_rounds = 10
dataset = "mnist"
DEVICE = "cpu"

# Global Model
model = TwoConvCNN().to(DEVICE)

# Centralized eval function
def get_eval_fn(model):

	# Load test data
	_, testloader, num_examples = load_data(dataset, batch_size)

	# Evaluate funcion
	def evaluate(server_round, weights, conf):
		model.set_weights(weights)  # Update model with the latest parameters
		loss, accuracy = test(model, testloader, device=DEVICE)

		return loss, {"accuracy": accuracy}

	return evaluate

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--malicious", action='store_true'
	)
	args = parser.parse_args()


	# SummaryWriter
	if args.malicious == True:
		writer = SummaryWriter(log_dir=f"./fl_logs/malicious")
	else:
		writer = SummaryWriter(log_dir=f"./fl_logs/non-malicious")

	writer.add_scalar("hp/batch_size", batch_size)
	writer.add_scalar("hp/num_rounds", num_rounds)


	# Optimization strategy
	strategy = TensorboardStrategy(
		min_fit_clients=4,
		min_available_clients=4,
		eval_fn=get_eval_fn(model),
		writer=writer
	)

	# Federation config
	config = fl.server.ServerConfig(
		num_rounds=num_rounds
	)
	
	fl.server.start_server(
		server_address="[::]:8080",
		config=config,
		strategy=strategy
	)