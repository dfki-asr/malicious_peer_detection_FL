from typing import Dict, Optional, Tuple
import argparse
import logging
import flwr as fl
import sys
from globals_mod import settings
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
from utils.datasets import load_data
from utils.models import CVAE
from utils.function import test

from strategies.MaliciousUpdateDetectionStrategy import MaliciousUpdateDetection
from strategies.TensorboardStrategy import TensorboardStrategy





torch.manual_seed(0)
# DEVICE='cpu'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Server device: {DEVICE}")
batch_size = 64
num_rounds =20
dataset = "mnist"


# Centralized eval function
def get_eval_fn(model):

	# Load test data
	_, testloader, num_examples = load_data(dataset, batch_size)

	# Evaluate funcion
	def evaluate(server_round, weights, conf):
		model.set_weights(weights)  # Update model with the latest parameters
		loss, c_loss, accuracy = test(model, testloader, device=DEVICE)

		return loss, {"accuracy": accuracy, "c_loss": c_loss}

	return evaluate


def fig_config(server_round: int):
	"""Return training configuration dict for each round."""
	config = {
		"batch_size": 64,
		"current_round": server_round,
		"local_epochs": 1,
	}
	return config


if __name__ == "__main__":
	logging.basicConfig(filename="log_traces/logfilename.log", level=logging.INFO)
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--strategy", type=str, default="detection_strategy", help="Set of strategies: fedavg, detection_strategy"
	)
	parser.add_argument(
		"--attack", type=str, default="label_flipping", help="Set of attacks"
	)
	args = parser.parse_args()
	print(f"Running {args.strategy} for {args.attack} attack. Total number of rounds: {num_rounds}")
	logging.info(f"Running {args.strategy} for {args.attack} attack. Total number of rounds: {num_rounds}")
	# Global Model
	model = CVAE(dim_x=(28, 28, 1), dim_y=10, dim_z=20).to(DEVICE)

	# SummaryWriter
	writer = SummaryWriter(log_dir=f"./fl_logs/")

	writer.add_scalar("hp/batch_size", batch_size)
	writer.add_scalar("hp/num_rounds", num_rounds)
	writer.add_text("hp/strategy", args.strategy)
	writer.add_text("hp/attack", args.attack)


	# Optimization strategy
	if args.strategy == "detection_strategy":
		strategy = MaliciousUpdateDetection(
			min_fit_clients=2,
			min_available_clients=2,
			eval_fn=get_eval_fn(model),
			writer=writer,
			on_fit_config_fn=fig_config,
		)
	elif args.strategy == "fedavg":
		strategy = TensorboardStrategy(
			min_fit_clients=2,
			min_available_clients=2,
			eval_fn=get_eval_fn(model),
			writer=writer,
			on_fit_config_fn=fig_config,
		)


	# Federation config
	config = fl.server.ServerConfig(
		num_rounds=num_rounds
	)
	
	fl.server.start_server(
		server_address="127.0.0.1:8080",
		config=config,
		strategy=strategy
	)