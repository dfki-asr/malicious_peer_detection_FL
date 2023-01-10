from typing import Union, Dict, List, Optional, Tuple
from functools import reduce
import numpy as np

import flwr as fl
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from torch.utils.tensorboard import SummaryWriter
from utils.models import CVAE, test
from utils.datasets import load_data

dataset = "mnist"
batch_size = 64
DEVICE = "cpu"

class MaliciousUpdateDetection(fl.server.strategy.FedAvg):
    def __repr__(self) -> str:
        return "MaliciousUpdateDetection"

    def __init__(
        self,
        min_fit_clients,
        min_available_clients,
        eval_fn,
        writer):

        super().__init__(min_fit_clients=min_fit_clients, 
                        min_available_clients=min_available_clients, 
                        evaluate_fn=eval_fn,
                        on_fit_config_fn=self.fit_config)
        self.writer = writer


    def fit_config(self, server_round: int):
        config = {
            "current_round": server_round,
        }
        return config


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Getting decoders
        if server_round == 3:
            n_decoders = 2
            cvaes = [CVAE(dim_x=(28, 28, 1), dim_y=10, dim_z=20) for i in range(n_decoders)]
            for i in range(n_decoders):
                cvaes[i].set_weights(weights_results[i][0])     # Load test data
            _, testloader, num_examples = load_data(dataset, batch_size)
            for i in range(n_decoders):
                loss, c_loss, accuracy = test(cvaes[i], testloader, device=DEVICE)
                print(i, " - loss: ", loss, ", c_loss: ", c_loss)


        parameters_aggregated = ndarrays_to_parameters(self.aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated


    def aggregate(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime


    def evaluate(self, server_round, parameters):
        """Evaluate model parameters using an evaluation function."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Write scalars
        self.writer.add_scalar("Training/test_loss", loss, server_round)
        self.writer.add_scalar("Training/test_accuracy", metrics["accuracy"], server_round)
        self.writer.add_scalar("Training/test_c_loss", metrics["c_loss"], server_round)

        return loss, metrics
