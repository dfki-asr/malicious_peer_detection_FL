from typing import Union, Dict, List, Optional, Tuple
from functools import reduce
import psutil

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

import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from utils.models import CVAE, CVAE_regression
from utils.datasets import load_data


dataset = "mnist"
batch_size = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cond_shape=10

class MaliciousUpdateDetection(fl.server.strategy.FedAvg):
    def __repr__(self) -> str:
        return "MaliciousUpdateDetection"

    def __init__(
        self,
        min_fit_clients,
        min_available_clients,
        fraction_fit,
        eval_fn,
        writer,
        on_fit_config_fn,
        server_lr,
        server_momentum,
        model_inst):

        super().__init__(min_fit_clients=min_fit_clients, 
                        min_available_clients=min_available_clients, 
                        fraction_fit=fraction_fit,
                        evaluate_fn=eval_fn,
                        on_fit_config_fn=on_fit_config_fn)
        self.writer = writer
        self.server_lr = server_lr
        self.server_momentum = server_momentum
        self.model_inst = model_inst
        self.global_parameters = []
        self.bytes_recv_init_counter = psutil.net_io_counters().bytes_recv
        self.bytes_sent_init_counter = psutil.net_io_counters().bytes_sent


    def configure_fit(
        self, server_round, parameters, client_manager
    ):
        """Configure the next round of training."""

        if server_round == 1:
            self.global_parameters = parameters_to_ndarrays(parameters)

        clients_conf = super().configure_fit(server_round, parameters, client_manager)

        self.writer.add_scalar("Training/total_num_clients", len(clients_conf), server_round)

        # Return client/config pairs
        return clients_conf

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
        # if not self.accept_failures and failures:
        #     return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Retreiving CVAEs from each client
        n_decoders = len(weights_results)
        if self.model_inst == CVAE:
            cvaes = [CVAE(dim_x=(28, 28, 1), dim_y=10, dim_z=20).to(DEVICE) for i in range(n_decoders)]
        elif self.model_inst == CVAE_regression:
            cvaes = [CVAE_regression(dim_x=(28, 28, 1), dim_y=10, dim_z=20, input_size=784, num_classes=10).to(DEVICE) for i in range(n_decoders)]

        for i in range(n_decoders):
            cvaes[i].set_weights(weights_results[i][0])

        # Evaluating performance of local classifiers on synthetic data, and discarding malicious updates
        benign_indices = self.eval_local_updates(cvaes, server_round)
        weights_results = [weights_results[i] for i in benign_indices]

        # Computing momentum vector
        if self.server_momentum > 0:
            pseudo_gradient: NDArrays = [
                x - y
                for x, y in zip(
                    self.global_parameters, self.aggregate(weights_results)
                )
            ]
            
            if server_round > 1:
                self.momentum_vector = [
                    self.server_momentum * x + y
                    for x, y in zip(self.momentum_vector, pseudo_gradient)
                ]
            else:
                self.momentum_vector = pseudo_gradient

            # Updating global model
            self.global_parameters = [
                x - self.server_lr * y
                for x, y in zip(
                    self.global_parameters, self.momentum_vector
                )
            ]

        else:
            # Updating global model
            if server_round == 1:
                self.global_parameters = self.aggregate(weights_results)
            else:
                self.global_parameters = [global_layer * (1 - self.server_lr) + local_layer * self.server_lr \
                    for global_layer, local_layer in zip(self.global_parameters, self.aggregate(weights_results))]

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        return ndarrays_to_parameters(self.global_parameters), metrics_aggregated


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
        self.writer.add_scalar("System/bytes_rcv", (psutil.net_io_counters().bytes_recv - self.bytes_recv_init_counter) / 1000000, server_round)
        self.writer.add_scalar("System/bytes_sent", (psutil.net_io_counters().bytes_sent - self.bytes_sent_init_counter) / 1000000, server_round)

        return loss, metrics


    def eval_local_updates(self, cvaes, server_round):
        """Evaluating performance of the local classifiers using synthetic data"""

        log_img_dir = f'fl_logs/img/server_generation/round-{server_round}'
        os.makedirs(log_img_dir, exist_ok=True)        

        print("Evaluating local classifiers on data generated by each local decoder")
        # generating 10 images per decoder to evaluate local classifiers 
        n_cvaes = len(cvaes)
        n_synthetic_data = 10
        classifier_accs = np.zeros((n_cvaes, n_cvaes))
        benign_indices = np.arange(n_cvaes)

        for decoder_index, cvae in enumerate(cvaes):
            for label in range(n_synthetic_data):
                # generating data using decoder i
                sample = torch.randn(1, 20).to(DEVICE)
                c = np.zeros(shape=(sample.shape[0],))
                c[:] = label
                c = torch.FloatTensor(c)
                c = c.to(torch.int64)
                c = c.to(DEVICE)
                c = F.one_hot(c, cond_shape)
                cvae.eval()
                with torch.inference_mode():
                    sample = cvae.decoder((sample, c)).to(DEVICE)
                    sample = sample.reshape([1, 1, 28, 28])
                    sample = sample.view(-1, 784)

                # testing each classifier with the generated data
                for classifier_index, model in enumerate(cvaes):
                    model.eval()
                    with torch.inference_mode():
                        if self.model_inst == CVAE:
                            c_out = model.classifier(sample)
                        elif self.model_inst == CVAE_regression:
                            c_out = model.linear(sample)

                    c_out = torch.argmax(c_out).item()
                    if c_out == label:
                        classifier_accs[classifier_index][decoder_index] += 1
            
            
            print(f"Data generated by Decoder {decoder_index}")
           
            for classifier_index in range(n_cvaes):
                print(f"Classifier {classifier_index} accuracy : {classifier_accs[classifier_index][decoder_index]/n_synthetic_data}")

        delete_list = []
        dynamic_threshold = np.mean(classifier_accs)/n_synthetic_data
        print(f'Setting Dynamic Threshold to {dynamic_threshold}')

        for classifier_index in range(n_cvaes):
            avg_acc = np.mean(classifier_accs[classifier_index])/n_synthetic_data
            print(f'Classifier {classifier_index}, average accuracy : {avg_acc}')
            if avg_acc < dynamic_threshold:
                delete_list.append(classifier_index)
                self.writer.add_scalar("Training/threshold", dynamic_threshold, server_round)
        
        if(len(delete_list)>0):
            print(f"Discarding classifiers {delete_list}, benign indices size {len(benign_indices)}")
            benign_indices = np.delete(benign_indices, delete_list)
        
        print(f'Indices : {benign_indices}')

        return benign_indices