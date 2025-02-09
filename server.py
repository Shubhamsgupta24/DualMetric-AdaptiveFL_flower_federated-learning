import flwr as fl
import tensorflow as tf
import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from flwr.common import Scalar, NDArrays, Parameters, FitRes, EvaluateRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

# Global variable
NUM_CLIENTS = 3

# Create CentralModel folder if it doesn't exist
os.makedirs('CentralModel', exist_ok=True)

class CustomStrategy(fl.server.strategy.FedAvg):

    def __init__(self, *args, **kwargs):
        '''
        Initializing the parent class i.e fl.server.strategy.FedAvg where args is the positional arguments in the form of a tuple and kwargs is the keyword arguments in the form of a dictionary.
        Here we have considered kwargs: {"min_available_clients": NUM_CLIENTS,"min_fit_clients": NUM_CLIENTS,"min_evaluate_clients": NUM_CLIENTS}.
        Also, we have initialized the central_model to None that will store weights later on.
        '''
        super().__init__(*args, **kwargs)
        self.central_model = None
    
    def aggregate_fit(self,server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: list[Union[tuple[ClientProxy, FitRes], BaseException]], ) -> tuple[Optional[Parameters], dict[str, Scalar]]: 
        '''
        This method is used to aggregate the results of the training process from all the clients and return the tuple of (aggregated weights of all clients,aggregated metrics of all the clients).
        Takes input as server_round, results, failures:
        1)server_round - Current round number
        2)results - List of tuples containing ClientProxy(unique client object id) and FitRes(Fit Response from client fit method includes parameters(weights),number of training examples and metric dictionary)
        3)failures - Union of List of exceptions and result tuple raised during the training process
        It returns two outputs :  
        1)Optional[Parameters] - Aggregated weights of all clients.These weights are returned to be used as the global model for the next round of federated learning.Parameters in other terms in FL are the model weights.
        2)dict[str, Scalar] - Aggregated metrics of all the clients. In this case, we are returning the aggregated loss and accuracy of all the clients in the form of a dictionary.This is optional and can be used for logging purposes.
        '''

        print(f"\n########### STEP 1 For Server Round {server_round} : Aggregating Weights and Metrics of the Fit Response given by all clients ################\n")
        if not results:
            return None
        
        # If there are any failures and we don't accept failures, return None
        if not self.accept_failures and failures:
            return None, {}
        
        # ========= Aggregating Weights =========

        weights = [] # Model parameters (weights) sent by each client.
        num_examples = [] # Number of training examples used by each client.

        for _, fit_res in results:
            weights.append(parameters_to_ndarrays(fit_res.parameters))
            num_examples.append(fit_res.num_examples)

        # Aggregate the collected weights which are converted to ndarrays(for more efficiency) using the aggregate function
        aggregated_ndarrays = aggregate(list(zip(weights, num_examples)))

        # Convert the aggregated result back to Parameters(i.e aggregated weights)
        aggregated_weights = ndarrays_to_parameters(aggregated_ndarrays)

        # ========= Aggregating Metrics =========

        metrics_aggregated = {"accuracy": 0.0, "loss": 0.0} # Aggregated metrics of all clients
        total_examples = 0 # Total number of training examples used by all clients

        print("\nIndividual Client Metrics:\n")
        for client_idx, (_, res) in enumerate(results):
            num_examples = res.num_examples
            metrics = res.metrics
            
            # Print individual client metrics
            print(f"Client {client_idx}: Number of training examples: {num_examples}, Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

            # Weighted sum of metrics
            metrics_aggregated["accuracy"] += num_examples * metrics["accuracy"]
            metrics_aggregated["loss"] += num_examples * metrics["loss"]
            total_examples += num_examples

        # Compute the weighted average for each metric
        metrics_aggregated["accuracy"] /= total_examples
        metrics_aggregated["loss"] /= total_examples

        print(f"\nRound {server_round} training set metrics - Aggregated Loss: {metrics_aggregated['loss']:.4f}, Aggregated Accuracy: {metrics_aggregated['accuracy']:.4f}\n")
        print(f"\n================= Sending updated model to client for testing and client side updation ===================\n")

        # self.central_model = aggregated_weights
        
        return aggregated_weights, metrics_aggregated

    def aggregate_evaluate(self,server_round: int,results: List[Tuple[ClientProxy, EvaluateRes]],failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        '''
        This method is used to aggregate the results of the evaluation process from all the clients and return the tuple of aggregated loss and dictionary of metrics containing aggregated accuracy.
        Takes input as server_round, results, failures:
        1)server_round - Current round number
        2)results - List of tuples containing ClientProxy(unique client object id) and EvaluateRes(Evaluate Response from client evaluate method includes loss,number of testing examples and metric dictionary.
        3)failures - Union of List of exceptions and result tuple raised during the testing process
        It returns two outputs :  
        1)float - Aggregated loss of all the clients.
        2)dict[str, Scalar] - Aggregated metrics of all the clients. In this case, we are returning the aggregated accuracy of all the clients in the form of a dictionary.
        '''

        print(f"\n################### STEP 2 For Server Round {server_round} : Aggregating Weights and Metrics of the Evaluate Response given by all clients##################\n")
        if not results:
            return None
        
        # If there are any failures and we don't accept failures, return None
        if not self.accept_failures and failures:
            return None, {}
        
        # ========= Aggregating Loss and Metrics =========
        total_loss = 0
        total_examples = 0
        total_accuracy = 0

        for client_idx, (_, evaluate_res) in enumerate(results):
            num_examples = evaluate_res.num_examples
            loss = evaluate_res.loss
            accuracy = evaluate_res.metrics["accuracy"]

            # Print individual client evaluation results
            print(f"Client {client_idx}: Number of testing examples: {num_examples}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

            # Accumulate loss and accuracy for aggregation
            total_loss += loss * num_examples  # Weighted sum of loss
            total_accuracy += accuracy * num_examples  # Weighted sum of accuracy
            total_examples += num_examples
        
        loss_aggregated = total_loss / total_examples
        accuracy_aggregated = total_accuracy / total_examples
        print(f"\nRound {server_round} Testing Set Metrics -  Aggregated Loss: {loss_aggregated:.4f}, Aggregated Accuracy: {accuracy_aggregated:.4f}\n")
        print(f"\n================= Aggregation Completed for Round {server_round} ===================\n")
        
        return loss_aggregated, {"accuracy": accuracy_aggregated}

    # def get_central_model(self) -> Optional[NDArrays]:
    #     return self.central_model

# Using the custom strategy which is inherited from fl.server.strategy.FedAvg.In this all clients are required to participate in the training (fit) step and evaluation in each round.Minimum number of clients must be available for a round of training to begin.
strategy = CustomStrategy(
    min_available_clients=NUM_CLIENTS,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS
)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)

print("\nFederated learning completed.")

exit()



# The below code is not in use now may be used in future references.

central_model_weights = strategy.get_central_model()

def create_central_model(input_shape: int, num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(1300, 64, input_length=input_shape),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

num_classes = 11  # Adjust this based on your actual number of classes

if central_model_weights is not None:
    central_model = create_central_model(25, num_classes)
    central_model.set_weights(central_model_weights)
    central_model.save(os.path.join('CentralModel', 'central_model.h5'))
    print("Central model saved successfully.")
else:
    print("Error: Central model weights not available.")
