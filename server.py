import flwr as fl
from typing import List, Tuple, Optional, Dict, Union
from flwr.common import Scalar, NDArrays, Parameters, FitRes, EvaluateRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from visualisations import visualize_global_accuracy_clients,visualize_local_accuracy_clients,plot_accuracy_fairness_tradeoff
import numpy as np
import json
import os

'''
CAUTION:
1) NUM_CLIENTS will be according to the number of datasets present in the Train folder which are created using different cases in data_prep.py
'''

# Global variables
NUM_CLIENTS = 11
NUM_ROUNDS = 50
VISUAL_DIR = "./Visualizations"

class CustomStrategy(fl.server.strategy.FedAvg):

    def __init__(self, *args, **kwargs):
        '''
        Initializing the parent class i.e fl.server.strategy.FedAvg where args is the positional arguments in the form of a tuple and kwargs is the keyword arguments in the form of a dictionary.
        Here we have considered kwargs: {"min_available_clients": NUM_CLIENTS,"min_fit_clients": NUM_CLIENTS,"min_evaluate_clients": NUM_CLIENTS}.
        Also, we have initialized the central_model to None that will store weights later on implementating pretraining.
        '''
        super().__init__(*args, **kwargs)
        self.central_model = None  # Central model (weights) to be used for pretraining
        # Add attributes for hyperparameter tuning
        self.client_local_accuracy_history = {}  # {client_id: [local_accuracy_history] containing local accuracy of all rounds}
        self.client_global_accuracy_history = {}  # {client_id: [global_accuracy_history] containing global accuracy of all rounds}
        self.client_local_accuracy_variance = {} # {round_number : [local_accuracy_variance] containing local accuracy variance of that particular round}
        self.client_mu = {}  # {client_id: mu_value}
        self.client_lr_factor = {}  # {client_id: lr_factor_value}
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """
        This method configure_fit is used to configure the fit process for each round of federated learning.
        It takes input as server_round, parameters, client_manager:
        1)server_round - Current round number
        2)parameters - Model weights to be sent to the clients for training
        3)client_manager - Client manager object which is used to sample clients for training
        It returns a list of (client, FitIns) tuples with updated config.
        It overrides configure_fit to send dynamic hyperparameters to clients via Fit Instruction.
        """
        # Sample clients
        sample_clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients
        )

        # Create client-specific FitIns
        fit_instructions = []
        for client in sample_clients:
            client_id = client.cid
            config = {
                "proximal_mu": self.client_mu.get(client_id, 0.01),  # Default to 0.01 if not set
                "lr_factor": self.client_lr_factor.get(client_id, 1.0),  # Default to 1.0 if not set
                "server_round": server_round
            }
            fit_instructions.append((client, fl.common.FitIns(parameters, config)))

        return fit_instructions

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

        print(f"\n########### STEP 1 For Server Round {server_round} : Aggregating Weights and Metrics of the Fit Response given by all clients ################\n", flush=True)
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

        print("\nIndividual Client Metrics:\n", flush=True)
        for client_idx, (_, res) in enumerate(results):
            num_examples = res.num_examples
            metrics = res.metrics
            
            # Print individual client metrics
            print(f"Client {client_idx}: Number of training examples: {num_examples}, Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}", flush=True)

            # Weighted sum of metrics
            metrics_aggregated["accuracy"] += num_examples * metrics["accuracy"]
            metrics_aggregated["loss"] += num_examples * metrics["loss"]
            total_examples += num_examples

        # Compute the weighted average for each metric
        metrics_aggregated["accuracy"] /= total_examples
        metrics_aggregated["loss"] /= total_examples

        print(f"\nRound {server_round} training set metrics - Aggregated Loss: {metrics_aggregated['loss']:.4f}, Aggregated Accuracy: {metrics_aggregated['accuracy']:.4f}\n", flush=True)
        print(f"\n================= Sending updated model to client for testing and client side updation ===================\n", flush=True)
        
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

        print(f"\n################### STEP 2 For Server Round {server_round} : Aggregating Metrics of the Evaluate Response given by all clients ##################\n", flush=True)
        if not results:
            return None
        
        # If there are any failures and we don't accept failures, return None
        if not self.accept_failures and failures:
            return None, {}
        
        # ========= PART 1 - Aggregating Local Client Training Set Loss and Metrics =========
        print("\n-> Individual Client Training Set Metrics on Clients Local Training Set:\n")
        total_local_loss = 0
        total_local_accuracy=0
        total_train_examples = 0

        local_accuracies = []

        for client_idx, (_, evaluate_res) in enumerate(results):
            num_train_examples = evaluate_res.num_examples
            local_loss = evaluate_res.loss
            local_accuracy = evaluate_res.metrics["local_accuracy"]
            local_accuracies.append(local_accuracy)

            # Print individual client evaluation results
            print(f"Client {client_idx}: Number of trianing examples: {num_train_examples}, Loss: {local_loss:.4f}, Accuracy: {local_accuracy:.4f}", flush=True)

            # Accumulate loss and accuracy for aggregation
            total_local_loss += local_loss * num_train_examples  # Weighted sum of loss
            total_local_accuracy += local_accuracy * num_train_examples  # Weighted sum of accuracy
            total_train_examples += num_train_examples # Total number of training examples

        local_loss_aggregated = total_local_loss / total_train_examples
        local_accuracy_aggregated = total_local_accuracy / total_train_examples

        variance = np.var(local_accuracies)
        self.client_local_accuracy_variance[server_round] = variance

        print(f"\nRound {server_round} Local Training Set Metrics - Aggregated Loss: {local_loss_aggregated:.4f}, Aggregated Accuracy: {local_accuracy_aggregated:.4f}, Variance: {variance}\n", flush=True)

        # ========= PART 2 - Aggregating Global Test Set Loss and Metrics =========
        print("\n-> Individual Client Testing Set Metrics on Global Test Set:\n", flush=True)
        total_global_loss = 0
        total_global_accuracy = 0
        total_test_examples = 0

        for client_idx, (_, evaluate_res) in enumerate(results):
            num_test_examples = evaluate_res.metrics["X_test_len"]
            global_loss = evaluate_res.metrics["global_loss"]
            global_accuracy = evaluate_res.metrics["global_accuracy"]

            # Print individual client evaluation results
            print(f"Client {client_idx}: Number of testing examples: {num_test_examples}, Loss: {global_loss:.4f}, Accuracy: {global_accuracy:.4f}", flush=True)

            # Accumulate loss and accuracy for aggregation
            total_global_loss += global_loss * num_test_examples # Weighted sum of loss
            total_global_accuracy += global_accuracy * num_test_examples # Weighted sum of accuracy
            total_test_examples += num_test_examples # Total number of testing examples
        
        global_loss_aggregated = total_global_loss / total_test_examples
        global_accuracy_aggregated = total_global_accuracy / total_test_examples
        print(f"\nRound {server_round} Global Testing Set Metrics - Aggregated Loss: {global_loss_aggregated:.4f}, Aggregated Accuracy: {global_accuracy_aggregated:.4f}\n", flush=True)

        # ========= PART 3 - Dynamic Hyperparameter Tuning for Each Client =================
        print(f"\n-> Server Guided Dynamic Hyperparameter Tuning for Better Client Adaption: \n", flush=True)

        for client_idx, (client_proxy, evaluate_res) in enumerate(results):
                client_id = client_proxy.cid  # Get unique client ID
                local_accuracy = evaluate_res.metrics["local_accuracy"]
                global_accuracy = evaluate_res.metrics["global_accuracy"]

                # Initialize client history if not present
                if client_id not in self.client_local_accuracy_history:
                    self.client_local_accuracy_history[client_id] = []
                    self.client_global_accuracy_history[client_id] = []
                    self.client_mu[client_id] = 0.01  # Initial mu
                    self.client_lr_factor[client_id] = 1.0  # Initial lr_factor

                # Append to client-specific history
                self.client_local_accuracy_history[client_id].append(local_accuracy)
                self.client_global_accuracy_history[client_id].append(global_accuracy)

        # Call the tuning function after collecting all metrics
        self.tune_client_hyperparameters(server_round)

        print(f"\n============= Aggregation Completed for Round {server_round} ==============\n", flush=True)
        
        return local_loss_aggregated, {"local_accuracy": local_accuracy_aggregated,"global_loss": global_loss_aggregated,"global_accuracy": global_accuracy_aggregated}
    
    def tune_client_hyperparameters(self, server_round: int):
        """
        Tune both mu and lr_factor for each client based on their global accuracy trend.
        Aim to improve accuracy beyond baseline with conservative adjustments.
        """
        # Skip tuning in the first round (no history yet)
        if server_round <= 1:
            return
        
        # Compute average local accuracy across all clients (for disparity-based adjustment)
        current_local_accuracies = {
            cid: self.client_local_accuracy_history[cid][-1]
            for cid in self.client_local_accuracy_history.keys()
        }
        avg_local_acc = sum(current_local_accuracies.values()) / len(current_local_accuracies)

        for client_id in self.client_local_accuracy_history.keys():
            
            # Calculate global accuracy improvement
            global_history = self.client_global_accuracy_history[client_id]
            curr_global_acc = global_history[-1]
            prev_global_acc = global_history[-2] if len(global_history) > 1 else curr_global_acc
            global_acc_diff = curr_global_acc - prev_global_acc

            # Calculate local accuracy improvement
            local_history = self.client_local_accuracy_history[client_id]
            curr_local_acc = local_history[-1]
            prev_local_acc = local_history[-2] if len(local_history) > 1 else curr_local_acc
            local_acc_diff = curr_local_acc - prev_local_acc
            
            # --- Tuning Conditions for mu ---
            mu_update_factor = 1.0

            # === Intra-client behavior: trends within the client ===
            # 1) Local ↓, Global ↓ or ↔
            if local_acc_diff < -0.03 and global_acc_diff <= 0:
                mu_update_factor *= 1.005  # Slight more regularization
                print(f"Round {server_round}, Client {client_id}: ↓ Local & Global : ↑ mu to {self.client_mu[client_id]:.4f}", flush=True)
            # 2) Local ↑, Global ↑ or ↔
            elif local_acc_diff > 0.05 and global_acc_diff >= 0:
                mu_update_factor *= 0.995  # Slightly More local freedom
                print(f"Round {server_round}, Client {client_id}: ↑ Local & Global : ↓ mu to {self.client_mu[client_id]:.4f}", flush=True)
            # 3) Local ↑, Global ↓ 
            elif local_acc_diff > 0.05 and global_acc_diff < -0.03:
                mu_update_factor *= 1.05  # Strong regularization increase
                print(f"Round {server_round}, Client {client_id}: ↑ Local, ↓ Global : ↓ mu (slightly) to {self.client_mu[client_id]:.4f}", flush=True)
            # 4) Local ↓, Global ↑
            elif local_acc_diff < -0.03 and global_acc_diff > 0.05:
                mu_update_factor *= 0.95  # Strong more local freedom
                print(f"Round {server_round}, Client {client_id}: ↓ Local, ↑ Global : ↑ mu (slightly) to {self.client_mu[client_id]:.4f}", flush=True)  

            # === Inter-client behavior: compare client to average ===
            disparity = curr_local_acc - avg_local_acc

            # 5) Local accuracy significantly below average
            if disparity < -0.05:
                mu_update_factor *= 0.85
                print(f"Round {server_round}, Client {client_id}: Local acc well below avg ({disparity:.2f}) → ↑ mu strongly (inter)", flush=True)
            
            # 7) Local accuracy slightly below average
            elif -0.05 < disparity < -0.02:
                mu_update_factor *= 0.95
                print(f"Round {server_round}, Client {client_id}: Local acc slightly below avg ({disparity:.2f}) → ↑ mu slightly (inter)", flush=True)

            # 6) Local accuracy significantly above average
            elif disparity > 0.05:
                mu_update_factor *= 1.15
                print(f"Round {server_round}, Client {client_id}: Local acc well above avg ({disparity:.2f}) → ↓ mu strongly (inter)", flush=True)
        

            # 8) Local accuracy slightly above average
            elif 0.02 < disparity < 0.05:
                mu_update_factor *= 1.05
                print(f"Round {server_round}, Client {client_id}: Local acc slightly above avg ({disparity:.2f}) → ↓ mu slightly (inter)", flush=True)

            # Update mu for the client
            self.client_mu[client_id] *= mu_update_factor


            # --- Tune lr_factor ---
            # if global_acc_diff < -0.05:
            #     self.client_lr_factor[client_id] *= 0.95
            #     print(f"Round {server_round}, Client {client_id}: ↓ Accuracy ({global_acc_diff:.4f}) → ↓ lr_factor to {self.client_lr_factor[client_id]:.4f}", flush=True)
            # elif global_acc_diff > 0.1:
            #     self.client_lr_factor[client_id] *= 1.055
            #     print(f"Round {server_round}, Client {client_id}: ↑ Accuracy ({global_acc_diff:.4f}) → ↑ lr_factor to {self.client_lr_factor[client_id]:.4f}", flush=True)
            # Otherwise, both remain unchanged

            # Cap values
            self.client_mu[client_id] = max(0.001, min(self.client_mu[client_id], 0.1))
            self.client_lr_factor[client_id] = max(0.5, min(self.client_lr_factor[client_id], 1.5))

            # Log current state
            print(f"Round {server_round}, Client {client_id}: Current global acc: {curr_global_acc:.4f}, mu: {self.client_mu[client_id]:.4f}, lr_factor: {self.client_lr_factor[client_id]:.4f}",flush=True)


# Using the custom strategy which is inherited from fl.server.strategy.FedAvg.In this all clients are required to participate in the training (fit) step and evaluation in each round.Minimum number of clients must be available for a round of training to begin.
strategy = CustomStrategy(
    min_available_clients=NUM_CLIENTS,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS
)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy
)

# Call function to visualize local and global accuracy trends
visualize_global_accuracy_clients(strategy.client_global_accuracy_history, VISUAL_DIR)
visualize_local_accuracy_clients(strategy.client_local_accuracy_history, VISUAL_DIR)
plot_accuracy_fairness_tradeoff(strategy.client_local_accuracy_history, strategy.client_global_accuracy_history, VISUAL_DIR)

# Directory to save the individual history files
HISTORY_DIR = os.path.join(VISUAL_DIR, "histories")
os.makedirs(HISTORY_DIR, exist_ok=True)

# Save client_local_accuracy_history
local_accuracy_history_path = os.path.join(HISTORY_DIR, "client_local_accuracy_history.json")
with open(local_accuracy_history_path, 'w') as f:
    json.dump(strategy.client_local_accuracy_history, f, indent=4)

# Save client_global_accuracy_history
global_accuracy_history_path = os.path.join(HISTORY_DIR, "client_global_accuracy_history.json")
with open(global_accuracy_history_path, 'w') as f:
    json.dump(strategy.client_global_accuracy_history, f, indent=4)

# Save client_local_accuracy_variance
local_accuracy_variance_path = os.path.join(HISTORY_DIR, "client_local_accuracy_variance.json")
with open(local_accuracy_variance_path, 'w') as f:
    json.dump(strategy.client_local_accuracy_variance, f, indent=4)


print("\nFederated learning completed.", flush=True)