import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics

# Global variable
NUM_CLIENTS = 2

class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print metrics
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"\nRound {server_round} Accuracy:")
        for i, (_, r) in enumerate(results):
            print(f"Client {i+1}: {r.metrics['accuracy']:.4f}")
        print(f"Aggregated: {accuracy_aggregated:.4f}\n")

        return 0.0, {"accuracy": accuracy_aggregated}

# Use the custom strategy
strategy = CustomStrategy(
    min_available_clients=NUM_CLIENTS,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS
)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)
