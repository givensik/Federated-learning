from __future__ import annotations

import argparse
from typing import Dict, Tuple

import flwr as fl
import torch

from shared.model import ActivityMLP
from shared.utils import evaluate_model, get_dataset_config, load_global_test_loader, set_parameters


def fit_config_fn(server_round: int) -> Dict[str, float]:
    """Provide training hyperparameters to clients."""
    return {
        "local_epochs": 2 if server_round < 3 else 3,
        "learning_rate": 1e-3,
    }


def evaluate_fn() -> fl.server.strategy.EvaluateFn:
    """Centralized evaluation across all client data."""
    config = get_dataset_config()
    test_loader = load_global_test_loader(batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(
        server_round: int,
        parameters,
        _: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        model = ActivityMLP(config["input_dim"], config["num_classes"]).to(device)
        set_parameters(model, parameters)
        loss, acc = evaluate_model(model, test_loader, device)
        return float(loss), {"accuracy": float(acc)}

    return evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower FL server for UCI HAR simulation")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080", help="gRPC server address")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        evaluate_fn=evaluate_fn(),
        on_fit_config_fn=fit_config_fn,
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

def aggregate_metrics(metrics) -> Dict[str, float]:
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"accuracy": 0.0}
    weighted_accuracy = sum(
        num_examples * metric.get("accuracy", 0.0) for num_examples, metric in metrics
    )
    return {"accuracy": weighted_accuracy / total_examples}
