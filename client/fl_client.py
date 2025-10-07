from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import flwr as fl
import torch
from torch import nn

from shared.model import ActivityMLP
from shared.utils import (
    evaluate_model,
    get_dataset_config,
    get_parameters,
    load_client_dataloaders,
    set_parameters,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
) -> None:
    """Run local client training."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model.to(DEVICE)

    def get_parameters(self, config: Dict | None = None):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        epochs = int(config.get("local_epochs", 2))
        lr = float(config.get("learning_rate", 1e-3))
        train(self.model, self.train_loader, DEVICE, epochs=epochs, lr=lr)
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate_model(self.model, self.test_loader, DEVICE)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower client for UCI HAR FL simulation")
    parser.add_argument("--client-id", type=int, default=1, help="Client identifier (1-5)")
    parser.add_argument("--server-address", type=str, default=None, help="Flower server address")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client_id = args.client_id
    server_address = args.server_address or os.environ.get("SERVER_ADDRESS", "0.0.0.0:8080")

    config = get_dataset_config()
    train_loader, test_loader = load_client_dataloaders(client_id, batch_size=32)
    model = ActivityMLP(config["input_dim"], config["num_classes"])

    client = FlowerClient(client_id, train_loader, test_loader, model)
    fl.client.start_numpy_client(server_address=server_address, client=client)


if __name__ == "__main__":
    main()
