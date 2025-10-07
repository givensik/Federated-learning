from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CLIENT_TEMPLATE = "client{client_id}.csv"
DEFAULT_CLASSES = 6


class SensorDataset(Dataset):
    """PyTorch dataset wrapping preprocessed sensor features."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def set_parameters(model: torch.nn.Module, parameters: Iterable[np.ndarray]) -> None:
    """Load a flat list of NumPy arrays into the model state dict."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict((k, torch.tensor(v)) for k, v in params_dict)
    model.load_state_dict(state_dict, strict=True)


def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """Extract model weights as a list of NumPy arrays."""
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected data file missing: {path}")


def _load_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    _ensure_file(path)
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    features = raw[:, :-1]
    labels = raw[:, -1].astype(np.int64)
    if labels.min() == 1:
        labels -= 1
    return features, labels


def _standardize(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    normalized = (features - mean) / std
    return normalized, mean.squeeze(), std.squeeze()


def _train_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    test_size = max(1, math.floor(len(indices) * test_ratio))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return (
        features[train_idx],
        labels[train_idx],
        features[test_idx],
        labels[test_idx],
    )


def _build_loaders(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = SensorDataset(train_features, train_labels)
    test_ds = SensorDataset(test_features, test_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_client_dataloaders(
    client_id: int,
    batch_size: int = 32,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Load normalized train/test dataloaders for a given client."""
    path = DATA_DIR / CLIENT_TEMPLATE.format(client_id=client_id)
    features, labels = _load_csv(path)
    normalized, _, _ = _standardize(features)
    train_features, train_labels, test_features, test_labels = _train_test_split(
        normalized, labels, test_ratio=test_ratio, seed=seed + client_id
    )
    return _build_loaders(
        train_features, train_labels, test_features, test_labels, batch_size
    )


def load_global_test_loader(batch_size: int = 64) -> DataLoader:
    """Combine client datasets to build a shared evaluation loader."""
    features_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    for client_id in range(1, 6):
        features, labels = _load_csv(DATA_DIR / CLIENT_TEMPLATE.format(client_id=client_id))
        features_list.append(features)
        labels_list.append(labels)

    merged_features = np.vstack(features_list)
    merged_labels = np.concatenate(labels_list)
    normalized, _, _ = _standardize(merged_features)
    dataset = TensorDataset(
        torch.from_numpy(normalized.astype(np.float32)),
        torch.from_numpy(merged_labels.astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_dataset_config() -> Dict[str, int]:
    sample_path = DATA_DIR / CLIENT_TEMPLATE.format(client_id=1)
    features, labels = _load_csv(sample_path)
    input_dim = features.shape[1]
    num_classes = int(labels.max()) + 1
    return {"input_dim": input_dim, "num_classes": num_classes}


def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    predicted = predictions.argmax(dim=1)
    correct = (predicted == labels).sum().item()
    return correct / len(labels)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Return loss and accuracy for the provided dataloader."""
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss_sum = 0.0
    count = 0
    correct_sum = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            correct_sum += (outputs.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return loss_sum / count, correct_sum / count
