import torch.nn as nn


class ActivityMLP(nn.Module):
    """Lightweight multilayer perceptron for activity recognition."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        hidden_dim = max(32, min(128, input_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)
