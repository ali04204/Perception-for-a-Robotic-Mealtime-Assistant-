import torch
import torch.nn as nn


class WindowMLP(nn.Module):
    """
    Simple MLP on top of window features.

    Input:  (batch_size, in_features)
    Output: (batch_size, 2) logits for [not_ready, ready]
    """

    def __init__(self, in_features: int, hidden_sizes=(64, 32), dropout: float = 0.3):
        super().__init__()

        layers = []
        last_dim = in_features

        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            last_dim = h

        # Final classification layer
        layers.append(nn.Linear(last_dim, 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        return self.net(x)


def build_window_mlp(in_features: int, num_classes: int = 2) -> nn.Module:
    if num_classes != 2:
        raise ValueError("This baseline is hard coded for 2 classes right now")
    return WindowMLP(in_features=in_features)


if __name__ == "__main__":
    # Quick smoke test
    in_features = 20  # matches your WINDOW_FEATURE_COLUMNS length
    model = build_window_mlp(in_features)
    x = torch.randn(4, in_features)
    logits = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)
