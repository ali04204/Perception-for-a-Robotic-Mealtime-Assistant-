import torch
import torch.nn as nn


class SeqWindowGRU(nn.Module):
    """
    GRU based sequence model that outputs a label for each time step.
    Input:  (B, T, F)
    Output: (B, T, 2) logits for [not_ready, ready]
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * num_directions, 2)

    def forward(self, x):
        """
        x: (B, T, F)
        returns logits: (B, T, 2)
        """
        out, _ = self.gru(x)      # out: (B, T, H * num_directions)
        logits = self.fc(out)     # (B, T, 2)
        return logits


def build_seq_model(in_features: int) -> nn.Module:
    """
    Helper to build the default sequence model.
    You can tweak hidden size and layers here.
    """
    model = SeqWindowGRU(
        in_features=in_features,
        hidden_size=64,
        num_layers=1,
        bidirectional=True,
        dropout=0.0,
    )
    return model


if __name__ == "__main__":
    # Tiny sanity check when run directly
    model = build_seq_model(in_features=20)
    x = torch.randn(2, 10, 20)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
