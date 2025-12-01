import json
from pathlib import Path
import numpy as np


def main():
    npz_path = Path("results/features/nn_windows_datasets.npz")
    assert npz_path.exists(), f"Missing dataset npz: {npz_path}"

    data = np.load(npz_path)
    X_train = data["X_train"]
    in_features = int(X_train.shape[1])

    config = {
        "model_type": "window_mlp",
        "input_dim": in_features,
        "hidden_sizes": [64, 32],
        "activation": "ReLU",
        "output_dim": 2,
        "loss": "CrossEntropyLoss",
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 40,
        "class_weights_file": "baseline_nn_class_weights.json",
        "dataset_file": "nn_windows_datasets.npz",
    }

    out_dir = Path("results") / "config"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_nn_windows.json"

    with out_path.open("w", encoding="utf8") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote NN config to {out_path}")
    print("Input dim:", in_features)


if __name__ == "__main__":
    main()
