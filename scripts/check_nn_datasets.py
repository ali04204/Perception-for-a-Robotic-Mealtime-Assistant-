import json
from pathlib import Path

import numpy as np

npz_path = Path("results/features/nn_windows_datasets.npz")
info_path = Path("results/config/baseline_nn_class_weights.json")

npz = np.load(npz_path)
print("NPZ keys:", list(npz.keys()))
for k in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
    arr = npz[k]
    print(k, "shape:", arr.shape, "dtype:", arr.dtype)

with info_path.open("r", encoding="utf8") as f:
    info = json.load(f)

print("Label map:", info["label_map"])
print("Counts:", info["counts"])
print("Class weights:", info["class_weights"])
