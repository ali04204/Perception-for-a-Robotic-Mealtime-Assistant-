import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler

from feature_config import WINDOW_FEATURE_COLUMNS


def main():
    all_windows_path = Path("results/features/all_windows.csv")
    assert all_windows_path.exists(), f"Missing {all_windows_path}"

    ds = pd.read_csv(all_windows_path)
    train = ds[ds["split"] == "train"]

    X_train = train[WINDOW_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    print("Train windows shape:", X_train.shape)

    scaler = StandardScaler()
    scaler.fit(X_train)

    out_dir = Path("results/config")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "seq_nn_window_scaler.joblib"
    dump(scaler, out_path)
    print(f"Saved sequence NN scaler to {out_path}")


if __name__ == "__main__":
    main()
