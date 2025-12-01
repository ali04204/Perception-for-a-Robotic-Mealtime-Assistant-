import numpy as np
import pandas as pd
from pathlib import Path


def summarize_group(df, split, label):
    if df.empty:
        print(f"{split} - {label}: no rows")
        return

    p = df["nn_p_ready"].to_numpy()
    print(f"{split} - {label}: count={len(p)}")
    print(f"  mean={p.mean():.3f} std={p.std():.3f}")
    print(f"  min={p.min():.3f} 25%={np.percentile(p, 25):.3f} "
          f"50%={np.percentile(p, 50):.3f} 75%={np.percentile(p, 75):.3f} max={p.max():.3f}")


def main():
    path = Path("results/features/all_windows_nn_smoothed.csv")
    assert path.exists(), f"Missing smoothed windows file: {path}"

    ds = pd.read_csv(path)

    for split in ["val", "test"]:
        df_split = ds[ds["split"] == split]
        print(f"\n=== {split.upper()} ===")
        summarize_group(df_split[df_split["label"] == "not_ready"], split, "not_ready")
        summarize_group(df_split[df_split["label"] == "ready"], split, "ready")


if __name__ == "__main__":
    main()
