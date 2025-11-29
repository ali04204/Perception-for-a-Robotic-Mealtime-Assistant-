import sys
import pathlib
import numpy as np
import pandas as pd


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/add_mouth_width_feature.py "
              "results\\features\\17_holistic.csv results\\features\\17_windows.csv")
        sys.exit(1)

    hol_path = pathlib.Path(sys.argv[1]).resolve()
    win_path = pathlib.Path(sys.argv[2]).resolve()

    assert hol_path.exists(), f"Missing holistic csv: {hol_path}"
    assert win_path.exists(), f"Missing windows csv: {win_path}"

    # Per frame landmarks from MediaPipe Pose
    hol = pd.read_csv(hol_path)
    ds = pd.read_csv(win_path)

    # Pose indices 9 and 10 are mouth left and mouth right
    p9x = hol["p9_x"]
    p9y = hol["p9_y"]
    p10x = hol["p10_x"]
    p10y = hol["p10_y"]

    # Simple mouth feature per frame  horizontal mouth width
    mouth_width = np.sqrt((p9x - p10x) ** 2 + (p9y - p10y) ** 2)
    hol["mouth_width"] = mouth_width

    rows = []
    for _, row in ds.iterrows():
        t0 = row["t0"]
        t1 = row["t1"]

        mask = (hol["time_sec"] >= t0) & (hol["time_sec"] <= t1)
        vals = hol.loc[mask, "mouth_width"].dropna().values

        if len(vals) == 0:
            mw_mean = np.nan
            mw_std = np.nan
            mw_min = np.nan
            mw_max = np.nan
        else:
            mw_mean = float(vals.mean())
            mw_std = float(vals.std())
            mw_min = float(vals.min())
            mw_max = float(vals.max())

        row = row.copy()
        row["mouth_width_mean"] = mw_mean
        row["mouth_width_std"] = mw_std
        row["mouth_width_min"] = mw_min
        row["mouth_width_max"] = mw_max

        rows.append(row)

    ds_new = pd.DataFrame(rows)
    out_path = win_path.parent / f"{win_path.stem}_mouth.csv"
    ds_new.to_csv(out_path, index=False)
    print(f"Saved dataset with mouth feature to {out_path}")


if __name__ == "__main__":
    main()
