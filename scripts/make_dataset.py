import sys, pathlib, numpy as np, pandas as pd

# usage:
# .\.venv\Scripts\python.exe scripts\make_dataset.py results\features\demo_features.csv results\features\demo_frame_labels.csv

if len(sys.argv) < 3:
    print("Give paths to features.csv and frame_labels.csv")
    sys.exit(1)

feat_path = pathlib.Path(sys.argv[1])
lab_path  = pathlib.Path(sys.argv[2])
assert feat_path.exists() and lab_path.exists()

feat = pd.read_csv(feat_path)
labs = pd.read_csv(lab_path)
df = feat.merge(labs, on=["frame","time_sec"], how="left")

# robust, hand-agnostic features
df["hand_to_mouth_min"] = np.fmin(df["lh_to_mouth"].values, df["rh_to_mouth"].values)
df["hand_speed_max"] = np.fmax(df["lh_speed"].values, df["rh_speed"].values)
df["wrist_rel_shoulder_y_min"] = np.fmin(df["lwrist_rel_shoulder_y"].values, df["rwrist_rel_shoulder_y"].values)
df["elbow_angle_deg_min"] = np.fmin(df["left_elbow_angle_deg"].values, df["right_elbow_angle_deg"].values)

# config
WINDOW_SEC = 1.0
STRIDE_SEC = 0.5

# get fps from time grid
dt = df["time_sec"].diff().replace(0, np.nan).dropna()
fps = float(round(1.0 / dt.median()))
win = int(WINDOW_SEC * fps)
stride = int(STRIDE_SEC * fps)

feature_cols = [
    "hand_to_mouth_min",
    "hand_speed_max",
    "wrist_rel_shoulder_y_min",
    "elbow_angle_deg_min",
]

rows = []
for start in range(0, len(df) - win + 1, stride):
    w = df.iloc[start:start+win]

    # keep windows with at least 60% labeled frames
    labeled = w["label"].isin(["ready","not_ready"]).sum()
    if labeled < int(0.6 * len(w)):
        continue

    # majority label between ready and not_ready
    lab = "ready" if (w["label"] == "ready").sum() >= (w["label"] == "not_ready").sum() else "not_ready"

    feats = {}
    for c in feature_cols:
        x = w[c].to_numpy()
        x = x[~np.isnan(x)]
        if x.size == 0:
            # if totally missing, skip this window
            feats = None
            break
        feats[f"{c}_mean"] = float(np.mean(x))
        feats[f"{c}_std"]  = float(np.std(x))
        feats[f"{c}_min"]  = float(np.min(x))
        feats[f"{c}_max"]  = float(np.max(x))
        # slope
        tw = w["time_sec"][~w[c].isna()].to_numpy()
        m = float(np.polyfit(tw, w[c].dropna().to_numpy(), 1)[0]) if tw.size > 1 else 0.0
        feats[f"{c}_slope"] = m

    if feats is None:
        continue

    feats["t0"] = float(w["time_sec"].iloc[0])
    feats["t1"] = float(w["time_sec"].iloc[-1])
    feats["label"] = lab
    rows.append(feats)

ds = pd.DataFrame(rows)
out = feat_path.parent / f"{feat_path.stem.replace('_features','')}_windows.csv"
ds.to_csv(out, index=False)

print(f"fps={fps}, windows={len(ds)}")
print(f"Saved dataset to {out}")
if len(ds) > 0:
    print(ds['label'].value_counts())
else:
    print("No windows kept. Check your label times cover enough of the video.")
