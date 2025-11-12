import sys
import pathlib
import pandas as pd
import numpy as np

# Usage:
# python scripts/align_segments_to_frames.py results\features\demo_features.csv data\labels\demo_segments.csv

if len(sys.argv) < 3:
    print("Give paths to the features CSV and the segments CSV")
    sys.exit(1)

feat_path = pathlib.Path(sys.argv[1]).resolve()
seg_path  = pathlib.Path(sys.argv[2]).resolve()
assert feat_path.exists(), f"Missing {feat_path}"
assert seg_path.exists(),  f"Missing {seg_path}"

df = pd.read_csv(feat_path)              # has frame and time_sec
segs = pd.read_csv(seg_path)             # video,start_sec,end_sec,label,notes

# start with unlabeled to keep gaps visible
df["label"] = "unlabeled"

# apply each segment to matching time range
for _, r in segs.iterrows():
    s = float(r["start_sec"])
    e = float(r["end_sec"])
    lab = str(r["label"]).strip()
    mask = (df["time_sec"] >= s) & (df["time_sec"] < e)
    df.loc[mask, "label"] = lab

out_path = feat_path.parent / f"{feat_path.stem.replace('_features','')}_frame_labels.csv"
df_out = df[["frame", "time_sec", "label"]].copy()
df_out.to_csv(out_path, index=False)

print(f"Saved frame labels to {out_path}")
print("Counts:")
print(df_out["label"].value_counts(dropna=False))
