import sys
import pathlib
import pandas as pd
import numpy as np


def align_one(feat_path: pathlib.Path, seg_path: pathlib.Path):
    feat_path = feat_path.resolve()
    seg_path = seg_path.resolve()
    assert feat_path.exists(), f"Missing {feat_path}"
    assert seg_path.exists(), f"Missing {seg_path}"

    print(f"Aligning {feat_path.name} with {seg_path.name}")

    df = pd.read_csv(feat_path)   # has frame and time_sec
    segs = pd.read_csv(seg_path)  # video,start_sec,end_sec,label,notes

    # start with unlabeled to keep gaps visible
    df["label"] = "unlabeled"

    # apply each segment to matching time range
    for _, r in segs.iterrows():
        s = float(r["start_sec"])
        e = float(r["end_sec"])
        lab = str(r["label"]).strip()
        mask = (df["time_sec"] >= s) & (df["time_sec"] < e)
        df.loc[mask, "label"] = lab

    # name output like before
    out_path = feat_path.parent / f"{feat_path.stem.replace('_features','')}_frame_labels.csv"
    if out_path.exists():
        print(f"  Output {out_path.name} already exists, overwriting")

    df_out = df[["frame", "time_sec", "label"]].copy()
    df_out.to_csv(out_path, index=False)

    print(f"  Saved frame labels to {out_path}")
    print("  Counts:")
    print(df_out["label"].value_counts(dropna=False))


def main():
    # Case 1. Old behaviour, align one pair given on command line
    if len(sys.argv) >= 3:
        feat_path = pathlib.Path(sys.argv[1])
        seg_path = pathlib.Path(sys.argv[2])
        align_one(feat_path, seg_path)
        print("Done.")
        return

    # Case 2. No arguments, batch mode over all videos
    repo = pathlib.Path(__file__).resolve().parents[1]
    feat_dir = repo / "results" / "features"
    seg_dir = repo / "data" / "labels"

    print(f"Batch mode. Features dir: {feat_dir}")
    print(f"Segments dir: {seg_dir}")

    if not feat_dir.exists():
        print(f"Features directory not found: {feat_dir}")
        sys.exit(1)
    if not seg_dir.exists():
        print(f"Segments directory not found: {seg_dir}")
        sys.exit(1)

    feature_files = sorted(feat_dir.glob("*.csv"))
    if not feature_files:
        print("No feature csv files found")
        sys.exit(1)

    for feat_path in feature_files:
        stem = feat_path.stem

        # Decide base name
        if stem.endswith("_features"):
            base = stem.replace("_features", "")
        elif stem.endswith("_holistic"):
            base = stem.replace("_holistic", "")
        else:
            # skip files that do not match expected patterns
            print(f"Skipping {feat_path.name} (unexpected name)")
            continue

        seg_path = seg_dir / f"{base}_segments.csv"
        if not seg_path.exists():
            print(f"Skipping {feat_path.name} (no segments file {seg_path.name})")
            continue

        out_path = feat_path.parent / f"{stem.replace('_features','')}_frame_labels.csv"
        if out_path.exists():
            print(f"Skipping {feat_path.name} (labels {out_path.name} already exist)")
            continue

        align_one(feat_path, seg_path)

    print("Batch alignment done.")


if __name__ == "__main__":
    main()
