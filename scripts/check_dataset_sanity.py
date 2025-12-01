import pathlib
import sys
from collections import defaultdict

import pandas as pd


def find_clip_ids_from_folder(folder, suffix):
    folder = pathlib.Path(folder)
    if not folder.exists():
        return set()
    ids = set()
    for p in folder.glob(f"*{suffix}"):
        stem = p.stem
        cid = None
        if stem.endswith("_holistic"):
            cid = stem.replace("_holistic", "")
        elif stem.endswith("_features"):
            cid = stem.replace("_features", "")
        elif stem.endswith("_windows"):
            cid = stem.replace("_windows", "")
        else:
            continue
        # ignore aggregate files like all_windows
        if cid == "all":
            continue
        ids.add(cid)
    return ids


def main():
    repo_root = pathlib.Path(__file__).resolve().parents[1]

    data_dir = repo_root / "data"
    raw_dir = data_dir / "raw"

    results_dir = repo_root / "results"
    holistic_dir = results_dir / "holistic"
    features_dir = results_dir / "features"

    dataset_index_path = data_dir / "dataset_index.csv"
    splits_path = data_dir / "splits.csv"
    all_windows_path = features_dir / "all_windows.csv"

    print("===================== BASIC FILE CHECKS =====================")
    print("dataset_index.csv:", "OK" if dataset_index_path.exists() else "MISSING")
    print("splits.csv:        ", "OK" if splits_path.exists() else "MISSING")
    print("all_windows.csv:   ", "OK" if all_windows_path.exists() else "MISSING")

    # Discover ids
    raw_ids = set()
    if raw_dir.exists():
        for p in raw_dir.glob("*.mp4"):
            raw_ids.add(p.stem)

    hol_ids = find_clip_ids_from_folder(holistic_dir, "_holistic.csv")
    feat_ids = find_clip_ids_from_folder(features_dir, "_features.csv")
    win_ids = find_clip_ids_from_folder(features_dir, "_windows.csv")

    print("\n===================== PER STAGE COUNTS =====================")
    print(f"Raw videos:        {len(raw_ids)}")
    print(f"Holistic CSVs:     {len(hol_ids)}")
    print(f"Feature CSVs:      {len(feat_ids)}")
    print(f"Window CSVs:       {len(win_ids)}")

    # dataset_index and splits
    idx_ids = set()
    if dataset_index_path.exists():
        ds_idx = pd.read_csv(dataset_index_path)
        if "clip_id" in ds_idx.columns:
            idx_ids = set(ds_idx["clip_id"].astype(str))
        elif "video_id" in ds_idx.columns:
            idx_ids = set(ds_idx["video_id"].astype(str))

    split_ids = set()
    split_by_split = defaultdict(int)
    if splits_path.exists():
        ds_splits = pd.read_csv(splits_path)
        key = "clip_id" if "clip_id" in ds_splits.columns else "video_id"
        split_ids = set(ds_splits[key].astype(str))
        if "split" in ds_splits.columns:
            for _, row in ds_splits.iterrows():
                split_by_split[str(row["split"])] += 1

    print("\n===================== ID COVERAGE CHECKS =====================")
    print("Total ids in raw/:          ", len(raw_ids))
    print("Ids in dataset_index:       ", len(idx_ids))
    print("Ids in splits.csv:          ", len(split_ids))

    missing_in_index = raw_ids - idx_ids
    missing_in_splits = raw_ids - split_ids

    if missing_in_index:
        print("\nRaw ids missing in dataset_index.csv:", sorted(missing_in_index))
    if missing_in_splits:
        print("\nRaw ids missing in splits.csv:", sorted(missing_in_splits))

    print("\n===================== PER CLIP PIPELINE CHECK =====================")
    all_ids = sorted(raw_ids | idx_ids | split_ids | hol_ids | feat_ids | win_ids)

    for cid in all_ids:
        flags = {
            "raw": cid in raw_ids,
            "idx": cid in idx_ids,
            "split": cid in split_ids,
            "hol": cid in hol_ids,
            "feat": cid in feat_ids,
            "win": cid in win_ids,
        }
        missing = [k for k, ok in flags.items() if not ok]
        if missing:
            print(f"Clip {cid}: missing {', '.join(missing)}")

    print("\n===================== GLOBAL WINDOW STATS =====================")
    if all_windows_path.exists():
        ds_all = pd.read_csv(all_windows_path)
        n_rows = len(ds_all)
        print("Total windows rows:", n_rows)

        if "label" in ds_all.columns:
            label_counts = ds_all["label"].value_counts(dropna=False).to_dict()
            print("Label counts:", label_counts)
            n_missing_label = ds_all["label"].isna().sum()
        else:
            print("No 'label' column in all_windows.csv")
            n_missing_label = n_rows

        if "split" in ds_all.columns:
            split_counts = ds_all["split"].value_counts(dropna=False).to_dict()
            print("Split counts:", split_counts)
            n_missing_split = ds_all["split"].isna().sum()
        else:
            print("No 'split' column in all_windows.csv")
            n_missing_split = n_rows

        if n_missing_label > 0:
            print(f"WARNING: {n_missing_label} rows have missing label")
        if n_missing_split > 0:
            print(f"WARNING: {n_missing_split} rows have missing split")

        id_col = None
        if "clip_id" in ds_all.columns:
            id_col = "clip_id"
        elif "video_id" in ds_all.columns:
            id_col = "video_id"

        if id_col:
            aw_ids = set(ds_all[id_col].astype(str))
            missing_aw_splits = aw_ids - split_ids
            if missing_aw_splits:
                print("WARNING: clip ids in all_windows not found in splits.csv:", sorted(missing_aw_splits))
        else:
            print("No clip_id or video_id column in all_windows.csv")
    else:
        print("No all_windows.csv found")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
