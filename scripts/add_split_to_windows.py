import pathlib

import pandas as pd


def main():
    # Root of the repo: scripts/..
    root = pathlib.Path(__file__).resolve().parent.parent

    splits_path = root / "data" / "splits.csv"
    features_dir = root / "results" / "features"

    assert splits_path.exists(), f"Missing splits file: {splits_path}"
    assert features_dir.exists(), f"Missing features dir: {features_dir}"

    splits = pd.read_csv(splits_path)

    all_windows = []

    for _, row in splits.iterrows():
        video_id = row["video_id"]
        person_id = row["person_id"]
        split = row["split"]

        win_path = features_dir / f"{video_id}_windows.csv"

        if not win_path.exists():
            print(f"[WARN] Missing windows file for video {video_id}: {win_path}")
            continue

        print(f"[INFO] Updating {win_path}")

        df = pd.read_csv(win_path)

        # Add or overwrite helper columns
        df["video_id"] = video_id
        df["person_id"] = person_id
        df["split"] = split

        df.to_csv(win_path, index=False)
        all_windows.append(df)

    if not all_windows:
        print("[WARN] No window files were updated.")
        return

    merged = pd.concat(all_windows, ignore_index=True)
    merged_path = features_dir / "all_windows.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[INFO] Wrote merged dataset with {len(merged)} windows to {merged_path}")


if __name__ == "__main__":
    main()
