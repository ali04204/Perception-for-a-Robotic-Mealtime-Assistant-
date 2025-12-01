from pathlib import Path
import pandas as pd

def main():
    all_windows_path = Path("results/features/all_windows.csv")
    assert all_windows_path.exists(), f"Missing {all_windows_path}"

    ds = pd.read_csv(all_windows_path)

    # How many unique clips in total
    print("Unique video_ids in all_windows:", ds["video_id"].nunique())

    # How many per split
    print("\nUnique video_ids per split:")
    print(ds.groupby("split")["video_id"].nunique())

    # Optional list of ids per split
    for split in ["train", "val", "test"]:
        ids = sorted(ds[ds["split"] == split]["video_id"].unique().tolist())
        print(f"\n{split}: {len(ids)} video_ids -> {ids}")


if __name__ == "__main__":
    main()
