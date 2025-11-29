import pathlib
import pandas as pd


def main():
    feat_dir = pathlib.Path("results/features")
    files = sorted(feat_dir.glob("*_windows_mouth.csv"))
    assert files, f"No *_windows_mouth.csv files found in {feat_dir}"

    all_rows = []
    for f in files:
        df = pd.read_csv(f)
        # Add a video name column so you know which windows came from which clip
        video_name = f.stem.replace("_windows_mouth", "")
        df["video"] = video_name
        all_rows.append(df)

    big = pd.concat(all_rows, ignore_index=True)
    out_path = feat_dir / "all_windows_mouth.csv"
    big.to_csv(out_path, index=False)
    print(f"Combined {len(files)} files into {out_path}")
    print(f"Total windows: {len(big)}")
    print("Label counts:")
    print(big["label"].value_counts())


if __name__ == "__main__":
    main()
