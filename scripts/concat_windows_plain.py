import pathlib
import pandas as pd


def main():
    feat_dir = pathlib.Path("results/features")
    files = sorted(feat_dir.glob("*_windows.csv"))

    # Just in case, ignore the mouth files
    files = [f for f in files if not f.name.endswith("_windows_mouth.csv")]

    if not files:
        raise SystemExit(f"No *_windows.csv files found in {feat_dir}")

    all_rows = []
    for f in files:
        df = pd.read_csv(f)

        # Example video_name: "4", "5", "17", "20210804_145925_1"
        video_name = f.stem.replace("_windows", "")
        df["video"] = video_name

        all_rows.append(df)

    big = pd.concat(all_rows, ignore_index=True)

    out_path = feat_dir / "all_windows.csv"
    big.to_csv(out_path, index=False)

    print(f"Combined {len(files)} files into {out_path}")
    print(f"Total windows: {len(big)}")
    print("Label counts:")
    if "label" in big.columns:
        print(big["label"].value_counts())
    else:
        print("Warning. No 'label' column found in combined DataFrame.")


if __name__ == "__main__":
    main()
