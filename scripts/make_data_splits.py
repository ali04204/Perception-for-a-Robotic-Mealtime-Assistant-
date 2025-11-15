import pathlib
import pandas as pd
import numpy as np


def load_video_data(features_path: pathlib.Path, labels_path: pathlib.Path) -> pd.DataFrame:
    features_path = features_path.resolve()
    labels_path = labels_path.resolve()
    assert features_path.exists(), f"Missing features file {features_path}"
    assert labels_path.exists(), f"Missing labels file {labels_path}"

    print(f"Loading {features_path.name} and {labels_path.name}")

    df_feat = pd.read_csv(features_path)
    df_lab = pd.read_csv(labels_path)

    # join on frame and time_sec
    df = pd.merge(df_feat, df_lab, on=["frame", "time_sec"], how="inner")
    return df


def main():
    repo = pathlib.Path(__file__).resolve().parents[1]
    feat_dir = repo / "results" / "features"

    if not feat_dir.exists():
        print(f"Features directory not found: {feat_dir}")
        return

    feature_files = sorted(feat_dir.glob("*_features.csv"))
    if not feature_files:
        print("No *_features.csv files found")
        return

    all_rows = []

    for feat_path in feature_files:
        base = feat_path.stem.replace("_features", "")
        labels_path = feat_dir / f"{base}_frame_labels.csv"

        if not labels_path.exists():
            print(f"Skipping {feat_path.name} because labels {labels_path.name} are missing")
            continue

        df_video = load_video_data(feat_path, labels_path)
        df_video["video"] = base
        all_rows.append(df_video)

    if not all_rows:
        print("No videos with both features and labels. Nothing to do.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)

    # drop unlabeled rows
    if "label" not in full_df.columns:
        print("No label column found. Did align_segments_to_frames.py run correctly?")
        return

    full_df = full_df[full_df["label"] != "unlabeled"].reset_index(drop=True)

    # save combined dataset
    full_path = feat_dir / "all_data.csv"
    full_df.to_csv(full_path, index=False)
    print(f"Saved combined dataset to {full_path}")

    # train and test split by video
    video_ids = sorted(full_df["video"].unique())
    print("Videos in dataset:", video_ids)

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(video_ids))
    n_train = max(1, int(0.8 * len(video_ids)))

    train_video_ids = [video_ids[i] for i in perm[:n_train]]
    test_video_ids = [video_ids[i] for i in perm[n_train:]]

    print("Train videos:", train_video_ids)
    print("Test videos:", test_video_ids)

    train_df = full_df[full_df["video"].isin(train_video_ids)].reset_index(drop=True)
    test_df = full_df[full_df["video"].isin(test_video_ids)].reset_index(drop=True)

    train_path = feat_dir / "train_data.csv"
    test_path = feat_dir / "test_data.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train split to {train_path}")
    print(f"Saved test split to {test_path}")
    print("Done.")


if __name__ == "__main__":
    main()
