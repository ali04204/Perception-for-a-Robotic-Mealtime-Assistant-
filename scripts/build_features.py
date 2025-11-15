import sys
import math
import pathlib
import numpy as np
import pandas as pd


def angle_deg(a, b, c):
    # angle at b formed by points a-b and c-b in 2D image coords
    if any(np.isnan(v) for v in [*a, *b, *c]):
        return np.nan
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na == 0 or nc == 0:
        return np.nan
    cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def build_features_for_file(in_csv: pathlib.Path):
    in_csv = in_csv.resolve()
    assert in_csv.exists(), f"Input CSV not found: {in_csv}"

    print(f"Building features for {in_csv.name}")

    df = pd.read_csv(in_csv)
    # estimate fps from time_sec
    dt = df["time_sec"].diff()
    fps = float(np.round(1.0 / np.median(dt.dropna().replace(0, np.nan))))

    # helpers for pose and hands
    def pxy(idx):
        return df[f"p{idx}_x"], df[f"p{idx}_y"]

    def hxy(prefix, idx):
        return df[f"{prefix}{idx}_x"], df[f"{prefix}{idx}_y"]

    # mouth center from pose 9 and 10
    p9x, p9y = pxy(9)
    p10x, p10y = pxy(10)
    mouth_x = (p9x + p10x) / 2.0
    mouth_y = (p9y + p10y) / 2.0

    # wrist positions from pose
    lwx, lwy = pxy(15)
    rwx, rwy = pxy(16)

    # shoulder mean height for relative wrist height
    lsx, lsy = pxy(11)
    rsx, rsy = pxy(12)
    shoulder_y = (lsy + rsy) / 2.0

    # fingertip positions from hand landmarks (index fingertip is 8)
    lhx, lhy = hxy("lh", 8)
    rhx, rhy = hxy("rh", 8)

    # distances hand tip to mouth in normalized image units
    df["lh_to_mouth"] = np.sqrt((lhx - mouth_x) ** 2 + (lhy - mouth_y) ** 2)
    df["rh_to_mouth"] = np.sqrt((rhx - mouth_x) ** 2 + (rhy - mouth_y) ** 2)

    # wrist height relative to shoulders (negative means below shoulders since y grows downward)
    df["lwrist_rel_shoulder_y"] = lwy - shoulder_y
    df["rwrist_rel_shoulder_y"] = rwy - shoulder_y

    # elbow angles using pose points
    lex_x, lex_y = pxy(13)   # left elbow
    rex_x, rex_y = pxy(14)   # right elbow

    # vectorized angle computation row by row
    left_angles = []
    right_angles = []
    for i in range(len(df)):
        left_angles.append(
            angle_deg((lsx.iloc[i], lsy.iloc[i]), (lex_x.iloc[i], lex_y.iloc[i]), (lwx.iloc[i], lwy.iloc[i]))
        )
        right_angles.append(
            angle_deg((rsx.iloc[i], rsy.iloc[i]), (rex_x.iloc[i], rex_y.iloc[i]), (rwx.iloc[i], rwy.iloc[i]))
        )
    df["left_elbow_angle_deg"] = left_angles
    df["right_elbow_angle_deg"] = right_angles

    # hand tip speeds in image units per second
    df["lh_speed"] = np.sqrt((lhx.diff())**2 + (lhy.diff())**2) * fps
    df["rh_speed"] = np.sqrt((rhx.diff())**2 + (rhy.diff())**2) * fps

    # simple smoothing to reduce jitter
    for col in ["lh_to_mouth", "rh_to_mouth",
                "lwrist_rel_shoulder_y", "rwrist_rel_shoulder_y",
                "left_elbow_angle_deg", "right_elbow_angle_deg",
                "lh_speed", "rh_speed"]:
        df[col] = df[col].rolling(window=5, min_periods=1, center=True).median()

    # keep only useful columns
    keep = ["frame", "time_sec",
            "lh_to_mouth", "rh_to_mouth",
            "lh_speed", "rh_speed",
            "lwrist_rel_shoulder_y", "rwrist_rel_shoulder_y",
            "left_elbow_angle_deg", "right_elbow_angle_deg"]
    out = df[keep].copy()

    out_path = in_csv.parent / f"{in_csv.stem.replace('_holistic', '')}_features.csv"
    out.to_csv(out_path, index=False)
    print(f"Estimated fps: {fps}")
    print(f"Saved engineered features to {out_path}")


def main():
    # Case 1: old behaviour, single file given on command line
    if len(sys.argv) > 1:
        in_col = pathlib.Path(sys.argv[1])
        build_features_for_file(in_col)
        print("Done.")
        return

    # Case 2: no argument, batch over all *_holistic.csv in results/features
    repo = pathlib.Path(__file__).resolve().parents[1]
    feat_dir = repo / "results" / "features"
    print(f"Batch mode. Looking for holistic CSVs in {feat_dir}")

    if not feat_dir.exists():
        print(f"Features directory not found: {feat_dir}")
        sys.exit(1)

    holistic_files = sorted(feat_dir.glob("*_holistic.csv"))
    if not holistic_files:
        print("No *_holistic.csv files found")
        sys.exit(1)

    for in_csv in holistic_files:
        out_path = in_csv.parent / f"{in_csv.stem.replace('_holistic', '')}_features.csv"
        if out_path.exists():
            print(f"Skipping {in_csv.name} since {out_path.name} already exists")
            continue
        build_features_for_file(in_csv)

    print("Batch feature building done.")


if __name__ == "__main__":
    main()
