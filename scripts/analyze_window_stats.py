import pathlib
from datetime import datetime

import pandas as pd

# Project root = parent of "scripts"
ROOT = pathlib.Path(__file__).resolve().parents[1]
WINDOWS_CSV = ROOT / "results" / "features" / "all_windows.csv"
REPORT_PATH = ROOT / "results" / "reports" / "dataset_window_stats.md"


def choose_time_column(df: pd.DataFrame) -> str | None:
    """Pick a column that defines window order in time."""
    preferred = [
        "window_start_sec",
        "window_start_frame",
        "start_frame",
        "t_start",
        "frame_idx",
        "frame_index",
        "window_index",
    ]
    for col in preferred:
        if col in df.columns:
            return col
    return None


def choose_clip_column(df: pd.DataFrame) -> str | None:
    """Pick a column that identifies clips or videos."""
    preferred = ["clip_id", "video_id", "id"]
    for col in preferred:
        if col in df.columns:
            return col
    return None


def classify_transition(labels: list[str]) -> str:
    """Classify a clip as no, single, or multiple transitions."""
    if len(labels) == 0:
        return "empty"

    transitions = 0
    prev = labels[0]
    for lab in labels[1:]:
        if lab != prev:
            transitions += 1
            prev = lab

    if transitions == 0:
        return "no_transition"

    # Single transition from not_ready to ready
    if (
        transitions == 1
        and labels[0] == "not_ready"
        and labels[-1] == "ready"
    ):
        return "single_transition"

    return "multiple_transitions"


def main():
    assert WINDOWS_CSV.exists(), f"Missing {WINDOWS_CSV}"
    df = pd.read_csv(WINDOWS_CSV)

    # We only require split and label now
    required_cols = ["split", "label"]
    for c in required_cols:
        assert c in df.columns, f"Expected column {c} in all_windows.csv"

    clip_col = choose_clip_column(df)
    assert clip_col is not None, (
        "Could not find a clip id column. "
        "Expected one of clip_id, video_id, id in all_windows.csv"
    )

    time_col = choose_time_column(df)
    if time_col is None:
        print("Warning: no explicit time column found, using CSV row order")
    else:
        print(f"Using time column: {time_col}")

    # Sort once for stable per clip sequences
    if time_col is not None:
        df = df.sort_values([clip_col, time_col]).reset_index(drop=True)
    else:
        df = df.sort_values([clip_col]).reset_index(drop=True)

    # 1. Clip transition types
    clip_rows = []
    for (clip_id, split), g in df.groupby([clip_col, "split"]):
        labels = g["label"].tolist()
        t_type = classify_transition(labels)
        clip_rows.append(
            {
                "clip_id": clip_id,
                "split": split,
                "transition_type": t_type,
                "num_windows": len(labels),
            }
        )

    clip_stats = pd.DataFrame(clip_rows)

    # counts per type per split
    clip_counts = (
        clip_stats
        .groupby(["split", "transition_type"])
        .size()
        .reset_index(name="num_clips")
        .sort_values(["split", "transition_type"])
    )

    print("\nClip counts per transition type per split:")
    print(clip_counts)

    # 2. Class balance per split
    class_rows = []
    for split, g in df.groupby("split"):
        total = len(g)
        for label, count in g["label"].value_counts().items():
            pct = 100.0 * count / total if total > 0 else 0.0
            class_rows.append(
                {
                    "split": split,
                    "label": label,
                    "count": count,
                    "percent": pct,
                }
            )

    class_stats = (
        pd.DataFrame(class_rows)
        .sort_values(["split", "label"])
        .reset_index(drop=True)
    )

    print("\nWindow class balance per split:")
    print(class_stats)

    # 3. Write markdown report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append("# Dataset window stats")
    lines.append("")
    lines.append(f"Generated on {now} by scripts/analyze_window_stats.py")
    lines.append("")
    lines.append("## Clip transition types per split")
    lines.append("")
    lines.append("| split | transition_type | num_clips |")
    lines.append("| ----- | ---------------- | --------- |")
    for _, row in clip_counts.iterrows():
        lines.append(
            f"| {row['split']} | {row['transition_type']} | {row['num_clips']} |"
        )

    lines.append("")
    lines.append("## Window class balance per split")
    lines.append("")
    lines.append("| split | label | count | percent |")
    lines.append("| ----- | ----- | ----- | ------- |")
    for _, row in class_stats.iterrows():
        lines.append(
            f"| {row['split']} | {row['label']} | {row['count']} | {row['percent']:.2f} |"
        )

    # short summary paragraph
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "Fill this paragraph with a one or two line summary about transitions and class balance "
        "once you have looked at the numbers."
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf8")
    print(f"\nWrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
