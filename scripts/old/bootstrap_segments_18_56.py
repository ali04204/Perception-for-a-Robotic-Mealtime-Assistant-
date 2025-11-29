import csv
import pathlib

# Project root = parent of the scripts folder
ROOT = pathlib.Path(__file__).resolve().parents[1]

labels_dir = ROOT / "data" / "labels"
labels_dir.mkdir(parents=True, exist_ok=True)

# All clip ids we want templates for
clip_ids = list(range(18, 57))  # 18 to 56 inclusive

for cid in clip_ids:
    seg_path = labels_dir / f"{cid}_segments.csv"

    # If you REALLY don't want to overwrite anything, uncomment this:
    # if seg_path.exists():
    #     print(f"Skipping {seg_path} (already exists)")
    #     continue

    video_name = f"{cid}.mp4"

    # Simple template: you will edit start/end times, labels, and notes
    rows = [
        ["video", "start_sec", "end_sec", "label", "notes"],
        [video_name, 0.0, 5.0, "not_ready", ""],
        [video_name, 5.0, 10.0, "ready", ""],
    ]

    with seg_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Created template {seg_path}")
