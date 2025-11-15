import pathlib, csv, cv2

root = pathlib.Path(__file__).resolve().parents[1]
raw_dir = root / "data" / "raw"
labels_dir = root / "data" / "labels"
labels_dir.mkdir(parents=True, exist_ok=True)

videos = sorted(raw_dir.glob("*.mp4"))
if not videos:
    print("No .mp4 files in data/raw")
    raise SystemExit

for v in videos:
    cap = cv2.VideoCapture(str(v))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur = float(n / fps) if fps > 0 else 0.0
    cap.release()

    out_csv = labels_dir / f"{v.stem}_segments.csv"
    if out_csv.exists():
        print(f"Exists, skip: {out_csv.name}")
        continue

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video","start_sec","end_sec","label","notes"])
        # example rows to edit
        w.writerow([v.name, 0.0, min(2.0, max(0.0, dur/4)), "not_ready", "example"])
        w.writerow([v.name, min(2.0, max(0.0, dur/4)), min(5.0, max(0.0, dur/2)), "ready", "example"])
    print(f"Made template: {out_csv.name} duration≈{dur:.1f}s")
print("Done.")
