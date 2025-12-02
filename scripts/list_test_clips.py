import pathlib
import pandas as pd

repo_root = pathlib.Path(__file__).resolve().parents[1]
all_windows_path = repo_root / "results" / "features" / "all_windows.csv"

ds = pd.read_csv(all_windows_path)

if "split" not in ds.columns:
    raise SystemExit("Expected column 'split' in all_windows.csv")

# Figure out which column holds the clip id
id_col = None
if "clip_id" in ds.columns:
    id_col = "clip_id"
elif "video_id" in ds.columns:
    id_col = "video_id"

if id_col is None:
    raise SystemExit("Expected a 'clip_id' or 'video_id' column in all_windows.csv")

test_ids = sorted(ds.loc[ds["split"] == "test", id_col].astype(str).unique())

print("Test split ids (using column:", id_col + "):")
for cid in test_ids:
    print(" ", cid)
