import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# This script:
# 1. Loads ALL *_windows.csv files in results/features
# 2. Concatenates them into all_data.csv
# 3. Splits by video into train_data.csv and test_data.csv

root = Path("results") / "features"

# Pick up every per video windows file
window_files = [
    p for p in root.glob("*_windows.csv")
    if p.name not in ["train_data.csv", "test_data.csv", "all_data.csv"]
]

if not window_files:
    print("No *_windows.csv files found in results/features. Run make_dataset.py first.")
    raise SystemExit

dfs = []
for p in window_files:
    df = pd.read_csv(p)
    # source is the video id, for example "4" or "demo"
    df["source"] = p.stem.replace("_windows", "")
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)

all_path = root / "all_data.csv"
all_data.to_csv(all_path, index=False)
print(f"Saved {all_path} with {len(all_data)} rows")

sources = sorted(all_data["source"].unique())
print("All videos:", sources)

# Split videos into train and test sets
train_sources, test_sources = train_test_split(
    sources,
    test_size=0.25,
    random_state=42
)

train_df = all_data[all_data["source"].isin(train_sources)].reset_index(drop=True)
test_df = all_data[all_data["source"].isin(test_sources)].reset_index(drop=True)

train_path = root / "train_data.csv"
test_path = root / "test_data.csv"

# Drop the string column "source" from the files used for training
cols_to_save = [c for c in train_df.columns if c != "source"]

train_df[cols_to_save].to_csv(train_path, index=False)
test_df[cols_to_save].to_csv(test_path, index=False)

print("Train videos:", train_sources)
print("Test videos:", test_sources)
print("Train rows:", len(train_df))
print("Test rows:", len(test_df))