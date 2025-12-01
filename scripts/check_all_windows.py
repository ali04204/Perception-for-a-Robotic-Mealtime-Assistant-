import pandas as pd

path = "results/features/all_windows.csv"
ds = pd.read_csv(path)

print("Shape:", ds.shape)
print("Columns:", [c for c in ds.columns if c in ["clip_id", "person_id", "split", "label"]])
print(ds["split"].value_counts())
print(ds["label"].value_counts())
