import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

# Usage:
# .\.venv\Scripts\python.exe scripts\plot_quicklook.py results\features\demo_features.csv results\features\demo_frame_labels.csv

if len(sys.argv) < 3:
    print("Give paths to features.csv and frame_labels.csv")
    sys.exit(1)

feat_path = pathlib.Path(sys.argv[1]).resolve()
lab_path  = pathlib.Path(sys.argv[2]).resolve()
assert feat_path.exists(), f"Missing {feat_path}"
assert lab_path.exists(), f"Missing {lab_path}"

feat = pd.read_csv(feat_path)
labs = pd.read_csv(lab_path)
df = feat.merge(labs, on=["frame", "time_sec"], how="left")

def add_spans(ax, df, value, alpha=0.15):
    run_id = (df["label"].ne(df["label"].shift())).cumsum()
    first = True
    for _, g in df.groupby(run_id):
        lab = g["label"].iloc[0]
        if lab != value:
            continue
        t0 = g["time_sec"].iloc[0]
        t1 = g["time_sec"].iloc[-1]
        ax.axvspan(t0, t1, alpha=alpha, label=value if first else None)
        first = False

out_dir = feat_path.parent

# Distances to mouth
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["time_sec"], df["lh_to_mouth"], label="lh_to_mouth")
ax.plot(df["time_sec"], df["rh_to_mouth"], label="rh_to_mouth")
add_spans(ax, df, "ready", alpha=0.18)
add_spans(ax, df, "not_ready", alpha=0.08)
ax.set_xlabel("time_sec")
ax.set_ylabel("distance (norm units)")
ax.legend(loc="upper right")
fig.tight_layout()
p1 = out_dir / "quicklook_dist.png"
fig.savefig(p1, dpi=140)

# Hand speeds
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df["time_sec"], df["lh_speed"], label="lh_speed")
ax2.plot(df["time_sec"], df["rh_speed"], label="rh_speed")
add_spans(ax2, df, "ready", alpha=0.18)
add_spans(ax2, df, "not_ready", alpha=0.08)
ax2.set_xlabel("time_sec")
ax2.set_ylabel("speed (pix per sec equiv)")
ax2.legend(loc="upper right")
fig2.tight_layout()
p2 = out_dir / "quicklook_speed.png"
fig2.savefig(p2, dpi=140)

print("Saved:", p1)
print("Saved:", p2)
