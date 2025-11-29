import sys
import pathlib
import yaml

print("Python:", sys.version)
try:
    import numpy, pandas, cv2, mediapipe, sklearn, matplotlib
    print("numpy:", numpy.__version__)
    print("pandas:", pandas.__version__)
    print("opencv:", cv2.__version__)
    print("mediapipe:", mediapipe.__version__)
except Exception as e:
    print("Import error:", e)
    raise

root = pathlib.Path(__file__).resolve().parents[1]
cfg_path = root / "configs" / "default.yaml"
assert cfg_path.exists(), f"Missing {cfg_path}"

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

print("Loaded config keys:", list(cfg.keys()))
for key in ["raw", "labels", "splits", "results"]:
    p = root / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    print(f"Ensured directory exists: {p}")

print("Smoke test passed.")
