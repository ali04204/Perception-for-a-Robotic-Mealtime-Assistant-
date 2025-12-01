import json
from pathlib import Path


def main():
    config = {
        "enter_threshold": 0.55,
        "exit_threshold": 0.45,
        "min_windows_enter": 2,
        "min_windows_exit": 2,
        "comment": "NN smoothing tuned to probability range ~0.3-0.7.",
    }

    out_dir = Path("results") / "config"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "smoothing_nn.json"
    with out_path.open("w", encoding="utf8") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote NN smoothing config to {out_path}")
    print("Config:", config)


if __name__ == "__main__":
    main()
