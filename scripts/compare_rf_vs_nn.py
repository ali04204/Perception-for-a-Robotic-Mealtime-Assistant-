import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        print(f"Missing file: {path}")
        return None
    with path.open("r", encoding="utf8") as f:
        return json.load(f)


def main():
    metrics_dir = Path("results") / "metrics"

    rf_path = metrics_dir / "baseline_rf_metrics.json"
    nn_path = metrics_dir / "baseline_nn_metrics.json"
    seq_path = metrics_dir / "seq_nn_metrics.json"

    rf_smooth_test_path = metrics_dir / "baseline_rf_test_windows_smoothed.json"
    nn_smooth_test_path = metrics_dir / "baseline_nn_test_windows_smoothed.json"

    rf = load_json(rf_path)
    nn = load_json(nn_path)
    seq = load_json(seq_path)
    rf_s = load_json(rf_smooth_test_path)
    nn_s = load_json(nn_smooth_test_path)

    print("\n=== Unsmooth baselines (test) ===")

    if rf is not None:
        print("\nRandom Forest:")
        print("  test_acc:", rf.get("test_acc"))
        print("  test_f1:", rf.get("test_f1"))
        print("  test_confusion:", rf.get("test_confusion"))
    else:
        print("\nRandom Forest metrics not found")

    if nn is not None:
        print("\nNeural Net (window MLP):")
        print("  test_acc:", nn.get("test_acc"))
        print("  test_f1:", nn.get("test_f1"))
        print("  test_confusion:", nn.get("test_confusion"))
        print("  best_epoch:", nn.get("best_epoch"))
        print("  best_val_f1:", nn.get("best_val_f1"))
    else:
        print("\nNeural Net metrics not found")

    if seq is not None:
        print("\nSeq NN (GRU on window sequences):")
        print("  test_acc:", seq.get("test_acc"))
        print("  test_f1:", seq.get("test_f1"))
        print("  test_confusion:", seq.get("test_confusion"))
        print("  best_val_f1:", seq.get("best_val_f1"))
    else:
        print("\nSeq NN metrics not found")

    print("\n=== Smoothed baselines (test) ===")

    if rf_s is not None:
        print("\nRandom Forest smoothed:")
        print("  accuracy:", rf_s.get("accuracy"))
        print("  per class:")
        print("    not_ready f1:", rf_s.get("per_class", {}).get("not_ready", {}).get("f1"))
        print("    ready     f1:", rf_s.get("per_class", {}).get("ready", {}).get("f1"))
        print("  confusion:", rf_s.get("confusion"))
    else:
        print("\nRandom Forest smoothed metrics not found")

    if nn_s is not None:
        print("\nNeural Net smoothed:")
        print("  accuracy:", nn_s.get("accuracy"))
        print("  per class:")
        print("    not_ready f1:", nn_s.get("per_class", {}).get("not_ready", {}).get("f1"))
        print("    ready     f1:", nn_s.get("per_class", {}).get("ready", {}).get("f1"))
        print("  confusion:", nn_s.get("confusion"))
    else:
        print("\nNeural Net smoothed metrics not found")

    if rf is not None or nn is not None or seq is not None:
        print("\n=== High level - unsmoothed test ===")
        if rf is not None:
            print(f"  RF      test_acc={rf.get('test_acc')}, test_f1={rf.get('test_f1')}")
        if nn is not None:
            print(f"  MLP NN  test_acc={nn.get('test_acc')}, test_f1={nn.get('test_f1')}")
        if seq is not None:
            print(f"  Seq NN  test_acc={seq.get('test_acc')}, test_f1={seq.get('test_f1')}")


if __name__ == "__main__":
    main()
