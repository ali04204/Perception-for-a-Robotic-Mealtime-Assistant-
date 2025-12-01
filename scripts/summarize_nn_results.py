import json
from pathlib import Path


def main():
    log_path = Path("results/metrics/baseline_nn_training_log.json")
    assert log_path.exists(), f"Missing log file: {log_path}"

    with log_path.open("r", encoding="utf8") as f:
        log = json.load(f)

    epochs = log.get("epochs", [])
    if not epochs:
        raise ValueError("No epochs found in training log")

    # Find epoch with best val_f1
    best_entry = max(
        epochs,
        key=lambda e: e.get("val_f1") if e.get("val_f1") is not None else -1.0,
    )

    best_epoch = best_entry["epoch"]
    best_val_loss = best_entry.get("val_loss")
    best_val_acc = best_entry.get("val_acc")
    best_val_f1 = best_entry.get("val_f1")

    test = log.get("test_metrics", {})

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "test_loss": test.get("loss"),
        "test_acc": test.get("acc"),
        "test_f1": test.get("f1"),
        "test_confusion": test.get("confusion"),
    }

    out_path = Path("results/metrics/baseline_nn_metrics.json")
    with out_path.open("w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    print("Best epoch:", best_epoch)
    print("Best val loss:", best_val_loss)
    print("Best val acc:", best_val_acc)
    print("Best val f1:", best_val_f1)
    print("Test loss:", summary["test_loss"])
    print("Test acc:", summary["test_acc"])
    print("Test f1:", summary["test_f1"])
    print("Test confusion:", summary["test_confusion"])
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
