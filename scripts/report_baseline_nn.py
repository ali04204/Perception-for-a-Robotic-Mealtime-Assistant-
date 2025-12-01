import json
from pathlib import Path


def main():
    metrics_path = Path("results/metrics/baseline_nn_metrics.json")
    class_info_path = Path("results/config/baseline_nn_class_weights.json")

    assert metrics_path.exists(), f"Missing metrics file: {metrics_path}"
    assert class_info_path.exists(), f"Missing class info file: {class_info_path}"

    with metrics_path.open("r", encoding="utf8") as f:
        m = json.load(f)

    with class_info_path.open("r", encoding="utf8") as f:
        ci = json.load(f)

    counts = ci.get("counts", {})
    total = ci.get("total_windows", None)

    not_ready = counts.get("not_ready", 0)
    ready = counts.get("ready", 0)

    best_epoch = m["best_epoch"]
    best_val_loss = m["best_val_loss"]
    best_val_acc = m["best_val_acc"]
    best_val_f1 = m["best_val_f1"]

    test_loss = m["test_loss"]
    test_acc = m["test_acc"]
    test_f1 = m["test_f1"]
    cm = m["test_confusion"] or [[0, 0], [0, 0]]

    nr_nr, nr_r = cm[0]
    r_nr, r_r = cm[1]

    out_dir = Path("results") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_nn_results.md"

    lines = []

    lines.append("# Baseline neural network on window features")
    lines.append("")
    lines.append("This report summarizes the performance of the window based MLP baseline.")
    lines.append("")
    lines.append("## Dataset summary")
    lines.append("")
    if total is not None:
        lines.append(f"- Total windows: **{total}**")
    lines.append(f"- not_ready windows: **{not_ready}**")
    lines.append(f"- ready windows: **{ready}**")
    lines.append("- Features per window: **20**")
    lines.append("")
    lines.append("## Model")
    lines.append("")
    lines.append("- Architecture: MLP on window feature vector")
    lines.append("- Output: logits for two classes [not_ready, ready]")
    lines.append("")
    lines.append("## Validation performance")
    lines.append("")
    lines.append(f"- Best epoch: **{best_epoch}**")
    lines.append(f"- Best val loss: **{best_val_loss:.4f}**")
    lines.append(f"- Best val accuracy: **{best_val_acc:.4f}**")
    lines.append(f"- Best val F1 (ready as positive): **{best_val_f1:.4f}**")
    lines.append("")
    lines.append("## Test performance")
    lines.append("")
    lines.append(f"- Test loss: **{test_loss:.4f}**")
    lines.append(f"- Test accuracy: **{test_acc:.4f}**")
    lines.append(f"- Test F1 (ready as positive): **{test_f1:.4f}**")
    lines.append("")
    lines.append("### Test confusion matrix")
    lines.append("")
    lines.append("|              | predicted not_ready | predicted ready |")
    lines.append("|--------------|---------------------|-----------------|")
    lines.append(f"| actual not_ready | {nr_nr} | {nr_r} |")
    lines.append(f"| actual ready     | {r_nr} | {r_r} |")
    lines.append("")
    lines.append("_Rows are actual labels and columns are predicted labels._")
    lines.append("")

    out_text = "\n".join(lines)

    with out_path.open("w", encoding="utf8") as f:
        f.write(out_text)

    print(f"Wrote NN baseline report to {out_path}")


if __name__ == "__main__":
    main()
