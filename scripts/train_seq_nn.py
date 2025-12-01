import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from baseline_seq_nn import build_seq_model


LABEL_MAP = {"not_ready": 0, "ready": 1}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


class SeqDataset(Dataset):
    def __init__(self, X, y, lengths):
        """
        X: (N, T, F)
        y: (N, T) with padding label -1
        lengths: (N,)
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.lengths = torch.from_numpy(lengths).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]


def make_loader(X, y, lengths, batch_size, shuffle):
    ds = SeqDataset(X, y, lengths)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def masked_loss(logits, y, criterion):
    """
    logits: (B, T, C)
    y: (B, T) with -1 for padding
    """
    B, T, C = logits.shape
    logits_flat = logits.view(B * T, C)
    y_flat = y.view(B * T)
    mask = y_flat != -1
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return criterion(logits_flat[mask], y_flat[mask])


def evaluate(model, loader, device, criterion):
    model.eval()
    losses = []
    all_true = []
    all_pred = []

    with torch.no_grad():
        for X_batch, y_batch, lengths in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = masked_loss(logits, y_batch, criterion)
            losses.append(loss.item())

            # collect per window predictions
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            y_flat = y_batch.view(B * T)
            mask = y_flat != -1
            if mask.sum() == 0:
                continue

            probs = torch.softmax(logits_flat[mask], dim=1)
            preds = probs.argmax(dim=1)

            all_true.append(y_flat[mask].cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    if not all_true:
        return {"loss": float(np.mean(losses)), "acc": None, "f1": None, "confusion": None}

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {"loss": float(np.mean(losses)), "acc": float(acc), "f1": float(f1), "confusion": cm}


def main():
    npz_path = Path("results/features/seq_nn_datasets.npz")
    class_w_path = Path("results/config/baseline_nn_class_weights.json")

    assert npz_path.exists(), f"Missing dataset npz: {npz_path}"
    assert class_w_path.exists(), f"Missing class weights: {class_w_path}"

    data = np.load(npz_path)

    X_train = data["X_train"]
    y_train = data["y_train"]
    len_train = data["len_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]
    len_val = data["len_val"]

    X_test = data["X_test"]
    y_test = data["y_test"]
    len_test = data["len_test"]

    in_features = X_train.shape[2]
    print("Train sequences:", X_train.shape, "Val sequences:", X_val.shape, "Test sequences:", X_test.shape)
    print("In features:", in_features)

    with class_w_path.open("r", encoding="utf8") as f:
        cw = json.load(f)

    # Handle both index keyed and label keyed formats
    if "0" in cw and "1" in cw:
        w0 = cw["0"]
        w1 = cw["1"]
    elif "not_ready" in cw and "ready" in cw:
        w0 = cw["not_ready"]
        w1 = cw["ready"]
    else:
        print("Warning: could not find expected keys in class weights JSON, using 1.0 for both classes")
        w0 = 1.0
        w1 = 1.0

    class_weights = torch.tensor([w0, w1], dtype=torch.float32)
    print("Using class weights:", class_weights.tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_seq_model(in_features=in_features)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    train_loader = make_loader(X_train, y_train, len_train, batch_size=4, shuffle=True)
    val_loader = make_loader(X_val, y_val, len_val, batch_size=4, shuffle=False)
    test_loader = make_loader(X_test, y_test, len_test, batch_size=4, shuffle=False)

    num_epochs = 80
    best_state = None
    best_val_f1 = -1.0

    logs = {"epochs": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []
        for X_batch, y_batch, lengths in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = masked_loss(logits, y_batch, criterion)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics = evaluate(model, val_loader, device, criterion)

        logs["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_f1": val_metrics["f1"],
            }
        )

        print(
            f"Epoch {epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] is not None and val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = model.state_dict()

    out_models = Path("results/models")
    out_models.mkdir(parents=True, exist_ok=True)

    if best_state is not None:
        best_path = out_models / "seq_nn.pt"
        torch.save(best_state, best_path)
        print(f"Saved best sequence model to {best_path}")
        model.load_state_dict(best_state)
    else:
        print("Warning: no best state recorded, using last epoch weights")

    # final test metrics using best model
    test_metrics = evaluate(model, test_loader, device, criterion)
    print("Test metrics:", test_metrics)

    out_metrics_dir = Path("results/metrics")
    out_metrics_dir.mkdir(parents=True, exist_ok=True)

    logs["best_val_f1"] = best_val_f1
    logs["test_metrics"] = test_metrics

    log_path = out_metrics_dir / "seq_nn_training_log.json"
    with log_path.open("w", encoding="utf8") as f:
        json.dump(logs, f, indent=2)
    print(f"Saved training log to {log_path}")

    # small summary for quick comparison script later
    summary = {
        "model_type": "seq_gru_window",
        "best_epoch": max(log["epoch"] for log in logs["epochs"]),
        "best_val_f1": best_val_f1,
        "test_acc": test_metrics["acc"],
        "test_f1": test_metrics["f1"],
        "test_confusion": test_metrics["confusion"],
    }
    summary_path = out_metrics_dir / "seq_nn_metrics.json"
    with summary_path.open("w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved seq NN summary to {summary_path}")


if __name__ == "__main__":
    main()
