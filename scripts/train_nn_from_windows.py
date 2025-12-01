import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from baseline_nn import build_window_mlp


def make_loader(X, y, batch_size=64, shuffle=False):
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item()
            total_batches += 1

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    if total_batches == 0:
        return {
            "loss": None,
            "acc": None,
            "f1": None,
            "confusion": None,
        }

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = total_loss / total_batches
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds).tolist()

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "f1": float(f1),
        "confusion": cm,
    }


def main(args):
    npz_path = Path(args.dataset)
    weights_path = Path(args.class_weights)

    assert npz_path.exists(), f"Missing dataset npz: {npz_path}"
    assert weights_path.exists(), f"Missing class weights json: {weights_path}"

    data = np.load(npz_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    in_features = X_train.shape[1]
    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)
    print("In features:", in_features)

    with weights_path.open("r", encoding="utf8") as f:
        info = json.load(f)

    # Build weight tensor in class index order [0, 1]
    w0 = info["class_weights"]["0"]
    w1 = info["class_weights"]["1"]
    class_weights = torch.tensor([w0, w1], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_window_mlp(in_features=in_features)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = make_loader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size=args.batch_size, shuffle=False)

    best_state = None
    best_val_f1 = -1.0
    logs = {"epochs": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        train_loss = running_loss / max(num_batches, 1)

        val_metrics = evaluate(model, val_loader, device, criterion)

        logs["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_f1": val_metrics["f1"],
            }
        )

        print(
            f"Epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] is not None and val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = model.state_dict()

    out_models = Path("results/models")
    out_metrics = Path("results/metrics")
    out_models.mkdir(parents=True, exist_ok=True)
    out_metrics.mkdir(parents=True, exist_ok=True)

    if best_state is not None:
        best_model_path = out_models / "baseline_nn.pt"
        torch.save(best_state, best_model_path)
        print(f"Saved best model to {best_model_path}")
        model.load_state_dict(best_state)
    else:
        print("Warning: no best state recorded, using last epoch model")

    test_metrics = evaluate(model, test_loader, device, criterion)
    print("Test metrics:", test_metrics)

    logs["best_val_f1"] = float(best_val_f1)
    logs["test_metrics"] = test_metrics

    log_path = out_metrics / "baseline_nn_training_log.json"
    with log_path.open("w", encoding="utf8") as f:
        json.dump(logs, f, indent=2)
    print(f"Saved training log to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="results/features/nn_windows_datasets.npz",
        help="Path to npz with X_train, y_train, X_val, y_val, X_test, y_test",
    )
    parser.add_argument(
        "--class_weights",
        default="results/config/baseline_nn_class_weights.json",
        help="Path to JSON with class_weights dict",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
