import pathlib
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import matplotlib.pyplot as plt


def load_train_test():
    repo = pathlib.Path(__file__).resolve().parents[1]
    feat_dir = repo / "results" / "features"

    train_path = feat_dir / "train_data.csv"
    test_path = feat_dir / "test_data.csv"

    assert train_path.exists(), f"Missing {train_path}"
    assert test_path.exists(), f"Missing {test_path}"

    print(f"Loading {train_path.name} and {test_path.name}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df, feat_dir


def prepare_xy(df):
    assert "label" in df.columns, "label column missing"

    # normalize labels
    lab = df["label"].astype(str).str.strip().str.lower()

    print("Raw label values:")
    print(lab.value_counts())

    mapping = {"not_ready": 0, "ready": 1}
    mask = lab.isin(mapping.keys())

    if not mask.any():
        raise ValueError("No rows with labels 'ready' or 'not_ready' found")

    if (~mask).any():
        print("Dropping rows with other labels:")
        print(lab[~mask].value_counts())

    lab = lab[mask]
    df = df.loc[mask].copy()

    y = lab.map(mapping).values

    drop_cols = ["frame", "time_sec", "video", "label"]
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feat_cols].values

    print("Using feature columns:", feat_cols)
    print("Class distribution after filtering:")
    print(pd.Series(y).value_counts())

    return X, y, feat_cols


def main():
    train_df, test_df, feat_dir = load_train_test()

    X_train, y_train, feat_cols = prepare_xy(train_df)
    X_test, y_test, _ = prepare_xy(test_df)

    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)
    print("Training done.")

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"\nTest accuracy: {acc:.3f}")

    print("\nConfusion matrix (rows true, cols predicted):")
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_test, pred, target_names=["not_ready", "ready"]))

    # save model
    model_path = feat_dir / "baseline_rf.joblib"
    dump(clf, model_path)
    print(f"\nSaved model to {model_path}")

    # feature importance csv
    imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    imp_path = feat_dir / "baseline_rf_feature_importance.csv"
    imp.to_csv(imp_path, header=["importance"])
    print(f"Saved importances to {imp_path}")

    # plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["not_ready", "ready"])
    ax.set_yticklabels(["not_ready", "ready"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    cm_path = feat_dir / "confusion_matrix.png"
    plt.tight_layout()
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix plot to {cm_path}")

    # plot feature importance bar chart
    fig, ax = plt.subplots()
    ax.bar(feat_cols, clf.feature_importances_)
    ax.set_ylabel("Importance")
    ax.set_title("Feature importance (RandomForest)")
    ax.set_xticks(range(len(feat_cols)))
    ax.set_xticklabels(feat_cols, rotation=45, ha="right")

    plt.tight_layout()
    fi_path = feat_dir / "feature_importance.png"
    fig.savefig(fi_path, dpi=300)
    plt.close(fig)
    print(f"Saved feature importance plot to {fi_path}")


if __name__ == "__main__":
    main()
