import sys, pathlib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# usage:
# .\.venv\Scripts\python.exe scripts\train_baseline.py results\features\demo_windows.csv

if len(sys.argv) < 2:
    print("Give path to the windows dataset csv")
    sys.exit(1)

ds_path = pathlib.Path(sys.argv[1])
assert ds_path.exists()
ds = pd.read_csv(ds_path)

y = ds["label"].map({"not_ready":0, "ready":1}).astype(int)
X = ds.drop(columns=["label","t0","t1"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Test accuracy: {acc:.3f}")
print(classification_report(y_test, pred, target_names=["not_ready","ready"]))

model_path = ds_path.parent / "baseline_rf.joblib"
dump(clf, model_path)
print(f"Saved model to {model_path}")

imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
imp_path = ds_path.parent / "baseline_rf_feature_importance.csv"
imp.to_csv(imp_path, header=["importance"])
print(f"Saved importances to {imp_path}")
