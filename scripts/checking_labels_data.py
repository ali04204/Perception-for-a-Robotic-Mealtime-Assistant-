import pandas as pd

train = pd.read_csv("results/features/train_data.csv")
test = pd.read_csv("results/features/test_data.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Train label counts:")
print(train["label"].value_counts())
print("Test label counts:")
print(test["label"].value_counts())
