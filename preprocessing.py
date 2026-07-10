#===========================================
# HEART ATTACK PREDICTION
# Data Preprocessing
#===========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("data/dataset.csv")

print("=" * 60)
print("ORIGINAL DATASET SHAPE")
print("=" * 60)
print(df.shape)

# -----------------------------------------
# Remove Duplicate Rows
# -----------------------------------------
duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

# -----------------------------------------
# Check Missing Values
# -----------------------------------------
print("\nMissing Values")
print(df.isnull().sum())

# -----------------------------------------
# Separate Features and Target
# -----------------------------------------
X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures Shape :", X.shape)
print("Target Shape   :", y.shape)

# -----------------------------------------
# Split Dataset
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data :", X_train.shape)
print("Testing Data  :", X_test.shape)

# -----------------------------------------
# Feature Scaling
# -----------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed Successfully.")

print("\nData Preprocessing Completed Successfully!")