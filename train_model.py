#===========================================
# HEART ATTACK PREDICTION
# Train Machine Learning Models
#===========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

# -----------------------------------------
# Load Dataset
# -----------------------------------------

df = pd.read_csv("data/dataset.csv")

# Remove duplicate rows
df = df.drop_duplicates()

# -----------------------------------------
# Separate Features and Target
# -----------------------------------------

X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------------------
# Train-Test Split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------------------
# Feature Scaling
# -----------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------------
# Models
# -----------------------------------------

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

best_model = None
best_accuracy = 0
best_name = ""

print("=" * 60)
print("MODEL ACCURACY")
print("=" * 60)

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(f"{name:<25} : {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

# -----------------------------------------
# Save Best Model
# -----------------------------------------

joblib.dump(best_model, "models/heart_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)
print("Model    :", best_name)
print("Accuracy :", round(best_accuracy * 100, 2), "%")

print("\nModel saved successfully!")