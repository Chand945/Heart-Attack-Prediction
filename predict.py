#===========================================
# HEART ATTACK PREDICTION
# Predict New Patient
#===========================================

import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

print("=" * 50)
print("ENTER PATIENT DETAILS")
print("=" * 50)

age = int(input("Age: "))
sex = int(input("Sex (1=Male, 0=Female): "))
cp = int(input("Chest Pain Type (0-3): "))
trestbps = int(input("Resting Blood Pressure: "))
chol = int(input("Cholesterol: "))
fbs = int(input("Fasting Blood Sugar (>120 mg/dl) (1/0): "))
restecg = int(input("Rest ECG (0-2): "))
thalach = int(input("Maximum Heart Rate: "))
exang = int(input("Exercise Induced Angina (1/0): "))
oldpeak = float(input("Oldpeak: "))
slope = int(input("Slope (0-2): "))
ca = int(input("Number of Major Vessels (0-4): "))
thal = int(input("Thal (0-3): "))

# Create input array
import pandas as pd

patient = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])

# Scale input
patient = scaler.transform(patient)

# Predict
prediction = model.predict(patient)

print("\n" + "=" * 50)

if prediction[0] == 1:
    print("Prediction: High Risk of Heart Disease")
else:
    print("Prediction: Low Risk of Heart Disease")

print("=" * 50)