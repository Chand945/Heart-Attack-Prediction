import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("❤️ Heart Disease Prediction")

st.write("Enter patient details below:")

age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate", 50, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("Predict"):

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

    patient = scaler.transform(patient)

    prediction = model.predict(patient)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")