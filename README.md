# ❤️ Heart Attack Prediction using Machine Learning

A Machine Learning project that predicts whether a patient is at **high risk** or **low risk** of heart disease based on medical attributes. The project includes data analysis, model training, evaluation, and a **Streamlit web application** for real-time predictions.

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help doctors and patients make informed decisions.

This project uses Machine Learning algorithms to analyze patient health data and predict the likelihood of heart disease.

---

🚀 Live Demo

Try the application here:

🔗 https://heart-attack-prediction-ndfbjwvykrjvtjajjcvkdg.streamlit.app/


## 🚀 Features

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Preprocessing
- 🤖 Multiple Machine Learning Models
- 📈 Model Performance Comparison
- 💾 Save and Load Trained Model
- ❤️ Heart Disease Prediction
- 🌐 Interactive Streamlit Web Application

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

## 📂 Project Structure

```
Heart-Attack-Prediction/
│
├── data/
│   └── dataset.csv
│
├── models/
│   ├── heart_model.pkl
│   └── scaler.pkl
│
├── screenshots/
│
├── app.py
├── main.py
├── visualization.py
├── preprocessing.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The dataset contains medical information such as:

| Feature | Description |
|---------|-------------|
| age | Age |
| sex | Gender |
| cp | Chest Pain Type |
| trestbps | Resting Blood Pressure |
| chol | Cholesterol |
| fbs | Fasting Blood Sugar |
| restecg | Rest ECG |
| thalach | Maximum Heart Rate |
| exang | Exercise Induced Angina |
| oldpeak | ST Depression |
| slope | Slope of ST Segment |
| ca | Number of Major Vessels |
| thal | Thalassemia |
| target | Heart Disease (0 = No, 1 = Yes) |

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Heart-Attack-Prediction.git
```

### Open Project

```bash
cd Heart-Attack-Prediction
```

### Create Virtual Environment

**Windows**

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Data Exploration

```bash
python main.py
```

### Data Visualization

```bash
python visualization.py
```

### Data Preprocessing

```bash
python preprocessing.py
```

### Train Machine Learning Models

```bash
python train_model.py
```

### Predict from Terminal

```bash
python predict.py
```

### Run Streamlit Web App

```bash
streamlit run app.py
```

---

## 📈 Machine Learning Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

### Model Accuracy

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 78.69% |
| Decision Tree | 72.13% |
| Random Forest | 78.69% |
| **KNN (Best Model)** | **80.33%** |

---

## 📷 Screenshots

### Home Page

Save a screenshot as:

```
screenshots/home.png
```

Then add:

```markdown
![Home Page](screenshots/home.png)
```

---

### Prediction Result

Save another screenshot as:

```
screenshots/result.png
```

Then add:

```markdown
![Prediction Result](screenshots/result.png)
```

---

## 💡 Future Improvements

- Hyperparameter Tuning
- Cross Validation
- Feature Engineering
- Prediction Probability
- Better User Interface
- Deploy on Streamlit Community Cloud
- Docker Support

---

## 👨‍💻 Author

**Adoni Chand Basha**

B.Tech Artificial Intelligence & Machine Learning Student

GitHub: https://github.com/Chand945

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

---