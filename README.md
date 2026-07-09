# 🎓 Student Performance Prediction Using Machine Learning

## 📌 Overview

This project predicts the final grade (G3) of students using Machine Learning algorithms.

The project includes:

- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Model Training
- Model Evaluation
- Streamlit Web Application

---

## 📂 Dataset

- UCI Student Performance Dataset
- 395 Student Records
- 33 Features
- Target Variable: G3 (Final Grade)

---

## 🚀 Features

- Data Cleaning
- Label Encoding
- Model Comparison
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Best Model Selection
- Student Grade Prediction
- Streamlit Web Application

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

## 🤖 Machine Learning Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor ⭐

---

## 📊 Model Performance

| Model | R² Score |
|--------|---------:|
| Linear Regression | 0.7546 |
| Decision Tree | 0.7241 |
| Random Forest | **0.83** |

Random Forest was selected as the final model because it achieved the best performance.

---

## 📁 Project Structure

```text
Student-Performance-Prediction/
│
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── student_performance_model.pkl
├── notebooks/
│   └── EDA.ipynb
├── reports/
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ▶️ Installation

```bash
git clone <repository-url>

cd Student-Performance-Prediction

pip install -r requirements.txt

streamlit run app.py
```

---

## 📸 Application

The Streamlit application allows users to:

- Enter student information
- Predict the final student grade
- Display performance level

---

## 📈 Future Improvements

- Better User Interface
- Feature Importance Visualization
- Model Comparison Dashboard
- Online Deployment using Streamlit Community Cloud

---

## 👨‍💻 Developer

**Aarya**

Final Year Electronics & Telecommunication Engineering Student

Interested in Data Science, Machine Learning and Artificial Intelligence.