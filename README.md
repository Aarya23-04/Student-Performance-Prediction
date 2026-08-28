# 🎓 Student Performance Prediction

A Machine Learning project that predicts a student's final academic grade using demographic, social, and academic information.

The project uses multiple regression algorithms and provides an interactive Streamlit web application for making predictions.

---

## 📌 Project Overview

Student performance can be influenced by several factors such as previous grades, study time, failures, absences, family background, and other student-related attributes.

This project analyzes these factors and builds Machine Learning models to predict the student's final grade (G3).

The trained model is integrated with a Streamlit web application where users can enter student information and receive a predicted final grade.

---

## 🎯 Objectives

- Analyze student performance data.
- Perform Exploratory Data Analysis (EDA).
- Preprocess categorical and numerical data.
- Train multiple Machine Learning regression models.
- Compare model performance using evaluation metrics.
- Identify important features affecting student performance.
- Build an interactive prediction web application.

---

## 🗂️ Dataset

The project uses the Student Performance Dataset containing information about students, including:

- School
- Gender
- Age
- Family size
- Parents' education
- Study time
- Travel time
- Past class failures
- Absences
- First period grade (G1)
- Second period grade (G2)
- Final grade (G3)

The target variable is:

**G3 — Final Grade**

---

## 🔍 Exploratory Data Analysis

The project includes EDA to understand patterns in the dataset.

Analysis includes:

- Dataset structure and dimensions
- Missing-value analysis
- Statistical summary
- Final grade distribution
- Study time vs final grade
- Absences vs final grade
- Correlation analysis
- Top features related to final grade

---

## 🤖 Machine Learning Models

Three regression algorithms were evaluated:

### 1. Linear Regression

Used as a baseline regression model.

### 2. Decision Tree Regressor

Used to capture non-linear relationships between student characteristics and final grade.

### 3. Random Forest Regressor

An ensemble learning method that combines multiple decision trees to improve prediction performance.

Random Forest was selected as the final model based on its evaluation performance.

---

## 📊 Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Random Forest achieved the best performance among the tested models.

The project also uses **5-Fold Cross-Validation** to evaluate model stability.

---

## ⭐ Feature Importance

Random Forest feature importance was used to identify the features that contribute most to the prediction.

The top features can be viewed in:

```text
reports/feature_importance.png