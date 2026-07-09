import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/student_performance_model.pkl")

# Load the processed dataset
df = pd.read_csv("data/processed/preprocessed_student_data.csv")

# Use the first student's data (excluding G3)
sample = df.drop("G3", axis=1).iloc[[0]]

# Make prediction
prediction = model.predict(sample)

print("Predicted Final Grade (G3):", round(prediction[0], 2))