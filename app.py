import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/student_performance_model.pkl")

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction")
st.write("Fill in the student details and click Predict.")

# -------------------------
# Basic Information
# -------------------------

col1, col2 = st.columns(2)

with col1:
    school = st.selectbox("School", ["GP", "MS"])
    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.slider("Age", 15, 22, 17)
    address = st.selectbox("Address", ["Urban", "Rural"])
    famsize = st.selectbox("Family Size", ["LE3", "GT3"])
    Pstatus = st.selectbox("Parents Status", ["Together", "Apart"])

with col2:
    Medu = st.slider("Mother Education", 0, 4, 2)
    Fedu = st.slider("Father Education", 0, 4, 2)
    traveltime = st.slider("Travel Time", 1, 4, 2)
    studytime = st.slider("Study Time", 1, 4, 2)
    failures = st.slider("Past Class Failures", 0, 4, 0)
    absences = st.number_input("Absences", 0, 100, 4)

st.subheader("Academic Scores")

G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
G2 = st.slider("Second Period Grade (G2)", 0, 20, 10)

# -------------------------
# Encode inputs
# -------------------------

school = 0 if school == "GP" else 1
sex = 0 if sex == "Female" else 1
address = 0 if address == "Urban" else 1
famsize = 0 if famsize == "LE3" else 1
Pstatus = 1 if Pstatus == "Together" else 0

# -------------------------
# Create Input Data
# -------------------------

input_data = {
    "school": school,
    "sex": sex,
    "age": age,
    "address": address,
    "famsize": famsize,
    "Pstatus": Pstatus,
    "Medu": Medu,
    "Fedu": Fedu,
    "traveltime": traveltime,
    "studytime": studytime,
    "failures": failures,
    "absences": absences,
    "G1": G1,
    "G2": G2
}


# Load training data
training_df = pd.read_csv(
    "data/processed/preprocessed_student_data.csv"
)

# Add missing features using their median values
# instead of setting everything to 0
for col in training_df.drop("G3", axis=1).columns:
    if col not in input_data:
        input_data[col] = training_df[col].median()

# Create input DataFrame
input_df = pd.DataFrame([input_data])

# Arrange columns in exactly the same order
# as the model expects
input_df = input_df[
    training_df.drop("G3", axis=1).columns
]
# -------------------------
# Prediction
# -------------------------

if st.button("Predict Final Grade"):

    prediction = model.predict(input_df)

    grade = prediction[0]

    st.success(f"🎯 Predicted Final Grade: {grade:.2f}")

    if grade >= 15:
        st.success("🟢 Excellent Performance")

    elif grade >= 10:
        st.warning("🟡 Average Performance")

    else:
        st.error("🔴 Needs Improvement")