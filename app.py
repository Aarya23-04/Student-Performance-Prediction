import streamlit as st
import pandas as pd
import joblib

# Prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

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

# -----------------------------
# Prediction
# -----------------------------

if st.button("🎯 Predict Final Grade"):

    prediction = model.predict(input_df)

    grade = float(prediction[0])

    # Keep grade within valid range
    grade = max(0, min(20, grade))

    # Estimate prediction uncertainty using individual trees
    tree_predictions = [
        tree.predict(input_df)[0]
        for tree in model.estimators_
    ]

    prediction_std = pd.Series(tree_predictions).std()

    lower_bound = max(0, grade - prediction_std)
    upper_bound = min(20, grade + prediction_std)

  # Store prediction history
history_record = {
    "Study Time": studytime,
    "Absences": absences,
    "G1": G1,
    "G2": G2,
    "Predicted Grade": round(grade, 2)
}

st.session_state.prediction_history.append(history_record)

st.session_state.prediction_history.append(history_record)
st.divider()

st.subheader("📊 Prediction Result")

col1, col2 = st.columns(2)

with col1:
        st.metric(
            label="Predicted Final Grade",
            value=f"{grade:.2f} / 20"
        )

        st.info(
            f"Estimated prediction range: "
            f"{lower_bound:.2f} – {upper_bound:.2f} / 20"
        )

with col2:
        if grade >= 15:
            status = "🟢 Excellent"
        elif grade >= 10:
            status = "🟡 Average"
        else:
            status = "🔴 Needs Improvement"

        st.metric(
            label="Performance Level",
            value=status
        )



with col2:
        if grade >= 15:
            status = "🟢 Excellent"
        elif grade >= 10:
            status = "🟡 Average"
        else:
            status = "🔴 Needs Improvement"

        st.metric(
            label="Performance Level",
            value=status
        )

    # Performance message
if grade >= 15:
        st.success(
            "Excellent performance! Keep maintaining your current study habits."
        )

elif grade >= 10:
        st.warning(
            "Average performance. Increasing study time and reducing absences "
            "may help improve the final grade."
        )

else:
        st.error(
            "The predicted performance is below the passing range. "
            "Focus on regular study, attendance, and improving previous grades."
        )

    # Progress bar
st.write("### 📈 Predicted Performance")

progress = grade / 20

st.progress(progress)

        # -----------------------------
    # Student Summary
    # -----------------------------

st.divider()

st.subheader("👤 Student Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
        st.write("**School:**", "GP" if school == 0 else "MS")
        st.write("**Gender:**", "Female" if sex == 0 else "Male")
        st.write("**Age:**", age)

with summary_col2:
        st.write("**Address:**", "Urban" if address == 0 else "Rural")
        st.write("**Family Size:**", "LE3" if famsize == 0 else "GT3")
        st.write("**Parents:**", "Together" if Pstatus == 1 else "Apart")

with summary_col3:
        st.write("**Study Time:**", studytime)
        st.write("**Absences:**", absences)
        st.write("**G1 Score:**", G1)
        st.write("**G2 Score:**", G2)

        # ==============================
# Model Performance
# ==============================

st.divider()

st.subheader("📊 Model Performance")

performance_df = pd.read_csv("reports/model_performance.csv")

st.dataframe(
    performance_df,
    use_container_width=True
)

# ==============================
# Model Comparison
# ==============================

st.subheader("📈 Model Comparison")

st.bar_chart(
    performance_df.set_index("Model")["R2 Score"]
)

# ==============================
# Feature Importance
# ==============================

st.divider()

st.subheader("🔍 Feature Importance")

st.write(
    "The chart below shows which factors contributed most "
    "to the Random Forest prediction."
)

st.image(
    "reports/feature_importance.png",
    use_container_width=True
)

# ==============================
# Prediction History
# ==============================

st.divider()

st.subheader("📜 Prediction History")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

else:
    st.info("No predictions have been made yet.")