import joblib

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pandas as pd

from sklearn.model_selection import train_test_split

# Load the preprocessed dataset
df = pd.read_csv("data/processed/preprocessed_student_data.csv")

print("Dataset Loaded Successfully!\n")

# Features (Input)
X = df.drop("G3", axis=1)

# Target (Output)
y = df["G3"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ==============================
# Decision Tree Regressor
# ==============================

decision_tree = DecisionTreeRegressor(random_state=42)

decision_tree.fit(X_train, y_train)

print("\nDecision Tree Model Trained Successfully!")

y_pred_dt = decision_tree.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("\n----- Decision Tree Performance -----")
print("Mean Absolute Error (MAE):", round(mae_dt, 2))
print("Mean Squared Error (MSE):", round(mse_dt, 2))
print("R² Score:", round(r2_dt, 4))

# ==============================
# Random Forest Regressor
# ==============================

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")

y_pred_rf = random_forest.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n----- Random Forest Performance -----")
print("Mean Absolute Error (MAE):", round(mae_rf, 2))
print("Mean Squared Error (MSE):", round(mse_rf, 2))
print("R² Score:", round(r2_rf, 4))

# Create Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully!")

# Make predictions
y_pred = linear_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n----- Linear Regression Performance -----")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("R² Score:", round(r2, 4))

# Save the best model (Random Forest)
joblib.dump(random_forest, "models/student_performance_model.pkl")

print("\nBest model saved successfully!")