import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)


# ==========================================
# Load Preprocessed Dataset
# ==========================================

df = pd.read_csv(
    "data/processed/preprocessed_student_data.csv"
)

print("Dataset Loaded Successfully!\n")


# ==========================================
# Features and Target
# ==========================================

X = df.drop("G3", axis=1)
y = df["G3"]


# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ==========================================
# Decision Tree Regressor
# ==========================================

decision_tree = DecisionTreeRegressor(
    random_state=42
)

decision_tree.fit(X_train, y_train)

print("\nDecision Tree Model Trained Successfully!")

y_pred_dt = decision_tree.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = mse_dt ** 0.5
r2_dt = r2_score(y_test, y_pred_dt)

print("\n----- Decision Tree Performance -----")
print("MAE:", round(mae_dt, 2))
print("MSE:", round(mse_dt, 2))
print("RMSE:", round(rmse_dt, 2))
print("R² Score:", round(r2_dt, 4))


# ==========================================
# Random Forest Regressor
# ==========================================

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")

y_pred_rf = random_forest.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mse_rf ** 0.5
r2_rf = r2_score(y_test, y_pred_rf)

print("\n----- Random Forest Performance -----")
print("MAE:", round(mae_rf, 2))
print("MSE:", round(mse_rf, 2))
print("RMSE:", round(rmse_rf, 2))
print("R² Score:", round(r2_rf, 4))


# ==========================================
# Linear Regression
# ==========================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully!")

y_pred_lr = linear_model.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = mse_lr ** 0.5
r2_lr = r2_score(y_test, y_pred_lr)

print("\n----- Linear Regression Performance -----")
print("MAE:", round(mae_lr, 2))
print("MSE:", round(mse_lr, 2))
print("RMSE:", round(rmse_lr, 2))
print("R² Score:", round(r2_lr, 4))


# ==========================================
# 5-Fold Cross Validation
# ==========================================

cv_scores = cross_val_score(
    random_forest,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("\n----- Random Forest 5-Fold Cross Validation -----")
print("Fold R² Scores:", cv_scores)
print("Mean Cross-Validation R²:",
      round(cv_scores.mean(), 4))


# ==========================================
# Model Comparison
# ==========================================

models = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest"
]

r2_scores = [
    r2_lr,
    r2_dt,
    r2_rf
]

plt.figure(figsize=(8, 5))

plt.bar(models, r2_scores)

plt.xlabel("Models")
plt.ylabel("R² Score")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "reports/model_comparison.png"
)

plt.close()


# ==========================================
# Random Forest Feature Importance
# ==========================================

feature_importance = pd.Series(
    random_forest.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n----- Top 10 Important Features -----")
print(feature_importance.head(10))


plt.figure(figsize=(10, 6))

feature_importance.head(10).plot(
    kind="bar"
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Top 10 Important Features")

plt.tight_layout()

plt.savefig(
    "reports/feature_importance.png"
)

plt.close()


# ==========================================
# Save Best Model
# ==========================================

joblib.dump(
    random_forest,
    "models/student_performance_model.pkl"
)

print("\nBest model saved successfully!")

print("\nTraining and evaluation completed successfully!")