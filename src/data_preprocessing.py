import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset
df = pd.read_csv("data/processed/clean_student_data.csv")

print("Dataset Loaded Successfully!\n")

# Display first 5 rows
print(df.head())

# Dataset shape
print("\nDataset Shape:", df.shape)

# Find categorical columns
categorical_columns = df.select_dtypes(include="object").columns

print("\nCategorical Columns:")
print(categorical_columns)

# Label Encoding
encoder = LabelEncoder()

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column])

print("\nCategorical columns encoded successfully!")

# Save preprocessed dataset
df.to_csv("data/processed/preprocessed_student_data.csv", index=False)

print("\nPreprocessed dataset saved successfully!")