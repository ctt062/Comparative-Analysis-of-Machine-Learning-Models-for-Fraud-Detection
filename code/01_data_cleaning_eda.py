# 01_data_cleaning_eda.py
# This script loads the raw data, performs basic cleaning and EDA,
# and saves the cleaned DataFrame for the next step.

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("--- Starting Data Cleaning and EDA ---")

# --- 1. Setup and Configuration ---
warnings.filterwarnings("ignore")
# Create directories for outputs if they don't exist
os.makedirs('graph', exist_ok=True)
os.makedirs('data', exist_ok=True)

# --- 2. Load Data ---
# The user specified that the CSV file is located in the 'code' folder.
DATA_PATH = "code/onlinefraud.csv"
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded successfully from '{DATA_PATH}'!")

# ------------------------------------------------------------------
# --- SAMPLE DATA (NEW SECTION TO REDUCE RUNTIME) ---
# ------------------------------------------------------------------
print("\n--- Sampling the Dataset ---")
original_shape = df.shape

# Randomly sample 10% of the data. Using random_state for reproducibility.
#df = df.sample(frac=1, random_state=42)
#df.to_csv("code/onlinefraud.csv", index=False)  # Overwrite with the smaller file

print(f"Original dataset shape: {original_shape}")
print(f"Sampled 10% of the data. New dataset shape: {df.shape}")
# ------------------------------------------------------------------


# --- 3. Understand the Data ---
print("\n1. First 5 Rows of the (Sampled) Dataset:")
print(df.head())

print("\n2. (Sampled) Dataset Dimensions (Rows, Columns):")
print(df.shape)

print("\n3. Column Information (Data Types, Non-Null Counts):")
df.info()

print("\n4. Missing Values Count per Column:")
print(df.isnull().sum())

# Drop rows with any missing values
df = df.dropna()
print("\nRows with missing values have been dropped.")

print("\n5. Descriptive Statistics for Numerical Columns:")
print(df.describe().T)

print("\n6. Descriptive Statistics for Categorical Columns:")
print(df.describe(include=['object']))

print("\n7. Distribution of the Target Variable ('isFraud'):")
fraud_counts = df['isFraud'].value_counts()
print(fraud_counts)
fraud_percentage = (fraud_counts.get(1, 0) / df.shape[0]) * 100
print(f"\nPercentage of Fraudulent Transactions: {fraud_percentage:.4f}%")

print("\n8. Distribution of Transaction Types ('type'):")
print(df['type'].value_counts())

# --- 4. Visualizations ---
print("\nGenerating and saving initial visualizations (on sampled data)...")

# Set plot style
sns.set(style="whitegrid")

# Plot Fraud Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='isFraud', data=df, palette='viridis')
plt.title('Distribution of Fraudulent Transactions (0: No, 1: Yes)')
plt.xlabel("Is Fraud?")
plt.ylabel("Number of Transactions")
plt.xticks([0, 1], ['Non-Fraudulent', 'Fraudulent'])
plt.savefig('graph/01_fraud_distribution.png')
plt.close()
print("Saved 'fraud_distribution.png' to 'graph' folder.")

# Plot Transaction Type Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='type', data=df, order=df['type'].value_counts().index, palette='magma')
plt.title('Distribution of Transaction Types')
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.savefig('graph/01_transaction_type_distribution.png')
plt.close()
print("Saved 'transaction_type_distribution.png' to 'graph' folder.")

# --- 5. Save Cleaned Data ---
# Save the cleaned dataframe to be used by the next script
df.to_pickle('data/cleaned_df.pkl')
print("\nCleaned (and sampled) DataFrame saved to 'data/cleaned_df.pkl'.")
print("--- Data Cleaning and EDA Complete ---")