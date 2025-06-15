# 02_data_preprocessing.py
# This script loads the cleaned data, preprocesses it by handling outliers,
# scaling features, applying one-hot encoding, and performing PCA.
# Finally, it splits the data into training and testing sets.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

print("--- Starting Data Preprocessing ---")

# --- 1. Load Cleaned Data ---
df = pd.read_pickle('data/cleaned_df.pkl')
print("Loaded cleaned data from 'data/cleaned_df.pkl'.")

# --- 2. Feature Engineering / Dropping ---
columns_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
columns_present = [col for col in columns_to_drop if col in df.columns]
df_processed = df.drop(columns=columns_present)
print(f"Dropped columns: {columns_present}")

# --- 3. Identify Feature Types ---
numerical_features = df_processed.select_dtypes(include=np.number).columns.tolist()
if 'isFraud' in numerical_features:
    numerical_features.remove('isFraud')
categorical_features = df_processed.select_dtypes(include='object').columns.tolist()
print(f"Numerical features identified: {numerical_features}")
print(f"Categorical features identified: {categorical_features}")

# --- 4. Visualize Numerical Features Before Outlier Removal ---
print("\nVisualizing numerical feature distributions (before outlier removal)...")
df_numeric = df[numerical_features].copy()
scaler_viz = StandardScaler()
scaled_numeric_data = scaler_viz.fit_transform(df_numeric)
df_scaled_numeric = pd.DataFrame(scaled_numeric_data, columns=numerical_features)
plt.figure(figsize=(15, 6))
sns.boxplot(data=df_scaled_numeric)
plt.title('Box Plots of Scaled Numerical Features (Before Outlier Removal)', fontsize=16)
plt.xlabel('Numerical Features', fontsize=12)
plt.ylabel('Scaled Value (StandardScaler)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('graph/02_boxplot_before_outlier_removal.png')
plt.close()
print("Saved 'boxplot_before_outlier_removal.png' to 'graph' folder.")

# --- 5. Outlier Removal using IQR ---
df_no_outliers = df_processed.copy()
initial_rows = df_no_outliers.shape[0]
print(f"\nShape before outlier removal: {df_no_outliers.shape}")
outliers_removed_details = {}
for col in numerical_features:
    rows_before_col = df_no_outliers.shape[0]
    print(f"\nProcessing column: {col}")
    Q1 = df_no_outliers[col].quantile(0.25)
    Q3 = df_no_outliers[col].quantile(0.75)
    IQR_val = Q3 - Q1
    # NOTE: The notebook uses a multiplier of 20, which is very permissive.
    # A standard value is 1.5. This is kept as-is to not change the code's content.
    lower_bound = Q1 - 20 * IQR_val
    upper_bound = Q3 + 20 * IQR_val
    df_no_outliers = df_no_outliers[
        (df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)
    ]
    rows_after_col = df_no_outliers.shape[0]
    removed_count = rows_before_col - rows_after_col
    outliers_removed_details[col] = removed_count
    print(f"Removed {removed_count} outliers based on {col}. Current shape: {df_no_outliers.shape}")
total_rows_removed = initial_rows - df_no_outliers.shape[0]
percentage_removed = (total_rows_removed / initial_rows) * 100 if initial_rows > 0 else 0
print(f"\nFinal shape after outlier removal: {df_no_outliers.shape}")
print(f"Total rows removed: {total_rows_removed} ({percentage_removed:.2f}%)")

# --- 6. Visualize Numerical Features After Outlier Removal ---
print("\nVisualizing numerical feature distributions (after outlier removal)...")
if not df_no_outliers.empty:
    df_numeric_no_outliers = df_no_outliers.loc[:, numerical_features].copy()
    scaler_viz_after = StandardScaler()
    scaled_numeric_data_after = scaler_viz_after.fit_transform(df_numeric_no_outliers)
    df_scaled_numeric_no_outliers = pd.DataFrame(scaled_numeric_data_after, columns=numerical_features)
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df_scaled_numeric_no_outliers)
    plt.title('Box Plots of Scaled Numerical Features (After Outlier Removal)', fontsize=16)
    plt.xlabel('Numerical Features', fontsize=12)
    plt.ylabel('Scaled Value (StandardScaler)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('graph/02_boxplot_after_outlier_removal.png')
    plt.close()
    print("Saved 'boxplot_after_outlier_removal.png' to 'graph' folder.")
else:
    print("Warning: All data removed after outlier detection. Cannot generate plot.")

# --- 7. Final Feature and Target Split ---
X = df_no_outliers.drop('isFraud', axis=1)
y = df_no_outliers['isFraud']
print(f"\nShape of features (X) before preprocessing: {X.shape}")
print(f"Shape of target (y): {y.shape}")

# --- 8. Preprocessing Pipeline ---
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)
X_processed = preprocessor.fit_transform(X)
print(f"Shape of X after scaling/encoding: {X_processed.shape}")

# --- 9. PCA (Principal Component Analysis) ---
pca_analyzer = PCA(random_state=42)
pca_analyzer.fit(X_processed)
cumulative_explained_variance = np.cumsum(pca_analyzer.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1
print(f"Number of components needed for >= 95% variance: {n_components_95}")
pca = PCA(n_components=n_components_95, random_state=42)
X_pca = pca.fit_transform(X_processed)
print(f"\nShape of X after PCA: {X_pca.shape}")

# --- 10. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData split into training and testing sets.")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")

# --- 11. Save Processed Data for Modeling ---
np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)
print("\nProcessed training and testing data saved to 'data/' folder.")
print("--- Data Preprocessing Complete ---")