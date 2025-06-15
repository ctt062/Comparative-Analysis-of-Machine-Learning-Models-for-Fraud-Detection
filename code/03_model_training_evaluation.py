# 03_model_training_evaluation.py
# This script loads the preprocessed data, trains multiple models
# (including a tuned Decision Tree, Logistic Regression, Random Forest, XGBoost, and a Neural Network),
# evaluates their performance on the test set, and generates a comparison report and ROC curve plot.

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import uniform, randint
from imblearn.over_sampling import SMOTE

print("--- Starting Model Training and Evaluation ---")
warnings.filterwarnings("ignore")

# --- 1. Load Preprocessed Data ---
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')
print("Loaded preprocessed data from 'data/' folder.")
print(f"Shapes: X_train={X_train.shape}, X_test={X_test.shape}")

# --- 2. SMOTE (for reference) ---
# NOTE: The notebook includes a SMOTE cell, but the RandomizedSearchCV uses
# the original imbalanced training data with `class_weight` or `scale_pos_weight` parameters.
# This suggests the author opted for class weighting over resampling.
# We will keep this cell to show the logic but use the original data for tuning
# to match the notebook's behavior.
print("\n--- SMOTE (for reference) ---")
print(f"Class distribution in y_train before SMOTE:\n{pd.Series(y_train).value_counts()}")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Shape of resampled data (not used for tuning): {X_train_resampled.shape}")
print(f"Class distribution in y_train after SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")

# --- 3. Model and Parameter Setup ---
param_dist = {
    "Decision Tree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 50),
        'min_samples_leaf': randint(1, 30)
    },
    "Logistic Regression": {
        'C': uniform(0.1, 10),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    "Random Forest": {
        'n_estimators': randint(50, 300),
        'max_depth': [None] + list(randint(5, 30).rvs(5)),
        'min_samples_split': randint(2, 50),
        'min_samples_leaf': randint(1, 30),
        'max_features': ['sqrt', 'log2', None]
    },
    "XGBoost": {
        'n_estimators': randint(50, 400),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 15),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5)
    }
}

fraud_count_train = np.sum(y_train == 1)
non_fraud_count_train = np.sum(y_train == 0)
scale_pos_weight = non_fraud_count_train / fraud_count_train if fraud_count_train > 0 else 1

base_models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
}
print(f"\nCalculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")

# --- 4. Hyperparameter Tuning with 5-Fold Cross-Validation ---
print("\n--- Starting Hyperparameter Tuning with 5-Fold Cross-Validation ---")
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_estimators = {}
tuning_results_list = []
n_iter_search = 20
tuning_scoring_metric = 'roc_auc'

for name in param_dist.keys():
    print(f"\n--- Tuning {name} ---")
    start_time = time.time()
    random_search = RandomizedSearchCV(
        estimator=base_models[name],
        param_distributions=param_dist[name],
        n_iter=n_iter_search,
        scoring=tuning_scoring_metric,
        cv=cv_strategy,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    # Using original imbalanced data with class weights, as in the notebook
    random_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    print(f"Tuning completed in {tuning_time:.2f} seconds.")

    best_estimators[name] = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    print(f"Best Parameters for {name}: {best_params}")
    print(f"Best Cross-Validation {tuning_scoring_metric.upper()} for {name}: {best_score:.4f}")

    tuning_results_list.append({
        "Model": name, "Best CV ROC_AUC": best_score, "Best Params": best_params, "Tuning Time (s)": tuning_time
    })

# --- 5. Neural Network Training ---
print("\n--- Training and Evaluating Neural Network (Fixed Architecture) ---")
input_shape = (X_train.shape[1],)
total_samples = len(y_train)
weight_for_0 = (1 / non_fraud_count_train) * (total_samples / 2.0)
weight_for_1 = (1 / fraud_count_train) * (total_samples / 2.0)
nn_class_weight = {0: weight_for_0, 1: weight_for_1}

nn_model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")],
    name="Fraud_Detection_NN"
)
nn_model.compile(optimizer='adam', loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                          tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='roc_auc')])
print("\nNeural Network Model Summary:")
nn_model.summary()
start_time = time.time()
history = nn_model.fit(
    X_train, y_train, batch_size=2048, epochs=15, validation_split=0.1,
    class_weight=nn_class_weight, verbose=2
)
nn_training_time = time.time() - start_time
best_estimators["Neural Network"] = nn_model
print(f"NN Training completed in {nn_training_time:.2f} seconds.")

# --- 6. Final Evaluation on Test Set ---
print("\n--- Evaluating All Models on the Hold-Out Test Set ---")
final_results_list = []
model_probabilities = {}

# Evaluate sklearn/XGB models
for name, model in best_estimators.items():
    if name == "Neural Network": continue
    print(f"\nEvaluating {name}...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    model_probabilities[name] = y_pred_proba
    eval_time = time.time() - start_time
    
    # CORRECTED LOGIC to get the tuning time
    training_time = 0  # Default value
    for result in tuning_results_list:
        if result['Model'] == name:
            training_time = result.get('Tuning Time (s)', 0)
            break
            
    final_results_list.append({
        "Model": name, "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred), "ROC AUC": roc_auc_score(y_test, y_pred_proba),
        "Training/Tuning Time (s)": training_time, # Using the corrected lookup
        "Prediction Time (s)": eval_time
    })

# Evaluate NN model
print("\nEvaluating Neural Network...")
start_time = time.time()
loss, acc, prec, rec, roc_auc = nn_model.evaluate(X_test, y_test, verbose=0)
y_pred_proba_nn = nn_model.predict(X_test, verbose=0).flatten()
y_pred_nn = (y_pred_proba_nn > 0.5).astype(int)
eval_time = time.time() - start_time
model_probabilities["Neural Network"] = y_pred_proba_nn
final_results_list.append({
    "Model": "Neural Network", "Accuracy": acc, "Precision": prec, "Recall": rec,
    "F1 Score": f1_score(y_test, y_pred_nn), "ROC AUC": roc_auc,
    "Training/Tuning Time (s)": nn_training_time, "Prediction Time (s)": eval_time
})

# --- 7. Display and Plot Results ---
final_results_df = pd.DataFrame(final_results_list).set_index("Model")
final_results_df_sorted = final_results_df.sort_values(by='ROC AUC', ascending=False)
print("\n--- Comparison of Final Model Results ---")
print(final_results_df_sorted.to_string(float_format="%.4f"))

# Plot ROC Curves
plt.figure(figsize=(12, 9))
for model_name in final_results_df_sorted.index:
    probabilities = model_probabilities[model_name]
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc_score = final_results_df_sorted.loc[model_name, 'ROC AUC']
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (AUC = 0.50)')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison (Test Set)', fontsize=14)
plt.legend(loc="lower right", fontsize=10); plt.grid(alpha=0.5)
plt.savefig('graph/03_roc_curve_comparison.png')
plt.close()
print("\nSaved 'roc_curve_comparison.png' to 'graph' folder.")

# Detailed Report for Best Model
best_model_name = final_results_df_sorted.index[0]
print(f"\n--- Detailed Report for Best Model ({best_model_name}) ---")
if best_model_name == "Neural Network":
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_nn)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred_nn, digits=4)}")
else:
    best_model_obj = best_estimators[best_model_name]
    y_pred_best = best_model_obj.predict(X_test)
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_best)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred_best, digits=4)}")

print("\n--- Model Training and Evaluation Complete ---")