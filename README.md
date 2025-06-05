# 🔍 Comparative Analysis of Machine Learning Models for Fraud Detection

(still uploading my code)

This project applies supervised learning techniques to detect rare fraudulent transactions in a highly imbalanced financial dataset (~0.8% fraud rate), completed for the **Predictive Analytics (IEDA3560)** course at **HKUST**.

## 🧠 Project Objectives
- Identify and classify fraudulent financial transactions using machine learning
- Address data imbalance using SMOTE
- Compare model performance focusing on **Recall** and **ROC AUC** due to the rarity of fraud cases

## ⚙️ Techniques & Tools
- **Languages & Libraries:** Python, Scikit-learn, XGBoost, imbalanced-learn, pandas, seaborn, matplotlib
- **Data Preprocessing:** 
  - IQR method for outlier handling (retain long-tail fraud patterns)
  - SMOTE to balance minority class
- **Models Compared:** 
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - MLP Neural Network

## 📊 Key Results
| Model             | Recall     | Precision | F1-Score | ROC AUC |
|------------------|------------|-----------|----------|---------|
| Neural Network    | 97.5%      | —         | —        | 0.9964  |
| XGBoost (Best F1) | —          | 12.4%     | 0.2171   | —       |

- Non-linear models (XGBoost, MLP) outperformed linear models
- ROC curves and confusion matrices confirmed robustness

## 📂 Files
- `notebooks/`: Jupyter notebooks for model training and evaluation
- `report.pdf`: Final project report
- `slides.pdf`: Presentation slides

## 🧰 Skills Demonstrated
Machine Learning · Feature Engineering · Imbalanced Classification · Fraud Detection · Model Evaluation

## 🔗 Connect
[LinkedIn](https://www.linkedin.com/in/tin-tak-chong) • [Email](mailto:chongtt062@gmail.com)

