# Machine Learning Credit Risk Prediction

## Overview
This project presents an end-to-end machine learning pipeline for credit risk classification using the Statlog German Credit dataset. The objective is to predict whether a loan applicant represents a good or bad credit risk through preprocessing, feature selection, dimensionality reduction, and model comparison.

## Dataset
- Source: UCI Machine Learning Repository
- Dataset: Statlog German Credit Data
- Instances: 1000
- Features: 20 (categorical and numerical)
- Target Distribution:
  - Good credit: 700
  - Bad credit: 300
- Target Encoding:
  - 0 → Good credit
  - 1 → Bad credit

## Data Preprocessing
- No missing values detected
- No duplicate records found
- Categorical feature encoding using LabelEncoder
- Feature correlation analysis
- Removal of weak and highly correlated features
- Outlier removal using IQR method (dataset reduced after cleaning)
- Feature scaling using StandardScaler
- Class imbalance handling using SMOTE

## Feature Engineering
- Dimensionality reduction using Principal Component Analysis (PCA)
- 10 principal components retained
- Feature selection based on correlation with target

## Models Implemented
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes
- Neural Network (MLP)
- K-Nearest Neighbors (KNN)
- Voting Ensemble (Hard & Soft)
- Gradient Boosting

## Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The system prioritizes recall for the high-risk class to minimize false negatives in credit risk detection.

## Key Findings
- Random Forest achieved the highest overall accuracy (~80%) without PCA
- PCA improved minority class recall for several models
- Ensemble methods provided balanced performance across classes
- SMOTE improved detection of bad credit applicants

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Imbalanced-learn

## Project Structure
