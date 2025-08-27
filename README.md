# sba-loan-default-prediction
Machine learning models to predict SBA loan defaults with profit-based evaluation (MSBA 207 final project)
SBA Loan Default Prediction (MSBA 207 Final Project)

This project develops and compares multiple machine learning models to predict Small Business Administration (SBA) loan defaults.
Because defaulting on a risky loan is far more costly than missing a safe approval, evaluation emphasizes profit impact and cost-sensitive decision thresholds rather than accuracy alone.

🔎 Problem Statement

Given application and loan attributes, predict whether a loan will be Paid in Full or Charged Off.
The business objective is to minimize expected loss by avoiding risky approvals while preserving profitable lending volume.

Asymmetric costs (from course brief): roughly +5% gain if a loan is repaid vs –25% loss if it defaults.

Goal: choose both a model and an operating threshold that maximize total profit.

📊 Dataset (SBA National)

Typical fields include DisbursementGross, GrAppv, SBA_Appv, Term, UrbanRural, NewExist, RevLineCr, LowDoc, and status label MIS_Status (Paid in Full / Charged Off).
The outcome is imbalanced (defaults ≈ 15–20% depending on filters).

Note: The raw dataset is not included in this repo due to size/licensing. Instructions below explain where to place it locally.

🧰 Methods & Models

Data Cleaning & Feature Engineering

Currency → numeric (remove symbols/commas), type fixes

Categorical encoding for RevLineCr, LowDoc, UrbanRural, etc.

Missing value handling; scaling for algorithms that require it

Stratified train/validation split

Modeling (with cross-validation and grid search)

Logistic Regression (L1/L2/ElasticNet, SAGA)

k-Nearest Neighbors (kNN)

Decision Tree

Random Forest

Gradient Boosting

Bagging (base estimator: Decision Tree)

Neural Network (MLP with early stopping)

Linear Discriminant Analysis (LDA)

Evaluation

ROC-AUC

Lift & Gain charts

Profit analysis under asymmetric costs

Threshold selection to maximize total profit

✅ Key Findings

Bagging achieved the highest net profit on the validation set, outperforming other models under the asymmetric cost structure.

Profit-optimal policy: lend to the safest ~70–75% of applicants, rather than everyone.

Recommended operating threshold (from notebook): approve only if predicted repayment probability ≥ ~0.83 (i.e., default probability < ~0.17).

Tree-ensemble methods (Bagging / Random Forest / Gradient Boosting) provided stronger rank-ordering than linear baselines in this dataset.

Exact AUC, lift, and profit numbers are shown in the notebook’s Results Summary, along with multi-model ROC and decile lift/gain plots.
