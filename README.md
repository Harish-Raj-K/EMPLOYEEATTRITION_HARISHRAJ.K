# Employee Attrition Prediction using Machine Learning

## Overview

This project builds a machine learning system that predicts whether an employee
is likely to leave the company, using the IBM HR Analytics Employee Attrition
dataset (1,470 employees, 35 columns). It covers data cleaning, exploratory
analysis, model training and comparison, evaluation, visualization, and
business recommendations for HR.

## Folder Contents

| File / Folder | Description |
|---|---|
| `analysis.ipynb` | Complete Jupyter Notebook covering all 7 tasks (data loading, cleaning, EDA, modeling, evaluation, visualization, HR insights). Fully executed with outputs and charts inline. |
| `HR_Attrition.csv` | The dataset used (IBM HR Analytics Employee Attrition, 1,470 rows). |
| `summary.docx` | 1-page summary written in plain, non-technical language for an HR Director. |
| `charts/` | All 5 chart images saved as `.png` files (also embedded in the notebook). |

## Charts Included

1. `chart1_attrition_by_dept_role.png` — Attrition rate by Department and Job Role
2. `chart2_income_boxplot.png` — Monthly Income: employees who left vs. stayed
3. `chart3_confusion_matrix.png` — Confusion matrix heatmap for the best model
4. `chart4_feature_importance.png` — Top 10 feature importances (best model)
5. `chart5_roc_curve.png` — ROC curve comparison across all 3 models (bonus)

## How to Run

1. Install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
2. Open `analysis.ipynb` in Jupyter Notebook or Google Colab.
3. Make sure `HR_Attrition.csv` is in the same directory as the notebook.
4. Run all cells. Charts will be regenerated into the `charts/` folder.

## Method Summary

- **Cleaning:** Dropped non-informative columns (`EmployeeNumber`, `Over18`,
  `StandardHours`, `EmployeeCount`); converted target to binary; one-hot
  encoded categorical features; scaled numeric features with `StandardScaler`.
- **Class imbalance:** Only 16.12% of employees left, so models were trained
  with `class_weight='balanced'` (Logistic Regression, Random Forest) or
  balanced sample weights (Gradient Boosting), rather than plain accuracy
  optimization.
- **Models compared:** Logistic Regression, Random Forest, Gradient Boosting
  — evaluated on Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

## Key Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression (best)** | 0.356 | 0.660 | 0.463 | **0.804** |
| Gradient Boosting | 0.407 | 0.468 | 0.436 | 0.779 |
| Random Forest | 0.500 | 0.085 | 0.145 | 0.769 |

**Best model: Logistic Regression** — highest ROC-AUC and by far the best
recall, meaning it catches the most employees who are actually at risk of
leaving. It's also the most interpretable model for HR stakeholders.

**Top predictors of attrition:** OverTime, frequent business travel, and job
role (Lab Technician, Sales Representative) — these outrank salary as
predictors, suggesting workload and burnout drive attrition at least as much
as pay.

**Highest-risk groups:** Sales Representatives (39.8% attrition), Lab
Technicians (23.9%), and employees within their first 2 years of tenure.

## Notes / Limitations

- The dataset is relatively small and imbalanced (only 237 "left" examples),
  so the model should be used to flag patterns worth a human conversation,
  not as a sole basis for individual employee decisions.
- Results reflect a single historical snapshot; the model should be retrained
  periodically as workforce conditions change.
