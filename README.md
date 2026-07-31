# Heart Disease Classification Model

Comparative machine learning study for AASD 4001 (Mathematical Concepts for Machine Learning), George Brown Polytechnic — evaluating five supervised classification algorithms on a 10,000-patient dataset with significant class imbalance (80/20).

## Overview

- **Dataset:** 10,000 patient records, 21 raw features → 15 encoded features after preprocessing
- **Models compared:** Decision Tree, Logistic Regression, Random Forest, SGD, SVM
- **Primary metric:** Recall (Class 1) — chosen over accuracy, since a model predicting "no heart disease" for every patient would score 80% accuracy while catching zero real cases
- **Key result:** SMOTE oversampling combined with feature ablation drove the best result to **72% recall** (SGD)

## Methodology

1. **EDA** — correlation heatmap, stratified boxplots, and stacked bar charts to understand feature distributions and class separation
2. **Preprocessing** — removed 5 low-value/high-missing features, imputed remaining missing values (median/mode), one-hot encoded categorical variables, scaled numeric features
3. **Baseline modeling** — trained all five classifiers with `class_weight='balanced'`
4. **Hyperparameter tuning** — GridSearchCV across all five models, optimized for recall
5. **Feature ablation** — tested whether Smoking and Diabetes contributed unique predictive signal
6. **Class rebalancing** — applied SMOTE to the training data and re-evaluated all models

## Results Summary

| Model | Recall (Baseline) | Recall (Tuned) | Recall (Ablation) |
|---|---|---|---|
| Decision Tree | 0.01 | 0.43 | 0.37 |
| Logistic Regression | 0.46 | 0.46 | 0.43 |
| Random Forest | 0.46 | 0.33 | 0.33 |
| SGD | 0.72 | 0.52 | 0.72 |
| SVM | 0.42 | 0.47 | 0.47 |

## Files

- `heart_disease_classification.ipynb` — full analysis and modeling code
- `report.pdf` — final written report with findings and visualizations
- `Heart_Disease_Classification_Presentation.pdf` — slide summary of the project

## Key Takeaways

- Accuracy is a misleading metric on imbalanced datasets — an 80% baseline accuracy masked a near-zero recall for the untuned Decision Tree
- Hyperparameter tuning does not guarantee recall improvement — Random Forest recall *decreased* after GridSearchCV tuning
- SMOTE does not universally help — it dramatically improved the Decision Tree (0.01 → 0.66 recall) but reduced recall for linear models like Logistic Regression and SGD
- Feature ablation confirmed redundancy — removing Smoking and Diabetes had minimal impact across most models, suggesting their signal was already captured by correlated features (BMI, Blood Pressure, Triglycerides)

## My Contribution

This was a team project completed as part of a 5-person group. My work spanned exploratory data analysis, preprocessing, hyperparameter tuning, and the SMOTE rebalancing analysis across all five models.
