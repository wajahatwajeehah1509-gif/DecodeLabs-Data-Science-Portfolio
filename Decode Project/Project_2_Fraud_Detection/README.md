# Project 2: Robust Fraud Detection Pipeline for Imbalanced Financial Datasets

## Project Mission: Algorithmic Precision Under Extreme Scarcity
This project implements a defensive enterprise payment infrastructure machine learning pipeline engineered to flag fraudulent transactions. Operating on a highly skewed dataset (284,807 transactions with an extreme imbalance of **99.83% Legitimate** vs. **0.17% Fraudulent**), the system rejects simple global error minimization. It introduces strict statistical controls to capture high-stakes anomalies without disrupting baseline traffic.

## Architectural Framework & Core Tasks

### 1. Advanced Data Rebalancing (Synthetic Interpolation)
- **Rejection of Destructive Methods:** Discarded random undersampling (which destroys baseline history) and traditional oversampling (which causes severe model overfitting via row duplication).
- **SMOTE Engine:** Implemented Synthetic Minority Over-sampling Technique to populate sparse regions of the minority feature space via mathematical interpolation:
  $$x_{\text{new}} = x_i + \lambda \times (x_{\text{nn}} - x_i)$$
  Where $\lambda \sim \text{Uniform}(0, 1)$, generating entirely new, unique data points along line segments of $k$-nearest neighbors.

### 2. Zero-Leakage Pipeline Architecture
- **The Data Leakage Trap:** Avoided the catastrophic error of scaling or applying SMOTE to the entire dataset prior to partitioning, which causes validation phases to test on synthetic points that already contain training metrics.
- **The Imblearn Imperative:** Deployed `imblearn.pipeline.Pipeline` instead of Scikit-Learn’s native pipeline to leverage the `fit_resample` interface. This natively ensures that SMOTE resampling and scaling occur *strictly* on training folds during cross-validation, maintaining an untouched, highly-imbalanced "Blind Exam" test environment.

### 3. Multi-Algorithm Benchmarking & Hyperparameter Tuning
- **Logistic Regression (LR) Engine:** Configured with an inline `StandardScaler` to protect regularization penalties from transaction variances. Generates a linear decision boundary with high coefficient transparency.
- **Random Forest (RF) Engine:** Built an ensemble network invariant to feature scaling due to ordinal partitioning logic. Generates complex, non-linear decision boundaries.
- **Holistic GridSearchCV:** Configured simultaneous grid searches tuning both preprocessor settings (`smote__k_neighbors: [3, 5, 7]`) and model parameters (`classifier__C` and `classifier__max_depth`) concurrently within isolated folds.

### 4. Evaluation Framework (Beyond Accuracy)
Abandoned global accuracy (which gives a deceptive 99.83% success rate on a broken default model) in favor of:
- **Precision:** Maximizing true positives over total flagged transactions to minimize false declines and customer frustration.
- **Recall:** Minimizing False Negatives to maximize direct financial asset recovery.
- **ROC-AUC:** Measuring spatial distribution separation boundaries across all potential classification thresholds.

## Tech Stack
- **Language:** Python
- **Core Libraries:** Imbalanced-Learn (`imblearn`), Scikit-Learn (`sklearn`), Pandas, NumPy, Matplotlib, Seaborn
