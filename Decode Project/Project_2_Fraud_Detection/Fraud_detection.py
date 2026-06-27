import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# 1. Load Data
df = pd.read_csv(r'C:\Users\Wajeehah Wajahat\Downloads\Decode Project\transformed_feature_space.csv')

# 2. Professional Feature Engineering (Fixing the Date/Address error)
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfWeek'] = df['Date'].dt.dayofweek
df = df.drop(columns=['Date', 'ShippingAddress'])

# 3. Create Target
top_10_percent = df['TotalPrice'].quantile(0.90)
df['IsFraud'] = np.where((df['OrderStatus_Returned'] == 1) & (df['TotalPrice'] > top_10_percent), 1, 0)

# 4. Stratified Split (Phase 1)
X = df.drop('IsFraud', axis=1)
y = df['IsFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Pipeline Construction (Phase 2)
pipe_lr = Pipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=42)), ('classifier', LogisticRegression())])
pipe_rf = Pipeline([('smote', SMOTE(random_state=42)), ('classifier', RandomForestClassifier(random_state=42))])

# 6. Holistic Optimization (Phase 3)
params_lr = {'smote__k_neighbors': [3, 5], 'classifier__C': [0.1, 1.0]}
params_rf = {'smote__k_neighbors': [3, 5], 'classifier__max_depth': [10, 20]}

for name, pipe, params in [("Logistic Regression", pipe_lr, params_lr), ("Random Forest", pipe_rf, params_rf)]:
    print(f"\n--- Tuning {name} ---")
    grid = GridSearchCV(pipe, params, cv=3, scoring='recall', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    # 7. Final Evaluation (Phase 4)
    preds = grid.predict(X_test)
    print(f"Best Recall Score: {grid.best_score_:.4f}")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, grid.predict_proba(X_test)[:,1]):.4f}")