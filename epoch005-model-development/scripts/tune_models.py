"""
Hyperparameter tuning for best performing models - Epoch 005
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             make_scorer)
import xgboost as xgb

# Paths
DATA_PATH = Path('/mnt/data/Preventative-Maintainance-Example/epoch004-feature-engineering')
MODEL_PATH = Path('/mnt/artifacts/epoch005-model-development/models')
RESULTS_PATH = Path('/mnt/artifacts/epoch005-model-development/results')

print("=" * 80)
print("HYPERPARAMETER TUNING")
print("=" * 80)
print()

# Load data
print("Loading train/test datasets...")
train_df = pd.read_parquet(DATA_PATH / 'fd001_train.parquet')
test_df = pd.read_parquet(DATA_PATH / 'fd001_test.parquet')

# Define features
exclude_cols = ['unit_id', 'time_cycles', 'RUL', 'failure_imminent', 'rul_category',
                'op_setting_1', 'op_setting_2', 'op_setting_3']
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

X_train = train_df[feature_cols]
y_train_reg = train_df['RUL']
y_train_clf = train_df['failure_imminent']

X_test = test_df[feature_cols]
y_test_reg = test_df['RUL']
y_test_clf = test_df['failure_imminent']

print(f"Training samples: {len(X_train):,}")
print(f"Features: {len(feature_cols)}")
print()

results = {}

# ========== TUNE RANDOM FOREST REGRESSOR ==========
print("=" * 80)
print("TUNING RANDOM FOREST REGRESSOR")
print("=" * 80)
print()

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

print("Parameter grid:")
for param, values in rf_param_grid.items():
    print(f"  {param}: {values}")
print()

print("Starting RandomizedSearchCV (30 iterations)...")
start_time = time.time()

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=rf_param_grid,
    n_iter=30,
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_search.fit(X_train, y_train_reg)
tuning_time = time.time() - start_time

print(f"\nTuning completed in {tuning_time:.2f}s")
print(f"Best parameters: {rf_search.best_params_}")
print()

# Train final model with best params
best_rf = rf_search.best_estimator_

# Predictions
y_pred_train_rf = best_rf.predict(X_train)
y_pred_test_rf = best_rf.predict(X_test)

# Metrics
rf_tuned_results = {
    'model': 'Random Forest Regressor (Tuned)',
    'best_params': rf_search.best_params_,
    'tuning_time_sec': tuning_time,
    'train_metrics': {
        'rmse': float(np.sqrt(mean_squared_error(y_train_reg, y_pred_train_rf))),
        'mae': float(mean_absolute_error(y_train_reg, y_pred_train_rf)),
        'r2': float(r2_score(y_train_reg, y_pred_train_rf))
    },
    'test_metrics': {
        'rmse': float(np.sqrt(mean_squared_error(y_test_reg, y_pred_test_rf))),
        'mae': float(mean_absolute_error(y_test_reg, y_pred_test_rf)),
        'r2': float(r2_score(y_test_reg, y_pred_test_rf))
    }
}

print("Tuned Model Performance:")
print(f"Train RMSE: {rf_tuned_results['train_metrics']['rmse']:.4f}, MAE: {rf_tuned_results['train_metrics']['mae']:.4f}, R²: {rf_tuned_results['train_metrics']['r2']:.6f}")
print(f"Test  RMSE: {rf_tuned_results['test_metrics']['rmse']:.4f}, MAE: {rf_tuned_results['test_metrics']['mae']:.4f}, R²: {rf_tuned_results['test_metrics']['r2']:.6f}")
print()

results['rf_regressor_tuned'] = rf_tuned_results

# Save tuned model
joblib.dump(best_rf, MODEL_PATH / 'rf_regressor_tuned.joblib')
print("Tuned model saved: rf_regressor_tuned.joblib")
print()

# ========== TUNE XGBOOST CLASSIFIER ==========
print("=" * 80)
print("TUNING XGBOOST CLASSIFIER")
print("=" * 80)
print()

# Calculate scale_pos_weight
scale_pos_weight = (y_train_clf == 0).sum() / (y_train_clf == 1).sum()

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}

print("Parameter grid:")
for param, values in xgb_param_grid.items():
    print(f"  {param}: {values}")
print()

print("Starting RandomizedSearchCV (30 iterations)...")
start_time = time.time()

xgb_search = RandomizedSearchCV(
    xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    ),
    param_distributions=xgb_param_grid,
    n_iter=30,
    cv=3,
    scoring='recall',  # Prioritize recall for failure detection
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train, y_train_clf)
tuning_time = time.time() - start_time

print(f"\nTuning completed in {tuning_time:.2f}s")
print(f"Best parameters: {xgb_search.best_params_}")
print()

# Train final model with best params
best_xgb = xgb_search.best_estimator_

# Predictions
y_pred_train_xgb = best_xgb.predict(X_train)
y_pred_test_xgb = best_xgb.predict(X_test)

# Metrics
xgb_tuned_results = {
    'model': 'XGBoost Classifier (Tuned)',
    'best_params': xgb_search.best_params_,
    'tuning_time_sec': tuning_time,
    'train_metrics': {
        'accuracy': float(accuracy_score(y_train_clf, y_pred_train_xgb)),
        'precision': float(precision_score(y_train_clf, y_pred_train_xgb)),
        'recall': float(recall_score(y_train_clf, y_pred_train_xgb)),
        'f1': float(f1_score(y_train_clf, y_pred_train_xgb))
    },
    'test_metrics': {
        'accuracy': float(accuracy_score(y_test_clf, y_pred_test_xgb)),
        'precision': float(precision_score(y_test_clf, y_pred_test_xgb)),
        'recall': float(recall_score(y_test_clf, y_pred_test_xgb)),
        'f1': float(f1_score(y_test_clf, y_pred_test_xgb))
    }
}

print("Tuned Model Performance:")
print(f"Train - Accuracy: {xgb_tuned_results['train_metrics']['accuracy']:.4f}, "
      f"Precision: {xgb_tuned_results['train_metrics']['precision']:.4f}, "
      f"Recall: {xgb_tuned_results['train_metrics']['recall']:.4f}, "
      f"F1: {xgb_tuned_results['train_metrics']['f1']:.4f}")
print(f"Test  - Accuracy: {xgb_tuned_results['test_metrics']['accuracy']:.4f}, "
      f"Precision: {xgb_tuned_results['test_metrics']['precision']:.4f}, "
      f"Recall: {xgb_tuned_results['test_metrics']['recall']:.4f}, "
      f"F1: {xgb_tuned_results['test_metrics']['f1']:.4f}")
print()

results['xgb_classifier_tuned'] = xgb_tuned_results

# Save tuned model
joblib.dump(best_xgb, MODEL_PATH / 'xgb_classifier_tuned.joblib')
print("Tuned model saved: xgb_classifier_tuned.joblib")
print()

# ========== SUMMARY ==========
print("=" * 80)
print("TUNING RESULTS SUMMARY")
print("=" * 80)
print()

print("FINAL MODEL SELECTION:")
print("-" * 80)
print(f"RUL Prediction (Regression): Random Forest (Tuned)")
print(f"  Test RMSE: {rf_tuned_results['test_metrics']['rmse']:.4f}")
print(f"  Test R²: {rf_tuned_results['test_metrics']['r2']:.6f}")
print()
print(f"Failure Detection (Classification): XGBoost (Tuned)")
print(f"  Test Accuracy: {xgb_tuned_results['test_metrics']['accuracy']:.4f}")
print(f"  Test Recall: {xgb_tuned_results['test_metrics']['recall']:.4f}")
print()

# Save tuning results
results_file = RESULTS_PATH / 'tuning_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Tuning results saved to: {results_file}")
print()

print("=" * 80)
print("HYPERPARAMETER TUNING COMPLETE")
print("=" * 80)
