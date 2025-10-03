"""
Train baseline models for predictive maintenance - Epoch 005
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import joblib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
import xgboost as xgb

# Paths
DATA_PATH = Path('/mnt/data/Preventative-Maintainance-Example/epoch004-feature-engineering')
MODEL_PATH = Path('/mnt/artifacts/epoch005-model-development/models')
RESULTS_PATH = Path('/mnt/artifacts/epoch005-model-development/results')

MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EPOCH 005: BASELINE MODEL TRAINING")
print("=" * 80)
print()

# Load data
print("Loading train/test datasets...")
train_df = pd.read_parquet(DATA_PATH / 'fd001_train.parquet')
test_df = pd.read_parquet(DATA_PATH / 'fd001_test.parquet')

print(f"Training set: {len(train_df):,} samples")
print(f"Testing set: {len(test_df):,} samples")
print()

# Define features (exclude targets and metadata)
exclude_cols = ['unit_id', 'time_cycles', 'RUL', 'failure_imminent', 'rul_category',
                'op_setting_1', 'op_setting_2', 'op_setting_3']
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

print(f"Total features for modeling: {len(feature_cols)}")
print()

# Prepare data for regression (RUL prediction)
X_train = train_df[feature_cols]
y_train_reg = train_df['RUL']
X_test = test_df[feature_cols]
y_test_reg = test_df['RUL']

# Prepare data for classification (failure_imminent)
y_train_clf = train_df['failure_imminent']
y_test_clf = test_df['failure_imminent']

print(f"Regression target (RUL): Train mean={y_train_reg.mean():.1f}, Test mean={y_test_reg.mean():.1f}")
print(f"Classification target: Train positive={y_train_clf.sum():,} ({y_train_clf.mean()*100:.1f}%), "
      f"Test positive={y_test_clf.sum():,} ({y_test_clf.mean()*100:.1f}%)")
print()

# Store results
results = {}

# ========== REGRESSION MODELS ==========
print("=" * 80)
print("PART 1: REGRESSION MODELS (RUL Prediction)")
print("=" * 80)
print()

# Model 1: Random Forest Regressor
print("Training Random Forest Regressor...")
start_time = time.time()

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_train_reg)

rf_train_time = time.time() - start_time

# Predictions
y_pred_train_rf = rf_reg.predict(X_train)
y_pred_test_rf = rf_reg.predict(X_test)

# Metrics
rf_results = {
    'model': 'Random Forest Regressor',
    'train_time_sec': rf_train_time,
    'train_metrics': {
        'rmse': np.sqrt(mean_squared_error(y_train_reg, y_pred_train_rf)),
        'mae': mean_absolute_error(y_train_reg, y_pred_train_rf),
        'r2': r2_score(y_train_reg, y_pred_train_rf)
    },
    'test_metrics': {
        'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_test_rf)),
        'mae': mean_absolute_error(y_test_reg, y_pred_test_rf),
        'r2': r2_score(y_test_reg, y_pred_test_rf)
    }
}

print(f"Training time: {rf_train_time:.2f}s")
print(f"Train RMSE: {rf_results['train_metrics']['rmse']:.2f}, MAE: {rf_results['train_metrics']['mae']:.2f}, R²: {rf_results['train_metrics']['r2']:.4f}")
print(f"Test  RMSE: {rf_results['test_metrics']['rmse']:.2f}, MAE: {rf_results['test_metrics']['mae']:.2f}, R²: {rf_results['test_metrics']['r2']:.4f}")
print()

results['rf_regressor'] = rf_results

# Save model
joblib.dump(rf_reg, MODEL_PATH / 'rf_regressor_baseline.joblib')
print("Model saved: rf_regressor_baseline.joblib")
print()

# Model 2: XGBoost Regressor
print("Training XGBoost Regressor...")
start_time = time.time()

xgb_reg = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_reg.fit(X_train, y_train_reg)

xgb_train_time = time.time() - start_time

# Predictions
y_pred_train_xgb = xgb_reg.predict(X_train)
y_pred_test_xgb = xgb_reg.predict(X_test)

# Metrics
xgb_results = {
    'model': 'XGBoost Regressor',
    'train_time_sec': xgb_train_time,
    'train_metrics': {
        'rmse': np.sqrt(mean_squared_error(y_train_reg, y_pred_train_xgb)),
        'mae': mean_absolute_error(y_train_reg, y_pred_train_xgb),
        'r2': r2_score(y_train_reg, y_pred_train_xgb)
    },
    'test_metrics': {
        'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_test_xgb)),
        'mae': mean_absolute_error(y_test_reg, y_pred_test_xgb),
        'r2': r2_score(y_test_reg, y_pred_test_xgb)
    }
}

print(f"Training time: {xgb_train_time:.2f}s")
print(f"Train RMSE: {xgb_results['train_metrics']['rmse']:.2f}, MAE: {xgb_results['train_metrics']['mae']:.2f}, R²: {xgb_results['train_metrics']['r2']:.4f}")
print(f"Test  RMSE: {xgb_results['test_metrics']['rmse']:.2f}, MAE: {xgb_results['test_metrics']['mae']:.2f}, R²: {xgb_results['test_metrics']['r2']:.4f}")
print()

results['xgb_regressor'] = xgb_results

# Save model
joblib.dump(xgb_reg, MODEL_PATH / 'xgb_regressor_baseline.joblib')
print("Model saved: xgb_regressor_baseline.joblib")
print()

# ========== CLASSIFICATION MODELS ==========
print("=" * 80)
print("PART 2: CLASSIFICATION MODELS (Failure Imminent)")
print("=" * 80)
print()

# Model 3: Random Forest Classifier
print("Training Random Forest Classifier...")
start_time = time.time()

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_clf.fit(X_train, y_train_clf)

rf_clf_train_time = time.time() - start_time

# Predictions
y_pred_train_rf_clf = rf_clf.predict(X_train)
y_pred_test_rf_clf = rf_clf.predict(X_test)

# Metrics
rf_clf_results = {
    'model': 'Random Forest Classifier',
    'train_time_sec': rf_clf_train_time,
    'train_metrics': {
        'accuracy': accuracy_score(y_train_clf, y_pred_train_rf_clf),
        'precision': precision_score(y_train_clf, y_pred_train_rf_clf),
        'recall': recall_score(y_train_clf, y_pred_train_rf_clf),
        'f1': f1_score(y_train_clf, y_pred_train_rf_clf)
    },
    'test_metrics': {
        'accuracy': accuracy_score(y_test_clf, y_pred_test_rf_clf),
        'precision': precision_score(y_test_clf, y_pred_test_rf_clf),
        'recall': recall_score(y_test_clf, y_pred_test_rf_clf),
        'f1': f1_score(y_test_clf, y_pred_test_rf_clf)
    },
    'test_confusion_matrix': confusion_matrix(y_test_clf, y_pred_test_rf_clf).tolist()
}

print(f"Training time: {rf_clf_train_time:.2f}s")
print(f"Train - Accuracy: {rf_clf_results['train_metrics']['accuracy']:.4f}, "
      f"Precision: {rf_clf_results['train_metrics']['precision']:.4f}, "
      f"Recall: {rf_clf_results['train_metrics']['recall']:.4f}, "
      f"F1: {rf_clf_results['train_metrics']['f1']:.4f}")
print(f"Test  - Accuracy: {rf_clf_results['test_metrics']['accuracy']:.4f}, "
      f"Precision: {rf_clf_results['test_metrics']['precision']:.4f}, "
      f"Recall: {rf_clf_results['test_metrics']['recall']:.4f}, "
      f"F1: {rf_clf_results['test_metrics']['f1']:.4f}")
print()

results['rf_classifier'] = rf_clf_results

# Save model
joblib.dump(rf_clf, MODEL_PATH / 'rf_classifier_baseline.joblib')
print("Model saved: rf_classifier_baseline.joblib")
print()

# Model 4: XGBoost Classifier
print("Training XGBoost Classifier...")
start_time = time.time()

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (y_train_clf == 0).sum() / (y_train_clf == 1).sum()

xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)
xgb_clf.fit(X_train, y_train_clf)

xgb_clf_train_time = time.time() - start_time

# Predictions
y_pred_train_xgb_clf = xgb_clf.predict(X_train)
y_pred_test_xgb_clf = xgb_clf.predict(X_test)

# Metrics
xgb_clf_results = {
    'model': 'XGBoost Classifier',
    'train_time_sec': xgb_clf_train_time,
    'train_metrics': {
        'accuracy': accuracy_score(y_train_clf, y_pred_train_xgb_clf),
        'precision': precision_score(y_train_clf, y_pred_train_xgb_clf),
        'recall': recall_score(y_train_clf, y_pred_train_xgb_clf),
        'f1': f1_score(y_train_clf, y_pred_train_xgb_clf)
    },
    'test_metrics': {
        'accuracy': accuracy_score(y_test_clf, y_pred_test_xgb_clf),
        'precision': precision_score(y_test_clf, y_pred_test_xgb_clf),
        'recall': recall_score(y_test_clf, y_pred_test_xgb_clf),
        'f1': f1_score(y_test_clf, y_pred_test_xgb_clf)
    },
    'test_confusion_matrix': confusion_matrix(y_test_clf, y_pred_test_xgb_clf).tolist()
}

print(f"Training time: {xgb_clf_train_time:.2f}s")
print(f"Train - Accuracy: {xgb_clf_results['train_metrics']['accuracy']:.4f}, "
      f"Precision: {xgb_clf_results['train_metrics']['precision']:.4f}, "
      f"Recall: {xgb_clf_results['train_metrics']['recall']:.4f}, "
      f"F1: {xgb_clf_results['train_metrics']['f1']:.4f}")
print(f"Test  - Accuracy: {xgb_clf_results['test_metrics']['accuracy']:.4f}, "
      f"Precision: {xgb_clf_results['test_metrics']['precision']:.4f}, "
      f"Recall: {xgb_clf_results['test_metrics']['recall']:.4f}, "
      f"F1: {xgb_clf_results['test_metrics']['f1']:.4f}")
print()

results['xgb_classifier'] = xgb_clf_results

# Save model
joblib.dump(xgb_clf, MODEL_PATH / 'xgb_classifier_baseline.joblib')
print("Model saved: xgb_classifier_baseline.joblib")
print()

# ========== SUMMARY ==========
print("=" * 80)
print("BASELINE MODEL COMPARISON")
print("=" * 80)
print()

print("REGRESSION MODELS (Lower is better for RMSE/MAE, Higher for R²):")
print("-" * 80)
print(f"Random Forest: Test RMSE={rf_results['test_metrics']['rmse']:.2f}, "
      f"MAE={rf_results['test_metrics']['mae']:.2f}, R²={rf_results['test_metrics']['r2']:.4f}")
print(f"XGBoost:       Test RMSE={xgb_results['test_metrics']['rmse']:.2f}, "
      f"MAE={xgb_results['test_metrics']['mae']:.2f}, R²={xgb_results['test_metrics']['r2']:.4f}")
print()

print("CLASSIFICATION MODELS (Higher is better):")
print("-" * 80)
print(f"Random Forest: Test Accuracy={rf_clf_results['test_metrics']['accuracy']:.4f}, "
      f"Recall={rf_clf_results['test_metrics']['recall']:.4f}, "
      f"F1={rf_clf_results['test_metrics']['f1']:.4f}")
print(f"XGBoost:       Test Accuracy={xgb_clf_results['test_metrics']['accuracy']:.4f}, "
      f"Recall={xgb_clf_results['test_metrics']['recall']:.4f}, "
      f"F1={xgb_clf_results['test_metrics']['f1']:.4f}")
print()

# Determine best models
best_regressor = 'XGBoost' if xgb_results['test_metrics']['rmse'] < rf_results['test_metrics']['rmse'] else 'Random Forest'
best_classifier = 'XGBoost' if xgb_clf_results['test_metrics']['recall'] > rf_clf_results['test_metrics']['recall'] else 'Random Forest'

print(f"Best Regressor: {best_regressor}")
print(f"Best Classifier: {best_classifier}")
print()

# Save results
results['summary'] = {
    'best_regressor': best_regressor,
    'best_classifier': best_classifier,
    'timestamp': datetime.now().isoformat()
}

results_file = RESULTS_PATH / 'baseline_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {results_file}")
print()

print("=" * 80)
print("BASELINE MODEL TRAINING COMPLETE")
print("=" * 80)
