# Epoch-Specific Code Templates

This file contains detailed code templates for each epoch to be used by specialized agents.

## Epoch 002: Data Wrangling Template

### Script Template: `generate_data.py`

```python
# epoch002-data-wrangling/scripts/generate_data.py

import os
import pandas as pd
import psutil
import gc
from pathlib import Path
from datetime import datetime

# Import shared utilities
# from src.{project}.error_handling import execute_with_error_handling
# from src.{project}.config import ENABLE_COST_TRACKING

# Configuration
MAX_FILE_SIZE_MB = 50
MAX_RAM_GB = 12
CHUNK_SIZE = 10000

# Validate prerequisites (see boilerplate_patterns.md for implementation)
# state = validate_epoch_prerequisites("002")

# Validate data directory (see boilerplate_patterns.md for implementation)
project_name = os.getenv('DOMINO_PROJECT_NAME', 'demo')
# data_dir = validate_data_directory(project_name, "epoch002-data-wrangling")

# Main processing with error handling
def generate_data():
    # Initialize checkpoint
    # checkpoint = load_checkpoint("002")
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint['progress_percent']}%")

    # Generate data in chunks
    chunks = []
    for i in range(0, total_rows, CHUNK_SIZE):
        # Memory check
        memory_gb = psutil.Process().memory_info().rss / (1024**3)
        if memory_gb > MAX_RAM_GB:
            raise MemoryError(f"Memory {memory_gb}GB exceeds {MAX_RAM_GB}GB limit")

        # Generate chunk
        chunk = generate_chunk(i, min(i + CHUNK_SIZE, total_rows))
        chunks.append(chunk)

        # Progress reporting
        progress = (i + CHUNK_SIZE) / total_rows * 100
        print(f"Generated {i + CHUNK_SIZE}/{total_rows} rows (Memory: {memory_gb:.2f} GB)")

        # Save checkpoint
        # save_checkpoint("002", progress, "data_generation", "data_validation", ["setup"], {})

        gc.collect()

    # Combine and validate
    df = pd.concat(chunks, ignore_index=True)

    # Size check
    estimated_size = df.memory_usage(deep=True).sum() / (1024**2)
    if estimated_size > MAX_FILE_SIZE_MB:
        sample_ratio = MAX_FILE_SIZE_MB / estimated_size
        df = df.sample(frac=sample_ratio, random_state=42)

    # Save
    output_path = data_dir / "data.parquet"
    df.to_parquet(output_path)

    # Update lineage
    # update_data_lineage("002", "data_generation", {...})

    # Update pipeline state
    # update_pipeline_state({"epoch_002_complete": True, ...})

    return df

# Execute with error handling
# result = execute_with_error_handling("002", "data_generation", generate_data)

# Run tests
# pytest.main(["/mnt/code/src/{project}/tests/test_data_processing.py", "-v"])
```

## Epoch 005: Model Development Templates

### Overview

Epoch 005 follows a **two-phase process**:
1. **Phase 1 (Notebooks)**: Train baseline models with ALL frameworks to detect initial signal
2. **Phase 2 (Scripts)**: Hyperparameter tuning ONLY the best performing model

### Phase 1: Baseline Training Notebook

**Notebook: `notebooks/model_training.ipynb`**

Key sections:
- Cell 1: Setup and Data Loading with GPU detection
- Cell 2: Define standardized metrics function
- Cells 3-8: Train each framework (sklearn, XGBoost, LightGBM, TensorFlow, PyTorch)
- Cell 9: Compare all models and select best
- Cell 10: Save comprehensive metrics to artifacts directory

**GPU Detection Pattern:**
```python
def detect_gpu():
    """Detect if Nvidia GPU is available"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gpu_available = result.returncode == 0
        if gpu_available:
            print("GPU detected: Using Nvidia GPU for training")
        else:
            print("No GPU detected: Using CPU for training")
        return gpu_available
    except:
        print("No GPU detected: Using CPU for training")
        return False

GPU_AVAILABLE = detect_gpu()
```

**Standardized Metrics Function:**
```python
def log_standardized_metrics(y_true, y_pred, y_pred_proba, model, X_test, problem_type="classification"):
    """Log standardized metrics for all models"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
    import time

    metrics = {}

    if problem_type == "classification":
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        if y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            metrics["log_loss"] = log_loss(y_true, y_pred_proba)
    else:  # regression
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = mean_squared_error(y_true, y_pred, squared=False)
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2_score"] = r2_score(y_true, y_pred)
        metrics["mape"] = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-10, None))) * 100

    # Inference time
    start = time.time()
    _ = model.predict(X_test[:100])
    inference_time = (time.time() - start) / 100 * 1000  # ms per prediction
    metrics["inference_time_ms"] = inference_time

    # Model size
    import joblib
    import os
    temp_path = "/tmp/temp_model.pkl"
    try:
        joblib.dump(model, temp_path)
        metrics["model_size_mb"] = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
    except:
        metrics["model_size_mb"] = 0

    # GPU usage
    metrics["gpu_used"] = 1 if GPU_AVAILABLE else 0

    return metrics
```

**Training Pattern for Each Framework:**
```python
# Example: XGBoost with GPU support
with mlflow.start_run(run_name="xgboost_baseline"):
    start_time = time.time()

    xgb_params = {
        'n_estimators': 100,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    if GPU_AVAILABLE:
        xgb_params['tree_method'] = 'gpu_hist'
        xgb_params['predictor'] = 'gpu_predictor'

    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = log_standardized_metrics(y_test, y_pred, y_pred_proba, model, X_test)
    metrics["training_time"] = training_time

    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature)
```

**Best Model Selection:**
```python
# Compare all models
experiment = mlflow.get_experiment_by_name(f"{project_name}_model")
runs_df = mlflow.search_runs(experiment.experiment_id)

primary_metric = 'f1_score'  # or 'r2_score' for regression
best_run = runs_df.loc[runs_df[f'metrics.{primary_metric}'].idxmax()]
best_model_name = best_run['tags.mlflow.runName']

print(f"\nBest performing model: {best_model_name}")
print(f"Best {primary_metric}: {best_run[f'metrics.{primary_metric}']:.4f}")

# Save for tuning phase
best_model_info = {
    'model_name': best_model_name,
    'primary_metric': primary_metric,
    'primary_metric_value': float(best_run[f'metrics.{primary_metric}']),
    'run_id': best_run['run_id']
}

with open('/mnt/artifacts/epoch005-model-development/best_model_info.json', 'w') as f:
    json.dump(best_model_info, f, indent=2)
```

### Phase 2: Hyperparameter Tuning Script

**Script: `scripts/tune_best_model.py`**

```python
# Universal hyperparameter tuning script - tunes ONLY the best model from training

import json
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

# GPU Detection
GPU_AVAILABLE = detect_gpu()

# Load best model info
with open('/mnt/artifacts/epoch005-model-development/best_model_info.json', 'r') as f:
    best_model_info = json.load(f)

best_model_name = best_model_info['model_name']
primary_metric = best_model_info['primary_metric']

print(f"\nTuning best model: {best_model_name}")
print(f"Baseline {primary_metric}: {best_model_info['primary_metric_value']:.4f}\n")

# Load data
# data_path = state["epoch_004_outputs"]["feature_data_path"]
# df = pd.read_parquet(data_path)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set experiment
mlflow.set_experiment(f"{project_name}_model")

# Dispatch to appropriate tuning function based on best model
if 'xgboost' in best_model_name:
    print("Tuning XGBoost...")

    from xgboost import XGBClassifier

    with mlflow.start_run(run_name="xgboost_tuning_parent") as parent_run:
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

        results = []
        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    for subsample in param_grid['subsample']:
                        for colsample in param_grid['colsample_bytree']:

                            with mlflow.start_run(run_name=f"xgb_n{n_est}_d{depth}_lr{lr}", nested=True):
                                start_time = time.time()

                                xgb_params = {
                                    'n_estimators': n_est,
                                    'max_depth': depth,
                                    'learning_rate': lr,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample,
                                    'random_state': 42,
                                    'eval_metric': 'logloss'
                                }
                                if GPU_AVAILABLE:
                                    xgb_params['tree_method'] = 'gpu_hist'
                                    xgb_params['predictor'] = 'gpu_predictor'

                                model = XGBClassifier(**xgb_params)
                                model.fit(X_train, y_train)

                                training_time = time.time() - start_time

                                y_pred = model.predict(X_test)
                                y_pred_proba = model.predict_proba(X_test)

                                metrics = log_standardized_metrics(y_test, y_pred, y_pred_proba, model, X_test)
                                metrics["training_time"] = training_time

                                mlflow.log_params({
                                    'n_estimators': n_est,
                                    'max_depth': depth,
                                    'learning_rate': lr,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample
                                })
                                mlflow.log_metrics(metrics)

                                results.append({
                                    'n_estimators': n_est,
                                    'max_depth': depth,
                                    'learning_rate': lr,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample,
                                    primary_metric: metrics[primary_metric]
                                })

# Generate summary charts (common for all models)
results_df = pd.DataFrame(results)

# Chart 1: Parameter impact
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
param_cols = [col for col in results_df.columns if col != primary_metric][:3]

for idx, param in enumerate(param_cols):
    axes[idx].scatter(results_df[param], results_df[primary_metric])
    axes[idx].set_xlabel(param)
    axes[idx].set_ylabel(primary_metric)

plt.tight_layout()
plt.savefig('/mnt/artifacts/epoch005-model-development/tuning_parameter_impact.png')
mlflow.log_artifact('/mnt/artifacts/epoch005-model-development/tuning_parameter_impact.png')

# Chart 2: Top 10 models
top_10 = results_df.nlargest(10, primary_metric)

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(range(len(top_10)), top_10[primary_metric])
ax.set_title(f'Top 10 Models - {primary_metric}')
plt.savefig('/mnt/artifacts/epoch005-model-development/top_10_models.png')
mlflow.log_artifact('/mnt/artifacts/epoch005-model-development/top_10_models.png')

# Save all results
results_df.to_csv('/mnt/artifacts/epoch005-model-development/all_tuning_results.csv', index=False)
mlflow.log_artifact('/mnt/artifacts/epoch005-model-development/all_tuning_results.csv')

print(f"\nAll artifacts saved to: /mnt/artifacts/epoch005-model-development/")
print(f"  - all_tuning_results.csv")
print(f"  - tuning_parameter_impact.png")
print(f"  - top_10_models.png")
```

### Artifacts Generated (Dual Storage)

**Training Phase:**
- `all_models_metrics.json` - Comprehensive metrics for all models
- `model_comparison_summary.csv` - Tabular comparison
- `all_models_comparison.png` - Visual comparison charts
- `best_model_info.json` - Best model for tuning

**Tuning Phase:**
- `all_tuning_results.json` - Every parameter combination
- `all_tuning_results.csv` - Tabular tuning results
- `tuning_parameter_impact.png` - Parameter visualization
- `top_10_models.png` - Top models comparison
- `tuned_model_info.json` - Best tuned model
- `tuning_summary_report.json` - Comprehensive statistics

All saved to both `/mnt/artifacts/epoch005-model-development/` AND MLflow.

## Reference

For implementation details, see:
- `/mnt/code/.reference/boilerplate_patterns.md` - Helper functions
- `/mnt/code/.reference/framework_configs.json` - Parameter grids
- `/mnt/code/.reference/visualization_standards.md` - Plotting standards
