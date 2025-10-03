# Epoch 005: Model Development

**Agent**: Model-Developer-Agent | **MLflow**: `{project}_model` | **Duration**: 4-5 hours

## Purpose

**Two-Phase Model Development**: Train baseline models across ALL frameworks to detect initial signal, then hyperparameter tune ONLY the best performing model.

**Note**: Starting from this epoch, we switch to the `{project}_model` MLflow experiment.

## Two-Phase Process

### Phase 1: Initial Signal Detection (Notebooks)
Train baseline models with ALL frameworks to identify which has the strongest signal:

**Classification Frameworks**:
- scikit-learn: LogisticRegression, RandomForest
- XGBoost: XGBClassifier with GPU support
- LightGBM: LGBMClassifier with GPU support
- TensorFlow: Keras Sequential neural network
- PyTorch: Custom neural network with GPU support

**Regression Frameworks** (additional):
- scikit-learn: LinearRegression, RandomForestRegressor
- Statsmodels: OLS, GLM

All models use **standardized metrics** for fair comparison.

### Phase 2: Hyperparameter Tuning (Scripts)
After identifying the best framework, perform comprehensive hyperparameter tuning:
- Framework-specific parameter grids
- Nested MLflow runs (parent + child runs for each combination)
- GPU-accelerated tuning when available
- Summary charts and improvement analysis

## How to Use This Epoch

### Phase 1: Training (Jupyter Notebook)

**Launch Jupyter**:
```bash
jupyter notebook /mnt/code/epoch005-model-development/notebooks/model_training.ipynb
```

**Or execute directly**:
```bash
cd /mnt/code/epoch005-model-development/notebooks
jupyter nbconvert --to notebook --execute model_training.ipynb --inplace
```

**What the notebook does**:
1. Detects GPU availability (nvidia-smi check)
2. Trains 5 frameworks for classification (6 for regression with Statsmodels)
3. Logs standardized metrics for all models to MLflow
4. Creates visual comparison charts (8 metrics)
5. Identifies best performing model
6. Saves all metrics to both MLflow and artifacts directory

### Phase 2: Tuning (Python Script)

**After Phase 1 completes, run tuning script**:
```bash
python /mnt/code/epoch005-model-development/scripts/tune_best_model.py
```

**What the script does**:
1. Reads `best_model_info.json` from Phase 1
2. Dispatches to appropriate tuning function (sklearn/XGBoost/LightGBM/TensorFlow/PyTorch)
3. Tests comprehensive parameter grid with nested MLflow runs
4. Generates summary charts (parameter impact, top 10 models)
5. Saves all tuning results to both MLflow and artifacts directory
6. Reports improvement over baseline

## Command Examples

**Via Claude Code Agent**:
```
"Train all frameworks for credit risk prediction and tune the best model"
"Develop models for patient readmission using all available frameworks"
"Build fraud detection models across sklearn, XGBoost, LightGBM, TF, and PyTorch"
```

## Outputs

### Dual Storage (MLflow + Artifacts Directory)

All outputs saved to **BOTH**:
1. MLflow experiments (logged as artifacts to runs)
2. `/mnt/artifacts/epoch005-model-development/` (persistent file storage)

### Training Phase Artifacts
- `all_models_metrics.json` - Comprehensive metrics for all trained models
- `model_comparison_summary.csv` - Tabular comparison of all models
- `all_models_comparison.png` - Visual comparison charts (8 metrics)
- `best_model_info.json` - Best model identification for tuning phase

### Tuning Phase Artifacts
- `all_tuning_results.json` - Detailed results for every parameter combination
- `all_tuning_results.csv` - Tabular tuning results
- `tuning_parameter_impact.png` - Parameter impact visualization
- `top_10_models.png` - Top 10 tuned models comparison
- `tuned_model_info.json` - Best tuned model parameters and improvement
- `tuning_summary_report.json` - Comprehensive tuning statistics

### MLflow Tracking
**Training runs** (one per framework):
- Run name: `sklearn_logistic_regression`, `sklearn_random_forest`, `xgboost_baseline`, `lightgbm_baseline`, `tensorflow_nn`, `pytorch_nn`
- Metrics: accuracy, precision, recall, f1_score, roc_auc, log_loss, training_time, inference_time_ms, model_size_mb, gpu_used
- Models: Logged with signatures and input examples

**Tuning run** (nested structure):
- Parent run: `{framework}_tuning_parent` (e.g., `xgboost_tuning_parent`)
- Child runs: Each parameter combination tested
- Summary run: `training_summary` with aggregated metrics

### Reusable Code
Extracted to `/mnt/code/src/{project}/model_utils.py`:
- `detect_gpu()` - GPU detection function
- `log_standardized_metrics()` - Consistent metric logging
- Framework-specific training functions
- Model loading/saving utilities

## Standardized Metrics

All models log the same metrics for fair comparison:

**Classification**:
- accuracy, precision, recall, f1_score
- roc_auc, log_loss
- training_time, inference_time_ms
- model_size_mb, gpu_used

**Regression**:
- mse, rmse, mae
- r2_score, mape
- training_time, inference_time_ms
- model_size_mb, gpu_used

## GPU Detection and Support

Automatic GPU detection using `nvidia-smi`:
- **XGBoost**: `tree_method='gpu_hist'`, `predictor='gpu_predictor'`
- **LightGBM**: `device='gpu'`
- **TensorFlow**: Automatic GPU usage
- **PyTorch**: `model.cuda()` when available
- GPU usage tracked in metrics (`gpu_used: 1` or `0`)

## Framework-Specific Parameter Grids

### scikit-learn LogisticRegression
- C: [0.001, 0.01, 0.1, 1, 10, 100]
- solver: ['lbfgs', 'liblinear', 'saga']
- max_iter: [500, 1000, 2000]

### scikit-learn RandomForest
- n_estimators: [50, 100, 200, 300]
- max_depth: [5, 10, 20, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

### XGBoost
- n_estimators: [50, 100, 200, 300]
- max_depth: [3, 5, 7, 10]
- learning_rate: [0.01, 0.05, 0.1, 0.3]
- subsample: [0.6, 0.8, 1.0]
- colsample_bytree: [0.6, 0.8, 1.0]

### LightGBM
- n_estimators: [50, 100, 200, 300]
- max_depth: [3, 5, 7, 10]
- learning_rate: [0.01, 0.05, 0.1, 0.3]
- num_leaves: [15, 31, 63, 127]
- min_child_samples: [5, 10, 20]

### TensorFlow/Keras
- layer1_units: [32, 64, 128]
- layer2_units: [16, 32, 64]
- dropout: [0.2, 0.3, 0.5]
- learning_rate: [0.001, 0.01, 0.1]
- batch_size: [16, 32, 64]

### PyTorch
- layer1_units: [32, 64, 128]
- layer2_units: [16, 32, 64]
- dropout: [0.2, 0.3, 0.5]
- learning_rate: [0.001, 0.01, 0.1]
- epochs: [50, 100, 200]

## Cross-Agent Communication

**Reads from Pipeline Context**:
- `epoch_004_outputs.feature_data_path` - Engineered features location
- `epoch_004_outputs.feature_names` - Feature list
- `epoch_004_outputs.target_variable` - Prediction target
- `epoch_001.recommended_algorithms` - Business Analyst suggestions

**Writes to Pipeline Context**:
```json
{
  "epoch_005_complete": true,
  "epoch_005_outputs": {
    "best_model_name": "xgboost_baseline",
    "best_model_params": {...},
    "tuning_parent_run_id": "abc123...",
    "primary_metric": "f1_score",
    "final_metric_value": 0.89,
    "baseline_metric_value": 0.85,
    "improvement_pct": 4.71
  }
}
```

**Used By**:
- **Epoch 006** (Model Testing): Best model and metrics for validation
- **Epoch 007** (Deployment): Model path and serving configuration
- **Epoch 008** (Retrospective): Training performance analysis

## Success Criteria

✅ All frameworks trained with baseline parameters (Phase 1)
✅ Best framework identified based on primary metric
✅ Comprehensive hyperparameter tuning completed (Phase 2)
✅ All metrics logged to both MLflow and artifacts directory
✅ Improvement over baseline calculated and documented
✅ GPU detected and used when available
✅ Visual comparison charts generated
✅ Reusable code extracted to `/mnt/code/src/`

## Model Development Checklist

**Phase 1: Training**
- [ ] Launch Jupyter notebook
- [ ] GPU detection runs successfully
- [ ] All 5 frameworks train (6 with Statsmodels for regression)
- [ ] Standardized metrics logged for all models
- [ ] Visual comparison chart created
- [ ] Best model identified
- [ ] All artifacts saved to both locations

**Phase 2: Tuning**
- [ ] Run tuning script
- [ ] Best model info loaded correctly
- [ ] Parameter grid tested with nested runs
- [ ] Summary charts generated
- [ ] Improvement calculated
- [ ] All tuning artifacts saved to both locations
- [ ] Pipeline state updated

## Troubleshooting

**GPU not detected**:
- Check: `nvidia-smi` command works
- Models will automatically fall back to CPU
- Training will be slower but still functional

**Out of memory during training**:
- Reduce batch size for neural networks
- Use smaller parameter grids
- Train fewer frameworks (comment out some)

**MLflow logging errors**:
- Check MLflow tracking URI is set
- Verify experiment exists
- Ensure write permissions to artifacts directory

## Next Steps

Proceed to **Epoch 006: Model Testing** for comprehensive validation including edge cases, robustness testing, and compliance checks.

---

**Ready to start?** Launch the training notebook or use Claude Code with: `"Train all frameworks for [prediction task] and tune the best model"`
