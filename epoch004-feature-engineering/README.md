# Epoch 004: Feature Engineering

**Agent**: Data-Scientist-Agent | **MLflow**: `{project}_data` | **Duration**: 2-3 hours

## Purpose

Transform raw features into ML-ready representations using insights from EDA to maximize model performance.

## What You'll Do

1. **Feature Creation**: Derive new features (interactions, aggregations, domain-specific)
2. **Encoding**: Convert categorical variables to numerical (one-hot, target, ordinal)
3. **Scaling**: Normalize/standardize numerical features
4. **Feature Selection**: Remove redundant, high-cardinality, or leaky features
5. **Data Splitting**: Create train/validation/test sets with proper stratification

## How to Use This Epoch

**Command Template**:
```
"Engineer features for [prediction task] using [strategy]"
```

**Example Commands**:
- `"Engineer features for credit default prediction using target encoding and standardization"`
- `"Create features for patient readmission using interaction terms and one-hot encoding"`
- `"Transform fraud detection features with robust scaling and polynomial features"`

## Feature Engineering Techniques

**Categorical Encoding**:
- **One-Hot**: Low cardinality (<10 categories), no ordinal relationship
- **Target Encoding**: High cardinality (>10 categories), potential predictive power
- **Ordinal**: Natural ordering (education levels, size categories)
- **Frequency**: Replace with occurrence counts
- **Binary**: Two categories (Yes/No, True/False)

**Numerical Transformations**:
- **Standardization**: Zero mean, unit variance (for linear models, neural nets)
- **Normalization**: Min-max scaling to [0,1] (for distance-based models)
- **Log Transform**: Right-skewed distributions
- **Binning**: Convert continuous to categorical (age groups, income brackets)

**Feature Creation**:
- **Interactions**: Multiply two features (income × debt_ratio)
- **Ratios**: Divide features (debt/income, sales/budget)
- **Aggregations**: Sum, mean, count by group
- **Domain-Specific**: Business logic features (RFM in retail, clinical scores)

## Outputs

**Generated Files**:
- `notebooks/feature_engineering.ipynb` - Transformation process
- `scripts/feature_pipeline.py` - Reusable feature engineering pipeline
- `/mnt/data/{DOMINO_PROJECT_NAME}/epoch004-feature-engineering/`
  - `engineered_features.parquet` - Complete transformed dataset
  - `train_data.parquet` - Training set (70%)
  - `val_data.parquet` - Validation set (15%)
  - `test_data.parquet` - Test set (15%)
  - `feature_mappings.json` - Encoder mappings for inference
- `/mnt/artifacts/epoch004-feature-engineering/`
  - `feature_engineering_report.html` - Transformations and impact
  - `feature_importance_preliminary.png` - Initial rankings
  - `requirements.txt` - Dependencies

**Logged to MLflow** (`{project}_data` experiment):
- Number of engineered features
- Encoding strategies used
- Scaling methods applied
- Train/val/test split sizes
- Feature transformation parameters

**Reusable Code**: `/mnt/code/src/feature_engineering.py`
- Feature transformation functions
- Custom encoders and scalers
- Feature creation utilities
- Pipeline for inference

## Cross-Agent Communication

**Reads from Pipeline Context**:
- `epoch003.encoding_strategy` - How to encode categoricals
- `epoch003.scaling_needed` - Which features to scale
- `epoch003.features_to_drop` - Features to remove
- `epoch003.recommended_features` - Important features
- `epoch002.data_path` - Raw data location

**Writes to Pipeline Context**:
```json
{
  "epoch004": {
    "engineered_data_path": "/mnt/data/{project}/epoch004-feature-engineering/engineered_features.parquet",
    "train_path": "...train_data.parquet",
    "val_path": "...val_data.parquet",
    "test_path": "...test_data.parquet",
    "num_features_original": 15,
    "num_features_engineered": 28,
    "feature_names": ["income_scaled", "debt_ratio", "age_group_encoded", ...],
    "target_variable": "default",
    "encoding_mappings": {...},
    "scaler_params": {...},
    "train_size": 70000,
    "val_size": 15000,
    "test_size": 15000
  }
}
```

**Used By**:
- **Epoch 005** (Model Development): Training data
- **Epoch 006** (Model Testing): Test data and feature info
- **Epoch 007** (Deployment): Feature pipeline for inference
- **Epoch 008** (App): Feature names for UI

## Success Criteria

✅ All categorical features encoded appropriately
✅ Numerical features scaled/normalized
✅ New derived features created
✅ Redundant features removed
✅ Data split with stratification
✅ Feature mappings saved for inference
✅ Reusable pipeline extracted to `/mnt/code/src/`

## Feature Engineering Checklist

- [ ] Load data and EDA insights from Epoch 003
- [ ] Apply categorical encoding (one-hot, target, ordinal)
- [ ] Create interaction and ratio features
- [ ] Apply scaling/normalization to numericals
- [ ] Handle outliers (cap, clip, transform)
- [ ] Drop redundant and leaky features
- [ ] Split into train/val/test with stratification
- [ ] Save feature mappings for deployment
- [ ] Document all transformations
- [ ] Extract reusable code to `/mnt/code/src/feature_engineering.py`
- [ ] Log to MLflow with parameters

## Common Pitfalls to Avoid

❌ **Data Leakage**: Don't include target-derived features
❌ **Fit on Test**: Only fit scalers/encoders on training data
❌ **Dropping Target**: Ensure target variable retained in splits
❌ **Ignoring Missing**: Handle missing values before encoding
❌ **Over-Engineering**: Too many features can cause overfitting

## Next Steps

Proceed to **Epoch 005: Model Development** to train ML models on engineered features.

---

**Ready to start?** Use the Data-Scientist-Agent with your feature engineering strategy.
