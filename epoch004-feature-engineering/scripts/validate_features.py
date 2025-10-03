"""
Validate engineered features for quality and predictive power
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

# Paths
DATA_PATH = Path('/mnt/data/Preventative-Maintainance-Example/epoch004-feature-engineering')
ARTIFACTS_PATH = Path('/mnt/artifacts/epoch004-feature-engineering')

print("=" * 80)
print("FEATURE VALIDATION")
print("=" * 80)
print()

# Load train dataset
print("Loading training dataset...")
train_df = pd.read_parquet(DATA_PATH / 'fd001_train.parquet')
print(f"Loaded {len(train_df):,} training samples")
print()

# Define feature groups
original_sensors = [col for col in train_df.columns if col.startswith('sensor_') and '_' not in col[7:]]
rolling_features = [col for col in train_df.columns if '_rolling_' in col]
degradation_features = [col for col in train_df.columns if any(x in col for x in ['_diff', '_rate_', '_ewma', '_cumsum_'])]
interaction_features = [col for col in train_df.columns if any(x in col for x in ['_div_', '_mult_', '_minus_'])]
lifecycle_features = ['lifecycle_pct', 'cycles_elapsed', 'cycles_remaining']
deviation_features = [col for col in train_df.columns if '_dev_from_mean' in col]

all_feature_cols = (original_sensors + rolling_features + degradation_features +
                   interaction_features + lifecycle_features + deviation_features)

print("Feature Group Counts:")
print("-" * 80)
print(f"Original sensors: {len(original_sensors)}")
print(f"Rolling features: {len(rolling_features)}")
print(f"Degradation features: {len(degradation_features)}")
print(f"Interaction features: {len(interaction_features)}")
print(f"Lifecycle features: {len(lifecycle_features)}")
print(f"Deviation features: {len(deviation_features)}")
print(f"Total features for modeling: {len(all_feature_cols)}")
print()

# Check for data quality issues
print("Step 1: Data Quality Validation")
print("-" * 80)

# Check for infinite values
inf_counts = np.isinf(train_df[all_feature_cols]).sum()
features_with_inf = inf_counts[inf_counts > 0]
if len(features_with_inf) > 0:
    print(f"WARNING: {len(features_with_inf)} features contain infinite values")
    for feat, count in features_with_inf.items():
        print(f"  {feat}: {count} infinite values")
else:
    print("No infinite values detected")

# Check for NaN values
nan_counts = train_df[all_feature_cols].isnull().sum()
features_with_nan = nan_counts[nan_counts > 0]
if len(features_with_nan) > 0:
    print(f"\nWARNING: {len(features_with_nan)} features contain NaN values")
    for feat, count in features_with_nan.items():
        print(f"  {feat}: {count} NaN values")
else:
    print("No NaN values detected")

# Check for constant features
print("\nChecking for zero-variance features...")
constant_features = []
for col in all_feature_cols:
    if train_df[col].nunique() == 1:
        constant_features.append(col)

if len(constant_features) > 0:
    print(f"WARNING: {len(constant_features)} constant features found")
    for feat in constant_features[:10]:
        print(f"  {feat}")
else:
    print("No constant features found")

print()

# Validate feature correlations with RUL
print("Step 2: Feature Importance Analysis")
print("-" * 80)

# Sample for faster computation
sample_size = min(10000, len(train_df))
sample_indices = np.random.choice(len(train_df), sample_size, replace=False)
X_sample = train_df.loc[sample_indices, all_feature_cols]
y_sample = train_df.loc[sample_indices, 'RUL']

# Calculate correlations
print("Calculating feature correlations with RUL...")
correlations = X_sample.corrwith(y_sample).abs().sort_values(ascending=False)

print("\nTop 20 Features by Correlation with RUL:")
for i, (feat, corr) in enumerate(correlations.head(20).items(), 1):
    print(f"  {i:2d}. {feat:45s} : {corr:.4f}")

# Train quick Random Forest for feature importance
print("\nTraining Random Forest for feature importance...")
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_sample, y_sample)

rf_importances = pd.Series(rf_model.feature_importances_, index=all_feature_cols).sort_values(ascending=False)

print("\nTop 20 Features by Random Forest Importance:")
for i, (feat, imp) in enumerate(rf_importances.head(20).items(), 1):
    print(f"  {i:2d}. {feat:45s} : {imp:.4f}")

print()

# Compare original vs engineered features
print("Step 3: Original vs Engineered Feature Performance")
print("-" * 80)

# Top original sensors
top_original_corr = correlations[correlations.index.isin(original_sensors)].head(5).mean()
print(f"Top 5 original sensors - Mean correlation: {top_original_corr:.4f}")

# Top rolling features
top_rolling_corr = correlations[correlations.index.isin(rolling_features)].head(5).mean()
print(f"Top 5 rolling features - Mean correlation: {top_rolling_corr:.4f}")

# Top degradation features
top_degradation_corr = correlations[correlations.index.isin(degradation_features)].head(5).mean()
print(f"Top 5 degradation features - Mean correlation: {top_degradation_corr:.4f}")

# Top interaction features
if len(interaction_features) > 0:
    top_interaction_corr = correlations[correlations.index.isin(interaction_features)].head(5).mean()
    print(f"Top 5 interaction features - Mean correlation: {top_interaction_corr:.4f}")

print()

# Save validation results
print("Step 4: Saving Validation Results")
print("-" * 80)

validation_results = {
    "data_quality": {
        "infinite_values": int(len(features_with_inf)),
        "nan_values": int(len(features_with_nan)),
        "constant_features": int(len(constant_features))
    },
    "feature_counts": {
        "original_sensors": len(original_sensors),
        "rolling_features": len(rolling_features),
        "degradation_features": len(degradation_features),
        "interaction_features": len(interaction_features),
        "lifecycle_features": len(lifecycle_features),
        "deviation_features": len(deviation_features),
        "total": len(all_feature_cols)
    },
    "top_20_by_correlation": {
        feat: float(corr) for feat, corr in correlations.head(20).items()
    },
    "top_20_by_rf_importance": {
        feat: float(imp) for feat, imp in rf_importances.head(20).items()
    },
    "performance_comparison": {
        "top_5_original_mean_corr": float(top_original_corr),
        "top_5_rolling_mean_corr": float(top_rolling_corr),
        "top_5_degradation_mean_corr": float(top_degradation_corr)
    }
}

validation_file = ARTIFACTS_PATH / 'feature_validation_results.json'
with open(validation_file, 'w') as f:
    json.dump(validation_results, f, indent=2)

print(f"Validation results saved: {validation_file}")

# Save feature importance rankings
importance_df = pd.DataFrame({
    'Feature': correlations.index,
    'Correlation': correlations.values,
    'RF_Importance': [rf_importances.get(feat, 0) for feat in correlations.index]
})

importance_file = ARTIFACTS_PATH / 'feature_importance_rankings.csv'
importance_df.to_csv(importance_file, index=False)
print(f"Feature importance rankings saved: {importance_file}")

print()
print("=" * 80)
print("FEATURE VALIDATION COMPLETE")
print("=" * 80)
print()
print("Summary:")
print(f"  Total features: {len(all_feature_cols)}")
print(f"  Data quality: {'PASS' if len(features_with_inf) == 0 and len(features_with_nan) == 0 else 'WARNING'}")
print(f"  Best feature correlation: {correlations.iloc[0]:.4f}")
print(f"  Engineered features improved over original: {'YES' if top_rolling_corr > top_original_corr else 'NO'}")
