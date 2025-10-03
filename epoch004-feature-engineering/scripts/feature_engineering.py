"""
Feature Engineering for Predictive Maintenance - Epoch 004
Creates temporal, degradation, and interaction features for RUL prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.insert(0, '/mnt/code/src')
from predictive_maintenance.data_utils import load_turbofan_data, calculate_rul

# Constants
DATA_PATH = Path('/mnt/data/Preventative-Maintainance-Example/epoch002-data-wrangling')
OUTPUT_PATH = Path('/mnt/data/Preventative-Maintainance-Example/epoch004-feature-engineering')
ARTIFACTS_PATH = Path('/mnt/artifacts/epoch004-feature-engineering')

# Create directories
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

# Top 10 sensors from Epoch 003
TOP_SENSORS = ['sensor_5', 'sensor_11', 'sensor_9', 'sensor_15', 'sensor_13',
               'sensor_16', 'sensor_10', 'sensor_21', 'sensor_19', 'sensor_17']

print("=" * 80)
print("EPOCH 004: FEATURE ENGINEERING")
print("=" * 80)
print()

# Load data
print("Loading dataset...")
df = load_turbofan_data(DATA_PATH / 'fd001_train.parquet')
df = calculate_rul(df)
print(f"Loaded {len(df):,} rows from {df['unit_id'].nunique()} engines")
print()

# Get all sensor columns
sensor_cols = [col for col in df.columns if col.startswith('sensor_')]

# Sort by unit_id and time_cycles for temporal features
df = df.sort_values(['unit_id', 'time_cycles']).reset_index(drop=True)

print("Step 1: Creating Temporal Features (Rolling Statistics)")
print("-" * 80)

# Rolling windows to test
rolling_windows = [5, 10, 20, 50]

for window in rolling_windows:
    print(f"Creating rolling features with window={window}...")

    for sensor in TOP_SENSORS:
        # Rolling mean
        df[f'{sensor}_rolling_mean_{window}'] = (
            df.groupby('unit_id')[sensor]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

        # Rolling std
        df[f'{sensor}_rolling_std_{window}'] = (
            df.groupby('unit_id')[sensor]
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )

        # Rolling min
        df[f'{sensor}_rolling_min_{window}'] = (
            df.groupby('unit_id')[sensor]
            .transform(lambda x: x.rolling(window, min_periods=1).min())
        )

        # Rolling max
        df[f'{sensor}_rolling_max_{window}'] = (
            df.groupby('unit_id')[sensor]
            .transform(lambda x: x.rolling(window, min_periods=1).max())
        )

print(f"Created {len(rolling_windows) * len(TOP_SENSORS) * 4} rolling features")
print()

print("Step 2: Creating Degradation Features (Rates of Change)")
print("-" * 80)

for sensor in TOP_SENSORS:
    # First derivative (change per cycle)
    df[f'{sensor}_diff'] = (
        df.groupby('unit_id')[sensor]
        .transform(lambda x: x.diff())
    )

    # Rate of change over last 10 cycles
    df[f'{sensor}_rate_10'] = (
        df.groupby('unit_id')[sensor]
        .transform(lambda x: (x - x.shift(10)) / 10)
    )

    # Exponentially weighted moving average
    df[f'{sensor}_ewma'] = (
        df.groupby('unit_id')[sensor]
        .transform(lambda x: x.ewm(span=10, adjust=False).mean())
    )

    # Cumulative sum of changes
    df[f'{sensor}_cumsum_diff'] = (
        df.groupby('unit_id')[sensor]
        .transform(lambda x: x.diff().fillna(0).cumsum())
    )

print(f"Created {len(TOP_SENSORS) * 4} degradation features")
print()

print("Step 3: Creating Cross-Sensor Interaction Features")
print("-" * 80)

# Key sensor pairs (based on domain knowledge)
sensor_pairs = [
    ('sensor_5', 'sensor_11'),   # Pressures
    ('sensor_9', 'sensor_13'),   # Speeds
    ('sensor_15', 'sensor_16'),  # Ratios
]

for sensor_a, sensor_b in sensor_pairs:
    # Ratio
    df[f'{sensor_a}_div_{sensor_b}'] = df[sensor_a] / (df[sensor_b] + 1e-6)

    # Product
    df[f'{sensor_a}_mult_{sensor_b}'] = df[sensor_a] * df[sensor_b]

    # Difference
    df[f'{sensor_a}_minus_{sensor_b}'] = df[sensor_a] - df[sensor_b]

print(f"Created {len(sensor_pairs) * 3} interaction features")
print()

print("Step 4: Creating Lifecycle Position Features")
print("-" * 80)

# Normalized time (percentage through lifecycle)
max_cycles = df.groupby('unit_id')['time_cycles'].transform('max')
df['lifecycle_pct'] = (df['time_cycles'] / max_cycles) * 100

# Cycle count features
df['cycles_elapsed'] = df['time_cycles']
df['cycles_remaining'] = df['RUL']

print("Created 3 lifecycle position features")
print()

print("Step 5: Creating Statistical Aggregation Features")
print("-" * 80)

# Per-engine statistics (useful for capturing engine-specific baseline)
for sensor in TOP_SENSORS[:5]:  # Top 5 to avoid too many features
    engine_mean = df.groupby('unit_id')[sensor].transform('mean')
    engine_std = df.groupby('unit_id')[sensor].transform('std')

    # Deviation from engine baseline
    df[f'{sensor}_dev_from_mean'] = (df[sensor] - engine_mean) / (engine_std + 1e-6)

print(f"Created {5} deviation features")
print()

print("Step 6: Creating Target Variables")
print("-" * 80)

# Regression target (already have RUL)

# Binary classification: failure imminent (RUL < 30 cycles)
df['failure_imminent'] = (df['RUL'] < 30).astype(int)

# Multi-class classification
def rul_bucket(rul):
    if rul < 30:
        return 0  # Critical
    elif rul < 60:
        return 1  # Warning
    elif rul < 90:
        return 2  # Caution
    else:
        return 3  # Healthy

df['rul_category'] = df['RUL'].apply(rul_bucket)

print("Created 2 additional target variables:")
print(f"  - failure_imminent: {df['failure_imminent'].sum():,} positive cases")
print(f"  - rul_category distribution:")
print(df['rul_category'].value_counts().sort_index())
print()

print("Step 7: Feature Summary")
print("-" * 80)

# Count features
original_features = len([col for col in df.columns if col.startswith('sensor_') and '_' not in col[7:]])
new_features = len([col for col in df.columns if '_rolling_' in col or '_diff' in col or
                    '_rate_' in col or '_ewma' in col or '_cumsum_' in col or
                    '_div_' in col or '_mult_' in col or '_minus_' in col or
                    '_dev_from_mean' in col])
lifecycle_features = 3
target_features = 2

print(f"Original sensor features: {original_features}")
print(f"Engineered features: {new_features}")
print(f"Lifecycle features: {lifecycle_features}")
print(f"Target variables: {target_features}")
print(f"Total features: {len(df.columns)}")
print()

print("Step 8: Handling Missing Values")
print("-" * 80)

# Fill NaN values created by diff() and rolling operations
missing_before = df.isnull().sum().sum()
df = df.fillna(method='bfill').fillna(0)
missing_after = df.isnull().sum().sum()

print(f"Missing values before: {missing_before:,}")
print(f"Missing values after: {missing_after:,}")
print()

print("Step 9: Saving Engineered Dataset")
print("-" * 80)

# Save full dataset
output_file = OUTPUT_PATH / 'fd001_engineered_features.parquet'
df.to_parquet(output_file, index=False)

file_size_mb = output_file.stat().st_size / (1024 * 1024)
print(f"Saved to: {output_file}")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Size: {file_size_mb:.2f} MB")
print()

# Save feature metadata
feature_metadata = {
    "created": datetime.now().isoformat(),
    "original_features": original_features,
    "engineered_features": new_features,
    "lifecycle_features": lifecycle_features,
    "target_variables": target_features,
    "total_features": len(df.columns),
    "feature_groups": {
        "rolling_statistics": len([c for c in df.columns if '_rolling_' in c]),
        "degradation_features": len([c for c in df.columns if '_diff' in c or '_rate_' in c or '_ewma' in c or '_cumsum_' in c]),
        "interaction_features": len([c for c in df.columns if '_div_' in c or '_mult_' in c or '_minus_' in c]),
        "lifecycle_features": lifecycle_features,
        "deviation_features": len([c for c in df.columns if '_dev_from_mean' in c])
    },
    "target_distribution": {
        "rul_mean": float(df['RUL'].mean()),
        "rul_std": float(df['RUL'].std()),
        "failure_imminent_positive_rate": float(df['failure_imminent'].mean()),
        "rul_category_counts": df['rul_category'].value_counts().to_dict()
    }
}

metadata_file = ARTIFACTS_PATH / 'feature_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(feature_metadata, f, indent=2)

print(f"Feature metadata saved to: {metadata_file}")
print()

print("=" * 80)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 80)
