#!/usr/bin/env python3
"""
Epoch 003: Exploratory Data Analysis (Simplified)
Analysis of NASA Turbofan data without visualization dependencies
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Setup paths
sys.path.insert(0, '/mnt/code/src')
from predictive_maintenance.data_utils import load_turbofan_data, calculate_rul

# Paths
DATA_PATH = Path('/mnt/data/Preventative-Maintainance-Example/epoch002-data-wrangling')
ARTIFACTS_PATH = Path('/mnt/artifacts/epoch003-exploratory-data-analysis')
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EPOCH 003: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Load data
print("\n[1/6] Loading data and calculating RUL...")
df = load_turbofan_data(DATA_PATH / 'fd001_train.parquet')
df = calculate_rul(df)

print(f"Dataset shape: {df.shape}")
print(f"Number of engines: {df['unit_id'].nunique()}")

# Identify columns
sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
op_setting_cols = [col for col in df.columns if col.startswith('op_setting')]

print(f"Sensor columns: {len(sensor_cols)}")

# Basic statistics
print("\n[2/6] Computing statistics...")
stats = df[sensor_cols].describe()
stats.to_csv(ARTIFACTS_PATH / 'sensor_statistics.csv')

rul_stats = {
    'mean': float(df['RUL'].mean()),
    'median': float(df['RUL'].median()),
    'min': int(df['RUL'].min()),
    'max': int(df['RUL'].max()),
    'std': float(df['RUL'].std())
}

print(f"RUL Statistics:")
for key, val in rul_stats.items():
    print(f"  {key}: {val}")

# Correlation analysis
print("\n[3/6] Computing correlations...")
correlation_matrix = df[sensor_cols].corr()
correlation_matrix.to_csv(ARTIFACTS_PATH / 'sensor_correlations.csv')

# Highly correlated pairs
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append({
                'sensor1': correlation_matrix.columns[i],
                'sensor2': correlation_matrix.columns[j],
                'correlation': float(correlation_matrix.iloc[i, j])
            })

print(f"  Highly correlated pairs (>0.9): {len(high_corr_pairs)}")

# RUL correlation (feature importance)
print("\n[4/6] Computing feature importance...")
rul_correlations = df[sensor_cols].corrwith(df['RUL']).abs().sort_values(ascending=False)
rul_correlations.to_csv(ARTIFACTS_PATH / 'rul_correlations.csv')

top_sensors = rul_correlations.head(10).to_dict()
print(f"\nTop 10 Most Predictive Sensors:")
for sensor, corr in list(top_sensors.items())[:10]:
    print(f"  {sensor}: {corr:.4f}")

# Engine lifetime analysis
print("\n[5/6] Analyzing engine lifetimes...")
engine_lifetimes = df.groupby('unit_id')['time_cycles'].max()
lifetime_stats = {
    'mean': float(engine_lifetimes.mean()),
    'median': float(engine_lifetimes.median()),
    'min': int(engine_lifetimes.min()),
    'max': int(engine_lifetimes.max()),
    'std': float(engine_lifetimes.std())
}

print(f"Engine Lifetime Statistics:")
for key, val in lifetime_stats.items():
    print(f"  {key}: {val}")

# Data quality
print("\n[6/6] Validating data quality...")
quality = {
    'completeness': 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
    'missing_values': int(df.isnull().sum().sum()),
    'duplicates': int(df.duplicated().sum())
}

print(f"Data Quality:")
print(f"  Completeness: {quality['completeness']:.2%}")
print(f"  Missing values: {quality['missing_values']}")
print(f"  Duplicates: {quality['duplicates']}")

# Save comprehensive insights
print("\nSaving insights summary...")
insights = {
    'dataset_info': {
        'total_rows': int(len(df)),
        'num_engines': int(df['unit_id'].nunique()),
        'num_sensors': len(sensor_cols),
        'time_range': f"{df['time_cycles'].min()} - {df['time_cycles'].max()} cycles"
    },
    'rul_statistics': rul_stats,
    'engine_lifetime_statistics': lifetime_stats,
    'data_quality': quality,
    'correlation_analysis': {
        'highly_correlated_pairs': high_corr_pairs,
        'num_high_correlations': len(high_corr_pairs)
    },
    'feature_importance': {
        'top_10_predictive_sensors': {k: float(v) for k, v in list(top_sensors.items())[:10]}
    },
    'key_findings': [
        f"Dataset contains {df['unit_id'].nunique()} engines with complete run-to-failure data",
        f"Average engine lifetime: {lifetime_stats['mean']:.1f} cycles (range: {lifetime_stats['min']}-{lifetime_stats['max']})",
        f"RUL ranges from {rul_stats['min']} to {rul_stats['max']} cycles",
        f"Data quality: {quality['completeness']:.1%} complete, {quality['missing_values']} missing values",
        f"Found {len(high_corr_pairs)} highly correlated sensor pairs (potential redundancy)",
        f"Top predictive sensor: {list(top_sensors.keys())[0]} (correlation: {list(top_sensors.values())[0]:.4f})",
        f"Recommended sensors for modeling: {', '.join(list(top_sensors.keys())[:5])}"
    ],
    'recommendations_for_epoch004': [
        "Focus feature engineering on top 10 sensors with highest RUL correlation",
        "Consider removing or combining highly correlated sensors to reduce multicollinearity",
        "Add rolling window features (mean, std, min, max) for temporal patterns",
        "Create degradation rate features (sensor change over time)",
        "Engineer RUL-based binary classification targets (e.g., failure within 30 cycles)"
    ]
}

with open(ARTIFACTS_PATH / 'eda_insights.json', 'w') as f:
    json.dump(insights, f, indent=2)

print(f"\n✓ Insights saved to: {ARTIFACTS_PATH / 'eda_insights.json'}")

print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)
print(f"\nGenerated files in {ARTIFACTS_PATH}:")
print("  - sensor_statistics.csv")
print("  - sensor_correlations.csv")
print("  - rul_correlations.csv")
print("  - eda_insights.json")
print("\nKey Findings:")
for finding in insights['key_findings']:
    print(f"  • {finding}")
print("\n✓ Ready for Epoch 004: Feature Engineering")
print("="*80)
