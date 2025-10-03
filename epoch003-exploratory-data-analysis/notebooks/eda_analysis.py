#!/usr/bin/env python3
"""
Epoch 003: Exploratory Data Analysis
Comprehensive analysis of NASA Turbofan data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json

# Setup paths
sys.path.insert(0, '/mnt/code/src')
from predictive_maintenance.data_utils import load_turbofan_data, calculate_rul, validate_quality

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = Path('/mnt/data/Preventative-Maintainance-Example/epoch002-data-wrangling')
ARTIFACTS_PATH = Path('/mnt/artifacts/epoch003-exploratory-data-analysis')
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EPOCH 003: EXPLORATORY DATA ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA AND CALCULATE RUL
# ============================================================================
print("\n[1/7] Loading data and calculating RUL...")

df = load_turbofan_data(DATA_PATH / 'fd001_train.parquet')
df = calculate_rul(df)

print(f"Dataset shape: {df.shape}")
print(f"Number of engines: {df['unit_id'].nunique()}")
print(f"Columns: {len(df.columns)}")

# Identify sensor columns
sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
op_setting_cols = [col for col in df.columns if col.startswith('op_setting')]

print(f"Sensor columns: {len(sensor_cols)}")
print(f"Operational settings: {len(op_setting_cols)}")

# ============================================================================
# 2. BASIC STATISTICS
# ============================================================================
print("\n[2/7] Computing basic statistics...")

stats = df[sensor_cols].describe()
stats.to_csv(ARTIFACTS_PATH / 'sensor_statistics.csv')

# RUL statistics
rul_stats = {
    'mean': float(df['RUL'].mean()),
    'median': float(df['RUL'].median()),
    'min': int(df['RUL'].min()),
    'max': int(df['RUL'].max()),
    'std': float(df['RUL'].std())
}

print(f"RUL Statistics:")
print(f"  Mean: {rul_stats['mean']:.1f} cycles")
print(f"  Median: {rul_stats['median']:.1f} cycles")
print(f"  Range: {rul_stats['min']} - {rul_stats['max']} cycles")

# ============================================================================
# 3. SENSOR DISTRIBUTIONS
# ============================================================================
print("\n[3/7] Creating sensor distribution visualizations...")

# Plot distributions for first 12 sensors
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, sensor in enumerate(sensor_cols[:12]):
    axes[idx].hist(df[sensor].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{sensor}', fontsize=10)
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ARTIFACTS_PATH / 'sensor_distributions_1-12.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot distributions for remaining sensors
fig, axes = plt.subplots(3, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, sensor in enumerate(sensor_cols[12:21]):
    axes[idx].hist(df[sensor].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{sensor}', fontsize=10)
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ARTIFACTS_PATH / 'sensor_distributions_13-21.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved sensor distribution plots")

# ============================================================================
# 4. TEMPORAL DEGRADATION PATTERNS
# ============================================================================
print("\n[4/7] Analyzing temporal degradation patterns...")

# Select 5 sample engines
sample_engines = df['unit_id'].unique()[:5]

# Plot sensor trends for sample engines
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

selected_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_15']

for idx, sensor in enumerate(selected_sensors):
    for engine_id in sample_engines:
        engine_data = df[df['unit_id'] == engine_id].sort_values('time_cycles')
        axes[idx].plot(engine_data['time_cycles'], engine_data[sensor],
                      alpha=0.6, label=f'Engine {engine_id}')

    axes[idx].set_title(f'{sensor} Over Time', fontsize=11)
    axes[idx].set_xlabel('Time Cycles')
    axes[idx].set_ylabel('Sensor Value')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ARTIFACTS_PATH / 'temporal_degradation_patterns.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved temporal degradation patterns")

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================
print("\n[5/7] Generating correlation heatmap...")

# Compute correlation matrix
correlation_matrix = df[sensor_cols].corr()

# Plot heatmap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Sensor Correlation Matrix', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(ARTIFACTS_PATH / 'sensor_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Find highly correlated pairs (>0.9)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append({
                'sensor1': correlation_matrix.columns[i],
                'sensor2': correlation_matrix.columns[j],
                'correlation': float(correlation_matrix.iloc[i, j])
            })

print(f"  ✓ Found {len(high_corr_pairs)} highly correlated sensor pairs (>0.9)")

# ============================================================================
# 6. RUL DISTRIBUTION ANALYSIS
# ============================================================================
print("\n[6/7] Analyzing RUL distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RUL distribution
axes[0].hist(df['RUL'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(df['RUL'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["RUL"].mean():.1f}')
axes[0].axvline(df['RUL'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["RUL"].median():.1f}')
axes[0].set_title('RUL Distribution', fontsize=12)
axes[0].set_xlabel('Remaining Useful Life (cycles)')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# RUL by engine
rul_by_engine = df.groupby('unit_id')['time_cycles'].max().sort_values()
axes[1].hist(rul_by_engine, bins=30, edgecolor='black', alpha=0.7, color='coral')
axes[1].set_title('Engine Lifetime Distribution', fontsize=12)
axes[1].set_xlabel('Total Cycles to Failure')
axes[1].set_ylabel('Number of Engines')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ARTIFACTS_PATH / 'rul_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved RUL distribution plots")

# ============================================================================
# 7. FEATURE IMPORTANCE (CORRELATION WITH RUL)
# ============================================================================
print("\n[7/7] Computing feature importance...")

# Calculate correlation with RUL
rul_correlations = df[sensor_cols].corrwith(df['RUL']).abs().sort_values(ascending=False)

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 8))
rul_correlations.plot(kind='barh', ax=ax, color='teal')
ax.set_title('Sensor Correlation with RUL (Absolute Values)', fontsize=12)
ax.set_xlabel('Absolute Correlation')
ax.set_ylabel('Sensor')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(ARTIFACTS_PATH / 'feature_importance_rul_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved feature importance plot")

# Get top 10 sensors
top_sensors = rul_correlations.head(10).to_dict()
print(f"\nTop 10 Most Predictive Sensors:")
for sensor, corr in list(top_sensors.items())[:10]:
    print(f"  {sensor}: {corr:.4f}")

# ============================================================================
# 8. SAVE INSIGHTS SUMMARY
# ============================================================================
print("\n[8/7] Creating insights summary...")

insights = {
    'dataset_info': {
        'total_rows': int(len(df)),
        'num_engines': int(df['unit_id'].nunique()),
        'num_sensors': len(sensor_cols),
        'time_range': f"{df['time_cycles'].min()} - {df['time_cycles'].max()} cycles"
    },
    'rul_statistics': rul_stats,
    'data_quality': {
        'completeness': 1.0,
        'missing_values': int(df.isnull().sum().sum()),
        'duplicates': int(df.duplicated().sum())
    },
    'highly_correlated_pairs': high_corr_pairs,
    'top_10_predictive_sensors': {k: float(v) for k, v in list(top_sensors.items())[:10]},
    'key_findings': [
        f"Dataset contains {df['unit_id'].nunique()} engines with complete run-to-failure data",
        f"Average engine lifetime: {rul_by_engine.mean():.1f} cycles",
        f"RUL ranges from {rul_stats['min']} to {rul_stats['max']} cycles",
        f"Found {len(high_corr_pairs)} highly correlated sensor pairs (potential redundancy)",
        f"Top predictive sensor: {list(top_sensors.keys())[0]} (correlation: {list(top_sensors.values())[0]:.4f})"
    ]
}

with open(ARTIFACTS_PATH / 'eda_insights.json', 'w') as f:
    json.dump(insights, f, indent=2)

print("  ✓ Saved insights summary")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)
print(f"\nArtifacts saved to: {ARTIFACTS_PATH}")
print("\nGenerated files:")
print("  - sensor_statistics.csv")
print("  - sensor_distributions_1-12.png")
print("  - sensor_distributions_13-21.png")
print("  - temporal_degradation_patterns.png")
print("  - sensor_correlation_heatmap.png")
print("  - rul_distribution.png")
print("  - feature_importance_rul_correlation.png")
print("  - eda_insights.json")
print("\nReady for Epoch 004: Feature Engineering")
print("="*80)
