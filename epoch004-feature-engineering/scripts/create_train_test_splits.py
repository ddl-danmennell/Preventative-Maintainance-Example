"""
Create train/test splits for Epoch 005 model development
Splits by unit_id to ensure no data leakage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

# Paths
DATA_PATH = Path('/mnt/data/Preventative-Maintainance-Example/epoch004-feature-engineering')
OUTPUT_PATH = Path('/mnt/data/Preventative-Maintainance-Example/epoch004-feature-engineering')
ARTIFACTS_PATH = Path('/mnt/artifacts/epoch004-feature-engineering')

print("=" * 80)
print("CREATING TRAIN/TEST SPLITS")
print("=" * 80)
print()

# Load engineered dataset
print("Loading engineered dataset...")
df = pd.read_parquet(DATA_PATH / 'fd001_engineered_features.parquet')
print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
print(f"Total engines: {df['unit_id'].nunique()}")
print()

# Get unique engine IDs
unique_engines = df['unit_id'].unique()
print(f"Splitting {len(unique_engines)} engines into train/test sets...")

# Train/test split (80/20 by engines)
train_engines, test_engines = train_test_split(
    unique_engines,
    test_size=0.2,
    random_state=42
)

print(f"Training engines: {len(train_engines)}")
print(f"Testing engines: {len(test_engines)}")
print()

# Split data
train_df = df[df['unit_id'].isin(train_engines)].copy()
test_df = df[df['unit_id'].isin(test_engines)].copy()

print(f"Training set: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
print(f"Testing set: {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")
print()

# Verify no overlap
assert len(set(train_engines) & set(test_engines)) == 0, "Data leakage: engines in both sets!"
print("Verified: No engine overlap between train and test sets")
print()

# Target distribution analysis
print("Target Variable Distribution:")
print("-" * 80)

print("\nRUL Statistics:")
print(f"  Train - Mean: {train_df['RUL'].mean():.1f}, Std: {train_df['RUL'].std():.1f}")
print(f"  Test  - Mean: {test_df['RUL'].mean():.1f}, Std: {test_df['RUL'].std():.1f}")

print("\nFailure Imminent (RUL < 30):")
print(f"  Train - Positive: {train_df['failure_imminent'].sum():,} ({train_df['failure_imminent'].mean()*100:.1f}%)")
print(f"  Test  - Positive: {test_df['failure_imminent'].sum():,} ({test_df['failure_imminent'].mean()*100:.1f}%)")

print("\nRUL Category Distribution:")
for category in range(4):
    train_count = (train_df['rul_category'] == category).sum()
    test_count = (test_df['rul_category'] == category).sum()
    print(f"  Category {category}: Train={train_count:,}, Test={test_count:,}")
print()

# Save splits
print("Saving train/test datasets...")

train_file = OUTPUT_PATH / 'fd001_train.parquet'
test_file = OUTPUT_PATH / 'fd001_test.parquet'

train_df.to_parquet(train_file, index=False)
test_df.to_parquet(test_file, index=False)

train_size_mb = train_file.stat().st_size / (1024 * 1024)
test_size_mb = test_file.stat().st_size / (1024 * 1024)

print(f"Train set saved: {train_file} ({train_size_mb:.2f} MB)")
print(f"Test set saved: {test_file} ({test_size_mb:.2f} MB)")
print()

# Save split metadata
split_metadata = {
    "train_engines": train_engines.tolist(),
    "test_engines": test_engines.tolist(),
    "train_size": len(train_df),
    "test_size": len(test_df),
    "split_ratio": 0.8,
    "random_seed": 42,
    "target_distributions": {
        "rul": {
            "train_mean": float(train_df['RUL'].mean()),
            "test_mean": float(test_df['RUL'].mean()),
            "train_std": float(train_df['RUL'].std()),
            "test_std": float(test_df['RUL'].std())
        },
        "failure_imminent": {
            "train_positive_rate": float(train_df['failure_imminent'].mean()),
            "test_positive_rate": float(test_df['failure_imminent'].mean())
        },
        "rul_category": {
            "train": train_df['rul_category'].value_counts().to_dict(),
            "test": test_df['rul_category'].value_counts().to_dict()
        }
    }
}

metadata_file = ARTIFACTS_PATH / 'train_test_split_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(split_metadata, f, indent=2)

print(f"Split metadata saved: {metadata_file}")
print()

print("=" * 80)
print("TRAIN/TEST SPLIT COMPLETE")
print("=" * 80)
