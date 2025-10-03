#!/usr/bin/env python3
"""
Generate NASA Turbofan-style synthetic data
Simplified version for direct execution
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
DATA_DIR = Path("/mnt/data/Preventative-Maintainance-Example/epoch002-data-wrangling")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Generating NASA Turbofan-style datasets...")
print(f"Output directory: {DATA_DIR}")

# Column names
index_names = ['unit_id', 'time_cycles']
operational_settings = ['op_setting_1', 'op_setting_2', 'op_setting_3']
sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
all_columns = index_names + operational_settings + sensor_columns

def generate_engine_data(engine_id, num_cycles, operating_condition='single'):
    """Generate data for one engine"""

    # Operating settings based on condition
    if operating_condition == 'single':
        op1 = np.full(num_cycles, 0.0)
        op2 = np.full(num_cycles, 0.0006)
        op3 = np.full(num_cycles, 100.0)
    else:  # multi
        condition_idx = np.random.randint(0, 6)
        op1 = np.random.uniform(-0.0007, 0.0007, num_cycles)
        op2 = np.random.uniform(0.0002, 0.0010, num_cycles)
        op3 = np.random.uniform(80, 120, num_cycles)

    # Base sensor values (typical ranges)
    base_values = [
        518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36,
        2388.06, 9046.19, 1.30, 47.47, 521.66, 2388.02, 8138.62,
        8.4195, 0.03, 392, 2388, 100.00, 38.86, 23.4190
    ]

    # Generate sensor readings with degradation
    data_rows = []
    for cycle in range(1, num_cycles + 1):
        degradation = 1 + (cycle / num_cycles) * 0.15  # 15% degradation over life

        row = {
            'unit_id': engine_id,
            'time_cycles': cycle,
            'op_setting_1': op1[cycle-1],
            'op_setting_2': op2[cycle-1],
            'op_setting_3': op3[cycle-1]
        }

        # Add sensor readings with noise and degradation
        for i, base in enumerate(base_values):
            noise = np.random.normal(0, base * 0.02)  # 2% noise
            row[f'sensor_{i+1}'] = base * degradation + noise

        data_rows.append(row)

    return pd.DataFrame(data_rows)

# Generate FD001 dataset (single operating condition, 100 engines)
print("\nGenerating FD001 dataset (100 engines, single condition)...")
fd001_data = []
for engine_id in range(1, 101):
    num_cycles = np.random.randint(128, 356)
    engine_df = generate_engine_data(engine_id, num_cycles, 'single')
    fd001_data.append(engine_df)
    if engine_id % 20 == 0:
        print(f"  Generated {engine_id}/100 engines...")

fd001_df = pd.concat(fd001_data, ignore_index=True)
fd001_path = DATA_DIR / "fd001_train.parquet"
fd001_df.to_parquet(fd001_path, compression='snappy')
fd001_size = fd001_path.stat().st_size / 1024 / 1024

print(f"✓ FD001 saved: {fd001_path}")
print(f"  Shape: {fd001_df.shape}")
print(f"  Size: {fd001_size:.2f} MB")

# Generate FD002 dataset (multiple operating conditions, 260 engines)
print("\nGenerating FD002 dataset (260 engines, multiple conditions)...")
fd002_data = []
for engine_id in range(1, 261):
    num_cycles = np.random.randint(128, 378)
    engine_df = generate_engine_data(engine_id, num_cycles, 'multi')
    fd002_data.append(engine_df)
    if engine_id % 50 == 0:
        print(f"  Generated {engine_id}/260 engines...")

fd002_df = pd.concat(fd002_data, ignore_index=True)
fd002_path = DATA_DIR / "fd002_train.parquet"
fd002_df.to_parquet(fd002_path, compression='snappy')
fd002_size = fd002_path.stat().st_size / 1024 / 1024

print(f"✓ FD002 saved: {fd002_path}")
print(f"  Shape: {fd002_df.shape}")
print(f"  Size: {fd002_size:.2f} MB")

print("\n" + "="*60)
print("DATA GENERATION COMPLETE")
print("="*60)
print(f"Total size: {fd001_size + fd002_size:.2f} MB")
print(f"Total rows: {len(fd001_df) + len(fd002_df):,}")
print(f"FD001 engines: {fd001_df['unit_id'].nunique()}")
print(f"FD002 engines: {fd002_df['unit_id'].nunique()}")
