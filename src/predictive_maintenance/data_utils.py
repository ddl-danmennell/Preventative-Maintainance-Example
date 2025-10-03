"""
Data Processing Utilities for Predictive Maintenance
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_turbofan_data(file_path):
    """Load turbofan parquet data"""
    return pd.read_parquet(file_path)

def calculate_rul(df, group_col='unit_id'):
    """Calculate Remaining Useful Life for each timestep"""
    df = df.copy()
    max_cycles = df.groupby(group_col)['time_cycles'].transform('max')
    df['RUL'] = max_cycles - df['time_cycles']
    return df

def add_rolling_features(df, sensor_cols, windows=[5, 10, 20]):
    """Add rolling mean and std for sensors"""
    df = df.copy()
    for sensor in sensor_cols:
        for window in windows:
            df[f'{sensor}_rolling_mean_{window}'] = (
                df.groupby('unit_id')[sensor]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df[f'{sensor}_rolling_std_{window}'] = (
                df.groupby('unit_id')[sensor]
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
    return df

def split_by_units(df, test_size=0.2, random_state=42):
    """Split data by unit_id to prevent leakage"""
    np.random.seed(random_state)
    unique_units = df['unit_id'].unique()
    n_test = int(len(unique_units) * test_size)
    test_units = np.random.choice(unique_units, n_test, replace=False)

    test_df = df[df['unit_id'].isin(test_units)].copy()
    train_df = df[~df['unit_id'].isin(test_units)].copy()
    return train_df, test_df

def validate_quality(df):
    """Validate data quality"""
    total_cells = len(df) * len(df.columns)
    missing = df.isnull().sum().sum()
    completeness = 1 - (missing / total_cells)

    return {
        'completeness': completeness,
        'missing_values': missing,
        'passed': completeness > 0.85
    }
