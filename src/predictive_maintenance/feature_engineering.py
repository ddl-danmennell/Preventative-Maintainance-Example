"""
Feature Engineering Utilities for Predictive Maintenance
Reusable functions for creating temporal, degradation, and interaction features
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

def add_rolling_features_extended(df: pd.DataFrame,
                                  sensor_cols: List[str],
                                  windows: List[int] = [5, 10, 20, 50],
                                  group_col: str = 'unit_id') -> pd.DataFrame:
    """
    Add comprehensive rolling statistics for sensor columns

    Args:
        df: DataFrame with sensor data
        sensor_cols: List of sensor column names
        windows: List of rolling window sizes
        group_col: Column to group by (default: 'unit_id')

    Returns:
        DataFrame with additional rolling features
    """
    df = df.copy()

    for sensor in sensor_cols:
        for window in windows:
            # Rolling mean
            df[f'{sensor}_rolling_mean_{window}'] = (
                df.groupby(group_col)[sensor]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )

            # Rolling std
            df[f'{sensor}_rolling_std_{window}'] = (
                df.groupby(group_col)[sensor]
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )

            # Rolling min
            df[f'{sensor}_rolling_min_{window}'] = (
                df.groupby(group_col)[sensor]
                .transform(lambda x: x.rolling(window, min_periods=1).min())
            )

            # Rolling max
            df[f'{sensor}_rolling_max_{window}'] = (
                df.groupby(group_col)[sensor]
                .transform(lambda x: x.rolling(window, min_periods=1).max())
            )

    return df


def add_degradation_features(df: pd.DataFrame,
                             sensor_cols: List[str],
                             group_col: str = 'unit_id') -> pd.DataFrame:
    """
    Add degradation rate features (derivatives, trends, EWMA)

    Args:
        df: DataFrame with sensor data
        sensor_cols: List of sensor column names
        group_col: Column to group by (default: 'unit_id')

    Returns:
        DataFrame with additional degradation features
    """
    df = df.copy()

    for sensor in sensor_cols:
        # First derivative (change per cycle)
        df[f'{sensor}_diff'] = (
            df.groupby(group_col)[sensor]
            .transform(lambda x: x.diff())
        )

        # Rate of change over last 10 cycles
        df[f'{sensor}_rate_10'] = (
            df.groupby(group_col)[sensor]
            .transform(lambda x: (x - x.shift(10)) / 10)
        )

        # Exponentially weighted moving average
        df[f'{sensor}_ewma'] = (
            df.groupby(group_col)[sensor]
            .transform(lambda x: x.ewm(span=10, adjust=False).mean())
        )

        # Cumulative sum of changes
        df[f'{sensor}_cumsum_diff'] = (
            df.groupby(group_col)[sensor]
            .transform(lambda x: x.diff().fillna(0).cumsum())
        )

    return df


def add_interaction_features(df: pd.DataFrame,
                             sensor_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Add cross-sensor interaction features (ratios, products, differences)

    Args:
        df: DataFrame with sensor data
        sensor_pairs: List of (sensor_a, sensor_b) tuples

    Returns:
        DataFrame with additional interaction features
    """
    df = df.copy()

    for sensor_a, sensor_b in sensor_pairs:
        # Ratio
        df[f'{sensor_a}_div_{sensor_b}'] = df[sensor_a] / (df[sensor_b] + 1e-6)

        # Product
        df[f'{sensor_a}_mult_{sensor_b}'] = df[sensor_a] * df[sensor_b]

        # Difference
        df[f'{sensor_a}_minus_{sensor_b}'] = df[sensor_a] - df[sensor_b]

    return df


def add_lifecycle_features(df: pd.DataFrame,
                           time_col: str = 'time_cycles',
                           group_col: str = 'unit_id') -> pd.DataFrame:
    """
    Add lifecycle position features

    Args:
        df: DataFrame with time series data
        time_col: Time/cycle column name
        group_col: Column to group by (default: 'unit_id')

    Returns:
        DataFrame with lifecycle features
    """
    df = df.copy()

    # Normalized time (percentage through lifecycle)
    max_cycles = df.groupby(group_col)[time_col].transform('max')
    df['lifecycle_pct'] = (df[time_col] / max_cycles) * 100

    # Cycle count features
    df['cycles_elapsed'] = df[time_col]

    return df


def add_deviation_features(df: pd.DataFrame,
                           sensor_cols: List[str],
                           group_col: str = 'unit_id') -> pd.DataFrame:
    """
    Add deviation from engine-specific baseline features

    Args:
        df: DataFrame with sensor data
        sensor_cols: List of sensor column names
        group_col: Column to group by (default: 'unit_id')

    Returns:
        DataFrame with deviation features
    """
    df = df.copy()

    for sensor in sensor_cols:
        engine_mean = df.groupby(group_col)[sensor].transform('mean')
        engine_std = df.groupby(group_col)[sensor].transform('std')

        # Z-score deviation from engine baseline
        df[f'{sensor}_dev_from_mean'] = (df[sensor] - engine_mean) / (engine_std + 1e-6)

    return df


def create_target_variables(df: pd.DataFrame,
                            rul_col: str = 'RUL',
                            failure_threshold: int = 30) -> pd.DataFrame:
    """
    Create multiple target variable formulations

    Args:
        df: DataFrame with RUL column
        rul_col: RUL column name
        failure_threshold: Threshold for binary classification

    Returns:
        DataFrame with additional target variables
    """
    df = df.copy()

    # Binary classification: failure imminent
    df['failure_imminent'] = (df[rul_col] < failure_threshold).astype(int)

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

    df['rul_category'] = df[rul_col].apply(rul_bucket)

    return df


def engineer_all_features(df: pd.DataFrame,
                          top_sensors: List[str],
                          all_sensors: List[str] = None,
                          rolling_windows: List[int] = [5, 10, 20, 50],
                          sensor_pairs: List[Tuple[str, str]] = None) -> pd.DataFrame:
    """
    Apply all feature engineering transformations

    Args:
        df: DataFrame with raw sensor data and RUL
        top_sensors: List of top sensors for intensive feature engineering
        all_sensors: List of all sensor columns (optional)
        rolling_windows: Window sizes for rolling features
        sensor_pairs: Sensor pairs for interaction features (optional)

    Returns:
        DataFrame with all engineered features
    """
    # Default sensor pairs if not provided
    if sensor_pairs is None:
        sensor_pairs = [
            (top_sensors[0], top_sensors[1]),
            (top_sensors[2], top_sensors[3]),
            (top_sensors[4], top_sensors[5]) if len(top_sensors) > 5 else (top_sensors[0], top_sensors[2])
        ]

    # Sort by unit_id and time for temporal features
    df = df.sort_values(['unit_id', 'time_cycles']).reset_index(drop=True)

    # Add rolling features
    df = add_rolling_features_extended(df, top_sensors, rolling_windows)

    # Add degradation features
    df = add_degradation_features(df, top_sensors)

    # Add interaction features
    df = add_interaction_features(df, sensor_pairs)

    # Add lifecycle features
    df = add_lifecycle_features(df)

    # Add deviation features (top 5 sensors only)
    df = add_deviation_features(df, top_sensors[:5])

    # Add target variables
    if 'RUL' in df.columns:
        df = create_target_variables(df)

    # Handle missing values from diff/rolling operations
    df = df.fillna(method='bfill').fillna(0)

    return df
