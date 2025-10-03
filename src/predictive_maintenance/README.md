# Predictive Maintenance Utilities

Reusable code extracted from Epoch 002 data wrangling.

## Installation

```python
import sys
sys.path.insert(0, '/mnt/code/src')
from predictive_maintenance.data_utils import *
```

## Functions

### load_turbofan_data(file_path)
Load turbofan parquet data

### calculate_rul(df, group_col='unit_id')
Calculate Remaining Useful Life for each timestep

### add_rolling_features(df, sensor_cols, windows=[5,10,20])
Add rolling mean and std for sensors

### split_by_units(df, test_size=0.2, random_state=42)
Split data by unit_id to prevent data leakage

### validate_quality(df)
Validate data quality (completeness, missing values)

## Example

```python
from predictive_maintenance.data_utils import *

# Load data
df = load_turbofan_data('/mnt/data/.../fd001_train.parquet')

# Add RUL target
df = calculate_rul(df)

# Add features
sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
df = add_rolling_features(df, sensor_cols[:5])

# Split data
train, test = split_by_units(df, test_size=0.2)
```
