#!/usr/bin/env python3
import pandas as pd
import sys
sys.path.insert(0, '/mnt/code/src')
from predictive_maintenance.data_utils import validate_quality

# Load datasets
fd001 = pd.read_parquet('/mnt/data/Preventative-Maintainance-Example/epoch002-data-wrangling/fd001_train.parquet')
fd002 = pd.read_parquet('/mnt/data/Preventative-Maintainance-Example/epoch002-data-wrangling/fd002_train.parquet')

print("="*60)
print("QUALITY GATE VALIDATION")
print("="*60)

# Validate FD001
fd001_quality = validate_quality(fd001)
print(f"\nFD001 Quality:")
print(f"  Completeness: {fd001_quality['completeness']:.2%} (threshold: >85%)")
print(f"  Missing values: {fd001_quality['missing_values']}")
print(f"  Status: {'✓ PASS' if fd001_quality['passed'] else '✗ FAIL'}")

# Validate FD002
fd002_quality = validate_quality(fd002)
print(f"\nFD002 Quality:")
print(f"  Completeness: {fd002_quality['completeness']:.2%} (threshold: >85%)")
print(f"  Missing values: {fd002_quality['missing_values']}")
print(f"  Status: {'✓ PASS' if fd002_quality['passed'] else '✗ FAIL'}")

# Overall
all_passed = fd001_quality['passed'] and fd002_quality['passed']
print(f"\n{'='*60}")
print(f"OVERALL STATUS: {'✓ ALL GATES PASSED' if all_passed else '✗ FAILED'}")
print(f"{'='*60}")

sys.exit(0 if all_passed else 1)
