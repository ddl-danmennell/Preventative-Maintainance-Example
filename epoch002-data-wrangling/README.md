# Epoch 002: Data Wrangling

**Agent**: Data-Wrangler-Agent | **MLflow**: `{project}_data` | **Duration**: 2-4 hours

## Purpose

Acquire or generate high-quality datasets with resource-safe execution (50MB limit, 12GB RAM monitoring).

## What You'll Do

1. **Data Sourcing**: Find data from APIs, databases, or generate synthetic data
2. **Quality Validation**: Assess completeness, accuracy, and consistency
3. **Data Cleaning**: Handle missing values, outliers, and duplicates
4. **Privacy Compliance**: Apply anonymization and comply with regulations
5. **Resource Management**: Enforce 50MB file limits and 12GB RAM monitoring

## How to Use This Epoch

**Command Template**:
```
"Generate synthetic [domain] data with [X] features and [Y] rows"
```

**Example Commands**:
- `"Generate synthetic credit card transaction data with 15 features and 100K rows"`
- `"Create synthetic patient records with demographics and lab results for 50K patients"`
- `"Generate financial trading data with OHLCV and technical indicators for 200K records"`

## Critical Resource Management

**50MB File Size Limit**:
- Automatic enforcement on all saved data files
- Parquet format with Snappy compression
- Automatic sampling if limit exceeded
- Verification after save

**12GB RAM Monitoring**:
- Continuous memory tracking with psutil
- Chunked processing (10K rows per chunk)
- Progress display: "Generated 50000/100000 rows (Memory: 8.5 GB)"
- MemoryError raised if limit exceeded

**Directory Validation**:
- Validates `/mnt/data/{DOMINO_PROJECT_NAME}/` exists
- Prompts user if missing: Type `create` or provide custom path
- No silent failures

## Outputs

**Generated Files**:
- `notebooks/data_generation.ipynb` - Data creation process
- `scripts/data_quality_check.py` - Quality validation
- `/mnt/data/{DOMINO_PROJECT_NAME}/epoch002-data-wrangling/`
  - `synthetic_data.parquet` - **≤50MB dataset**
  - `data_quality_report.json` - Quality metrics
- `/mnt/artifacts/epoch002-data-wrangling/`
  - `data_profile.html` - Interactive data report
  - `requirements.txt` - Dependencies

**Logged to MLflow** (`{project}_data` experiment):
- Dataset statistics (rows, columns, memory usage, file size)
- Data quality metrics (completeness, uniqueness, validity)
- Missing value patterns
- Resource usage (peak memory, generation time)

**Reusable Code**: `/mnt/code/src/data_utils.py`
- Data loading and validation
- Quality assessment functions
- Chunked data generation
- Memory monitoring utilities

## Cross-Agent Communication

**Reads from Pipeline Context**:
- `epoch001.business_requirements` - Data specifications
- `epoch001.regulatory_frameworks` - Privacy constraints

**Writes to Pipeline Context**:
```json
{
  "epoch002": {
    "data_path": "/mnt/data/{project}/epoch002-data-wrangling/synthetic_data.parquet",
    "num_rows": 100000,
    "num_features": 15,
    "target_variable": "is_fraud",
    "file_size_mb": 48.5,
    "memory_usage_gb": 9.2,
    "quality_score": 0.95
  }
}
```

**Used By**:
- **Epoch 003** (EDA): Data for analysis
- **Epoch 004** (Feature Engineering): Raw features
- **Epoch 005+**: Training datasets

## Success Criteria

✅ Dataset saved to validated directory
✅ File size ≤50MB with compression
✅ Memory usage stayed under 12GB
✅ Data quality score ≥0.90
✅ Privacy compliance verified
✅ MLflow tracking complete

## Troubleshooting

**Memory Error**: Reduce rows or features, increase chunk size limit
**File Too Large**: Agent auto-samples, but you can regenerate with fewer rows
**Directory Missing**: Agent prompts for creation or custom path

## Next Steps

Proceed to **Epoch 003: Exploratory Data Analysis** to understand patterns and relationships.

---

**Ready to start?** Use the Data-Wrangler-Agent with your data specifications.
