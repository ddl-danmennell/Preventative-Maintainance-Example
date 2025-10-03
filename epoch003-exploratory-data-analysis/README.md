# Epoch 003: Exploratory Data Analysis

**Agent**: Data-Scientist-Agent | **MLflow**: `{project}_data` | **Duration**: 2-3 hours

## Purpose

Perform comprehensive statistical analysis to understand data patterns, relationships, and quality for informed feature engineering.

## What You'll Do

1. **Statistical Summaries**: Distribution analysis, central tendency, variance
2. **Correlation Analysis**: Feature relationships and multicollinearity detection
3. **Visualization Creation**: Histograms, box plots, scatter plots, heatmaps
4. **Outlier Detection**: Identify anomalies and data quality issues
5. **Insight Generation**: Actionable recommendations for feature engineering

## How to Use This Epoch

**Command Template**:
```
"Perform EDA on [dataset] focusing on [target variable]"
```

**Example Commands**:
- `"Perform comprehensive EDA on credit data focusing on default prediction"`
- `"Analyze patient dataset with emphasis on readmission patterns"`
- `"Explore transaction data to identify fraud indicators"`

## Key Analysis Areas

**Univariate Analysis**:
- Distribution shapes (normal, skewed, bimodal)
- Missing value patterns and frequencies
- Outlier identification using IQR and Z-scores
- Cardinality for categorical features

**Bivariate Analysis**:
- Feature-target correlations
- Categorical vs numerical relationships
- Interaction effects identification

**Multivariate Analysis**:
- Correlation matrices and heatmaps
- Dimensionality reduction (PCA visualization)
- Feature importance preliminary rankings

## Outputs

**Generated Files**:
- `notebooks/exploratory_data_analysis.ipynb` - Comprehensive EDA with visualizations
- `scripts/data_profiling.py` - Automated profiling script
- `/mnt/artifacts/epoch003-exploratory-data-analysis/`
  - `visualizations/` - All plots and charts (PNG/HTML)
  - `statistical_summary.html` - Interactive data report
  - `insights_report.md` - Key findings and recommendations
  - `requirements.txt` - Dependencies

**Logged to MLflow** (`{project}_data` experiment):
- Dataset statistics and distributions
- Correlation coefficients
- Missing value percentages
- Outlier counts
- Key insights as metrics

**Reusable Code**: **None** (pure analysis epoch)

## Cross-Agent Communication

**Reads from Pipeline Context**:
- `epoch002.data_path` - Dataset location
- `epoch002.target_variable` - Variable to analyze
- `epoch001.business_requirements` - Analysis focus areas

**Writes to Pipeline Context**:
```json
{
  "epoch003": {
    "key_insights": [
      "Income and education highly correlated with default",
      "Age shows non-linear relationship with target",
      "Missing values primarily in employment_length (15%)"
    ],
    "recommended_features": ["income", "debt_ratio", "age_group"],
    "features_to_drop": ["customer_id", "application_date"],
    "encoding_strategy": {
      "employment_type": "one_hot",
      "education": "ordinal",
      "state": "target_encoding"
    },
    "scaling_needed": ["income", "debt_amount", "credit_score"],
    "outliers_detected": 342,
    "correlation_threshold": 0.8
  }
}
```

**Used By**:
- **Epoch 004** (Feature Engineering): Transformation strategy
- **Epoch 005** (Model Development): Feature selection
- **Epoch 006** (Model Testing): Expected value ranges

## Success Criteria

✅ All features profiled with distributions
✅ Correlation analysis completed
✅ Outliers identified and documented
✅ Missing value patterns analyzed
✅ Insights generated for feature engineering
✅ Visualizations created and saved
✅ Recommendations documented

## Key Deliverables for Next Epoch

**For Feature Engineering (Epoch 004)**:
1. Features requiring encoding (categorical → numerical)
2. Features requiring scaling (standardization/normalization)
3. Features with outliers needing treatment
4. Potential interaction features to create
5. Features to drop (high cardinality, leakage, redundancy)

## Analysis Checklist

- [ ] Load data from Epoch 002
- [ ] Generate descriptive statistics for all features
- [ ] Create distribution plots for numerical features
- [ ] Analyze categorical feature frequencies
- [ ] Calculate correlation matrix
- [ ] Identify and visualize outliers
- [ ] Analyze missing value patterns
- [ ] Test statistical assumptions (normality, homoscedasticity)
- [ ] Document insights and recommendations
- [ ] Log all findings to MLflow

## Next Steps

Proceed to **Epoch 004: Feature Engineering** using insights to create ML-ready features.

---

**Ready to start?** Use the Data-Scientist-Agent with your analysis focus.
