# Visualization Standards for Jupyter Notebooks

All JupyterLab notebooks must include informative visualizations relevant to the use case.

## Required Visualizations by Epoch

### Epoch 002 (Data Wrangling)
- Data distribution plots (histograms, box plots)
- Missing value heatmaps
- Target variable distribution
- Feature correlation preview
- Data quality summary charts

### Epoch 003 (EDA)
- Comprehensive correlation heatmaps
- Feature distribution plots (by target class)
- Outlier detection visualizations (scatter plots, box plots)
- Bivariate analysis plots (pair plots for key features)
- Time series plots (if temporal data)
- Statistical summary visualizations
- Class imbalance visualization

### Epoch 004 (Feature Engineering)
- Feature importance bar charts (preliminary)
- Before/after feature transformation comparisons
- Polynomial feature correlation matrices
- Interaction effect visualizations
- Feature scaling distribution plots
- Dimensionality reduction plots (PCA, t-SNE if applicable)

### Epoch 005 (Model Development)
- Training vs validation loss curves
- Confusion matrices
- ROC curves and AUC scores
- Precision-Recall curves
- Feature importance rankings (model-specific)
- Learning curves
- Residual plots (for regression)
- Cross-validation score distributions

### Epoch 006 (Model Testing)
- Edge case performance heatmaps
- Prediction distribution by input range
- Calibration plots
- Error analysis by feature bins
- Robustness testing results (perturbation impact)
- Performance across different data segments

### Epoch 007 (Deployment & Apps)
- API latency histograms
- Prediction confidence distributions
- Real-time monitoring dashboards
- Model drift detection plots
- A/B test results (if applicable)

### Epoch 008 (Retrospective)
- Timeline visualization (actual vs estimated)
- Resource utilization over time
- Performance metric progression across epochs
- Cost breakdown by epoch
- Success criteria achievement radar chart

## Standard Imports and Setup

```python
# Standard imports for all notebooks
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set style for consistency
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Figure size standards
SMALL_FIG = (8, 6)
MEDIUM_FIG = (12, 8)
LARGE_FIG = (16, 10)
```

## Reusable Visualization Functions

### Feature Importance Plot

```python
def plot_feature_importance(importance_df, title="Feature Importance", top_n=20):
    """
    Create informative feature importance plot

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        title: Plot title
        top_n: Number of top features to display
    """
    # Sort and select top N
    top_features = importance_df.nlargest(top_n, 'importance')

    # Create horizontal bar chart
    fig = px.bar(
        top_features,
        y='feature',
        x='importance',
        orientation='h',
        title=title,
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        height=max(400, top_n * 25),  # Dynamic height based on features
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )

    fig.show()

    # Save to artifacts
    fig.write_html(f"/mnt/artifacts/{epoch}/feature_importance.html")
    fig.write_image(f"/mnt/artifacts/{epoch}/feature_importance.png")
```

### Confusion Matrix

```python
def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """
    Create annotated confusion matrix
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels or [f"Class {i}" for i in range(len(cm))],
        y=labels or [f"Class {i}" for i in range(len(cm))],
        title=title,
        text_auto=True,
        color_continuous_scale='Blues'
    )

    fig.update_layout(height=600, width=700)
    fig.show()

    # Save
    fig.write_html(f"/mnt/artifacts/{epoch}/confusion_matrix.html")
```

### Distribution Comparison

```python
def plot_distribution_comparison(df, feature, target, title=None):
    """
    Compare feature distribution across target classes
    """
    fig = px.histogram(
        df,
        x=feature,
        color=target,
        marginal="box",
        title=title or f"{feature} Distribution by {target}",
        barmode='overlay',
        opacity=0.7
    )

    fig.update_layout(height=500, width=900)
    fig.show()

    # Save
    fig.write_html(f"/mnt/artifacts/{epoch}/{feature}_distribution.html")
```

### Interactive Correlation Heatmap

```python
def plot_correlation_heatmap(df, title="Feature Correlation Matrix"):
    """
    Create interactive correlation heatmap
    """
    corr = df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=title,
        height=800,
        width=900,
        xaxis={'side': 'bottom'}
    )

    fig.show()

    # Save
    fig.write_html(f"/mnt/artifacts/{epoch}/correlation_heatmap.html")
```

### ROC Curve

```python
def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """
    Create ROC curve visualization
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.2f})',
        line=dict(color='darkorange', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=600,
        width=700
    )

    fig.show()

    # Save
    fig.write_html(f"/mnt/artifacts/{epoch}/roc_curve.html")
```

### Learning Curve

```python
def plot_learning_curve(train_scores, val_scores, train_sizes, title="Learning Curve"):
    """
    Create learning curve visualization
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_scores,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_scores,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='orange', width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Training Examples',
        yaxis_title='Score',
        height=500,
        width=800
    )

    fig.show()

    # Save
    fig.write_html(f"/mnt/artifacts/{epoch}/learning_curve.html")
```

### Missing Values Heatmap

```python
def plot_missing_values_heatmap(df, title="Missing Values Heatmap"):
    """
    Visualize missing values pattern
    """
    import numpy as np

    missing = df.isnull()

    fig = px.imshow(
        missing.T,
        labels=dict(x="Sample Index", y="Features", color="Missing"),
        title=title,
        color_continuous_scale=['lightblue', 'darkred'],
        aspect='auto'
    )

    fig.update_layout(height=max(400, len(df.columns) * 10), width=1000)
    fig.show()

    # Save
    fig.write_html(f"/mnt/artifacts/{epoch}/missing_values_heatmap.html")
```

### Distribution Grid

```python
def plot_distribution_grid(df, features, ncols=3, title="Feature Distributions"):
    """
    Create grid of distribution plots
    """
    import numpy as np
    import math

    nrows = math.ceil(len(features) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
    axes = axes.flatten() if nrows > 1 else [axes]

    for idx, feature in enumerate(features):
        if idx < len(axes):
            df[feature].hist(bins=30, ax=axes[idx], edgecolor='black')
            axes[idx].set_title(feature)
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')

    # Hide unused subplots
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"/mnt/artifacts/{epoch}/distribution_grid.png", dpi=150, bbox_inches='tight')
    plt.show()
```

## Notebook Structure Standards

```python
# Cell 1: Title and Overview
"""
# Epoch 00X: [Epoch Name]
## Project: [Project Name]
### Date: [Date]

**Objective:** [Clear objective statement]

**Key Questions:**
1. Question 1
2. Question 2
3. Question 3
"""

# Cell 2: Imports and Setup
# [All imports with clear organization]

# Cell 3: Load Data and Initial Inspection
# [Data loading with shape, head, info]

# Cells 4-N: Analysis with Visualizations
# Each analysis section should include:
# - Markdown cell explaining what you're investigating
# - Code cell performing analysis
# - Visualization cell with informative plot
# - Markdown cell with insights and findings

# Final Cell: Summary and Next Steps
"""
## Summary of Findings

1. **Key Finding 1:** Description
2. **Key Finding 2:** Description
3. **Key Finding 3:** Description

## Recommendations for Next Epoch

- Recommendation 1
- Recommendation 2
- Recommendation 3

## Artifacts Saved

- `/mnt/artifacts/{epoch}/plot1.html`
- `/mnt/artifacts/{epoch}/plot2.png`
- `/mnt/data/{project}/{epoch}/processed_data.parquet`
"""
```

## Quality Standards

- All plots must have clear titles, axis labels, and legends
- Use color palettes appropriate for the use case (e.g., diverging for correlations)
- Save interactive HTML versions for exploration
- Save static PNG versions for reports
- Include statistical annotations where relevant (p-values, confidence intervals)
- Use appropriate plot types for data (scatter for continuous, bar for categorical)
- Ensure plots are accessible (consider colorblind-friendly palettes)
- Add context with text annotations for key insights

## Notebook Execution Requirements

**ALWAYS run all cells** in the notebook after creation and **ALWAYS save the notebook** with output cells intact.

This ensures:
- Code validation (confirms notebooks execute without errors)
- Output preservation (images, tables, and results saved in notebook)
- Immediate visibility (stakeholders can view results without re-running)
- Reproducibility verification (confirms environment and dependencies work)

### Execution Pattern

```python
# After creating notebook, execute all cells programmatically:
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

def execute_and_save_notebook(notebook_path):
    """Execute all cells in notebook and save with outputs"""
    # Load notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Execute notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': str(Path(notebook_path).parent)}})
        print(f"Successfully executed: {notebook_path}")
    except Exception as e:
        print(f"Error executing {notebook_path}: {e}")
        raise

    # Save with outputs
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"Notebook saved with outputs: {notebook_path}")

# Usage after creating notebook:
notebook_path = "/mnt/code/epoch003-exploratory-data-analysis/notebooks/eda_analysis.ipynb"
execute_and_save_notebook(notebook_path)
```

### Alternative: Manual execution via bash

```bash
# Execute notebook and save outputs
jupyter nbconvert --to notebook --execute --inplace /mnt/code/epoch003-exploratory-data-analysis/notebooks/eda_analysis.ipynb
```
