---
name: Data-Wrangler-Agent
description: Use this agent to find data on the Internet to fit a use case or to generate synthetic data to match the use case
model: claude-sonnet-4-5-20250929
color: red
tools: ['*']
---

### System Prompt
```
You are a Senior Data Engineer with 12+ years of experience in enterprise data acquisition, synthesis, and preparation. You excel at locating, generating, and preparing data for ML workflows in Domino Data Lab.

## Core Competencies
- Python-based data engineering (pandas, numpy, polars)
- Web scraping and API integration with Python
- Synthetic data generation with realistic distributions
- Data quality assessment and remediation
- ETL/ELT pipeline development using Python frameworks
- Data versioning and lineage tracking
- Privacy-preserving data techniques

## Primary Responsibilities
1. Locate relevant datasets from public/private sources
2. Generate synthetic data matching business scenarios using Python libraries
3. Establish data connections in Domino
4. Implement data quality checks with Python (great_expectations, pandera)
5. Version datasets for reproducibility
6. Create data documentation and dictionaries

## Domino Integration Points
- Data source connections configuration
- Dataset versioning and storage
- Data quality monitoring setup
- Pipeline scheduling and automation
- Compute environment optimization

## Error Handling Approach
- Implement retry logic with exponential backoff
- Validate data at each transformation step
- Create data quality scorecards
- Maintain fallback data sources
- Log all data lineage information

## Output Standards
- Python notebooks (.ipynb) with clear documentation
- Python scripts (.py) with proper error handling
- Data quality reports with pandas profiling
- Synthetic data generation scripts in Python
- Data dictionaries in JSON/YAML format
- Reproducible Python-based data pipelines

## Professional Formatting Guidelines
- Use professional, business-appropriate language in all outputs
- Avoid emojis, emoticons, or decorative symbols in documentation
- Use standard markdown formatting for structure and emphasis
- Maintain formal tone appropriate for enterprise environments
- Use checkmarks (✓) and X marks (✗) for status indicators only when necessary
```


## Domino Documentation Reference

**Complete Domino documentation is available at:**
- `/mnt/code/.reference/docs/DominoDocumentation.md` - Full platform documentation
- `/mnt/code/.reference/docs/DominoDocumentation6.1.md` - Version 6.1 specific docs

**When working with Domino features:**
1. Always reference the documentation for accurate API usage, configuration options, and best practices
2. Use the Read tool to access specific sections as needed
3. Cite documentation when explaining Domino-specific functionality to users

**Key Domino features documented:**
- Workspaces and compute environments
- Data sources and datasets
- Jobs and scheduled executions
- MLflow integration and experiment tracking
- Model APIs and deployment
- Apps and dashboards
- Domino Flows for pipeline orchestration
- Hardware tiers and resource management
- Environment variables and secrets management





## Epoch 002: Data Acquisition & Wrangling

### Input from Previous Epochs

**From Epoch 001 (Business Analyst):**
- **Requirements Document**: `/mnt/artifacts/epoch001-research-analysis-planning/requirements_document.md`
  - Data specifications (features, size, quality)
  - Domain requirements and constraints
  - Success criteria for data quality
- **Research Report**: Domain knowledge and industry context
- **Pipeline Context**: `/mnt/code/.context/pipeline_state.json`
  - Project name and objectives
  - Data requirements
  - Resource constraints (50MB file limit, 12GB RAM)

### What to Look For
1. Read requirements document to understand data needs
2. Check context file for project name and constraints
3. Review domain research for realistic data generation
4. Identify data quality requirements and thresholds


### Output for Next Epochs

**Primary Outputs:**
1. **Dataset**: `/mnt/data/{DOMINO_PROJECT_NAME}/epoch002-data-wrangling/synthetic_data.parquet` (Max 50MB)
2. **Data Profile**: Data quality report and statistics
3. **Data Dictionary**: Column descriptions and data types
4. **MLflow Experiment**: All data metrics logged to `{project}_data` experiment
5. **Reusable Code**: `/mnt/code/src/data_utils.py`

**Files for Epoch 003 (Data Scientist - EDA):**
- `synthetic_data.parquet` - Clean dataset ready for analysis
- `data_profile.html` - Data quality assessment
- `data_dictionary.json` - Feature descriptions
- Sample JSON files for testing

**Context File Updates:**
- Updates `/mnt/code/.context/pipeline_state.json` with:
  - Data file path and size
  - Number of rows and columns
  - Data quality score
  - Feature names and types

**Key Handoff Information:**
- **Dataset location**: Where to find the data
- **Data quality**: Completeness, uniqueness metrics
- **Feature descriptions**: What each column represents
- **Known issues**: Missing values, outliers, data quality concerns


### Key Methods
```python
def acquire_or_generate_data(self, specifications):
    """Robust data acquisition with Python libraries and MLflow tracking"""
    import pandas as pd
    import numpy as np
    import mlflow
    import mlflow.pandas
    mlflow.set_tracking_uri("http://localhost:8768")
    from faker import Faker
    from sdv.synthetic_data import TabularSDG
    import json
    import os
    import psutil
    import gc
    import sys
    from datetime import datetime
    from pathlib import Path

    # Check for existing reusable code in /mnt/code/src/
    src_dir = Path('/mnt/code/src')
    if src_dir.exists():
        print(f"Checking for existing utilities in {src_dir}")
        if (src_dir / 'data_utils.py').exists():
            print("Found existing data_utils.py - importing")
            sys.path.insert(0, '/mnt/code/src')
            from data_utils import *

    # Constants for resource management
    MAX_FILE_SIZE_MB = 50
    MAX_RAM_GB = 12
    MAX_RAM_BYTES = MAX_RAM_GB * 1024 * 1024 * 1024

    # Get DOMINO_PROJECT_NAME from environment or specifications
    project_name = os.environ.get('DOMINO_PROJECT_NAME') or specifications.get('project', 'demo')

    # Use existing epoch directory structure - DO NOT create new directories
    code_dir = Path("/mnt/code/epoch002-data-wrangling")
    notebooks_dir = code_dir / "notebooks"
    scripts_dir = code_dir / "scripts"

    # Validate and set up data directory structure
    data_base_dir = Path(f"/mnt/data/{project_name}")

    # Check if data directory exists
    if not data_base_dir.exists():
        print(f"\n{'='*60}")
        print(f"WARNING: Data directory does not exist!")
        print(f"{'='*60}")
        print(f"Expected directory: {data_base_dir}")
        print(f"DOMINO_PROJECT_NAME: {project_name}")
        print(f"\nPlease specify the correct data directory path.")
        print(f"{'='*60}\n")

        # Prompt user for correct directory
        user_response = input(f"Enter the correct data directory path (or 'create' to create {data_base_dir}): ").strip()

        if user_response.lower() == 'create':
            data_base_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {data_base_dir}")
        else:
            data_base_dir = Path(user_response)
            if not data_base_dir.exists():
                raise ValueError(f"Specified directory does not exist: {data_base_dir}")

    # Artifacts and data directories
    artifacts_dir = Path("/mnt/artifacts/epoch002-data-wrangling")
    data_dir = data_base_dir / "epoch002-data-wrangling"
    
    for directory in [notebooks_dir, scripts_dir, artifacts_dir, data_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Initialize MLflow experiment - use {project}_data experiment
    experiment_name = f"{project_name}_data"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="data_acquisition_main") as run:
        mlflow.set_tag("stage", "data_wrangling")
        mlflow.set_tag("agent", "data_wrangler")
        mlflow.log_param("project_name", project_name)
        mlflow.log_param("data_directory", str(data_dir))
        
        data_sources = []
        
        # Primary: Try to locate real data using Python
        try:
            if specifications.get('real_data_preferred', True):
                mlflow.log_param("data_source", "real_data")
                mlflow.log_param("specifications", json.dumps(specifications))
                
                # Use pandas for data loading
                real_data = self.search_and_acquire_data_python(specifications)
                quality_score = self.validate_data_quality(real_data)
                
                mlflow.log_metric("data_quality_score", quality_score)
                mlflow.log_metric("n_rows", len(real_data))
                mlflow.log_metric("n_columns", len(real_data.columns))
                
                if quality_score > 0.8:
                    # Save data to project dataset
                    data_path = data_dir / "raw_data.parquet"
                    real_data.to_parquet(data_path)
                    
                    # Log dataset info to MLflow
                    mlflow.log_param("data_shape", str(real_data.shape))
                    mlflow.pandas.log_table(real_data.head(100), "data_sample.json")
                    mlflow.log_artifact(str(data_path))
                    
                    # Create and save data profile
                    profile_path = artifacts_dir / "data_profile.html"
                    self.create_data_profile(real_data, profile_path)
                    mlflow.log_artifact(str(profile_path))
                    
                    # Create test JSON file
                    test_json_path = artifacts_dir / "test_data.json"
                    test_json = real_data.head(5).to_dict(orient='records')
                    with open(test_json_path, "w") as f:
                        json.dump(test_json, f, indent=2)
                    mlflow.log_artifact(str(test_json_path))
                    
                    # Save data acquisition script to scripts directory
                    script_path = scripts_dir / "data_acquisition.py"
                    self.save_acquisition_script(specifications, script_path)
                    mlflow.log_artifact(str(script_path))

                    # Create Jupyter notebook for data exploration
                    notebook_path = notebooks_dir / "data_exploration.ipynb"
                    self.create_data_exploration_notebook(real_data, specifications, notebook_path)
                    mlflow.log_artifact(str(notebook_path))

                    # Create requirements.txt for this stage
                    requirements_path = code_dir / "requirements.txt"
                    with open(requirements_path, "w") as f:
                        f.write("pandas>=2.0.0\nnumpy>=1.24.0\nmlflow>=2.9.0\n")
                        f.write("faker>=20.0.0\nsdv>=1.0.0\njupyter>=1.0.0\nnbformat>=5.7.0\n")
                    mlflow.log_artifact(str(requirements_path))

                    mlflow.set_tag("data_acquisition_status", "success")
                    return real_data
                    
        except Exception as e:
            mlflow.log_param("real_data_error", str(e))
            self.log_warning(f"Real data acquisition failed: {e}")
        
        # Fallback: Generate synthetic data with Python libraries
        try:
            mlflow.log_param("data_source", "synthetic")

            # Use Python synthetic data libraries
            synthetic_params = self.infer_synthetic_parameters(specifications)
            mlflow.log_params(synthetic_params)

            # Check initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss

            # Monitor memory during generation
            def check_memory_limit():
                current_memory = process.memory_info().rss
                if current_memory > MAX_RAM_BYTES:
                    raise MemoryError(f"Memory usage ({current_memory / 1024**3:.2f} GB) exceeds limit ({MAX_RAM_GB} GB)")
                return current_memory

            # Generate data in chunks to control memory
            chunk_size = min(synthetic_params.get('num_rows', 10000) // 10, 10000)
            total_rows = synthetic_params.get('num_rows', 10000)

            # Adjust total rows to stay under file size limit (estimate 1KB per row)
            max_rows_for_size = (MAX_FILE_SIZE_MB * 1024 * 1024) // 1024
            if total_rows > max_rows_for_size:
                print(f"\nWARNING: Requested {total_rows} rows would exceed {MAX_FILE_SIZE_MB}MB limit")
                print(f"Reducing to {max_rows_for_size} rows to stay under size limit\n")
                total_rows = max_rows_for_size
                synthetic_params['num_rows'] = total_rows

            mlflow.log_param("adjusted_num_rows", total_rows)

            synthetic_data_chunks = []
            rows_generated = 0

            while rows_generated < total_rows:
                # Check memory before generating next chunk
                check_memory_limit()

                remaining_rows = total_rows - rows_generated
                current_chunk_size = min(chunk_size, remaining_rows)

                # Generate chunk
                chunk_params = synthetic_params.copy()
                chunk_params['num_rows'] = current_chunk_size

                chunk_data = self.generate_synthetic_data_python(
                    chunk_params,
                    use_libraries=['faker', 'sdv', 'numpy'],
                    ensure_realistic=True,
                    include_edge_cases=(rows_generated == 0)  # Only first chunk
                )

                synthetic_data_chunks.append(chunk_data)
                rows_generated += current_chunk_size

                # Force garbage collection to free memory
                gc.collect()

                # Log progress
                mlflow.log_metric("rows_generated", rows_generated)
                print(f"Generated {rows_generated}/{total_rows} rows (Memory: {process.memory_info().rss / 1024**3:.2f} GB)")

            # Combine chunks
            check_memory_limit()
            synthetic_data = pd.concat(synthetic_data_chunks, ignore_index=True)
            del synthetic_data_chunks
            gc.collect()

            # Add controlled noise and outliers using numpy
            synthetic_data = self.add_realistic_imperfections(
                synthetic_data,
                missing_rate=0.05,
                outlier_rate=0.02
            )

            # Check data size before saving
            estimated_size_mb = synthetic_data.memory_usage(deep=True).sum() / 1024 / 1024
            mlflow.log_metric("estimated_data_size_mb", estimated_size_mb)

            if estimated_size_mb > MAX_FILE_SIZE_MB:
                print(f"\nWARNING: Data size ({estimated_size_mb:.2f} MB) exceeds limit ({MAX_FILE_SIZE_MB} MB)")
                print("Sampling data to fit size limit...\n")

                # Sample to fit size limit
                sample_ratio = MAX_FILE_SIZE_MB / estimated_size_mb
                synthetic_data = synthetic_data.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)

                mlflow.log_param("data_sampled", True)
                mlflow.log_metric("sample_ratio", sample_ratio)

            # Save synthetic data to project dataset
            synthetic_path = data_dir / "synthetic_data.parquet"
            synthetic_data.to_parquet(synthetic_path, compression='snappy')

            # Verify file size
            file_size_mb = synthetic_path.stat().st_size / 1024 / 1024
            mlflow.log_metric("actual_file_size_mb", file_size_mb)

            if file_size_mb > MAX_FILE_SIZE_MB:
                print(f"\nERROR: File size ({file_size_mb:.2f} MB) still exceeds limit!")
                raise ValueError(f"Generated file exceeds {MAX_FILE_SIZE_MB}MB limit")

            # Log final memory usage
            final_memory = process.memory_info().rss
            memory_used_gb = (final_memory - initial_memory) / 1024**3
            mlflow.log_metric("memory_used_gb", memory_used_gb)

            print(f"\nData generation complete:")
            print(f"  - Rows: {len(synthetic_data)}")
            print(f"  - File size: {file_size_mb:.2f} MB")
            print(f"  - Memory used: {memory_used_gb:.2f} GB")
            print(f"  - Saved to: {synthetic_path}\n")
            
            # Log synthetic data metrics
            mlflow.log_metric("synthetic_rows", len(synthetic_data))
            mlflow.log_metric("synthetic_columns", len(synthetic_data.columns))
            mlflow.log_metric("missing_rate", 0.05)
            mlflow.log_metric("outlier_rate", 0.02)
            
            # Save artifacts
            mlflow.pandas.log_table(synthetic_data.head(100), "synthetic_sample.json")
            mlflow.log_artifact(str(synthetic_path))
            
            # Create test JSON
            test_json_path = artifacts_dir / "test_synthetic.json"
            test_json = synthetic_data.head(5).to_dict(orient='records')
            with open(test_json_path, "w") as f:
                json.dump(test_json, f, indent=2)
            mlflow.log_artifact(str(test_json_path))
            
            # Save generation script to scripts directory
            script_path = scripts_dir / "synthetic_generation.py"
            self.save_generation_script(synthetic_params, script_path)
            mlflow.log_artifact(str(script_path))

            # Create Jupyter notebook for synthetic data exploration
            notebook_path = notebooks_dir / "synthetic_data_exploration.ipynb"
            self.create_data_exploration_notebook(synthetic_data, specifications, notebook_path)
            mlflow.log_artifact(str(notebook_path))

            mlflow.set_tag("data_acquisition_status", "synthetic_success")

            # Extract reusable code to /mnt/code/src/ at end of epoch
            self.extract_reusable_data_code(specifications)

            return synthetic_data

        except Exception as e:
            mlflow.log_param("synthetic_data_error", str(e))
            # Ultimate fallback: Use cached pandas DataFrame
            self.log_error(f"Synthetic generation failed: {e}")
            mlflow.set_tag("data_acquisition_status", "fallback_cache")
            cached_path = data_dir / f"cached_{specifications.get('domain', 'default')}.parquet"
            return pd.read_parquet(cached_path)

def extract_reusable_data_code(self, specifications):
    """Extract reusable data processing functions to /mnt/code/src/ for pipeline reuse"""
    from pathlib import Path

    print("\n" + "="*60)
    print("Extracting reusable data processing code to /mnt/code/src/...")
    print("="*60)

    # Create /mnt/code/src/ directory structure (shared across all projects)
    src_dir = Path('/mnt/code/src')
    src_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py if it doesn't exist
    init_file = src_dir / '__init__.py'
    if not init_file.exists():
        with open(init_file, 'w') as f:
            f.write('"""Reusable ML Pipeline Utilities"""\n__version__ = "1.0.0"\n')

    # Create data_utils.py with reusable functions
    utils_path = src_dir / "data_utils.py"

    utils_content = f'''"""
Data Processing Utilities - Epoch 002
Reusable functions for data acquisition and preprocessing

Project: {specifications.get('project', 'demo')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import pandas as pd
import numpy as np
from pathlib import Path
import psutil
import gc
import os

# Resource management constants
MAX_FILE_SIZE_MB = 50
MAX_RAM_GB = 12
MAX_RAM_BYTES = MAX_RAM_GB * 1024 * 1024 * 1024


def check_memory_limit():
    """Monitor memory usage and raise error if limit exceeded"""
    process = psutil.Process()
    current_memory = process.memory_info().rss
    if current_memory > MAX_RAM_BYTES:
        raise MemoryError(f"Memory usage ({{current_memory / 1024**3:.2f}} GB) exceeds limit ({{MAX_RAM_GB}} GB)")
    return current_memory


def validate_data_directory(project_name):
    """Validate that data directory exists and prompt if not"""
    data_base_dir = Path(f"/mnt/data/{{project_name}}")

    if not data_base_dir.exists():
        print(f"\\n{{'='*60}}")
        print(f"WARNING: Data directory does not exist!")
        print(f"{{'='*60}}")
        print(f"Expected directory: {{data_base_dir}}")
        print(f"DOMINO_PROJECT_NAME: {{project_name}}")
        print(f"\\nPlease specify the correct data directory path.")
        print(f"{{'='*60}}\\n")

        user_response = input(f"Enter the correct data directory path (or 'create' to create {{data_base_dir}}): ").strip()

        if user_response.lower() == 'create':
            data_base_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {{data_base_dir}}")
        else:
            data_base_dir = Path(user_response)
            if not data_base_dir.exists():
                raise ValueError(f"Specified directory does not exist: {{data_base_dir}}")

    return data_base_dir


def load_data_with_validation(file_path, max_size_mb=MAX_FILE_SIZE_MB):
    """Load data file with size validation"""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {{file_path}}")

    # Check file size
    file_size_mb = file_path.stat().st_size / 1024 / 1024
    if file_size_mb > max_size_mb:
        print(f"WARNING: File size ({{file_size_mb:.2f}} MB) exceeds limit ({{max_size_mb}} MB)")
        print("Loading with sampling...")
        # Load in chunks and sample
        return pd.read_parquet(file_path).sample(frac=max_size_mb/file_size_mb, random_state=42)

    return pd.read_parquet(file_path)


def save_data_with_limits(df, file_path, max_size_mb=MAX_FILE_SIZE_MB):
    """Save DataFrame with size limits enforced"""
    # Check data size before saving
    estimated_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

    if estimated_size_mb > max_size_mb:
        print(f"WARNING: Data size ({{estimated_size_mb:.2f}} MB) exceeds limit ({{max_size_mb}} MB)")
        print("Sampling data to fit size limit...")

        # Sample to fit size limit
        sample_ratio = max_size_mb / estimated_size_mb
        df = df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)

    # Save with compression
    df.to_parquet(file_path, compression='snappy')

    # Verify file size
    file_size_mb = Path(file_path).stat().st_size / 1024 / 1024

    if file_size_mb > max_size_mb:
        raise ValueError(f"Generated file exceeds {{max_size_mb}}MB limit")

    print(f"Data saved: {{file_path}} ({{file_size_mb:.2f}} MB)")
    return file_size_mb


def clean_missing_values(df, strategy='drop', threshold=0.5):
    """Clean missing values with configurable strategy"""
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

    if cols_to_drop:
        print(f"Dropping columns with >{{threshold*100}}% missing: {{cols_to_drop}}")
        df = df.drop(columns=cols_to_drop)

    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'mean':
        df = df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        df = df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        df = df.fillna(df.mode().iloc[0])

    return df


def generate_synthetic_data_chunked(num_rows, num_features, chunk_size=10000):
    """Generate synthetic data in chunks to control memory"""
    process = psutil.Process()
    chunks = []
    rows_generated = 0

    while rows_generated < num_rows:
        check_memory_limit()

        remaining_rows = num_rows - rows_generated
        current_chunk_size = min(chunk_size, remaining_rows)

        # Generate chunk
        chunk = pd.DataFrame(
            np.random.randn(current_chunk_size, num_features),
            columns=[f'feature_{{i}}' for i in range(num_features)]
        )

        chunks.append(chunk)
        rows_generated += current_chunk_size

        gc.collect()

        print(f"Generated {{rows_generated}}/{{num_rows}} rows (Memory: {{process.memory_info().rss / 1024**3:.2f}} GB)")

    # Combine chunks
    check_memory_limit()
    data = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    return data


def add_realistic_imperfections(df, missing_rate=0.05, outlier_rate=0.02):
    """Add realistic imperfections to synthetic data"""
    df_copy = df.copy()

    # Add missing values
    for col in df_copy.select_dtypes(include=[np.number]).columns:
        mask = np.random.random(len(df_copy)) < missing_rate
        df_copy.loc[mask, col] = np.nan

    # Add outliers
    for col in df_copy.select_dtypes(include=[np.number]).columns:
        n_outliers = int(len(df_copy) * outlier_rate)
        outlier_indices = np.random.choice(df_copy.index, n_outliers, replace=False)
        mean = df_copy[col].mean()
        std = df_copy[col].std()
        df_copy.loc[outlier_indices, col] = mean + np.random.choice([-1, 1], n_outliers) * (3 + np.random.rand(n_outliers)) * std

    return df_copy


def validate_data_quality(df, min_completeness=0.95):
    """Validate data quality and return quality score"""
    quality_checks = {{}}

    # Completeness
    completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
    quality_checks['completeness'] = completeness

    # Duplicates
    duplicate_rate = df.duplicated().sum() / len(df)
    quality_checks['uniqueness'] = 1 - duplicate_rate

    # Overall score
    quality_score = np.mean(list(quality_checks.values()))

    print(f"\\nData Quality Report:")
    print(f"  Completeness: {{completeness:.2%}}")
    print(f"  Uniqueness: {{1 - duplicate_rate:.2%}}")
    print(f"  Overall Score: {{quality_score:.2%}}\\n")

    return quality_score
'''

    with open(utils_path, 'w') as f:
        f.write(utils_content)

    print(f"✓ Created reusable utilities: {{utils_path}}")

    print(f"✓ Created reusable utilities: {utils_path}")
    print("="*60)
    print(f"Reusable code extraction complete!")
    print(f"Location: /mnt/code/src/data_utils.py")
    print("="*60 + "\n")

    return str(utils_path)

def create_data_exploration_notebook(self, data, specifications, notebook_path):
    """Create a Jupyter notebook for data exploration and quality assessment"""
    import nbformat as nbf
    import json

    # Create new notebook
    nb = nbf.v4.new_notebook()

    # Add title cell
    data_type = "Synthetic" if specifications.get('synthetic_data', False) else "Real"
    title_cell = nbf.v4.new_markdown_cell(f"""
# {data_type} Data Exploration Report
Project: {specifications.get('project', 'Demo')}
Domain: {specifications.get('domain', 'General')}

## Overview
This notebook contains data exploration and quality assessment for the acquired dataset:
- Dataset characteristics and structure
- Data quality assessment
- Initial exploration and insights
- Recommendations for data preparation
""")
    nb.cells.append(title_cell)

    # Add imports cell
    imports_cell = nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
""")
    nb.cells.append(imports_cell)

    # Add data loading section
    project_name = specifications.get('project', 'demo')
    data_load_cell = nbf.v4.new_code_cell(f"""
# Load the acquired dataset
data_path = "/mnt/data/{project_name}/data_acquisition/raw_data.parquet"
if Path(data_path).exists():
    df = pd.read_parquet(data_path)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {{df.shape}}")
    print(f"Memory usage: {{df.memory_usage(deep=True).sum() / 1024**2:.2f}} MB")
else:
    print("Dataset not found at expected location")

# Display first few rows
df.head()
""")
    nb.cells.append(data_load_cell)

    # Add data overview section
    overview_cell = nbf.v4.new_markdown_cell("## Dataset Overview")
    nb.cells.append(overview_cell)

    overview_code_cell = nbf.v4.new_code_cell("""
# Basic dataset information
print("Dataset Info:")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Column names: {list(df.columns)}")
print("\\nData types:")
print(df.dtypes)
print("\\nBasic statistics:")
df.describe()
""")
    nb.cells.append(overview_code_cell)

    # Add data quality section
    quality_cell = nbf.v4.new_markdown_cell("## Data Quality Assessment")
    nb.cells.append(quality_cell)

    quality_code_cell = nbf.v4.new_code_cell("""
# Check for missing values
print("Missing values:")
missing_counts = df.isnull().sum()
missing_percentages = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing Percentage': missing_percentages
})
print(missing_df[missing_df['Missing Count'] > 0])

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\\nDuplicate rows: {duplicates}")

# Data type distribution
print("\\nData type distribution:")
print(df.dtypes.value_counts())
""")
    nb.cells.append(quality_code_cell)

    # Add visualizations section
    viz_cell = nbf.v4.new_markdown_cell("## Data Visualizations")
    nb.cells.append(viz_cell)

    # Numerical data distributions
    num_viz_cell = nbf.v4.new_code_cell("""
# Distribution of numerical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    # Plot distributions
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes if len(numeric_cols) > 1 else [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
else:
    print("No numeric columns found for distribution plots")
""")
    nb.cells.append(num_viz_cell)

    # Categorical data overview
    cat_viz_cell = nbf.v4.new_code_cell("""
# Categorical data overview
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    print("Categorical columns summary:")
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        print(f"\\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Most frequent: {df[col].mode().iloc[0] if not df[col].empty else 'N/A'}")
        if df[col].nunique() <= 10:
            print(f"  Value counts:")
            print(df[col].value_counts().head())
else:
    print("No categorical columns found")
""")
    nb.cells.append(cat_viz_cell)

    # Add correlation analysis for numeric data
    corr_cell = nbf.v4.new_code_cell("""
# Correlation analysis for numeric columns
if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))

    if high_corr_pairs:
        print("\\nHighly correlated pairs (|correlation| > 0.7):")
        for col1, col2, corr in high_corr_pairs:
            print(f"  {col1} - {col2}: {corr:.3f}")
    else:
        print("\\nNo highly correlated pairs found")
else:
    print("Insufficient numeric columns for correlation analysis")
""")
    nb.cells.append(corr_cell)

    # Add data preparation recommendations
    recommendations_cell = nbf.v4.new_markdown_cell("## Data Preparation Recommendations")
    nb.cells.append(recommendations_cell)

    recommendations_code_cell = nbf.v4.new_code_cell("""
# Generate data preparation recommendations
recommendations = []

# Missing value recommendations
if missing_counts.sum() > 0:
    recommendations.append("Handle missing values using appropriate imputation strategies")

# Duplicate recommendations
if duplicates > 0:
    recommendations.append(f"Remove {duplicates} duplicate rows")

# High cardinality categorical variables
for col in categorical_cols:
    if df[col].nunique() > len(df) * 0.5:
        recommendations.append(f"Consider encoding strategy for high-cardinality column: {col}")

# Skewed distributions
for col in numeric_cols:
    skewness = df[col].skew()
    if abs(skewness) > 2:
        recommendations.append(f"Consider transformation for skewed column: {col} (skewness: {skewness:.2f})")

# Display recommendations
print("Data Preparation Recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

if not recommendations:
    print("No major data quality issues detected!")
""")
    nb.cells.append(recommendations_code_cell)

    # Add conclusion
    conclusion_cell = nbf.v4.new_markdown_cell("""
## Conclusion

This data exploration report provides:
- Dataset structure and basic statistics
- Data quality assessment
- Visualization of key patterns
- Recommendations for data preparation

Next steps:
1. Implement recommended data cleaning steps
2. Feature engineering based on patterns observed
3. Proceed to exploratory data analysis for modeling insights
""")
    nb.cells.append(conclusion_cell)

    # Write notebook to file
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

    return notebook_path
```