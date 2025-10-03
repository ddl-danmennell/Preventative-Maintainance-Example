# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains a collection of specialized Claude Code agents designed for building end-to-end machine learning demonstrations on the Domino Data Lab platform. The agents work together to create production-ready ML solutions with resource-safe execution, automated code reusability, and standardized Streamlit styling.

## Critical Operational Requirements

### Automated Quality Gates
**Each agent must validate prerequisites before starting:**
- Check `pipeline_state.json` for required inputs from previous epochs
- Validate data quality thresholds before proceeding
- Verify resource availability (disk space, memory)
- Confirm all dependencies from previous epochs are met

**Handoff criteria by epoch:**
- **Epoch 002 → 003**: Data file exists, size < 50MB, basic schema validation passed
- **Epoch 003 → 004**: EDA complete, data quality report generated, no critical issues
- **Epoch 004 → 005**: Features engineered, feature importance calculated, train/test split created
- **Epoch 005 → 006**: Model trained, baseline metrics logged, model artifact saved
- **Epoch 006 → 007**: Advanced testing complete, edge cases documented, compliance validated
- **Epoch 007 → 008**: Deployment successful, monitoring active, app functional

**Quality gate implementation:**
```python
def validate_epoch_prerequisites(current_epoch, pipeline_state_path="/mnt/code/.context/pipeline_state.json"):
    """Validate prerequisites before starting epoch"""
    import json
    from pathlib import Path

    if not Path(pipeline_state_path).exists():
        raise FileNotFoundError(f"Pipeline state not found. Run Epoch 001 first.")

    with open(pipeline_state_path) as f:
        state = json.load(f)

    required_epochs = {
        "003": ["002"],
        "004": ["002", "003"],
        "005": ["002", "003", "004"],
        "006": ["002", "003", "004", "005"],
        "007": ["002", "003", "004", "005", "006"],
        "008": ["001", "002", "003", "004", "005", "006", "007"]
    }

    if current_epoch in required_epochs:
        for req_epoch in required_epochs[current_epoch]:
            if not state.get(f"epoch_{req_epoch}_complete", False):
                raise ValueError(f"Epoch {req_epoch} must be completed before starting Epoch {current_epoch}")

    return state
```

### Data Directory Management
**ALWAYS validate data directory before proceeding:**
- Data MUST be saved to `/mnt/data/{DOMINO_PROJECT_NAME}/epoch00X-xxx/`
- Get project name from `DOMINO_PROJECT_NAME` environment variable
- If directory doesn't exist, PROMPT user with options:
  - Type `create` to create the directory
  - Enter custom path if different location needed
- **Never proceed without valid directory**

### Incremental Checkpointing
**All agents must implement checkpoint saving for long-running operations:**
- Save progress checkpoints every 10% or every 5 minutes
- Store in `/mnt/code/.context/checkpoints/epoch{XXX}_checkpoint.json`
- Enable resume from last checkpoint on failure
- Include metadata: timestamp, progress %, next step, intermediate artifacts

**Checkpoint structure:**
```python
{
  "epoch": "003",
  "checkpoint_id": "epoch003_step2_20251003_103045",
  "timestamp": "2025-10-03T10:30:45Z",
  "progress_percent": 40,
  "current_step": "feature_correlation_analysis",
  "next_step": "outlier_detection",
  "completed_steps": ["data_load", "missing_value_analysis"],
  "intermediate_artifacts": {
    "partial_eda_report": "/mnt/artifacts/epoch003/partial_report.html",
    "processed_data": "/mnt/code/.context/checkpoints/epoch003_data_step2.parquet"
  },
  "can_resume": true
}
```

**Resume implementation:**
```python
def load_checkpoint(epoch):
    """Load latest checkpoint for epoch"""
    from pathlib import Path
    import json

    checkpoint_dir = Path("/mnt/code/.context/checkpoints")
    checkpoints = list(checkpoint_dir.glob(f"epoch{epoch}_*.json"))

    if not checkpoints:
        return None

    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        return json.load(f)

def save_checkpoint(epoch, progress_percent, current_step, next_step, completed_steps, artifacts):
    """Save progress checkpoint"""
    import json
    from datetime import datetime
    from pathlib import Path

    checkpoint_dir = Path("/mnt/code/.context/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_id = f"epoch{epoch}_{current_step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint = {
        "epoch": epoch,
        "checkpoint_id": checkpoint_id,
        "timestamp": datetime.now().isoformat(),
        "progress_percent": progress_percent,
        "current_step": current_step,
        "next_step": next_step,
        "completed_steps": completed_steps,
        "intermediate_artifacts": artifacts,
        "can_resume": True
    }

    with open(checkpoint_dir / f"{checkpoint_id}.json", 'w') as f:
        json.dump(checkpoint, f, indent=2)
```

### Resource Management (Prevent Workspace Crashes)
**Mandatory limits for ALL data operations:**
- **File Size Limit**: 50MB maximum per file
- **RAM Limit**: 12GB maximum during operations
- **Chunked Processing**: 10,000 rows per chunk for data generation
- **Memory Monitoring**: Use psutil to check before each operation
- **Automatic Sampling**: Reduce data if limits exceeded
- **Progress Reporting**: Display "Generated X/Y rows (Memory: Z GB)"
- **Checkpoint Integration**: Save checkpoint every 10% progress or 5 minutes

### Cost and Resource Tracking (Optional/Configurable)
**Agents can optionally track compute costs and resource utilization:**

**Enable/Disable Configuration:**
```python
# In config.py or environment variable
ENABLE_COST_TRACKING = os.getenv("ENABLE_COST_TRACKING", "false").lower() == "true"
```

**Resource tracking implementation (when enabled):**
```python
import psutil
import time
from datetime import datetime
from pathlib import Path
import json

class ResourceTracker:
    """Optional resource and cost tracking"""

    def __init__(self, epoch, enabled=False):
        self.epoch = epoch
        self.enabled = enabled
        if not enabled:
            return

        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / (1024**3)  # GB
        self.metrics = {
            "epoch": epoch,
            "start_time": datetime.now().isoformat(),
            "compute_tier": os.getenv("DOMINO_HARDWARE_TIER", "unknown"),
            "samples": []
        }

    def sample(self):
        """Sample current resource usage"""
        if not self.enabled:
            return

        sample = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": time.time() - self.start_time,
            "memory_gb": psutil.Process().memory_info().rss / (1024**3),
            "cpu_percent": psutil.cpu_percent(interval=1)
        }
        self.metrics["samples"].append(sample)

    def finalize(self):
        """Calculate final metrics and optionally estimate cost"""
        if not self.enabled:
            return None

        elapsed_hours = (time.time() - self.start_time) / 3600
        peak_memory = max(s["memory_gb"] for s in self.metrics["samples"])
        avg_cpu = sum(s["cpu_percent"] for s in self.metrics["samples"]) / len(self.metrics["samples"])

        self.metrics.update({
            "end_time": datetime.now().isoformat(),
            "elapsed_hours": elapsed_hours,
            "peak_memory_gb": peak_memory,
            "avg_cpu_percent": avg_cpu
        })

        # Optional: Estimate cost (user can override with their own pricing)
        # This is disabled by default - users can implement their own cost calculation
        if os.getenv("ENABLE_COST_ESTIMATION", "false").lower() == "true":
            cost_per_hour = float(os.getenv("COST_PER_HOUR", "0"))
            self.metrics["estimated_cost_usd"] = elapsed_hours * cost_per_hour

        # Log to MLflow if enabled
        if os.getenv("LOG_RESOURCES_TO_MLFLOW", "true").lower() == "true":
            import mlflow
            mlflow.log_metrics({
                f"epoch_{self.epoch}_hours": elapsed_hours,
                f"epoch_{self.epoch}_peak_memory_gb": peak_memory,
                f"epoch_{self.epoch}_avg_cpu": avg_cpu
            })

        # Save to context
        context_path = Path("/mnt/code/.context/resource_tracking.json")
        if context_path.exists():
            with open(context_path) as f:
                tracking = json.load(f)
        else:
            tracking = {"epochs": {}}

        tracking["epochs"][self.epoch] = self.metrics

        with open(context_path, 'w') as f:
            json.dump(tracking, f, indent=2)

        return self.metrics

# Usage in agent code:
tracker = ResourceTracker(epoch="005", enabled=ENABLE_COST_TRACKING)

# Sample periodically during execution
tracker.sample()  # Call this in loops or after major steps

# Finalize at end
final_metrics = tracker.finalize()
if final_metrics:
    print(f"Epoch 005 completed in {final_metrics['elapsed_hours']:.2f} hours")
    print(f"Peak memory: {final_metrics['peak_memory_gb']:.2f} GB")
```

**Aggregate reporting (Epoch 008):**
```python
def generate_resource_report():
    """Generate resource utilization report across all epochs"""
    with open("/mnt/code/.context/resource_tracking.json") as f:
        tracking = json.load(f)

    total_hours = sum(e["elapsed_hours"] for e in tracking["epochs"].values())
    total_cost = sum(e.get("estimated_cost_usd", 0) for e in tracking["epochs"].values())

    report = {
        "total_compute_hours": total_hours,
        "total_estimated_cost_usd": total_cost,
        "by_epoch": tracking["epochs"]
    }

    return report
```

**Notes:**
- Cost tracking is **disabled by default**
- Easy to disable: set `ENABLE_COST_TRACKING=false` or don't set it
- Users can implement custom cost calculation logic
- Resource metrics logged to MLflow for analysis
- Useful for project planning and optimization in Epoch 008

### Data Lineage Tracking
**All agents must track data transformations in `/mnt/code/.context/data_lineage.json`:**
- Document every transformation applied to the data
- Track feature provenance (which features came from where)
- Enable traceability from final predictions back to source data
- Log data quality changes at each step

**Implementation pattern:**
```python
def update_data_lineage(epoch, transformation_type, details):
    """Update data lineage with new transformation"""
    import json
    from pathlib import Path
    from datetime import datetime

    lineage_path = Path("/mnt/code/.context/data_lineage.json")

    # Load existing lineage or create new
    if lineage_path.exists():
        with open(lineage_path) as f:
            lineage = json.load(f)
    else:
        lineage = {"source_data": {}, "transformations": [], "feature_provenance": {}}

    # Add transformation
    transformation = {
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "type": transformation_type,
        **details
    }
    lineage["transformations"].append(transformation)

    # Save updated lineage
    with open(lineage_path, 'w') as f:
        json.dump(lineage, f, indent=2)

# Example usage in Epoch 004:
update_data_lineage(
    epoch="004",
    transformation_type="feature_engineering",
    details={
        "features_added": ["credit_utilization_ratio", "payment_velocity"],
        "features_removed": ["raw_payment_amount"],
        "method": "polynomial_interactions",
        "rows_before": 100000,
        "rows_after": 100000
    }
)

# Document feature provenance
def add_feature_provenance(feature_name, source_features, formula, created_in_epoch):
    """Document how a feature was created"""
    import json
    from pathlib import Path

    lineage_path = Path("/mnt/code/.context/data_lineage.json")
    with open(lineage_path) as f:
        lineage = json.load(f)

    lineage["feature_provenance"][feature_name] = {
        "created_in_epoch": created_in_epoch,
        "source_features": source_features,
        "formula": formula,
        "timestamp": datetime.now().isoformat()
    }

    with open(lineage_path, 'w') as f:
        json.dump(lineage, f, indent=2)
```

### Code Reusability
**At end of each epoch, extract reusable code to `/mnt/code/src/{project}/`:**
- Create proper Python package structure with `__init__.py`
- Organize by epoch:
  - Epoch 002: `data_processing_utils.py`, `data_loading_pipeline.py`
  - Epoch 003: `feature_engineering.py`
  - Epoch 004: `model_utils.py`
  - Epoch 005: `validation.py`
  - Epoch 006: `deployment.py`, `monitoring.py`, `serving.py`
  - Epoch 007: `app_utils.py`, `streamlit_components.py`
- Include `README.md` with usage instructions
- Use relative imports (e.g., `from .data_processing_utils import *`)
- **Include unit tests** for all reusable functions

### Automated Testing Framework
**All agents must implement automated testing for their outputs:**

**Test Structure:**
```
/mnt/code/src/{project}/tests/
├── test_data_processing.py      # Epoch 002 tests
├── test_feature_engineering.py  # Epoch 004 tests
├── test_model_utils.py          # Epoch 005 tests
├── test_validation.py           # Epoch 006 tests
├── test_deployment.py           # Epoch 007 tests
└── conftest.py                  # Shared fixtures
```

**Testing Requirements by Epoch:**

**Epoch 002 (Data Wrangler):**
```python
import pytest
import pandas as pd
from pathlib import Path

def test_data_file_exists():
    """Verify data file was created"""
    data_path = Path(f"/mnt/data/{project_name}/epoch002-data-wrangling/data.parquet")
    assert data_path.exists(), "Data file not found"

def test_data_size_limit():
    """Verify data file < 50MB"""
    data_path = Path(f"/mnt/data/{project_name}/epoch002-data-wrangling/data.parquet")
    size_mb = data_path.stat().st_size / (1024 * 1024)
    assert size_mb < 50, f"Data file {size_mb}MB exceeds 50MB limit"

def test_data_schema():
    """Verify all required columns present"""
    df = pd.read_parquet(data_path)
    required_cols = ["feature1", "feature2", "target"]
    assert all(col in df.columns for col in required_cols)

def test_missing_values():
    """Verify missing values < 5%"""
    df = pd.read_parquet(data_path)
    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    assert missing_pct < 0.05, f"Missing values {missing_pct*100}% > 5%"
```

**Epoch 004 (Feature Engineering):**
```python
def test_feature_count():
    """Verify expected number of features"""
    df = pd.read_parquet(feature_data_path)
    assert len(df.columns) >= 40, "Insufficient features engineered"

def test_feature_provenance():
    """Verify all features documented in lineage"""
    import json
    with open("/mnt/code/.context/data_lineage.json") as f:
        lineage = json.load(f)
    assert len(lineage["feature_provenance"]) > 0
```

**Epoch 005 (Model Development):**
```python
def test_model_artifact_exists():
    """Verify model artifact saved"""
    import mlflow
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models(f"name='{project_name}_model'")
    assert len(models) > 0, "Model not registered in MLflow"

def test_baseline_performance():
    """Verify model meets baseline performance"""
    # Load model metrics from MLflow
    assert test_accuracy > 0.70, "Model accuracy below baseline"

def test_model_signature():
    """Verify model has valid signature"""
    model = mlflow.pyfunc.load_model(f"models:/{project_name}_model/latest")
    assert model.metadata.signature is not None
```

**Epoch 006 (Model Testing):**
```python
def test_edge_cases():
    """Verify model handles edge cases"""
    edge_cases = generate_edge_cases()
    predictions = model.predict(edge_cases)
    assert all(p in [0, 1] for p in predictions), "Invalid predictions"

def test_model_robustness():
    """Verify model stable under input perturbations"""
    X_perturbed = add_noise(X_test, noise_level=0.01)
    preds_original = model.predict(X_test)
    preds_perturbed = model.predict(X_perturbed)
    stability = (preds_original == preds_perturbed).mean()
    assert stability > 0.95, f"Model stability {stability} < 0.95"
```

**Epoch 007 (Deployment):**
```python
def test_api_endpoint():
    """Verify API endpoint responding"""
    import requests
    response = requests.post(api_url, json=test_payload)
    assert response.status_code == 200

def test_prediction_latency():
    """Verify prediction latency < 100ms"""
    import time
    start = time.time()
    response = requests.post(api_url, json=test_payload)
    latency = time.time() - start
    assert latency < 0.1, f"Latency {latency}s > 0.1s"

def test_monitoring_active():
    """Verify monitoring endpoint active"""
    response = requests.get(f"{api_url}/health")
    assert response.status_code == 200
```

**Integration Testing:**
```python
def test_end_to_end_pipeline():
    """Test complete pipeline from data to prediction"""
    # Load data
    df = pd.read_parquet(data_path)
    # Engineer features
    X = engineer_features(df)
    # Load model
    model = mlflow.pyfunc.load_model(f"models:/{project_name}_model/Production")
    # Predict
    predictions = model.predict(X[:10])
    assert len(predictions) == 10
```

**Running Tests:**
```bash
# Run all tests
pytest /mnt/code/src/{project}/tests/ -v

# Run specific epoch tests
pytest /mnt/code/src/{project}/tests/test_model_utils.py -v

# Run with coverage
pytest /mnt/code/src/{project}/tests/ --cov=/mnt/code/src/{project} --cov-report=html
```

### Standardized Error Handling
**All agents must use consistent error handling and recovery patterns:**

**Error taxonomy:**
```python
class PipelineError(Exception):
    """Base exception for all pipeline errors"""
    pass

class DataQualityError(PipelineError):
    """Data does not meet quality thresholds"""
    pass

class ResourceLimitError(PipelineError):
    """Resource limits exceeded (memory, disk, file size)"""
    pass

class DependencyError(PipelineError):
    """Required dependency from previous epoch missing"""
    pass

class ValidationError(PipelineError):
    """Validation checks failed"""
    pass

class CheckpointError(PipelineError):
    """Checkpoint save/load failed"""
    pass
```

**Error handling pattern:**
```python
def execute_with_error_handling(epoch, operation_name, operation_func, *args, **kwargs):
    """Execute operation with standardized error handling"""
    import traceback
    from pathlib import Path
    import json
    from datetime import datetime

    try:
        result = operation_func(*args, **kwargs)
        return result

    except ResourceLimitError as e:
        # Automatic degradation - sample data and retry
        print(f"Resource limit exceeded in {operation_name}. Attempting graceful degradation...")
        if "memory" in str(e).lower():
            # Sample data to reduce memory
            print("Sampling data to fit within memory constraints...")
            # Implement sampling logic
        return None

    except DependencyError as e:
        # Log blocker and halt
        update_pipeline_state(epoch, blocker=str(e))
        print(f"BLOCKER: {e}")
        print(f"Cannot proceed with Epoch {epoch}. Please complete dependencies first.")
        raise

    except (DataQualityError, ValidationError) as e:
        # Log warning, allow user to decide
        update_pipeline_state(epoch, warning=str(e))
        print(f"WARNING: {e}")
        user_input = input("Continue anyway? (yes/no): ")
        if user_input.lower() != "yes":
            raise

    except Exception as e:
        # Unknown error - log and attempt checkpoint recovery
        error_log = {
            "epoch": epoch,
            "operation": operation_name,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }

        error_log_path = Path("/mnt/code/.context/error_log.json")
        if error_log_path.exists():
            with open(error_log_path) as f:
                errors = json.load(f)
        else:
            errors = []

        errors.append(error_log)
        with open(error_log_path, 'w') as f:
            json.dump(errors, f, indent=2)

        print(f"ERROR in {operation_name}: {e}")
        print("Checking for checkpoint to resume from...")

        checkpoint = load_checkpoint(epoch)
        if checkpoint and checkpoint.get("can_resume"):
            print(f"Found checkpoint at {checkpoint['progress_percent']}% completion")
            user_input = input("Resume from checkpoint? (yes/no): ")
            if user_input.lower() == "yes":
                return checkpoint
        raise

def update_pipeline_state(epoch, blocker=None, warning=None):
    """Update pipeline state with blockers or warnings"""
    import json
    from pathlib import Path

    state_path = Path("/mnt/code/.context/pipeline_state.json")
    with open(state_path) as f:
        state = json.load(f)

    if blocker:
        if "blockers" not in state:
            state["blockers"] = []
        state["blockers"].append({"epoch": epoch, "message": blocker})

    if warning:
        if "warnings" not in state:
            state["warnings"] = []
        state["warnings"].append({"epoch": epoch, "message": warning})

    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
```

### Directory Structure Usage
**Use EXISTING epoch directories - DO NOT create new ones:**
- Code files go in `/mnt/code/epoch00X-xxx/` directories
- Notebooks in `epoch00X-xxx/notebooks/`
- Scripts in `epoch00X-xxx/scripts/`
- Apps in `epoch007-application-development/app/`

## Directory Structure

The repository follows a standardized structure:

```
/mnt/
├── code/
│   ├── src/{project}/                        # Reusable code (auto-extracted)
│   │   ├── __init__.py
│   │   ├── data_processing_utils.py          # Epoch 002
│   │   ├── data_loading_pipeline.py          # Epoch 002
│   │   ├── feature_engineering.py            # Epoch 003
│   │   ├── model_utils.py                    # Epoch 004
│   │   ├── validation.py                     # Epoch 005
│   │   ├── deployment.py                     # Epoch 006
│   │   ├── monitoring.py                     # Epoch 006
│   │   ├── serving.py                        # Epoch 006
│   │   ├── app_utils.py                      # Epoch 007
│   │   ├── streamlit_components.py           # Epoch 007
│   │   ├── error_handling.py                 # Standardized error handling
│   │   ├── config.py                         # Project configuration
│   │   ├── tests/                            # Unit tests for reusable code
│   │   │   ├── test_data_processing.py
│   │   │   ├── test_feature_engineering.py
│   │   │   └── test_model_utils.py
│   │   └── README.md
│   ├── .context/                             # Shared agent communication
│   │   ├── pipeline_state.json               # Cross-agent state management
│   │   ├── data_lineage.json                 # Data transformation tracking
│   │   └── checkpoints/                      # Incremental progress checkpoints
│   │       ├── epoch002_checkpoint.json
│   │       ├── epoch003_checkpoint.json
│   │       └── ...
│   ├── epoch001-research-analysis-planning/  # Business requirements & project planning
│   │   ├── notebooks/
│   │   ├── scripts/
│   │   ├── project_plan.json                 # Comprehensive plan for Epochs 002-008
│   │   ├── success_criteria.json             # Definition of done for each epoch
│   │   ├── resource_estimates.json           # Time, compute, memory estimates
│   │   ├── risk_assessment.json              # Risks and mitigation strategies
│   │   └── README.md
│   ├── epoch002-data-wrangling/              # Data acquisition (resource-safe)
│   │   ├── notebooks/
│   │   ├── scripts/
│   │   └── README.md
│   ├── epoch003-exploratory-data-analysis/   # EDA (memory-aware)
│   │   ├── notebooks/
│   │   ├── scripts/
│   │   └── README.md
│   ├── epoch004-feature-engineering/         # Feature engineering
│   │   ├── notebooks/
│   │   ├── scripts/
│   │   └── README.md
│   ├── epoch005-model-development/           # Model training with integrated testing
│   │   ├── notebooks/
│   │   ├── scripts/
│   │   └── README.md
│   ├── epoch006-model-testing/               # Advanced model validation
│   │   ├── notebooks/
│   │   ├── scripts/
│   │   └── README.md
│   ├── epoch007-application-development/     # Deployment, MLOps & Apps
│   │   ├── notebooks/
│   │   ├── scripts/
│   │   ├── app/                              # Streamlit apps
│   │   │   ├── app.py
│   │   │   └── style.css
│   │   └── README.md
│   └── epoch008-retrospective/               # Complete lifecycle retrospective
│       ├── notebooks/                        # Analysis notebooks from all agents
│       ├── scripts/                          # Retrospective analysis scripts
│       ├── lessons_learned/                  # Agent-specific retrospectives
│       │   ├── business_analysis.md
│       │   ├── data_wrangling.md
│       │   ├── eda_feature_engineering.md
│       │   ├── model_development.md
│       │   ├── model_testing.md
│       │   ├── mlops_deployment.md
│       │   └── overall_recommendations.md
│       └── README.md
├── artifacts/epoch00X-xxx/      # Models, reports, visualizations
│   ├── models/                  # Saved model files
│   └── visualizations/          # Generated plots and reports
└── data/{DOMINO_PROJECT_NAME}/  # Project-specific datasets (validated)
    ├── epoch002-data-wrangling/
    │   └── synthetic_data.parquet           # Max 50MB
    ├── epoch003-exploratory-data-analysis/
    │   └── eda_dataset.parquet
    ├── epoch004-feature-engineering/
    │   └── engineered_features.parquet
    └── epoch005-model-development/
        ├── train_data.parquet
        └── val_data.parquet
```

## Available Agents

### Core Agents
- **Business-Analyst-Agent**: Epoch 001 - **Project coordinator and requirements owner**. Reads and validates requirements from `requirements.md`, asks clarification questions for ambiguities, makes recommendations for unclear requirements, translates business requirements to technical specifications, coordinates with other agents to plan all epochs, creates comprehensive project specifications with success criteria, resource estimates, and risk assessments, and produces executive reporting (uses `{project}_data` MLflow experiment)

  **Epoch 001 Workflow:**
  1. Read `/mnt/code/epoch001-research-analysis-planning/requirements.md`
  2. Validate all requirements are clear and complete
  3. Identify ambiguities or unclear requirements
  4. Ask user for clarification on critical uncertainties
  5. Make recommendations with rationale for unclear items
  6. Update requirements.md with Business Analyst review section
  7. Proceed with planning only after requirements validated

  **Epoch 001 Outputs:**
  - Updated `requirements.md` with Business Analyst review and validation
  - `project_plan.json`: Comprehensive plan for Epochs 002-008 with agent assignments
  - `success_criteria.json`: Definition of done for each epoch with measurable criteria
  - `resource_estimates.json`: Estimated time, compute, memory, and cost per epoch
  - `risk_assessment.json`: Identified risks with mitigation strategies
  - Executive summary report with project overview and recommendations
- **Data-Wrangler-Agent**: Epoch 002 - Data acquisition with 50MB/12GB limits, chunked generation (uses `{project}_data` MLflow experiment)
- **Data-Scientist-Agent**: Epoch 003 & 004 - EDA with memory management (Epoch 003), Feature engineering (Epoch 004) (uses `{project}_data` MLflow experiment)
- **Model-Developer-Agent**: Epoch 005 - Initial signal detection across ALL frameworks, then hyperparameter tuning ONLY the best model (uses `{project}_model` MLflow experiment)

  **Epoch 005 Two-Phase Process:**

  **Phase 1 - Initial Signal Detection (Notebooks):**
  - Train baseline models with ALL frameworks to detect initial signal:
    - **Classification**: scikit-learn (LogisticRegression, RandomForest), XGBoost, LightGBM, TensorFlow, PyTorch
    - **Regression**: scikit-learn (LinearRegression, RandomForest), XGBoost, LightGBM, TensorFlow, PyTorch, Statsmodels
  - All models use same standardized metrics for fair comparison
  - Identify best performing framework based on primary metric
  - Save best model info to `/mnt/artifacts/epoch005-model-development/best_model_info.json`
  - Visual comparison charts across all frameworks

  **Phase 2 - Hyperparameter Tuning (Scripts):**
  - Read best model from Phase 1
  - Tune ONLY the best performing model with comprehensive parameter grid
  - Nested MLflow runs (parent tuning run with child runs for each parameter combination)
  - Generate summary charts comparing all tuning iterations
  - Save tuned model info with improvement metrics

  **Technical Details:**
  - **GPU Detection**: Automatically detect and use Nvidia GPU when available
    - XGBoost: `tree_method='gpu_hist'`, `predictor='gpu_predictor'`
    - LightGBM: `device='gpu'`
    - TensorFlow/PyTorch: Automatic GPU usage
    - Log GPU usage to MLflow metrics (gpu_used: 1 or 0)
  - **Standardized Metrics**: All models log same base metrics for comparison
    - Classification: accuracy, precision, recall, f1_score, roc_auc, log_loss
    - Regression: mse, rmse, mae, r2_score, mape
    - Additional: training_time, inference_time_ms, model_size_mb, gpu_used
- **Model-Tester-Agent**: Epoch 006 - Advanced testing - edge cases, compliance, robustness (uses `{project}_model` MLflow experiment)
- **MLOps-Engineer-Agent**: Epoch 007 - Deployment pipelines, monitoring, and Streamlit apps with standardized styling (#BAD2DE, #CBE2DA, #E5F0EC) (uses `{project}_model` MLflow experiment)
- **All-Agents-Collaborative**: Epoch 008 - Comprehensive retrospective of entire ML development lifecycle - all agents review their work, analyze performance metrics, generate automated reports, document lessons learned, and create reusable playbooks for future projects (uses `{project}_model` MLflow experiment)

  **Epoch 008 Deliverables:**
  - Agent-specific retrospective reports in `/mnt/code/epoch008-retrospective/lessons_learned/`
  - Performance comparison reports (actual vs estimated metrics)
  - Automated "what-if" analysis (alternative approaches and their potential impact)
  - Reusable playbook for similar future projects
  - Resource utilization analysis with optimization recommendations
  - Complete timeline and audit trail visualization

### Reference Documentation
- **Agent-Interaction-Protocol**: Communication patterns between agents
- **Example-Demonstration-Flows**: Workflow examples and file organization

## Domino Documentation Reference

**Complete Domino Data Lab documentation is available at:**
- `/mnt/code/.reference/docs/DominoDocumentation.md` - Full platform documentation
- `/mnt/code/.reference/docs/DominoDocumentation6.1.md` - Version 6.1 specific docs

**All agents have access to this documentation and should reference it when:**
- Working with Domino-specific features (Workspaces, Jobs, Flows, etc.)
- Configuring compute environments and hardware tiers
- Setting up data sources and datasets
- Using MLflow integration
- Deploying Model APIs
- Creating Apps and dashboards
- Managing environment variables and secrets

## Technology Stack

- **Primary Language**: Python 3.8+ for all ML operations
- **Resource Monitoring**: psutil for memory management
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch
- **UI Framework**: Streamlit with standardized styling (muted professional theme)
  - Colors: #BAD2DE (table), #CBE2DA (view), #E5F0EC (mv)
  - Semantic UI integration
  - Wide layout, 3-column grids
  - Performance caching with `@st.experimental_singleton` and `@st.experimental_memo`
- **Experiment Tracking**: MLflow with 2-experiment structure (`{project}_data` and `{project}_model`)
  - **DUAL STORAGE**: All metrics, artifacts, and results saved to BOTH MLflow AND `/mnt/artifacts/`
  - Training metrics: JSON, CSV, and PNG files in artifacts directory + MLflow logging
  - Tuning results: Complete parameter grids, summary reports, and charts in both locations
  - Models logged to MLflow with signatures and input examples
- **Deployment**: FastAPI, Flask, Docker, Domino Flows
- **Code Reusability**: Automatic extraction to `/mnt/code/src/` (shared across all projects)

## Key Patterns

### Business Analyst Planning Process (Epoch 001)
**The Business Analyst coordinates with other agents to create a comprehensive project plan:**

**Step 1: Requirements Gathering**
- Collaborate with user to understand business problem
- Define objectives, constraints, and success metrics
- Research industry best practices and similar projects

**Step 2: Agent Consultation**
- Query Data Wrangler about data availability and generation strategy
- Consult Data Scientist on analytical approach and features
- Engage Model Developer on model type recommendations
- Discuss Model Tester on validation requirements
- Review MLOps Engineer on deployment constraints
- Confirm Front-End requirements for Epoch 008 retrospective

**Step 3: Project Plan Creation**
Create `project_plan.json`:
```json
{
  "project_name": "credit_risk_model",
  "business_objective": "Predict credit default risk",
  "epochs": {
    "002": {
      "agent": "Data-Wrangler",
      "description": "Generate synthetic credit data",
      "estimated_duration_hours": 2,
      "dependencies": [],
      "deliverables": ["synthetic_data.parquet", "data_quality_report"]
    },
    "003": {
      "agent": "Data-Scientist",
      "description": "Exploratory data analysis",
      "estimated_duration_hours": 3,
      "dependencies": ["002"],
      "deliverables": ["eda_report.html", "data_quality_analysis"]
    }
    // ... continues for all epochs
  }
}
```

**Step 4: Success Criteria Definition**
Create `success_criteria.json`:
```json
{
  "002": {
    "criteria": [
      "Data file < 50MB",
      "100,000+ rows generated",
      "All required features present",
      "< 5% missing values",
      "Data quality score > 0.90"
    ]
  },
  "003": {
    "criteria": [
      "EDA report generated",
      "All features analyzed",
      "Correlations documented",
      "Outliers identified",
      "Feature distributions plotted"
    ]
  }
  // ... continues for all epochs
}
```

**Step 5: Resource Estimation**
Create `resource_estimates.json`:
```json
{
  "002": {"time_hours": 2, "memory_gb": 4, "compute": "small"},
  "003": {"time_hours": 3, "memory_gb": 6, "compute": "medium"},
  "004": {"time_hours": 2, "memory_gb": 8, "compute": "medium"},
  "005": {"time_hours": 4, "memory_gb": 10, "compute": "large"},
  "006": {"time_hours": 2, "memory_gb": 8, "compute": "medium"},
  "007": {"time_hours": 3, "memory_gb": 6, "compute": "medium"},
  "008": {"time_hours": 2, "memory_gb": 4, "compute": "small"},
  "total_estimated_hours": 18,
  "total_estimated_cost_usd": 150
}
```

**Step 6: Risk Assessment**
Create `risk_assessment.json`:
```json
{
  "risks": [
    {
      "risk": "Insufficient synthetic data quality",
      "probability": "medium",
      "impact": "high",
      "mitigation": "Use advanced generation techniques, validate with domain expert"
    },
    {
      "risk": "Model overfitting on synthetic data",
      "probability": "high",
      "impact": "medium",
      "mitigation": "Extensive cross-validation, holdout test set, synthetic data realism checks"
    },
    {
      "risk": "Memory limits exceeded during training",
      "probability": "low",
      "impact": "high",
      "mitigation": "Chunked processing, automatic sampling, checkpoint recovery"
    }
  ]
}
```

### Agent Coordination
- **Business-Analyst-Agent is project coordinator**: In Epoch 001, the Business Analyst works with the user to understand requirements, then coordinates with other agents to plan the entire project lifecycle (Epochs 002-008)
- **User triggers execution**: After Business Analyst completes planning, user manually triggers each subsequent agent for their epoch
- Individual agents work independently for specific tasks following the plan
- All agents use EXISTING epoch directories (no new directory creation)
- Each stage produces requirements.txt for dependency management
- **Reusable Code Extraction**: Each agent extracts code to `/mnt/code/src/` at completion
- **Code Reuse**: Each agent checks `/mnt/code/src/` for existing utilities at start
- **Cross-Agent Communication**: Agents share context via `/mnt/code/.context/pipeline_state.json`

### Resource-Safe Data Operations
```python
# Automatic implementation in agents:
import psutil
import gc

MAX_FILE_SIZE_MB = 50
MAX_RAM_GB = 12
MAX_RAM_BYTES = MAX_RAM_GB * 1024 * 1024 * 1024

# Memory monitoring
process = psutil.Process()
if process.memory_info().rss > MAX_RAM_BYTES:
    raise MemoryError(f"Memory exceeds {MAX_RAM_GB}GB limit")

# Chunked generation
chunk_size = 10000
for chunk in generate_chunks(total_rows, chunk_size):
    # Check memory before each chunk
    # Generate chunk
    # Garbage collect
    gc.collect()

# File size validation
estimated_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
if estimated_size_mb > MAX_FILE_SIZE_MB:
    # Sample to fit limit
    sample_ratio = MAX_FILE_SIZE_MB / estimated_size_mb
    df = df.sample(frac=sample_ratio, random_state=42)
```

### Directory Validation Pattern
```python
import os
from pathlib import Path

# Get project name
project_name = os.environ.get('DOMINO_PROJECT_NAME') or specifications.get('project', 'demo')

# Validate data directory
data_base_dir = Path(f"/mnt/data/{project_name}")

if not data_base_dir.exists():
    print(f"\nWARNING: Data directory does not exist!")
    print(f"Expected directory: {data_base_dir}")
    print(f"DOMINO_PROJECT_NAME: {project_name}\n")

    user_response = input(f"Enter 'create' to create {data_base_dir} or provide path: ").strip()

    if user_response.lower() == 'create':
        data_base_dir.mkdir(parents=True, exist_ok=True)
    else:
        data_base_dir = Path(user_response)
        if not data_base_dir.exists():
            raise ValueError(f"Directory does not exist: {data_base_dir}")
```

### Streamlit Styling Pattern
```python
import streamlit as st

# Page config
st.set_page_config(layout="wide", page_title="App Title", page_icon="📊")

# CSS integration
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")
st.markdown('<link href="https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.css" rel="stylesheet">', unsafe_allow_html=True)

# Performance caching
@st.experimental_singleton
def load_model():
    return mlflow.pyfunc.load_model("models:/model_name/latest")

@st.experimental_memo(ttl=600)
def load_data(query):
    return fetch_data(query)

# 3-column layout
col1, col2, col3 = st.columns(3)

# Custom cards
def create_card(title, content, bg_class="table-bg"):
    html = f"""
    <div class="ui card {bg_class}" style="width: 100%;">
        <div class="content">
            <div class="header">{title}</div>
            <div class="description">{content}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
```

### Code Extraction Pattern
```python
def extract_reusable_code(specifications):
    """Extract reusable code to /mnt/code/src/{project}/"""
    from pathlib import Path

    project_name = specifications.get('project', 'default')

    # Create src directory structure
    src_base_dir = Path('/mnt/code/src')
    project_src_dir = src_base_dir / project_name
    project_src_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py
    init_file = project_src_dir / '__init__.py'
    if not init_file.exists():
        with open(init_file, 'w') as f:
            f.write(f'"""{project_name} - Reusable ML Pipeline Components"""\n__version__ = "1.0.0"\n')

    # Create utility modules
    # ... (specific to each epoch)

    # Create README
    readme_path = project_src_dir / "README.md"
    # ... (with usage instructions)
```

### MLflow Integration and Model Registry
**2-Experiment Structure:**
- `{project_name}_data`: All data processing (Epochs 001-004) - Business Analyst, Data Wrangler, Data Scientist (EDA & Feature Engineering)
- `{project_name}_model`: All model operations (Epochs 005-008) - Model Developer, Model Tester, MLOps (incl. Apps), Front-End (Retrospective)

**Model Registry Usage:**
```python
import mlflow
from mlflow.models.signature import infer_signature

# Epoch 005: Register model with metadata
with mlflow.start_run(run_name=f"{model_type}_baseline") as run:
    # Train model
    model.fit(X_train, y_train)

    # Log parameters
    mlflow.log_params(model.get_params())

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # Infer signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log model with signature and input example
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train[:5],
        registered_model_name=f"{project_name}_model"
    )

    # Tag with epoch metadata
    mlflow.set_tags({
        "epoch": "005",
        "model_type": model_type,
        "training_date": datetime.now().isoformat(),
        "data_version": "v1.0"
    })

# Epoch 006: Promote model after testing
client = mlflow.tracking.MlflowClient()
latest_version = client.get_latest_versions(f"{project_name}_model", stages=["None"])[0]

# After successful testing, transition to Staging
client.transition_model_version_stage(
    name=f"{project_name}_model",
    version=latest_version.version,
    stage="Staging",
    archive_existing_versions=False
)

# Add testing metadata
client.set_model_version_tag(
    name=f"{project_name}_model",
    version=latest_version.version,
    key="testing_complete",
    value="true"
)

client.set_model_version_tag(
    name=f"{project_name}_model",
    version=latest_version.version,
    key="edge_cases_passed",
    value="15/15"
)

# Epoch 007: Promote to Production after deployment
client.transition_model_version_stage(
    name=f"{project_name}_model",
    version=latest_version.version,
    stage="Production",
    archive_existing_versions=True
)

# Add deployment metadata
client.set_model_version_tag(
    name=f"{project_name}_model",
    version=latest_version.version,
    key="deployment_date",
    value=datetime.now().isoformat()
)

client.set_model_version_tag(
    name=f"{project_name}_model",
    version=latest_version.version,
    key="api_endpoint",
    value="/predict"
)
```

**Model Registry Stages:**
- **None**: Initial registration in Epoch 005
- **Staging**: After passing Epoch 006 tests
- **Production**: After successful Epoch 007 deployment
- **Archived**: Previous versions when new model promoted

**Model Versioning Strategy:**
- Automatic versioning on each registration
- Tag versions with epoch, data version, and test results
- Enable A/B testing by loading specific versions
- Track lineage from data version to model version

**Additional MLflow Features:**
- All metrics, parameters, and artifacts logged
- Comprehensive artifact tracking at each stage
- Resource metrics logged (memory usage, file sizes, training time)
- **Dual Storage**: Everything saved to both `/mnt/artifacts/` AND MLflow
- Enable model comparison across versions

### File Organization
- Each agent uses existing epoch directories under `/mnt/code/epoch00X-xxx/`
- Notebooks go in `epoch00X-xxx/notebooks/`, scripts in `epoch00X-xxx/scripts/`
- Artifacts saved to `/mnt/artifacts/epoch00X-xxx/`
- Data organized by project: `/mnt/data/{DOMINO_PROJECT_NAME}/epoch00X-xxx/`
- **Reusable code** automatically extracted to `/mnt/code/src/` (shared across all projects)

### Cross-Agent Communication
Agents communicate through shared context files in `/mnt/code/.context/`:

**pipeline_state.json** - Master state tracking:
```json
{
  "project_name": "credit_risk_model",
  "epoch_001_complete": true,
  "epoch_002_complete": true,
  "epoch_002_outputs": {
    "data_path": "/mnt/data/project/epoch002-data-wrangling/data.parquet",
    "data_size_mb": 45.2,
    "row_count": 100000,
    "column_count": 25,
    "quality_score": 0.95,
    "schema": {"features": ["age", "income", "credit_score"], "target": "default"}
  },
  "epoch_003_complete": true,
  "epoch_003_outputs": {
    "eda_report_path": "/mnt/artifacts/epoch003-exploratory-data-analysis/eda_report.html",
    "missing_value_pct": 0.02,
    "outlier_count": 150,
    "data_quality_issues": []
  },
  "epoch_004_outputs": {
    "feature_count": 42,
    "train_test_split": {"train": 0.8, "test": 0.2},
    "feature_importance_top5": ["credit_utilization", "payment_history", ...]
  },
  "epoch_005_outputs": {
    "model_type": "XGBoost",
    "model_uri": "runs:/abc123/model",
    "test_accuracy": 0.87,
    "baseline_metrics": {"precision": 0.85, "recall": 0.82, "f1": 0.83}
  },
  "resource_estimates": {
    "epoch_006_memory_gb": 8,
    "epoch_007_compute_time_min": 15
  },
  "blockers": [],
  "warnings": ["High class imbalance detected in Epoch 003"]
}
```

**data_lineage.json** - Data transformation tracking:
```json
{
  "source_data": {
    "epoch": "002",
    "path": "/mnt/data/project/epoch002-data-wrangling/data.parquet",
    "generation_method": "synthetic",
    "timestamp": "2025-10-03T10:00:00Z"
  },
  "transformations": [
    {
      "epoch": "003",
      "step": "outlier_removal",
      "rows_before": 100000,
      "rows_after": 99850,
      "method": "IQR"
    },
    {
      "epoch": "004",
      "step": "feature_engineering",
      "features_added": ["credit_utilization_ratio", "payment_velocity"],
      "features_removed": ["raw_payment_amount"],
      "method": "polynomial_interactions"
    }
  ],
  "feature_provenance": {
    "credit_utilization_ratio": {
      "created_in_epoch": "004",
      "source_features": ["total_balance", "credit_limit"],
      "formula": "total_balance / credit_limit"
    }
  }
}
```

**checkpoints/** - Incremental progress saves:
- Enable resume capability if epoch fails mid-execution
- Store intermediate artifacts to prevent complete rework
- Each checkpoint includes timestamp, progress percentage, and next step

**Communication features:**
- **State Management**: Each agent reads outputs from previous stages
- **Metadata Sharing**: Model info, data schemas, performance metrics
- **Handoff Coordination**: Clear transitions with validation
- **Error Recovery**: Access to previous successful states and checkpoints
- **Traceability**: Complete audit trail of pipeline execution
- **Resource Planning**: Estimated requirements for upcoming epochs
- **Blocker Tracking**: Document and track issues preventing progress

## Common Usage Patterns

```python
# Epoch-by-epoch execution (Business Analyst coordinates planning in Epoch 001)
"Business Analyst: Plan credit risk model project"                          # Epoch 001 - Coordinates with all agents to plan Epochs 002-008
# After Epoch 001, user triggers each agent based on the plan:
"Data Wrangler: Generate synthetic credit data"                             # Epoch 002 (Max 50MB, 12GB RAM)
"Data Scientist: Perform exploratory data analysis on credit data"          # Epoch 003
"Data Scientist: Engineer features for credit risk prediction"              # Epoch 004
"Model Developer: Train and test credit risk models"                        # Epoch 005 (Integrated testing)
"Model Tester: Run advanced validation and edge case testing"               # Epoch 006
"MLOps Engineer: Deploy model, add monitoring, and build Streamlit app"     # Epoch 007
"All Agents: Conduct comprehensive retrospective of entire ML lifecycle"    # Epoch 008

# Specific agent tasks
"Generate synthetic financial data for fraud detection"                     # Data Wrangler
"Create polynomial and interaction features"                                # Data Scientist (Feature Engineering)
"Perform comprehensive model testing with edge cases"                       # Model Tester
"Deploy model with monitoring and interactive dashboard"                    # MLOps Engineer
```

## Agent Templates and Boilerplate

**Each epoch has standardized notebook and script templates to accelerate development:**

### Epoch 002 Template (Data Wrangling)
```python
# epoch002-data-wrangling/scripts/generate_data.py

import os
import pandas as pd
import psutil
import gc
from pathlib import Path
from datetime import datetime

# Import shared utilities
from src.{project}.error_handling import execute_with_error_handling
from src.{project}.config import ENABLE_COST_TRACKING

# Configuration
MAX_FILE_SIZE_MB = 50
MAX_RAM_GB = 12
CHUNK_SIZE = 10000

# Initialize resource tracking
from resource_tracker import ResourceTracker
tracker = ResourceTracker(epoch="002", enabled=ENABLE_COST_TRACKING)

# Validate prerequisites
from quality_gates import validate_epoch_prerequisites
state = validate_epoch_prerequisites("002")

# Validate data directory
project_name = os.getenv('DOMINO_PROJECT_NAME', state.get('project_name'))
data_dir = Path(f"/mnt/data/{project_name}/epoch002-data-wrangling")
if not data_dir.exists():
    user_input = input(f"Create {data_dir}? (yes/no): ")
    if user_input.lower() == "yes":
        data_dir.mkdir(parents=True, exist_ok=True)

# Main processing with error handling
def generate_data():
    # Initialize checkpoint
    checkpoint = load_checkpoint("002")
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint['progress_percent']}%")

    # Generate data in chunks
    chunks = []
    for i in range(0, total_rows, CHUNK_SIZE):
        # Memory check
        memory_gb = psutil.Process().memory_info().rss / (1024**3)
        if memory_gb > MAX_RAM_GB:
            raise ResourceLimitError(f"Memory {memory_gb}GB exceeds {MAX_RAM_GB}GB limit")

        # Generate chunk
        chunk = generate_chunk(i, min(i + CHUNK_SIZE, total_rows))
        chunks.append(chunk)

        # Progress reporting
        progress = (i + CHUNK_SIZE) / total_rows * 100
        print(f"Generated {i + CHUNK_SIZE}/{total_rows} rows (Memory: {memory_gb:.2f} GB)")

        # Save checkpoint
        save_checkpoint("002", progress, "data_generation", "data_validation", ["setup"], {})
        tracker.sample()

        gc.collect()

    # Combine and validate
    df = pd.concat(chunks, ignore_index=True)

    # Size check
    estimated_size = df.memory_usage(deep=True).sum() / (1024**2)
    if estimated_size > MAX_FILE_SIZE_MB:
        sample_ratio = MAX_FILE_SIZE_MB / estimated_size
        df = df.sample(frac=sample_ratio, random_state=42)

    # Save
    output_path = data_dir / "data.parquet"
    df.to_parquet(output_path)

    # Update lineage
    update_data_lineage("002", "data_generation", {
        "method": "synthetic",
        "rows": len(df),
        "columns": len(df.columns)
    })

    # Update pipeline state
    update_pipeline_state({
        "epoch_002_complete": True,
        "epoch_002_outputs": {
            "data_path": str(output_path),
            "row_count": len(df),
            "column_count": len(df.columns)
        }
    })

    return df

# Execute with error handling
result = execute_with_error_handling("002", "data_generation", generate_data)

# Finalize tracking
metrics = tracker.finalize()

# Run tests
import pytest
pytest.main(["/mnt/code/src/{project}/tests/test_data_processing.py", "-v"])
```

### Epoch 005 Template (Model Development)

**Artifacts Generated (Dual Storage - MLflow + Artifacts Directory):**

**Training Phase Artifacts:**
- `all_models_metrics.json` - Comprehensive metrics for all trained models
- `model_comparison_summary.csv` - Tabular comparison of all models
- `all_models_comparison.png` - Visual comparison charts (8 metrics)
- `best_model_info.json` - Best model identification for tuning phase

**Tuning Phase Artifacts:**
- `all_tuning_results.json` - Detailed results for every parameter combination
- `all_tuning_results.csv` - Tabular tuning results
- `tuning_parameter_impact.png` - Parameter impact visualization
- `top_10_models.png` - Top 10 tuned models comparison
- `tuned_model_info.json` - Best tuned model parameters and improvement
- `tuning_summary_report.json` - Comprehensive tuning statistics and analysis

All artifacts saved to both:
1. `/mnt/artifacts/epoch005-model-development/` (persistent file storage)
2. MLflow experiments (logged as artifacts to runs)

**Notebooks for Training** (`notebooks/model_training.ipynb`):

**IMPORTANT: Initial Signal Detection Process**
1. Train baseline models with ALL frameworks to detect initial signal:
   - **Classification**: scikit-learn (LogisticRegression, RandomForest), XGBoost, LightGBM, TensorFlow, PyTorch
   - **Regression**: scikit-learn (LinearRegression, RandomForest), XGBoost, LightGBM, TensorFlow, PyTorch, Statsmodels
2. Compare all models using standardized metrics
3. Identify best performing framework based on primary metric (accuracy, f1_score, r2_score, etc.)
4. Proceed to hyperparameter tuning with ONLY the best performing framework

```python
# Cell 1: Setup and Data Loading
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import time

# GPU Detection
def detect_gpu():
    """Detect if Nvidia GPU is available"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gpu_available = result.returncode == 0
        if gpu_available:
            print("GPU detected: Using Nvidia GPU for training")
        else:
            print("No GPU detected: Using CPU for training")
        return gpu_available
    except:
        print("No GPU detected: Using CPU for training")
        return False

GPU_AVAILABLE = detect_gpu()

# Load data
data_path = state["epoch_004_outputs"]["feature_data_path"]
df = pd.read_parquet(data_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cell 2: Define Standardized Metrics Function
def log_standardized_metrics(y_true, y_pred, y_pred_proba, model, X_test, problem_type="classification"):
    """Log standardized metrics for all models"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
    import time

    metrics = {}

    if problem_type == "classification":
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        if y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            metrics["log_loss"] = log_loss(y_true, y_pred_proba)
    else:  # regression
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = mean_squared_error(y_true, y_pred, squared=False)
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2_score"] = r2_score(y_true, y_pred)
        metrics["mape"] = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-10, None))) * 100

    # Inference time
    start = time.time()
    _ = model.predict(X_test[:100])
    inference_time = (time.time() - start) / 100 * 1000  # ms per prediction
    metrics["inference_time_ms"] = inference_time

    # Model size
    import joblib
    import os
    temp_path = "/tmp/temp_model.pkl"
    try:
        joblib.dump(model, temp_path)
        metrics["model_size_mb"] = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
    except:
        metrics["model_size_mb"] = 0  # For models that can't be pickled

    # GPU usage
    metrics["gpu_used"] = 1 if GPU_AVAILABLE else 0

    return metrics

# Cell 3: Train scikit-learn LogisticRegression
mlflow.set_experiment(f"{project_name}_model")

with mlflow.start_run(run_name="sklearn_logistic_regression"):
    start_time = time.time()

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = log_standardized_metrics(y_test, y_pred, y_pred_proba, model, X_test)
    metrics["training_time"] = training_time

    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature)

# Cell 4: Train scikit-learn RandomForest
with mlflow.start_run(run_name="sklearn_random_forest"):
    start_time = time.time()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = log_standardized_metrics(y_test, y_pred, y_pred_proba, model, X_test)
    metrics["training_time"] = training_time

    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature)

# Cell 5: Train XGBoost
with mlflow.start_run(run_name="xgboost_baseline"):
    start_time = time.time()

    xgb_params = {
        'n_estimators': 100,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    if GPU_AVAILABLE:
        xgb_params['tree_method'] = 'gpu_hist'
        xgb_params['predictor'] = 'gpu_predictor'

    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = log_standardized_metrics(y_test, y_pred, y_pred_proba, model, X_test)
    metrics["training_time"] = training_time

    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature)

# Cell 6: Train LightGBM
with mlflow.start_run(run_name="lightgbm_baseline"):
    start_time = time.time()

    lgbm_params = {
        'n_estimators': 100,
        'random_state': 42,
        'verbose': -1
    }
    if GPU_AVAILABLE:
        lgbm_params['device'] = 'gpu'

    model = LGBMClassifier(**lgbm_params)
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = log_standardized_metrics(y_test, y_pred, y_pred_proba, model, X_test)
    metrics["training_time"] = training_time

    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature)

# Cell 7: Train TensorFlow/Keras Neural Network
with mlflow.start_run(run_name="tensorflow_nn"):
    start_time = time.time()

    # Simple neural network
    tf_model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
    ])

    tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train with early stopping
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = tf_model.fit(X_train, y_train, epochs=100, batch_size=32,
                          validation_split=0.2, callbacks=[early_stop], verbose=0)

    training_time = time.time() - start_time

    y_pred_proba = tf_model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Wrap for consistency
    class TFModelWrapper:
        def __init__(self, model):
            self.model = model
        def predict(self, X):
            return np.argmax(self.model.predict(X, verbose=0), axis=1)

    model = TFModelWrapper(tf_model)
    metrics = log_standardized_metrics(y_test, y_pred, y_pred_proba, model, X_test)
    metrics["training_time"] = training_time

    mlflow.log_params({
        'layers': 3,
        'units_layer1': 64,
        'units_layer2': 32,
        'dropout': 0.3,
        'optimizer': 'adam',
        'epochs': len(history.history['loss'])
    })
    mlflow.log_metrics(metrics)

    mlflow.tensorflow.log_model(tf_model, "model")

# Cell 8: Train PyTorch Neural Network
with mlflow.start_run(run_name="pytorch_nn"):
    start_time = time.time()

    # Convert to PyTorch tensors
    X_train_torch = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train)
    y_train_torch = torch.LongTensor(y_train.values if hasattr(y_train, 'values') else y_train)
    X_test_torch = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test)

    # Simple neural network
    class PyTorchNN(nn.Module):
        def __init__(self, input_size, num_classes):
            super(PyTorchNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.fc3(x)
            return x

    pytorch_model = PyTorchNN(X_train.shape[1], len(np.unique(y_train)))

    if GPU_AVAILABLE and torch.cuda.is_available():
        pytorch_model = pytorch_model.cuda()
        X_train_torch = X_train_torch.cuda()
        y_train_torch = y_train_torch.cuda()
        X_test_torch = X_test_torch.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pytorch_model.parameters())

    # Train
    pytorch_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = pytorch_model(X_train_torch)
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()

    training_time = time.time() - start_time

    # Predict
    pytorch_model.eval()
    with torch.no_grad():
        outputs = pytorch_model(X_test_torch)
        y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
        y_pred = np.argmax(y_pred_proba, axis=1)

    # Wrap for consistency
    class PyTorchModelWrapper:
        def __init__(self, model, device):
            self.model = model
            self.device = device
        def predict(self, X):
            self.model.eval()
            with torch.no_grad():
                X_torch = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
                if self.device == 'cuda':
                    X_torch = X_torch.cuda()
                outputs = self.model(X_torch)
                return np.argmax(torch.softmax(outputs, dim=1).cpu().numpy(), axis=1)

    model = PyTorchModelWrapper(pytorch_model, 'cuda' if GPU_AVAILABLE and torch.cuda.is_available() else 'cpu')
    metrics = log_standardized_metrics(y_test, y_pred, y_pred_proba, model, X_test)
    metrics["training_time"] = training_time

    mlflow.log_params({
        'layers': 3,
        'units_layer1': 64,
        'units_layer2': 32,
        'dropout': 0.3,
        'optimizer': 'adam',
        'epochs': 100
    })
    mlflow.log_metrics(metrics)

    mlflow.pytorch.log_model(pytorch_model, "model")

# Cell 9: Compare All Models and Select Best
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Ensure artifacts directory exists
Path('/mnt/artifacts/epoch005-model-development').mkdir(parents=True, exist_ok=True)

# Fetch all runs from experiment
experiment = mlflow.get_experiment_by_name(f"{project_name}_model")
runs_df = mlflow.search_runs(experiment.experiment_id)

# Determine primary metric based on problem type
primary_metric = 'f1_score'  # For classification, use 'r2_score' for regression

# Find best model
best_run = runs_df.loc[runs_df[f'metrics.{primary_metric}'].idxmax()]
best_model_name = best_run['tags.mlflow.runName']

print(f"\nBest performing model: {best_model_name}")
print(f"Best {primary_metric}: {best_run[f'metrics.{primary_metric}']:.4f}")

# Plot comparison
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
metrics_to_compare = ['accuracy', 'f1_score', 'roc_auc', 'log_loss', 'training_time', 'inference_time_ms', 'model_size_mb', 'gpu_used']

for idx, metric in enumerate(metrics_to_compare):
    ax = axes[idx // 4, idx % 4]
    if f'metrics.{metric}' in runs_df.columns:
        runs_df.plot(x='tags.mlflow.runName', y=f'metrics.{metric}', kind='bar', ax=ax, legend=False)
        ax.set_title(metric)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/mnt/artifacts/epoch005-model-development/all_models_comparison.png', dpi=150, bbox_inches='tight')
mlflow.log_artifact('/mnt/artifacts/epoch005-model-development/all_models_comparison.png')
plt.show()

# Save best model info for tuning script (to artifacts)
best_model_info = {
    'model_name': best_model_name,
    'primary_metric': primary_metric,
    'primary_metric_value': float(best_run[f'metrics.{primary_metric}']),
    'run_id': best_run['run_id']
}

with open('/mnt/artifacts/epoch005-model-development/best_model_info.json', 'w') as f:
    json.dump(best_model_info, f, indent=2)

print(f"\nProceed to hyperparameter tuning with: {best_model_name}")

# Cell 10: Save All Metrics to Artifacts Directory
# Create comprehensive metrics report for all models

# Prepare detailed metrics dictionary
all_models_metrics = {}

for idx, row in runs_df.iterrows():
    model_name = row['tags.mlflow.runName']
    all_models_metrics[model_name] = {
        'run_id': row['run_id'],
        'metrics': {
            'accuracy': row.get('metrics.accuracy', None),
            'precision': row.get('metrics.precision', None),
            'recall': row.get('metrics.recall', None),
            'f1_score': row.get('metrics.f1_score', None),
            'roc_auc': row.get('metrics.roc_auc', None),
            'log_loss': row.get('metrics.log_loss', None),
            'training_time': row.get('metrics.training_time', None),
            'inference_time_ms': row.get('metrics.inference_time_ms', None),
            'model_size_mb': row.get('metrics.model_size_mb', None),
            'gpu_used': row.get('metrics.gpu_used', None)
        }
    }

# Save comprehensive metrics to artifacts directory
with open('/mnt/artifacts/epoch005-model-development/all_models_metrics.json', 'w') as f:
    json.dump(all_models_metrics, f, indent=2)

# Log to MLflow as well
with mlflow.start_run(run_name="training_summary") as summary_run:
    mlflow.log_artifact('/mnt/artifacts/epoch005-model-development/all_models_metrics.json')
    mlflow.log_artifact('/mnt/artifacts/epoch005-model-development/best_model_info.json')

    # Log summary statistics
    mlflow.log_param('total_models_trained', len(runs_df))
    mlflow.log_param('best_model', best_model_name)
    mlflow.log_metric('best_' + primary_metric, best_model_info['primary_metric_value'])

# Display comprehensive comparison table
summary_df = runs_df[[
    'tags.mlflow.runName',
    'metrics.accuracy',
    'metrics.f1_score',
    'metrics.roc_auc',
    'metrics.training_time',
    'metrics.inference_time_ms',
    'metrics.model_size_mb',
    'metrics.gpu_used'
]].sort_values(by=f'metrics.{primary_metric}', ascending=False)

summary_df.columns = ['Model', 'Accuracy', 'F1 Score', 'ROC AUC', 'Training Time', 'Inference (ms)', 'Size (MB)', 'GPU']

# Save summary table to CSV in artifacts directory
summary_df.to_csv('/mnt/artifacts/epoch005-model-development/model_comparison_summary.csv', index=False)

# Log summary CSV to MLflow
with mlflow.start_run(run_id=summary_run.info.run_id):
    mlflow.log_artifact('/mnt/artifacts/epoch005-model-development/model_comparison_summary.csv')

print("\n=== Model Comparison Summary ===")
print(summary_df.to_string(index=False))
print(f"\n✓ All metrics saved to: /mnt/artifacts/epoch005-model-development/")
print(f"  - all_models_metrics.json")
print(f"  - model_comparison_summary.csv")
print(f"  - all_models_comparison.png")
print(f"  - best_model_info.json")
print(f"\n✓ All artifacts also logged to MLflow")
```

**Scripts for Tuning** - Tune ONLY the best performing model from training phase

**IMPORTANT: Dynamic Tuning Based on Best Model**
- Script reads `/mnt/artifacts/epoch005-model-development/best_model_info.json`
- Tunes ONLY the best performing model identified during training
- Each framework has its own tuning script with appropriate hyperparameters
- All tuning uses nested MLflow runs with summary charts

**Script: `scripts/tune_best_model.py`** (Universal tuner that dispatches to appropriate framework):
```python
# Universal hyperparameter tuning script - tunes ONLY the best model from training

import json
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import subprocess

# GPU Detection
def detect_gpu():
    """Detect if Nvidia GPU is available"""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gpu_available = result.returncode == 0
        if gpu_available:
            print("GPU detected: Using Nvidia GPU for tuning")
        else:
            print("No GPU detected: Using CPU for tuning")
        return gpu_available
    except:
        print("No GPU detected: Using CPU for tuning")
        return False

GPU_AVAILABLE = detect_gpu()

# Load best model info
with open('/mnt/artifacts/epoch005-model-development/best_model_info.json', 'r') as f:
    best_model_info = json.load(f)

best_model_name = best_model_info['model_name']
primary_metric = best_model_info['primary_metric']

print(f"\nTuning best model: {best_model_name}")
print(f"Baseline {primary_metric}: {best_model_info['primary_metric_value']:.4f}\n")

# Load data
data_path = state["epoch_004_outputs"]["feature_data_path"]
df = pd.read_parquet(data_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardized metrics function (same as notebook)
def log_standardized_metrics(y_true, y_pred, y_pred_proba, model, X_test, problem_type="classification"):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

    metrics = {}

    if problem_type == "classification":
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        if y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            metrics["log_loss"] = log_loss(y_true, y_pred_proba)
    else:  # regression
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = mean_squared_error(y_true, y_pred, squared=False)
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2_score"] = r2_score(y_true, y_pred)
        metrics["mape"] = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-10, None))) * 100

    # Inference time
    start = time.time()
    _ = model.predict(X_test[:100])
    inference_time = (time.time() - start) / 100 * 1000
    metrics["inference_time_ms"] = inference_time

    # Model size
    import joblib
    import os
    temp_path = "/tmp/temp_model.pkl"
    try:
        joblib.dump(model, temp_path)
        metrics["model_size_mb"] = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
    except:
        metrics["model_size_mb"] = 0

    metrics["gpu_used"] = 1 if GPU_AVAILABLE else 0

    return metrics

# Set experiment
mlflow.set_experiment(f"{project_name}_model")

# Dispatch to appropriate tuning function based on best model
if 'sklearn_logistic_regression' in best_model_name:
    print("Tuning scikit-learn LogisticRegression...")

    from sklearn.linear_model import LogisticRegression

    with mlflow.start_run(run_name="sklearn_logistic_tuning_parent") as parent_run:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'max_iter': [500, 1000, 2000]
        }

        results = []
        for C in param_grid['C']:
            for solver in param_grid['solver']:
                for max_iter in param_grid['max_iter']:

                    with mlflow.start_run(run_name=f"lr_C{C}_s{solver}_i{max_iter}", nested=True):
                        start_time = time.time()

                        model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42)
                        model.fit(X_train, y_train)

                        training_time = time.time() - start_time

                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)

                        metrics = log_standardized_metrics(y_test, y_pred, y_pred_proba, model, X_test)
                        metrics["training_time"] = training_time

                        mlflow.log_params({'C': C, 'solver': solver, 'max_iter': max_iter})
                        mlflow.log_metrics(metrics)

                        results.append({'C': C, 'solver': solver, 'max_iter': max_iter, primary_metric: metrics[primary_metric]})

elif 'sklearn_random_forest' in best_model_name:
    print("Tuning scikit-learn RandomForest...")

    from sklearn.ensemble import RandomForestClassifier

    with mlflow.start_run(run_name="sklearn_rf_tuning_parent") as parent_run:
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        results = []
        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for split in param_grid['min_samples_split']:
                    for leaf in param_grid['min_samples_leaf']:

                        with mlflow.start_run(run_name=f"rf_n{n_est}_d{depth}_s{split}_l{leaf}", nested=True):
                            start_time = time.time()

                            model = RandomForestClassifier(
                                n_estimators=n_est,
                                max_depth=depth,
                                min_samples_split=split,
                                min_samples_leaf=leaf,
                                random_state=42
                            )
                            model.fit(X_train, y_train)

                            training_time = time.time() - start_time

                            y_pred = model.predict(X_test)
                            y_pred_proba = model.predict_proba(X_test)

                            metrics = log_standardized_metrics(y_test, y_pred, y_pred_proba, model, X_test)
                            metrics["training_time"] = training_time

                            mlflow.log_params({
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'min_samples_split': split,
                                'min_samples_leaf': leaf
                            })
                            mlflow.log_metrics(metrics)

                            results.append({
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'min_samples_split': split,
                                'min_samples_leaf': leaf,
                                primary_metric: metrics[primary_metric]
                            })

elif 'xgboost' in best_model_name:
    print("Tuning XGBoost...")

    from xgboost import XGBClassifier

    with mlflow.start_run(run_name="xgboost_tuning_parent") as parent_run:
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

        results = []
        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    for subsample in param_grid['subsample']:
                        for colsample in param_grid['colsample_bytree']:

                            with mlflow.start_run(run_name=f"xgb_n{n_est}_d{depth}_lr{lr}", nested=True):
                                start_time = time.time()

                                xgb_params = {
                                    'n_estimators': n_est,
                                    'max_depth': depth,
                                    'learning_rate': lr,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample,
                                    'random_state': 42,
                                    'eval_metric': 'logloss'
                                }
                                if GPU_AVAILABLE:
                                    xgb_params['tree_method'] = 'gpu_hist'
                                    xgb_params['predictor'] = 'gpu_predictor'

                                model = XGBClassifier(**xgb_params)
                                model.fit(X_train, y_train)

                                training_time = time.time() - start_time

                                y_pred = model.predict(X_test)
                                y_pred_proba = model.predict_proba(X_test)

                                metrics = log_standardized_metrics(y_test, y_pred, y_pred_proba, model, X_test)
                                metrics["training_time"] = training_time

                                mlflow.log_params({
                                    'n_estimators': n_est,
                                    'max_depth': depth,
                                    'learning_rate': lr,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample
                                })
                                mlflow.log_metrics(metrics)

                                results.append({
                                    'n_estimators': n_est,
                                    'max_depth': depth,
                                    'learning_rate': lr,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample,
                                    primary_metric: metrics[primary_metric]
                                })

elif 'lightgbm' in best_model_name:
    print("Tuning LightGBM...")

    from lightgbm import LGBMClassifier

    with mlflow.start_run(run_name="lightgbm_tuning_parent") as parent_run:
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'num_leaves': [15, 31, 63, 127],
            'min_child_samples': [5, 10, 20]
        }

        results = []
        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    for leaves in param_grid['num_leaves']:
                        for samples in param_grid['min_child_samples']:

                            with mlflow.start_run(run_name=f"lgbm_n{n_est}_d{depth}_lr{lr}", nested=True):
                                start_time = time.time()

                                lgbm_params = {
                                    'n_estimators': n_est,
                                    'max_depth': depth,
                                    'learning_rate': lr,
                                    'num_leaves': leaves,
                                    'min_child_samples': samples,
                                    'random_state': 42,
                                    'verbose': -1
                                }
                                if GPU_AVAILABLE:
                                    lgbm_params['device'] = 'gpu'

                                model = LGBMClassifier(**lgbm_params)
                                model.fit(X_train, y_train)

                                training_time = time.time() - start_time

                                y_pred = model.predict(X_test)
                                y_pred_proba = model.predict_proba(X_test)

                                metrics = log_standardized_metrics(y_test, y_pred, y_pred_proba, model, X_test)
                                metrics["training_time"] = training_time

                                mlflow.log_params(lgbm_params)
                                mlflow.log_metrics(metrics)

                                results.append({**lgbm_params, primary_metric: metrics[primary_metric]})

elif 'tensorflow' in best_model_name:
    print("Tuning TensorFlow Neural Network...")

    import tensorflow as tf
    from tensorflow import keras

    with mlflow.start_run(run_name="tensorflow_tuning_parent") as parent_run:
        param_grid = {
            'layer1_units': [32, 64, 128],
            'layer2_units': [16, 32, 64],
            'dropout': [0.2, 0.3, 0.5],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64]
        }

        results = []
        for units1 in param_grid['layer1_units']:
            for units2 in param_grid['layer2_units']:
                for dropout in param_grid['dropout']:
                    for lr in param_grid['learning_rate']:
                        for batch_size in param_grid['batch_size']:

                            with mlflow.start_run(run_name=f"tf_u1{units1}_u2{units2}_d{dropout}", nested=True):
                                start_time = time.time()

                                tf_model = keras.Sequential([
                                    keras.layers.Dense(units1, activation='relu', input_shape=(X_train.shape[1],)),
                                    keras.layers.Dropout(dropout),
                                    keras.layers.Dense(units2, activation='relu'),
                                    keras.layers.Dropout(dropout),
                                    keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
                                ])

                                optimizer = keras.optimizers.Adam(learning_rate=lr)
                                tf_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                                early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                                history = tf_model.fit(X_train, y_train, epochs=100, batch_size=batch_size,
                                                      validation_split=0.2, callbacks=[early_stop], verbose=0)

                                training_time = time.time() - start_time

                                y_pred_proba = tf_model.predict(X_test, verbose=0)
                                y_pred = np.argmax(y_pred_proba, axis=1)

                                class TFModelWrapper:
                                    def __init__(self, model):
                                        self.model = model
                                    def predict(self, X):
                                        return np.argmax(self.model.predict(X, verbose=0), axis=1)

                                model = TFModelWrapper(tf_model)
                                metrics = log_standardized_metrics(y_test, y_pred, y_pred_proba, model, X_test)
                                metrics["training_time"] = training_time

                                mlflow.log_params({
                                    'layer1_units': units1,
                                    'layer2_units': units2,
                                    'dropout': dropout,
                                    'learning_rate': lr,
                                    'batch_size': batch_size,
                                    'epochs': len(history.history['loss'])
                                })
                                mlflow.log_metrics(metrics)

                                results.append({
                                    'layer1_units': units1,
                                    'layer2_units': units2,
                                    'dropout': dropout,
                                    'learning_rate': lr,
                                    'batch_size': batch_size,
                                    primary_metric: metrics[primary_metric]
                                })

elif 'pytorch' in best_model_name:
    print("Tuning PyTorch Neural Network...")

    import torch
    import torch.nn as nn

    with mlflow.start_run(run_name="pytorch_tuning_parent") as parent_run:
        param_grid = {
            'layer1_units': [32, 64, 128],
            'layer2_units': [16, 32, 64],
            'dropout': [0.2, 0.3, 0.5],
            'learning_rate': [0.001, 0.01, 0.1],
            'epochs': [50, 100, 200]
        }

        results = []
        for units1 in param_grid['layer1_units']:
            for units2 in param_grid['layer2_units']:
                for dropout in param_grid['dropout']:
                    for lr in param_grid['learning_rate']:
                        for epochs in param_grid['epochs']:

                            with mlflow.start_run(run_name=f"pt_u1{units1}_u2{units2}_d{dropout}", nested=True):
                                start_time = time.time()

                                X_train_torch = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train)
                                y_train_torch = torch.LongTensor(y_train.values if hasattr(y_train, 'values') else y_train)
                                X_test_torch = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test)

                                class PyTorchNN(nn.Module):
                                    def __init__(self, input_size, num_classes, u1, u2, drop):
                                        super(PyTorchNN, self).__init__()
                                        self.fc1 = nn.Linear(input_size, u1)
                                        self.fc2 = nn.Linear(u1, u2)
                                        self.fc3 = nn.Linear(u2, num_classes)
                                        self.relu = nn.ReLU()
                                        self.dropout = nn.Dropout(drop)

                                    def forward(self, x):
                                        x = self.dropout(self.relu(self.fc1(x)))
                                        x = self.dropout(self.relu(self.fc2(x)))
                                        x = self.fc3(x)
                                        return x

                                pytorch_model = PyTorchNN(X_train.shape[1], len(np.unique(y_train)), units1, units2, dropout)

                                if GPU_AVAILABLE and torch.cuda.is_available():
                                    pytorch_model = pytorch_model.cuda()
                                    X_train_torch = X_train_torch.cuda()
                                    y_train_torch = y_train_torch.cuda()
                                    X_test_torch = X_test_torch.cuda()

                                criterion = nn.CrossEntropyLoss()
                                optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=lr)

                                pytorch_model.train()
                                for epoch in range(epochs):
                                    optimizer.zero_grad()
                                    outputs = pytorch_model(X_train_torch)
                                    loss = criterion(outputs, y_train_torch)
                                    loss.backward()
                                    optimizer.step()

                                training_time = time.time() - start_time

                                pytorch_model.eval()
                                with torch.no_grad():
                                    outputs = pytorch_model(X_test_torch)
                                    y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
                                    y_pred = np.argmax(y_pred_proba, axis=1)

                                class PyTorchModelWrapper:
                                    def __init__(self, model, device):
                                        self.model = model
                                        self.device = device
                                    def predict(self, X):
                                        self.model.eval()
                                        with torch.no_grad():
                                            X_torch = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
                                            if self.device == 'cuda':
                                                X_torch = X_torch.cuda()
                                            outputs = self.model(X_torch)
                                            return np.argmax(torch.softmax(outputs, dim=1).cpu().numpy(), axis=1)

                                model = PyTorchModelWrapper(pytorch_model, 'cuda' if GPU_AVAILABLE and torch.cuda.is_available() else 'cpu')
                                metrics = log_standardized_metrics(y_test, y_pred, y_pred_proba, model, X_test)
                                metrics["training_time"] = training_time

                                mlflow.log_params({
                                    'layer1_units': units1,
                                    'layer2_units': units2,
                                    'dropout': dropout,
                                    'learning_rate': lr,
                                    'epochs': epochs
                                })
                                mlflow.log_metrics(metrics)

                                results.append({
                                    'layer1_units': units1,
                                    'layer2_units': units2,
                                    'dropout': dropout,
                                    'learning_rate': lr,
                                    'epochs': epochs,
                                    primary_metric: metrics[primary_metric]
                                })

else:
    raise ValueError(f"Unknown model type: {best_model_name}")

# Generate summary charts (common for all models)
results_df = pd.DataFrame(results)

# Chart 1: Parameter impact visualization
fig, axes = plt.subplots(1, min(3, len(results_df.columns)-1), figsize=(15, 5))
param_cols = [col for col in results_df.columns if col != primary_metric][:3]

for idx, param in enumerate(param_cols):
    if idx < len(axes):
        axes[idx].scatter(results_df[param], results_df[primary_metric])
        axes[idx].set_xlabel(param)
        axes[idx].set_ylabel(primary_metric)
        axes[idx].set_title(f'{primary_metric} vs {param}')

plt.tight_layout()
plt.savefig('/mnt/artifacts/epoch005-model-development/tuning_parameter_impact.png', dpi=150, bbox_inches='tight')
mlflow.log_artifact('/mnt/artifacts/epoch005-model-development/tuning_parameter_impact.png')

# Chart 2: Top 10 models comparison
top_10 = results_df.nlargest(10, primary_metric)

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(range(len(top_10)), top_10[primary_metric])
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels([f"Rank {i+1}" for i in range(len(top_10))])
ax.set_xlabel(primary_metric)
ax.set_title(f'Top 10 Models - {primary_metric} Comparison')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('/mnt/artifacts/epoch005-model-development/top_10_models.png', dpi=150, bbox_inches='tight')
mlflow.log_artifact('/mnt/artifacts/epoch005-model-development/top_10_models.png')

# Log best parameters to parent run
best_idx = results_df[primary_metric].idxmax()
best_params = results_df.loc[best_idx].to_dict()

mlflow.log_params({f"best_{k}": v for k, v in best_params.items() if k != primary_metric})
mlflow.log_metrics({"best_" + primary_metric: best_params[primary_metric]})

print(f"\n=== Best Tuned Parameters ===")
for k, v in best_params.items():
    if k != primary_metric:
        print(f"{k}: {v}")
print(f"\nBest {primary_metric}: {best_params[primary_metric]:.4f}")
print(f"Improvement over baseline: {(best_params[primary_metric] - best_model_info['primary_metric_value']) * 100:.2f}%")

# Save all tuning results to artifacts directory
tuning_results_detailed = []
for idx, row in results_df.iterrows():
    result_dict = row.to_dict()
    tuning_results_detailed.append(result_dict)

# Save detailed tuning results
with open('/mnt/artifacts/epoch005-model-development/all_tuning_results.json', 'w') as f:
    json.dump(tuning_results_detailed, f, indent=2)

# Save tuning results as CSV
results_df.to_csv('/mnt/artifacts/epoch005-model-development/all_tuning_results.csv', index=False)

# Log tuning results to MLflow
mlflow.log_artifact('/mnt/artifacts/epoch005-model-development/all_tuning_results.json')
mlflow.log_artifact('/mnt/artifacts/epoch005-model-development/all_tuning_results.csv')

# Save best tuned model info
tuned_model_info = {
    'model_name': best_model_name,
    'best_params': best_params,
    'primary_metric': primary_metric,
    'tuned_metric_value': float(best_params[primary_metric]),
    'baseline_metric_value': best_model_info['primary_metric_value'],
    'improvement_pct': float((best_params[primary_metric] - best_model_info['primary_metric_value']) * 100),
    'total_tuning_iterations': len(results_df)
}

with open('/mnt/artifacts/epoch005-model-development/tuned_model_info.json', 'w') as f:
    json.dump(tuned_model_info, f, indent=2)

# Log tuned model info to MLflow
mlflow.log_artifact('/mnt/artifacts/epoch005-model-development/tuned_model_info.json')

# Create comprehensive tuning summary report
tuning_summary = {
    'best_model': best_model_name,
    'baseline_performance': {
        'metric': primary_metric,
        'value': best_model_info['primary_metric_value']
    },
    'tuned_performance': {
        'metric': primary_metric,
        'value': float(best_params[primary_metric])
    },
    'improvement': {
        'absolute': float(best_params[primary_metric] - best_model_info['primary_metric_value']),
        'percentage': float((best_params[primary_metric] - best_model_info['primary_metric_value']) * 100)
    },
    'best_parameters': {k: v for k, v in best_params.items() if k != primary_metric},
    'tuning_statistics': {
        'total_iterations': len(results_df),
        'parameter_space': {col: results_df[col].nunique() for col in results_df.columns if col != primary_metric},
        'best_iteration': int(best_idx),
        'metric_range': {
            'min': float(results_df[primary_metric].min()),
            'max': float(results_df[primary_metric].max()),
            'mean': float(results_df[primary_metric].mean()),
            'std': float(results_df[primary_metric].std())
        }
    }
}

with open('/mnt/artifacts/epoch005-model-development/tuning_summary_report.json', 'w') as f:
    json.dump(tuning_summary, f, indent=2)

# Log summary report to MLflow
mlflow.log_artifact('/mnt/artifacts/epoch005-model-development/tuning_summary_report.json')

# Update pipeline state
update_pipeline_state({
    "epoch_005_complete": True,
    "epoch_005_outputs": {
        "best_model_name": best_model_name,
        "best_model_params": best_params,
        "tuning_parent_run_id": parent_run.info.run_id,
        "primary_metric": primary_metric,
        "final_metric_value": float(best_params[primary_metric]),
        "baseline_metric_value": best_model_info['primary_metric_value'],
        "improvement_pct": tuned_model_info['improvement_pct']
    }
})

print(f"\n✓ All tuning metrics saved to: /mnt/artifacts/epoch005-model-development/")
print(f"  - all_tuning_results.json")
print(f"  - all_tuning_results.csv")
print(f"  - tuning_parameter_impact.png")
print(f"  - top_10_models.png")
print(f"  - tuned_model_info.json")
print(f"  - tuning_summary_report.json")
print(f"\n✓ All artifacts also logged to MLflow")
```

### Common Boilerplate
All agent scripts should include:
1. Import shared utilities from `/mnt/code/src/{project}/`
2. Validate prerequisites with quality gates
3. Initialize resource tracking (if enabled)
4. Load checkpoint (if resuming)
5. Execute with error handling wrapper
6. Save checkpoints periodically
7. Update data lineage
8. Update pipeline state
9. Run automated tests
10. Finalize resource tracking

### Jupyter Notebook Standards
**All JupyterLab notebooks must include informative visualizations relevant to the use case:**

**Required Visualizations by Epoch:**

**Epoch 002 (Data Wrangling):**
- Data distribution plots (histograms, box plots)
- Missing value heatmaps
- Target variable distribution
- Feature correlation preview
- Data quality summary charts

**Epoch 003 (EDA):**
- Comprehensive correlation heatmaps
- Feature distribution plots (by target class)
- Outlier detection visualizations (scatter plots, box plots)
- Bivariate analysis plots (pair plots for key features)
- Time series plots (if temporal data)
- Statistical summary visualizations
- Class imbalance visualization

**Epoch 004 (Feature Engineering):**
- Feature importance bar charts (preliminary)
- Before/after feature transformation comparisons
- Polynomial feature correlation matrices
- Interaction effect visualizations
- Feature scaling distribution plots
- Dimensionality reduction plots (PCA, t-SNE if applicable)

**Epoch 005 (Model Development):**
- Training vs validation loss curves
- Confusion matrices
- ROC curves and AUC scores
- Precision-Recall curves
- Feature importance rankings (model-specific)
- Learning curves
- Residual plots (for regression)
- Cross-validation score distributions

**Epoch 006 (Model Testing):**
- Edge case performance heatmaps
- Prediction distribution by input range
- Calibration plots
- Error analysis by feature bins
- Robustness testing results (perturbation impact)
- Performance across different data segments

**Epoch 007 (Deployment & Apps):**
- API latency histograms
- Prediction confidence distributions
- Real-time monitoring dashboards
- Model drift detection plots
- A/B test results (if applicable)

**Epoch 008 (Retrospective):**
- Timeline visualization (actual vs estimated)
- Resource utilization over time
- Performance metric progression across epochs
- Cost breakdown by epoch
- Success criteria achievement radar chart

**Visualization Best Practices:**

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

# Example: Feature importance plot
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

# Example: Confusion matrix with annotations
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

# Example: Distribution comparison
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

# Example: Interactive correlation heatmap
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

**Notebook Structure Standards:**

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

**Quality Standards:**
- All plots must have clear titles, axis labels, and legends
- Use color palettes appropriate for the use case (e.g., diverging for correlations)
- Save interactive HTML versions for exploration
- Save static PNG versions for reports
- Include statistical annotations where relevant (p-values, confidence intervals)
- Use appropriate plot types for data (scatter for continuous, bar for categorical)
- Ensure plots are accessible (consider colorblind-friendly palettes)
- Add context with text annotations for key insights

**Notebook Execution Requirements:**
- **ALWAYS run all cells** in the notebook after creation
- **ALWAYS save the notebook** with output cells intact
- This ensures:
  - Code validation (confirms notebooks execute without errors)
  - Output preservation (images, tables, and results saved in notebook)
  - Immediate visibility (stakeholders can view results without re-running)
  - Reproducibility verification (confirms environment and dependencies work)

**Execution Pattern:**
```python
# After creating notebook, execute all cells programmatically:
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

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

**Alternative: Manual execution via bash:**
```bash
# Execute notebook and save outputs
jupyter nbconvert --to notebook --execute --inplace /mnt/code/epoch003-exploratory-data-analysis/notebooks/eda_analysis.ipynb
```

## Development Guidelines

- **Business-Analyst-Agent coordinates project planning**: In Epoch 001, Business Analyst works with user to understand requirements, then collaborates with other agents to create comprehensive project plan for Epochs 002-008
- **User triggers execution**: After planning is complete, user manually triggers each subsequent agent for their epoch based on the plan
- **Use templates**: Start with epoch-specific templates to ensure all requirements met
- Always specify project names for proper organization
- Include business context for better agent recommendations
- **Validate data directories** before operations
- **Monitor resource usage** through logged metrics
- **Use extracted code** from `/mnt/code/src/` for reusability across projects
- **Check src/ directory** at start of each epoch for existing utilities
- Test with small datasets before scaling
- Leverage MLOps-Engineer-Agent's standardized Streamlit styling in Epoch 007
- Use Model-Tester-Agent for advanced validation beyond basic metrics
- **Human-in-the-Loop Validation Points**: Mandatory review gates at critical junctions
- **Epoch 008 Retrospective**: All agents participate to review the entire ML lifecycle
  - Each agent reviews their specific contributions and challenges
  - Identify what worked well and what could be improved
  - Document lessons learned for future projects
  - Provide actionable recommendations for process improvements

### Human-in-the-Loop Validation Points
**Mandatory review gates require user approval before proceeding:**

**Review Point 1: After Epoch 001 (Planning Complete)**
- **Trigger**: Business Analyst completes project plan
- **Review**: User reviews and approves:
  - Project plan (epochs, agents, deliverables)
  - Success criteria for each epoch
  - Resource estimates (time, compute, cost)
  - Risk assessment and mitigation strategies
- **Action**: User explicitly approves plan or requests modifications
- **Gate**: No subsequent epochs start until plan approved

**Review Point 2: After Epoch 003 (Data Quality)**
- **Trigger**: EDA complete, data quality report generated
- **Review**: User/Business Analyst reviews:
  - Data quality metrics (missing values, outliers, distributions)
  - Feature correlations and relationships
  - Identified data issues and proposed solutions
  - Data quality score and whether it meets threshold
- **Action**: Approve to proceed to feature engineering, or return to Epoch 002
- **Gate**: Prevents feature engineering on poor-quality data

**Review Point 3: After Epoch 004 (Feature Readiness)**
- **Trigger**: Features engineered, ready for modeling
- **Review**: User/Data Scientist reviews:
  - Engineered features and their business meaning
  - Feature importance rankings
  - Train/test split strategy
  - Data leakage checks
- **Action**: Approve feature set or request modifications
- **Gate**: Prevents training models with incorrect features

**Review Point 4: After Epoch 006 (Model Validation)**
- **Trigger**: Advanced testing complete
- **Review**: User/Model Tester reviews:
  - Comprehensive test results (accuracy, edge cases, robustness)
  - Compliance validation results
  - Known model limitations and failure modes
  - Recommendation for production deployment
- **Action**: Approve for deployment or return for retraining
- **Gate**: Prevents deploying insufficiently tested models

**Review Point 5: After Epoch 007 (Deployment Verification)**
- **Trigger**: Model deployed, monitoring active, app functional
- **Review**: User/MLOps Engineer reviews:
  - Deployment success metrics
  - API endpoint functionality and latency
  - Monitoring dashboards and alert configuration
  - Streamlit app usability and performance
- **Action**: Approve deployment or rollback
- **Gate**: Ensures production system meets requirements before retrospective

**Implementation Pattern:**
```python
def request_human_approval(epoch, review_point_name, review_data):
    """Request human approval at validation gate"""
    print(f"\n{'='*80}")
    print(f"HUMAN REVIEW REQUIRED: {review_point_name}")
    print(f"{'='*80}\n")

    # Display review data
    print("Review Summary:")
    for key, value in review_data.items():
        print(f"  {key}: {value}")

    print(f"\nPlease review the {review_point_name} outputs.")
    print("This is a mandatory validation gate.")

    # Request approval
    while True:
        response = input(f"\nApprove to proceed to next epoch? (yes/no/details): ").lower()

        if response == "yes":
            # Log approval
            log_approval(epoch, review_point_name, "approved")
            update_pipeline_state({f"{epoch}_review_approved": True})
            print(f"✓ Approval granted. Proceeding to next epoch.")
            return True

        elif response == "no":
            log_approval(epoch, review_point_name, "rejected")
            print(f"✗ Approval denied. Please review and address issues.")
            return False

        elif response == "details":
            # Show detailed report
            show_detailed_report(epoch, review_data)
        else:
            print("Please enter 'yes', 'no', or 'details'")

def log_approval(epoch, review_point, decision):
    """Log approval decision to pipeline state"""
    import json
    from pathlib import Path
    from datetime import datetime

    approval_log_path = Path("/mnt/code/.context/approval_log.json")
    if approval_log_path.exists():
        with open(approval_log_path) as f:
            log = json.load(f)
    else:
        log = {"approvals": []}

    log["approvals"].append({
        "epoch": epoch,
        "review_point": review_point,
        "decision": decision,
        "timestamp": datetime.now().isoformat(),
        "reviewer": os.getenv("DOMINO_USER_NAME", "unknown")
    })

    with open(approval_log_path, 'w') as f:
        json.dump(log, f, indent=2)

# Example usage after Epoch 003:
review_data = {
    "Missing Values": "2.3%",
    "Data Quality Score": "0.93",
    "Outliers Detected": "145",
    "Critical Issues": "None"
}

approved = request_human_approval(
    epoch="003",
    review_point_name="Data Quality Assessment",
    review_data=review_data
)

if not approved:
    print("Returning to Epoch 002 for data quality improvements...")
    # Handle rejection
```

**Optional Review Points:**
- Users can request ad-hoc reviews at any point
- Business Analyst can schedule additional reviews in project plan
- Set `SKIP_OPTIONAL_REVIEWS=true` to bypass non-mandatory gates (not recommended)

## Important Reminders

**Do what has been asked; nothing more, nothing less.**

**NEVER create files unless they're absolutely necessary for achieving your goal.**

**ALWAYS prefer editing an existing file to creating a new one.**

**NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.**

**NEVER use emojis or emoticons in any outputs, code, documentation, or communication.**

## Epoch 008: Comprehensive Retrospective Framework

**All agents participate in Epoch 008 to conduct a thorough post-mortem analysis:**

### Retrospective Components

**1. Agent-Specific Retrospectives**

Each agent creates a detailed report in `/mnt/code/epoch008-retrospective/lessons_learned/`:

**business_analysis.md** (Business Analyst):
- Accuracy of initial requirements understanding
- Quality of project plan vs actual execution
- Resource estimate accuracy (predicted vs actual)
- Risk assessment effectiveness
- Recommendations for future planning

**data_wrangling.md** (Data Wrangler):
- Data quality achieved vs targets
- Synthetic data generation effectiveness
- Resource constraints encountered
- Data size and memory management lessons
- Recommendations for data generation

**eda_feature_engineering.md** (Data Scientist):
- Feature engineering effectiveness
- Feature importance vs expectations
- Data quality issues discovered
- Analysis techniques that worked well
- Recommendations for feature selection

**model_development.md** (Model Developer):
- Model performance vs baseline
- Algorithm selection rationale and outcomes
- Hyperparameter tuning effectiveness
- Training time and resource usage
- Recommendations for model selection

**model_testing.md** (Model Tester):
- Testing coverage achieved
- Edge cases discovered
- Model robustness assessment
- Compliance validation results
- Recommendations for testing strategy

**mlops_deployment.md** (MLOps Engineer):
- Deployment success metrics
- Monitoring effectiveness
- API performance and latency
- Streamlit app usability feedback
- Recommendations for deployment

**2. Automated Performance Analysis**

```python
# epoch008-retrospective/scripts/generate_performance_report.py

def generate_performance_comparison():
    """Compare estimated vs actual metrics across all epochs"""
    import json
    import pandas as pd

    # Load estimates from Epoch 001
    with open("/mnt/code/epoch001-research-analysis-planning/resource_estimates.json") as f:
        estimates = json.load(f)

    # Load actuals from resource tracking
    with open("/mnt/code/.context/resource_tracking.json") as f:
        actuals = json.load(f)["epochs"]

    comparison = []
    for epoch_num in ["002", "003", "004", "005", "006", "007"]:
        comparison.append({
            "epoch": epoch_num,
            "estimated_hours": estimates.get(epoch_num, {}).get("time_hours", 0),
            "actual_hours": actuals.get(epoch_num, {}).get("elapsed_hours", 0),
            "estimated_memory_gb": estimates.get(epoch_num, {}).get("memory_gb", 0),
            "actual_peak_memory_gb": actuals.get(epoch_num, {}).get("peak_memory_gb", 0),
            "variance_hours": None,
            "variance_memory": None
        })

    df = pd.DataFrame(comparison)
    df["variance_hours"] = ((df["actual_hours"] - df["estimated_hours"]) / df["estimated_hours"] * 100).round(1)
    df["variance_memory"] = ((df["actual_peak_memory_gb"] - df["estimated_memory_gb"]) / df["estimated_memory_gb"] * 100).round(1)

    return df
```

**3. What-If Analysis**

```python
def generate_what_if_analysis():
    """Analyze alternative approaches and potential impacts"""

    analyses = []

    # What if we used different model?
    analyses.append({
        "scenario": "Use Random Forest instead of XGBoost",
        "potential_impact": {
            "training_time": "+30% slower",
            "accuracy": "-2% lower",
            "interpretability": "+40% better"
        },
        "recommendation": "Consider for next project if interpretability is priority"
    })

    # What if we had more features?
    analyses.append({
        "scenario": "Engineer 20% more features",
        "potential_impact": {
            "feature_eng_time": "+25%",
            "model_accuracy": "+1-3%",
            "inference_latency": "+15%"
        },
        "recommendation": "Worthwhile if accuracy improvement needed"
    })

    # What if we had more data?
    analyses.append({
        "scenario": "Generate 2x more training data",
        "potential_impact": {
            "data_gen_time": "+100%",
            "model_accuracy": "+3-5%",
            "training_time": "+80%"
        },
        "recommendation": "High ROI for accuracy improvement"
    })

    return analyses
```

**4. Reusable Playbook Generation**

```python
def generate_playbook():
    """Create reusable playbook for similar future projects"""

    playbook = {
        "project_type": "credit_risk_classification",
        "success_factors": [
            "Comprehensive planning in Epoch 001 saved time later",
            "Chunked data processing prevented memory issues",
            "Early feature engineering reduced model iterations"
        ],
        "pitfalls_avoided": [
            "Quality gates caught data issues before modeling",
            "Checkpointing prevented rework after failures",
            "Human review after EDA saved training on bad features"
        ],
        "recommended_timeline": {
            "001_planning": "2-3 hours",
            "002_data": "2-3 hours",
            "003_eda": "3-4 hours",
            "004_features": "2-3 hours",
            "005_modeling": "4-5 hours",
            "006_testing": "2-3 hours",
            "007_deployment": "3-4 hours",
            "008_retrospective": "2 hours",
            "total": "20-27 hours"
        },
        "tech_stack_recommendations": {
            "data_format": "Parquet (fast, compressed)",
            "model_framework": "XGBoost (high performance)",
            "deployment": "FastAPI + Streamlit",
            "monitoring": "Prometheus + Grafana"
        },
        "resource_requirements": {
            "peak_memory_gb": 10.5,
            "storage_gb": 2.0,
            "compute_tier": "medium to large"
        },
        "key_learnings": [
            "Always validate data directories before operations",
            "Resource tracking helps optimize future projects",
            "Human review gates prevent costly mistakes",
            "Automated testing catches issues early"
        ]
    }

    # Save as both JSON and Markdown
    with open("/mnt/code/epoch008-retrospective/project_playbook.json", 'w') as f:
        json.dump(playbook, f, indent=2)

    # Generate markdown report
    generate_markdown_playbook(playbook)

    return playbook
```

**5. Timeline Visualization**

Generate visual timeline showing:
- Actual time spent per epoch
- Checkpoint/restart events
- Human review gate decisions
- Blocker/warning occurrences
- Resource utilization peaks

**6. Final Recommendations Document**

```markdown
# Overall Recommendations for Future Projects

## Process Improvements
1. **Increase planning time**: Estimate +20% for Epoch 001 given complexity
2. **Add intermediate checkpoints**: More frequent saves in long-running epochs
3. **Enhanced monitoring**: Add real-time resource alerts

## Technical Improvements
1. **Feature engineering automation**: Build reusable feature library
2. **Model selection automation**: Implement AutoML comparison in Epoch 005
3. **Deployment templates**: Standardize FastAPI and Streamlit templates

## Resource Optimization
1. **Memory management**: Implement dynamic chunk sizing based on available RAM
2. **Parallel processing**: Identify opportunities for parallel execution
3. **Cost reduction**: Use spot instances for non-critical training runs

## Quality Improvements
1. **Enhanced testing**: Add adversarial testing in Epoch 006
2. **Data quality**: Implement automated data quality scoring
3. **Documentation**: Auto-generate API docs from code

## Team Collaboration
1. **Agent coordination**: Earlier involvement of downstream agents in planning
2. **Review efficiency**: Standardize review checklist templates
3. **Knowledge sharing**: Export lessons learned to central repository
```

## Resource Management Best Practices

### Before Data Operations
1. Validate `/mnt/data/{DOMINO_PROJECT_NAME}/` exists
2. Check available memory with psutil
3. Estimate final data size

### During Data Generation
1. Use chunked processing (10K rows default)
2. Monitor memory before each chunk
3. Display progress with memory usage
4. Force garbage collection between chunks

### After Data Operations
1. Verify file size < 50MB
2. Sample if necessary
3. Log metrics to MLflow
4. Extract reusable code to `/mnt/code/src/`

## Streamlit Styling Reference

Based on: https://github.com/mydgd/snowflake-table-catalog

### Color Palette
- Table Background: `#BAD2DE`
- View Background: `#CBE2DA`
- Materialized View: `#E5F0EC`
- Text Color: `rgb(90, 90, 90)`

### Required Files
- `app.py` - Main Streamlit application
- `style.css` - Standard stylesheet with theme colors

### Key Features
- Wide layout configuration
- Semantic UI integration
- 3-column responsive grids
- Performance caching
- Session state management
- Custom card components
