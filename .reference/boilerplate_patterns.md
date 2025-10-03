# Boilerplate Patterns Reference

This file contains standardized code patterns and templates used across all agents.

## Resource Tracking Implementation

### ResourceTracker Class

```python
import psutil
import time
from datetime import datetime
from pathlib import Path
import json
import os

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
```

## Error Handling Patterns

### Exception Classes

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

### Error Handling Wrapper

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

## Checkpoint Management

### Checkpoint Functions

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

## Data Lineage Tracking

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

def add_feature_provenance(feature_name, source_features, formula, created_in_epoch):
    """Document how a feature was created"""
    import json
    from pathlib import Path
    from datetime import datetime

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

## Quality Gates

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

## Resource Management

```python
import psutil
import gc

MAX_FILE_SIZE_MB = 50
MAX_RAM_GB = 12
MAX_RAM_BYTES = MAX_RAM_GB * 1024 * 1024 * 1024

def check_memory_limit():
    """Check if memory usage is within limits"""
    process = psutil.Process()
    if process.memory_info().rss > MAX_RAM_BYTES:
        raise MemoryError(f"Memory exceeds {MAX_RAM_GB}GB limit")

def chunked_data_generation(total_rows, chunk_size=10000):
    """Generate data in memory-safe chunks"""
    chunks = []
    for i in range(0, total_rows, chunk_size):
        check_memory_limit()

        # Generate chunk
        chunk = generate_chunk(i, min(i + chunk_size, total_rows))
        chunks.append(chunk)

        # Progress reporting
        memory_gb = psutil.Process().memory_info().rss / (1024**3)
        print(f"Generated {i + chunk_size}/{total_rows} rows (Memory: {memory_gb:.2f} GB)")

        gc.collect()

    return chunks

def validate_file_size(df, max_size_mb=MAX_FILE_SIZE_MB):
    """Validate and sample data if it exceeds size limit"""
    estimated_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    if estimated_size_mb > max_size_mb:
        sample_ratio = max_size_mb / estimated_size_mb
        print(f"Data size {estimated_size_mb:.2f}MB exceeds limit. Sampling to {sample_ratio:.2%}")
        df = df.sample(frac=sample_ratio, random_state=42)
    return df
```

## Directory Validation

```python
import os
from pathlib import Path

def validate_data_directory(project_name=None, epoch=None):
    """Validate and create data directory if needed"""
    if not project_name:
        project_name = os.environ.get('DOMINO_PROJECT_NAME', 'demo')

    data_base_dir = Path(f"/mnt/data/{project_name}")

    if epoch:
        data_base_dir = data_base_dir / epoch

    if not data_base_dir.exists():
        print(f"\nWARNING: Data directory does not exist!")
        print(f"Expected directory: {data_base_dir}")
        print(f"DOMINO_PROJECT_NAME: {project_name}\n")

        user_response = input(f"Enter 'create' to create {data_base_dir} or provide path: ").strip()

        if user_response.lower() == 'create':
            data_base_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {data_base_dir}")
        else:
            data_base_dir = Path(user_response)
            if not data_base_dir.exists():
                raise ValueError(f"Directory does not exist: {data_base_dir}")

    return data_base_dir
```

## Code Extraction Pattern

```python
def extract_reusable_code(project_name, epoch, module_name, code_content, description):
    """Extract reusable code to /mnt/code/src/{project}/"""
    from pathlib import Path

    # Create src directory structure
    src_base_dir = Path('/mnt/code/src')
    project_src_dir = src_base_dir / project_name
    project_src_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py if needed
    init_file = project_src_dir / '__init__.py'
    if not init_file.exists():
        with open(init_file, 'w') as f:
            f.write(f'"""{project_name} - Reusable ML Pipeline Components"""\n__version__ = "1.0.0"\n')

    # Write module
    module_path = project_src_dir / f"{module_name}.py"
    with open(module_path, 'w') as f:
        f.write(f'"""{description}"""\n\n')
        f.write(code_content)

    print(f"Extracted reusable code to: {module_path}")
    return module_path
```

## Human-in-the-Loop Validation

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
            print(f"Approval granted. Proceeding to next epoch.")
            return True

        elif response == "no":
            log_approval(epoch, review_point_name, "rejected")
            print(f"Approval denied. Please review and address issues.")
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
    import os

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
```
