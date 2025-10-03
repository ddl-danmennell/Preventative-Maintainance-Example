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

**Implementation:** See `/mnt/code/.reference/boilerplate_patterns.md` for quality gate code.

### Data Directory Management
**ALWAYS validate data directory before proceeding:**
- Data MUST be saved to `/mnt/data/{DOMINO_PROJECT_NAME}/epoch00X-xxx/`
- Get project name from `DOMINO_PROJECT_NAME` environment variable
- If directory doesn't exist, PROMPT user with options:
  - Type `create` to create the directory
  - Enter custom path if different location needed
- **Never proceed without valid directory**

**Implementation:** See `/mnt/code/.reference/boilerplate_patterns.md` for directory validation code.

### Incremental Checkpointing
**All agents must implement checkpoint saving for long-running operations:**
- Save progress checkpoints every 10% or every 5 minutes
- Store in `/mnt/code/.context/checkpoints/epoch{XXX}_checkpoint.json`
- Enable resume from last checkpoint on failure
- Include metadata: timestamp, progress %, next step, intermediate artifacts

**Implementation:** See `/mnt/code/.reference/boilerplate_patterns.md` for checkpoint management code.

### Resource Management (Prevent Workspace Crashes)
**Mandatory limits for ALL data operations:**
- **File Size Limit**: 50MB maximum per file
- **RAM Limit**: 12GB maximum during operations
- **Chunked Processing**: 10,000 rows per chunk for data generation
- **Memory Monitoring**: Use psutil to check before each operation
- **Automatic Sampling**: Reduce data if limits exceeded
- **Progress Reporting**: Display "Generated X/Y rows (Memory: Z GB)"
- **Checkpoint Integration**: Save checkpoint every 10% progress or 5 minutes

**Implementation:** See `/mnt/code/.reference/boilerplate_patterns.md` for resource management code.

### Cost and Resource Tracking (Optional/Configurable)
**Agents can optionally track compute costs and resource utilization:**
- Enable/Disable via `ENABLE_COST_TRACKING` environment variable (default: false)
- Track memory, CPU, elapsed time per epoch
- Optional cost estimation via `ENABLE_COST_ESTIMATION` and `COST_PER_HOUR`
- Log metrics to MLflow if `LOG_RESOURCES_TO_MLFLOW=true`

**Implementation:** See `/mnt/code/.reference/boilerplate_patterns.md` for ResourceTracker class.

### Data Lineage Tracking
**All agents must track data transformations in `/mnt/code/.context/data_lineage.json`:**
- Document every transformation applied to the data
- Track feature provenance (which features came from where)
- Enable traceability from final predictions back to source data
- Log data quality changes at each step

**Implementation:** See `/mnt/code/.reference/boilerplate_patterns.md` for lineage tracking functions.

### Code Reusability
**At end of each epoch, extract reusable code to `/mnt/code/src/{project}/`:**
- Create proper Python package structure with `__init__.py`
- Organize by epoch with appropriate module names
- Include `README.md` with usage instructions
- Use relative imports (e.g., `from .data_processing_utils import *`)
- **Include unit tests** for all reusable functions

**Implementation:** See `/mnt/code/.reference/boilerplate_patterns.md` for code extraction pattern.

### Automated Testing Framework
**All agents must implement automated testing for their outputs:**
- Test structure: `/mnt/code/src/{project}/tests/`
- Run tests with: `pytest /mnt/code/src/{project}/tests/ -v`
- Tests verify: data quality, model performance, deployment functionality
- Integration tests verify end-to-end pipeline

### Standardized Error Handling
**All agents must use consistent error handling and recovery patterns:**
- Exception taxonomy: PipelineError, DataQualityError, ResourceLimitError, etc.
- Automatic degradation for resource limits
- Log blockers and warnings to pipeline state
- Checkpoint recovery on unexpected errors

**Implementation:** See `/mnt/code/.reference/boilerplate_patterns.md` for error handling patterns.

### Human-in-the-Loop Validation Points
**Mandatory review gates require user approval before proceeding:**

**Review Point 1: After Epoch 001 (Planning Complete)**
- User reviews project plan, success criteria, resource estimates, risk assessment
- No subsequent epochs start until plan approved

**Review Point 2: After Epoch 003 (Data Quality)**
- User reviews data quality metrics, correlations, identified issues
- Prevents feature engineering on poor-quality data

**Review Point 3: After Epoch 004 (Feature Readiness)**
- User reviews engineered features, importance rankings, data leakage checks
- Prevents training models with incorrect features

**Review Point 4: After Epoch 006 (Model Validation)**
- User reviews comprehensive test results, compliance, limitations
- Prevents deploying insufficiently tested models

**Review Point 5: After Epoch 007 (Deployment Verification)**
- User reviews deployment metrics, API functionality, monitoring
- Ensures production system meets requirements

**Implementation:** See `/mnt/code/.reference/boilerplate_patterns.md` for human approval functions.

### Directory Structure Usage
**Use EXISTING epoch directories - DO NOT create new ones:**
- Code files go in `/mnt/code/epoch00X-xxx/` directories
- Notebooks in `epoch00X-xxx/notebooks/`
- Scripts in `epoch00X-xxx/scripts/`
- Apps in `epoch007-application-development/app/`

## Directory Structure

```
/mnt/
├── code/
│   ├── .reference/                           # Reference documentation
│   │   ├── boilerplate_patterns.md           # Code templates and patterns
│   │   ├── visualization_standards.md        # Visualization guidelines
│   │   ├── framework_configs.json            # Framework configurations
│   │   ├── DominoDocumentation.md            # Domino platform docs
│   │   └── DominoDocumentation6.1.md         # Version 6.1 docs
│   ├── src/{project}/                        # Reusable code (auto-extracted)
│   │   ├── __init__.py
│   │   ├── data_processing_utils.py          # Epoch 002
│   │   ├── feature_engineering.py            # Epoch 004
│   │   ├── model_utils.py                    # Epoch 005
│   │   ├── tests/                            # Unit tests
│   │   └── README.md
│   ├── .context/                             # Shared agent communication
│   │   ├── pipeline_state.json               # Cross-agent state management
│   │   ├── data_lineage.json                 # Data transformation tracking
│   │   ├── resource_tracking.json            # Resource utilization
│   │   ├── approval_log.json                 # Human review decisions
│   │   ├── error_log.json                    # Error tracking
│   │   └── checkpoints/                      # Incremental progress
│   ├── epoch001-research-analysis-planning/  # Business requirements
│   ├── epoch002-data-wrangling/              # Data acquisition
│   ├── epoch003-exploratory-data-analysis/   # EDA
│   ├── epoch004-feature-engineering/         # Feature engineering
│   ├── epoch005-model-development/           # Model training
│   ├── epoch006-model-testing/               # Model validation
│   ├── epoch007-application-development/     # Deployment & Apps
│   └── epoch008-retrospective/               # Lifecycle retrospective
├── artifacts/epoch00X-xxx/      # Models, reports, visualizations
└── data/{DOMINO_PROJECT_NAME}/  # Project-specific datasets
```

## Available Agents

### Core Agents
- **Business-Analyst-Agent**: Epoch 001 - Project coordinator, requirements owner, planning
- **Data-Wrangler-Agent**: Epoch 002 - Data acquisition with resource limits
- **Data-Scientist-Agent**: Epoch 003 & 004 - EDA and feature engineering
- **Model-Developer-Agent**: Epoch 005 - Two-phase: baseline all frameworks, tune best
- **Model-Tester-Agent**: Epoch 006 - Advanced testing, edge cases, compliance
- **MLOps-Engineer-Agent**: Epoch 007 - Deployment, monitoring, Streamlit apps
- **All-Agents-Collaborative**: Epoch 008 - Comprehensive retrospective

### Reference Documentation
- **Agent-Interaction-Protocol**: Communication patterns between agents
- **Example-Demonstration-Flows**: Workflow examples and file organization

## Domino Documentation Reference

**Complete Domino Data Lab documentation available at:**
- `/mnt/code/.reference/docs/DominoDocumentation.md` - Full platform documentation
- `/mnt/code/.reference/docs/DominoDocumentation6.1.md` - Version 6.1 specific docs

## Technology Stack

- **Primary Language**: Python 3.8+ for all ML operations
- **Resource Monitoring**: psutil for memory management
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch
- **UI Framework**: Streamlit with standardized styling (muted professional theme)
  - Colors: #BAD2DE (table), #CBE2DA (view), #E5F0EC (mv)
  - See `/mnt/code/.reference/framework_configs.json` for full styling config
- **Experiment Tracking**: MLflow with 2-experiment structure (`{project}_data` and `{project}_model`)
  - **DUAL STORAGE**: All metrics, artifacts, and results saved to BOTH MLflow AND `/mnt/artifacts/`
- **Deployment**: FastAPI, Flask, Docker, Domino Flows
- **Code Reusability**: Automatic extraction to `/mnt/code/src/` (shared across all projects)

## Key Patterns

### Business Analyst Planning Process (Epoch 001)
**The Business Analyst coordinates with other agents to create a comprehensive project plan:**

**Step 1: Requirements Gathering**
- Collaborate with user to understand business problem
- Define objectives, constraints, and success metrics
- Research industry best practices

**Step 2: Agent Consultation**
- Query Data Wrangler about data availability
- Consult Data Scientist on analytical approach
- Engage Model Developer on model recommendations
- Discuss Model Tester on validation requirements
- Review MLOps Engineer on deployment constraints

**Step 3-6: Plan Creation**
Creates comprehensive planning documents:
- `project_plan.json`: Epochs 002-008 with agent assignments
- `success_criteria.json`: Definition of done for each epoch
- `resource_estimates.json`: Time, compute, memory, cost estimates
- `risk_assessment.json`: Risks with mitigation strategies

### Agent Coordination
- **Business-Analyst-Agent is project coordinator**: Plans entire lifecycle in Epoch 001
- **User triggers execution**: Manually triggers each agent for their epoch after planning
- All agents use EXISTING epoch directories
- **Reusable Code Extraction**: Each agent extracts code to `/mnt/code/src/` at completion
- **Cross-Agent Communication**: Agents share context via `/mnt/code/.context/pipeline_state.json`

### Resource-Safe Data Operations
See `/mnt/code/.reference/boilerplate_patterns.md` for:
- Memory monitoring patterns
- Chunked generation code
- File size validation

### Streamlit Styling Pattern
See `/mnt/code/.reference/framework_configs.json` for:
- Color palette
- Page configuration
- Performance caching decorators
- Layout patterns

### MLflow Integration and Model Registry
**2-Experiment Structure:**
- `{project_name}_data`: All data processing (Epochs 001-004)
- `{project_name}_model`: All model operations (Epochs 005-008)

**Model Registry Stages:**
- **None**: Initial registration in Epoch 005
- **Staging**: After passing Epoch 006 tests
- **Production**: After successful Epoch 007 deployment
- **Archived**: Previous versions when new model promoted

**Key Features:**
- Automatic versioning on each registration
- Tag versions with epoch, data version, test results
- Enable A/B testing by loading specific versions
- Track lineage from data version to model version
- **Dual Storage**: Everything saved to both `/mnt/artifacts/` AND MLflow

### File Organization
- Code: `/mnt/code/epoch00X-xxx/`
- Notebooks: `epoch00X-xxx/notebooks/`
- Scripts: `epoch00X-xxx/scripts/`
- Artifacts: `/mnt/artifacts/epoch00X-xxx/`
- Data: `/mnt/data/{DOMINO_PROJECT_NAME}/epoch00X-xxx/`
- Reusable code: `/mnt/code/src/` (shared across projects)

### Cross-Agent Communication
Agents communicate through shared context files in `/mnt/code/.context/`:

**pipeline_state.json** - Master state tracking
**data_lineage.json** - Data transformation tracking
**checkpoints/** - Incremental progress saves

**Communication features:**
- State management across epochs
- Metadata sharing (model info, schemas, metrics)
- Handoff coordination with validation
- Error recovery with checkpoint access
- Complete audit trail
- Resource planning for upcoming epochs
- Blocker tracking

## Common Usage Patterns

```python
# Epoch-by-epoch execution
"Business Analyst: Plan credit risk model project"                          # Epoch 001
"Data Wrangler: Generate synthetic credit data"                             # Epoch 002
"Data Scientist: Perform exploratory data analysis on credit data"          # Epoch 003
"Data Scientist: Engineer features for credit risk prediction"              # Epoch 004
"Model Developer: Train and test credit risk models"                        # Epoch 005
"Model Tester: Run advanced validation and edge case testing"               # Epoch 006
"MLOps Engineer: Deploy model, add monitoring, and build Streamlit app"     # Epoch 007
"All Agents: Conduct comprehensive retrospective of entire ML lifecycle"    # Epoch 008
```

## Reference Files

### Boilerplate Patterns
`/mnt/code/.reference/boilerplate_patterns.md` contains:
- ResourceTracker class
- Error handling patterns
- Checkpoint management
- Data lineage tracking
- Quality gates
- Resource management
- Directory validation
- Code extraction
- Human-in-the-loop validation

### Visualization Standards
`/mnt/code/.reference/visualization_standards.md` contains:
- Required visualizations by epoch
- Reusable plotting functions
- Notebook structure standards
- Execution requirements

### Framework Configurations
`/mnt/code/.reference/framework_configs.json` contains:
- GPU detection code
- Framework-specific parameters
- Baseline and tuning parameter grids
- Standardized metrics
- MLflow integration patterns
- Streamlit styling configuration
- Resource limits

## Development Guidelines

- **Business-Analyst-Agent coordinates project planning**: Plans Epochs 002-008 in Epoch 001
- **User triggers execution**: Manually triggers each agent after planning complete
- **Use templates**: Reference files for epoch-specific requirements
- Always specify project names for proper organization
- **Validate data directories** before operations
- **Monitor resource usage** through logged metrics
- **Use extracted code** from `/mnt/code/src/` for reusability
- **Check src/ directory** at start of each epoch for existing utilities
- Test with small datasets before scaling
- **Mandatory review gates** at critical junctions
- **Epoch 008 Retrospective**: All agents review entire ML lifecycle

## Epoch 008: Comprehensive Retrospective Framework

**All agents participate in Epoch 008 to conduct thorough post-mortem analysis:**

### Retrospective Components

**1. Agent-Specific Retrospectives**
Each agent creates detailed report in `/mnt/code/epoch008-retrospective/lessons_learned/`:
- business_analysis.md
- data_wrangling.md
- eda_feature_engineering.md
- model_development.md
- model_testing.md
- mlops_deployment.md
- overall_recommendations.md

**2. Automated Performance Analysis**
Compare estimated vs actual metrics across all epochs

**3. What-If Analysis**
Analyze alternative approaches and potential impacts

**4. Reusable Playbook Generation**
Create playbook for similar future projects with:
- Success factors
- Pitfalls avoided
- Recommended timeline
- Tech stack recommendations
- Resource requirements
- Key learnings

**5. Timeline Visualization**
Visual timeline showing time spent, checkpoints, reviews, blockers

**6. Final Recommendations**
Process improvements, technical improvements, resource optimization, quality improvements

## Important Reminders

**Do what has been asked; nothing more, nothing less.**

**NEVER create files unless they're absolutely necessary for achieving your goal.**

**ALWAYS prefer editing an existing file to creating a new one.**

**NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.**

**NEVER use emojis or emoticons in any outputs, code, documentation, or communication.**
