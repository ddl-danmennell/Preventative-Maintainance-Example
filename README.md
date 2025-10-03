# Machine Learning Project Development Template

> A structured framework for building end-to-end machine learning projects with comprehensive planning, quality gates, automated testing, checkpoint recovery, and resource-safe execution powered by Claude Sonnet 4.5

## What You'll Build

This template guides you through creating a complete, production-ready ML solution with:

- **Comprehensive project planning** with Business Analyst coordination and success criteria
- **Automated quality gates** preventing progression with incomplete dependencies
- **Automated data pipelines** with lineage tracking and resource management
- **Advanced ML models** with MLflow Model Registry integration
- **Rigorous testing framework** including unit, integration, and automated validation
- **Interactive dashboards** with standardized Streamlit styling
- **Production deployment** with monitoring and human-in-the-loop validation
- **Automated code extraction** to reusable modules with unit tests
- **Incremental checkpointing** for failure recovery
- **Cross-agent communication** via shared context files
- **Resource-safe execution** with 50MB file limits and 12GB RAM monitoring
- **Comprehensive retrospective** with performance analysis and playbook generation
- **Full documentation** and reproducibility

## Project Lifecycle (8 Epochs)

The Business Analyst coordinates planning in Epoch 001, then you trigger each subsequent epoch.

### Epoch 001: Research Analysis and Planning
**Agent**: Business-Analyst-Agent | **MLflow**: `{project}_data` | **Duration**: 2-3 hours

**PROJECT COORDINATOR** - Business Analyst works with you and other agents to plan the entire ML lifecycle.

**What You'll Do**:
- Translate business requirements to technical specifications
- Coordinate with all agents to plan Epochs 002-008
- Create success criteria and resource estimates for each epoch
- Assess risks and define mitigation strategies
- Generate executive research report

**Outputs**:
- `project_plan.json` - Comprehensive plan with agent assignments
- `success_criteria.json` - Definition of done for each epoch
- `resource_estimates.json` - Time, compute, memory, cost estimates
- `risk_assessment.json` - Identified risks with mitigations
- Executive summary report

**Usage**: `"Plan [use case] project with comprehensive requirements analysis"`

**Human Review Gate**: User approves plan before proceeding to Epoch 002

---

### Epoch 002: Data Wrangling
**Agent**: Data-Wrangler-Agent | **MLflow**: `{project}_data` | **Duration**: 2-3 hours

Acquire or generate high-quality datasets with resource-safe execution and automated validation.

**What You'll Do**:
- Validate prerequisites (Epoch 001 complete)
- Source data from APIs, databases, or generate synthetic data
- Clean and validate data quality (chunked processing: 10K rows/chunk)
- Track data lineage and transformations
- Save to `/mnt/data/{DOMINO_PROJECT_NAME}/epoch002-data-wrangling/`
- Extract reusable code to `/mnt/code/src/{project}/`
- Run automated tests

**Outputs**: `synthetic_data.parquet` (≤50MB), data quality report, unit tests

**Quality Gate**: Data file < 50MB, basic schema validation passed

**Usage**: `"Generate synthetic [domain] data with [X] features and [Y] rows"`

---

### Epoch 003: Exploratory Data Analysis
**Agent**: Data-Scientist-Agent | **MLflow**: `{project}_data` | **Duration**: 3-4 hours

Perform comprehensive statistical analysis with informative visualizations in executed notebooks.

**What You'll Do**:
- Validate prerequisites (Epoch 002 complete, data file exists)
- Statistical summaries and distribution analysis
- Feature correlation matrices and relationship visualization
- Identify outliers, missing patterns, and data quality issues
- Create comprehensive visualizations (distributions, heatmaps, pair plots)
- Execute all notebook cells and save with outputs
- Generate actionable insights for feature engineering

**Outputs**: Executed EDA notebook with embedded plots, visualizations (HTML + PNG), statistical reports

**Quality Gate**: EDA complete, data quality report generated, no critical issues

**Human Review Gate**: User reviews data quality before feature engineering

**Usage**: `"Perform comprehensive EDA on [dataset] focusing on [target variable]"`

---

### Epoch 004: Feature Engineering
**Agent**: Data-Scientist-Agent | **MLflow**: `{project}_data` | **Duration**: 2-3 hours

Transform raw features into ML-ready representations with feature provenance tracking.

**What You'll Do**:
- Validate prerequisites (Epoch 003 complete, EDA passed review)
- Create derived features (interactions, aggregations, domain-specific)
- Encode categorical variables (one-hot, target, ordinal)
- Scale numerical features (standardization, normalization)
- Document feature provenance in data lineage
- Split data into train/validation/test sets
- Extract feature engineering utilities

**Outputs**: `engineered_features.parquet`, feature provenance, `/mnt/code/src/{project}/feature_engineering.py`

**Quality Gate**: Features engineered, feature importance calculated, train/test split created

**Human Review Gate**: User reviews engineered features before model training

**Usage**: `"Engineer features for [prediction task] using [encoding/scaling strategy]"`

---

### Epoch 005: Model Development
**Agent**: Model-Developer-Agent | **MLflow**: `{project}_model` | **Duration**: 4-5 hours

**Two-Phase Process**: Initial signal detection across ALL frameworks, then hyperparameter tuning ONLY the best model.

**Phase 1 - Initial Signal Detection (Notebooks)**:
- Train baseline models with ALL frameworks to detect initial signal:
  - **Classification**: scikit-learn (LogisticRegression, RandomForest), XGBoost, LightGBM, TensorFlow, PyTorch
  - **Regression**: scikit-learn (LinearRegression, RandomForest), XGBoost, LightGBM, TensorFlow, PyTorch, Statsmodels
- GPU detection and automatic configuration for all frameworks
- All models use standardized metrics for fair comparison
- Identify best performing framework based on primary metric
- Execute all notebook cells and save with outputs

**Phase 2 - Hyperparameter Tuning (Scripts)**:
- Tune ONLY the best performing model from Phase 1
- Comprehensive parameter grids for each framework
- Nested MLflow runs (parent tuning run with child runs for each combination)
- Generate summary charts and improvement analysis

**Dual Storage**: All metrics, artifacts, and results saved to BOTH MLflow AND `/mnt/artifacts/`

**Outputs**:
- Training: `all_models_metrics.json`, `model_comparison_summary.csv`, `all_models_comparison.png`, `best_model_info.json`
- Tuning: `all_tuning_results.json`, `all_tuning_results.csv`, `tuning_parameter_impact.png`, `top_10_models.png`, `tuned_model_info.json`, `tuning_summary_report.json`
- Registered models in MLflow Model Registry (stage: None)
- Reusable model utilities with unit tests

**Quality Gate**: All frameworks tested, best model identified and tuned, metrics logged

**Usage**:
- Notebook: `jupyter notebook /mnt/code/epoch005-model-development/notebooks/model_training.ipynb`
- Tuning: `python /mnt/code/epoch005-model-development/scripts/tune_best_model.py`

---

### Epoch 006: Model Testing
**Agent**: Model-Tester-Agent | **MLflow**: `{project}_model` | **Duration**: 2-3 hours

Conduct advanced validation and promote model to Staging in Model Registry.

**What You'll Do**:
- Validate prerequisites (Epoch 005 complete)
- Functional testing: Business requirements verification
- Performance testing: Latency, throughput, resource usage
- Edge case testing: Boundary conditions, adversarial inputs
- Robustness testing: Input perturbation stability
- Promote model to Staging stage after passing tests
- Tag model with test results

**Outputs**: Comprehensive test report, Model Registry stage: Staging, unit tests

**Quality Gate**: Advanced testing complete, edge cases documented, compliance validated

**Human Review Gate**: User reviews test results before deployment

**Usage**: `"Run comprehensive testing on [model] including edge cases and robustness"`

---

### Epoch 007: Application Development
**Agent**: MLOps-Engineer-Agent | **MLflow**: `{project}_model` | **Duration**: 3-4 hours

Deploy model with monitoring, create Streamlit app, promote to Production in Model Registry.

**What You'll Do**:
- Validate prerequisites (Epoch 006 complete, model in Staging)
- Create deployment pipelines (Docker, Kubernetes, Domino)
- Set up monitoring and alerting (drift detection, performance tracking)
- Build model serving APIs (FastAPI, Flask)
- Build Streamlit dashboard with standardized styling (#BAD2DE theme)
- Promote model to Production stage
- Tag model with deployment metadata
- Extract deployment and app utilities

**Outputs**: Deployment pipeline, monitoring dashboard, API endpoints, Streamlit app (`app.py`, `style.css`)

**Quality Gate**: Deployment successful, monitoring active, app functional

**Human Review Gate**: User verifies deployment before retrospective

**Usage**: `"Deploy [model] with monitoring, API serving, and interactive dashboard"`

---

### Epoch 008: Retrospective
**Agent**: All-Agents-Collaborative | **MLflow**: `{project}_model` | **Duration**: 2 hours

Comprehensive review of entire ML lifecycle with all agents participating.

**What You'll Do**:
- Each agent reviews their specific contributions
- Generate performance comparison reports (estimated vs actual)
- Create "what-if" analysis for alternative approaches
- Generate reusable playbook for similar projects
- Analyze resource utilization and optimization opportunities
- Document lessons learned and recommendations
- Create timeline visualization

**Outputs**:
- Agent-specific retrospectives in `lessons_learned/`
- Performance comparison reports
- What-if analysis
- Reusable project playbook
- Resource utilization analysis
- Overall recommendations document

**Usage**: `"Conduct comprehensive retrospective of entire ML lifecycle"`

## Agent Architecture

All agents powered by Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)

### Business Analyst (Project Coordinator)
- Coordinates with all agents to plan entire lifecycle
- Creates comprehensive project plan with success criteria
- Estimates resources and assesses risks
- Generates executive summaries
- **Human Review Gate**: Plan approval required before execution

### Data Wrangler
- Resource-safe data generation (50MB, 12GB limits)
- Chunked processing with checkpointing
- Data lineage tracking
- Automated quality validation
- Unit testing

### Data Scientist
- EDA with comprehensive visualizations
- Feature engineering with provenance tracking
- Executed notebooks with embedded outputs
- Statistical analysis and insights

### Model Developer
- Two-phase process: signal detection → tuning best model
- ALL frameworks tested: scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch (+ Statsmodels for regression)
- GPU detection and automatic configuration
- Standardized metrics for fair comparison
- MLflow Model Registry integration with nested runs
- Dual storage: MLflow + artifacts directory
- Model versioning and tagging

### Model Tester
- Comprehensive testing framework
- Edge case and robustness validation
- Model Registry promotion to Staging
- Test result tagging

### MLOps Engineer
- Deployment pipeline creation
- Monitoring and alerting setup
- Streamlit app development
- Model Registry promotion to Production
- API serving with FastAPI

### All Agents (Retrospective)
- Collaborative review of entire lifecycle
- Performance analysis and recommendations
- Playbook generation for future projects

## Key Features

### Automated Quality Gates
- Prerequisite validation before each epoch
- Handoff criteria enforcement
- Dependency checking via `pipeline_state.json`
- Prevents progression with incomplete work

### Incremental Checkpointing
- Automatic checkpoint saves every 10% or 5 minutes
- Resume capability on failure
- Checkpoint metadata includes progress, next step, intermediate artifacts
- Stored in `/mnt/code/.context/checkpoints/`

### Data Lineage Tracking
- Complete transformation history in `data_lineage.json`
- Feature provenance (source features, formulas)
- Data quality changes at each step
- Traceability from predictions to source data

### Standardized Error Handling
- Error taxonomy (DataQualityError, ResourceLimitError, DependencyError)
- Graceful degradation for resource limits
- Automatic checkpoint recovery
- Blocker and warning tracking

### Model Registry Integration
- Automatic registration with signatures
- Stage transitions: None → Staging → Production → Archived
- Comprehensive tagging (epoch, testing, deployment metadata)
- Version tracking and lineage

### Automated Testing Framework
- Unit tests for all reusable code
- Integration tests for end-to-end pipeline
- Epoch-specific validation tests
- Test execution with pytest

### Human-in-the-Loop Validation
- 5 mandatory review gates:
  1. After Epoch 001: Plan approval
  2. After Epoch 003: Data quality assessment
  3. After Epoch 004: Feature readiness
  4. After Epoch 006: Model validation
  5. After Epoch 007: Deployment verification
- Approval logging with timestamps
- Option to reject and return to previous epoch

### Cost and Resource Tracking (Optional)
- Disabled by default (`ENABLE_COST_TRACKING=false`)
- Tracks compute time, memory, CPU usage
- Logs to MLflow and `resource_tracking.json`
- Aggregate reporting in Epoch 008
- Easy to customize with own cost calculation

### Jupyter Notebook Standards
- Required visualizations by epoch
- Interactive plots (Plotly) and static plots (Matplotlib/Seaborn)
- Dual save format (HTML + PNG)
- All cells executed and saved with outputs
- Standardized notebook structure with findings

## Project Structure

```
Your Project/
├── src/{project}/                        # Reusable code modules with tests
│   ├── __init__.py
│   ├── data_processing_utils.py          # Epoch 002
│   ├── data_loading_pipeline.py          # Epoch 002
│   ├── feature_engineering.py            # Epoch 004
│   ├── model_utils.py                    # Epoch 005
│   ├── validation.py                     # Epoch 006
│   ├── deployment.py                     # Epoch 007
│   ├── monitoring.py                     # Epoch 007
│   ├── serving.py                        # Epoch 007
│   ├── app_utils.py                      # Epoch 007
│   ├── streamlit_components.py           # Epoch 007
│   ├── error_handling.py                 # Standardized error handling
│   ├── config.py                         # Project configuration
│   ├── tests/                            # Unit tests
│   │   ├── test_data_processing.py
│   │   ├── test_feature_engineering.py
│   │   └── test_model_utils.py
│   └── README.md
├── .context/                             # Cross-agent communication
│   ├── pipeline_state.json               # Master state tracking
│   ├── data_lineage.json                 # Transformation tracking
│   ├── resource_tracking.json            # Resource usage (optional)
│   ├── approval_log.json                 # Human review decisions
│   ├── error_log.json                    # Error history
│   └── checkpoints/                      # Progress checkpoints
│       ├── epoch002_checkpoint.json
│       └── ...
├── epoch001-research-analysis-planning/  # Business requirements & planning
│   ├── notebooks/
│   ├── scripts/
│   ├── project_plan.json
│   ├── success_criteria.json
│   ├── resource_estimates.json
│   ├── risk_assessment.json
│   └── README.md
├── epoch002-data-wrangling/             # Data pipelines & quality
│   ├── notebooks/
│   ├── scripts/
│   └── README.md
├── epoch003-exploratory-data-analysis/   # EDA & insights
│   ├── notebooks/                        # Executed with outputs
│   ├── scripts/
│   └── README.md
├── epoch004-feature-engineering/         # Feature transformations
│   ├── notebooks/
│   ├── scripts/
│   └── README.md
├── epoch005-model-development/          # ML training & optimization
│   ├── notebooks/
│   ├── scripts/
│   └── README.md
├── epoch006-model-testing/              # Comprehensive testing
│   ├── notebooks/
│   ├── scripts/
│   └── README.md
├── epoch007-application-development/    # Deployment & apps
│   ├── notebooks/
│   ├── scripts/
│   ├── app/
│   │   ├── app.py                       # Streamlit app
│   │   └── style.css                    # Standard styling
│   └── README.md
├── epoch008-retrospective/              # Lifecycle review
│   ├── notebooks/
│   ├── scripts/
│   ├── lessons_learned/
│   │   ├── business_analysis.md
│   │   ├── data_wrangling.md
│   │   ├── eda_feature_engineering.md
│   │   ├── model_development.md
│   │   ├── model_testing.md
│   │   ├── mlops_deployment.md
│   │   └── overall_recommendations.md
│   ├── project_playbook.json
│   └── README.md
└── artifacts/                           # Models, reports, visualizations
    ├── epoch001-research-analysis-planning/
    ├── epoch002-data-wrangling/
    ├── epoch003-exploratory-data-analysis/
    ├── epoch004-feature-engineering/
    ├── epoch005-model-development/
    ├── epoch006-model-testing/
    ├── epoch007-application-development/
    └── epoch008-retrospective/
```

## Data Storage Structure

```
/mnt/data/{DOMINO_PROJECT_NAME}/
├── epoch002-data-wrangling/
│   └── data.parquet                      # Max 50MB
├── epoch003-exploratory-data-analysis/
│   └── eda_dataset.parquet
├── epoch004-feature-engineering/
│   └── engineered_features.parquet
└── epoch005-model-development/
    ├── train_data.parquet
    └── val_data.parquet
```

## Technology Stack

### Core Technologies
- **Python 3.8+** - Primary development language
- **MLflow** - Experiment tracking and Model Registry
- **psutil** - Memory monitoring for resource management
- **nbformat** - Notebook execution and validation
- **pytest** - Automated testing

### ML Frameworks
- **scikit-learn** - Classical ML algorithms
- **XGBoost/LightGBM** - Gradient boosting
- **TensorFlow/PyTorch** - Deep learning
- **Optuna** - Hyperparameter optimization

### Visualization
- **Matplotlib/Seaborn** - Static visualizations
- **Plotly** - Interactive visualizations
- **Jupyter** - Notebook environment

### Deployment & UI
- **FastAPI** - High-performance Python APIs
- **Streamlit** - Interactive dashboards (standardized styling)
- **Semantic UI** - Professional UI components
- **Docker** - Containerization

## Getting Started

### Prerequisites
- Domino workspace access
- Python environment with required packages
- `/mnt/data/{DOMINO_PROJECT_NAME}/` directory (will prompt if missing)

### Your First Project

1. **Start with Business Analyst for planning**
   ```
   "Plan a credit risk model project with comprehensive requirements analysis"
   ```

2. **Review and approve the plan**
   - Project plan with agent assignments
   - Success criteria for each epoch
   - Resource estimates
   - Risk assessment
   - Approve to proceed

3. **Follow the planned epochs**
   - Each agent triggered based on plan
   - Quality gates validate prerequisites
   - Human review at critical junctions
   - Checkpoints enable recovery

4. **Complete with retrospective**
   - All agents review contributions
   - Performance analysis
   - Playbook generation

## Resource Management

### Memory Safety
- 12GB RAM limit with continuous monitoring
- Chunked processing (10K rows default)
- Automatic garbage collection
- Real-time progress with memory display

### File Size Control
- 50MB limit on all data files
- Automatic sampling when exceeded
- Parquet compression
- Post-save validation

### Checkpoint Recovery
- Automatic saves every 10% progress
- Resume from last checkpoint on failure
- Checkpoint metadata includes next step

## Streamlit Styling Standards

### Color Palette
- Table Background: `#BAD2DE` (muted blue)
- View Background: `#CBE2DA` (muted green)
- Materialized View: `#E5F0EC` (light green)
- Text Color: `rgb(90, 90, 90)` (neutral gray)

### Standard Features
- Wide layout with responsive design
- Semantic UI integration
- 3-column grids
- Performance caching
- Session state management
- Custom card components

## Support & Resources

### Documentation
- Main guidance: `CLAUDE.md`
- Agent protocols: `.claude/agents/`
- Epoch guides: `epoch00X-xxx/README.md`
- Reusable code: `src/{project}/README.md`
- Domino platform: `.reference/docs/DominoDocumentation.md`

---

<div align="center">
  <strong>Comprehensive Planning • Quality Gates • Checkpoint Recovery • Automated Testing • Human Oversight • MLflow Registry • Production-Ready</strong>
  <br>
  <em>Powered by Claude Sonnet 4.5</em>
</div>
