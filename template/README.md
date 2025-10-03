# ML Project Template

> Production-ready machine learning project template with automated code reusability, cross-agent communication, and comprehensive governance

## Overview

This template provides a complete framework for building end-to-end ML projects with:
- **7 Epoch Structure**: Research → Data → EDA → Development → Testing → Deployment → Retrospective
- **8 Specialized Agents**: All powered by Claude Sonnet 4.5
- **Automated Code Extraction**: Reusable utilities saved to `/mnt/code/src/{project}/`
- **Cross-Agent Communication**: Seamless state sharing via pipeline context
- **Governance Compliance**: Built-in regulatory framework support
- **Multiple UI Options**: React, Next.js, Vue.js, or Python frameworks

## Quick Start

### 1. Define Your Project
```python
"Create a [domain] ML model for [use case] with [compliance requirements]"
```

### 2. Pipeline Automatically Executes
- Epoch 001: Research analysis and planning
- Epoch 002: Data wrangling
- Epoch 003: Data exploration
- Epoch 004: Model development
- Epoch 005: Model testing
- Epoch 006: Application development
- Epoch 007: Retrospective

### 3. Access Reusable Code
After completion, find extracted utilities in:
```
/mnt/code/src/{your_project}/
├── data_processing.py
├── feature_engineering.py
├── model_utils.py
├── validation.py
├── deployment.py
├── monitoring.py
├── serving.py
└── config.py
```

## Features

### Automated Code Reusability
- Functions and classes automatically extracted from each epoch
- Organized into reusable Python modules
- Import and use in future projects: `from src.{project} import model_utils`

### Cross-Agent Communication
- Agents share state via `/mnt/code/.context/{project}_pipeline_state.json`
- Seamless data flow between stages
- Complete audit trail and traceability

### Governance & Compliance
- Automatic identification of applicable frameworks (NIST RMF, Model Risk Management, GDPR, HIPAA)
- Built-in compliance testing and validation
- Professional PDF reports for stakeholder review

### Technology Selection
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch
- **UI Frameworks**: React, Next.js, Vue.js, Streamlit, Dash, Gradio
- **Deployment**: FastAPI, Docker, Domino Flows
- **Tracking**: MLflow for experiments and model registry

## Documentation

- See `/mnt/code/README.md` for complete project documentation
- See `/mnt/code/CLAUDE.md` for agent usage guidelines
- See `/mnt/code/.claude/agents/` for agent specifications
- See `/mnt/code/.context/README.md` for context management

## Support

For issues or questions about this template, refer to the main project documentation or agent interaction protocols.

---

**Powered by Claude Sonnet 4.5** • Research-Driven • Rigorously Tested • Compliance-Ready • Production-Proven
