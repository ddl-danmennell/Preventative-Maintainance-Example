# Epoch 007: Application Development

**Agent**: MLOps-Engineer-Agent | **MLflow**: `{project}_model` | **Duration**: 3-5 hours

## Purpose

Deploy production-ready model with monitoring, CI/CD pipelines, and scalable serving infrastructure.

## What You'll Do

1. **Deployment Pipeline**: Create containerized deployment workflow (Docker, Kubernetes, Domino)
2. **Model Serving**: Build REST API with FastAPI or Flask
3. **Monitoring Setup**: Implement drift detection and performance tracking
4. **CI/CD Integration**: Automate testing and deployment
5. **Infrastructure**: Configure scaling, load balancing, and logging

## How to Use This Epoch

**Command Template**:
```
"Deploy [model] with monitoring and API serving on [platform]"
```

**Example Commands**:
- `"Deploy XGBoost credit model with FastAPI and Evidently drift monitoring on Domino"`
- `"Create Kubernetes deployment for fraud detection model with Prometheus monitoring"`
- `"Build Flask API for patient readmission model with automated retraining pipeline"`

## Deployment Options

### 1. Domino Model API
**Best for**: Enterprise deployments on Domino platform
- Automated scaling and version management
- Built-in monitoring and logging
- Secure endpoint with authentication
- Model registry integration

### 2. FastAPI + Docker
**Best for**: High-performance REST APIs
- Automatic OpenAPI documentation
- Async request handling
- Type validation with Pydantic
- Easy containerization

### 3. Flask + Gunicorn
**Best for**: Simple, production-ready APIs
- Lightweight and flexible
- WSGI production server
- Easy integration with existing systems

### 4. Kubernetes Deployment
**Best for**: Large-scale, cloud-native deployments
- Auto-scaling based on load
- Rolling updates and rollbacks
- Health checks and self-healing
- Multi-region deployment

## Deployment Components

### Model Serving API
**Endpoints**:
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /health` - Health check
- `GET /model/info` - Model metadata
- `GET /metrics` - Performance metrics

**Request Format**:
```json
{
  "features": {
    "income": 75000,
    "debt_ratio": 0.35,
    "credit_score": 720,
    ...
  }
}
```

**Response Format**:
```json
{
  "prediction": 0,
  "probability": 0.23,
  "model_version": "1.0.0",
  "inference_time_ms": 45
}
```

### Monitoring & Observability

**Drift Detection**:
- Feature drift monitoring (distribution changes)
- Prediction drift (output distribution)
- Concept drift (model performance degradation)
- Alert thresholds and notifications

**Performance Metrics**:
- Request latency (p50, p95, p99)
- Throughput (requests/second)
- Error rates
- Model accuracy on live data

**Logging**:
- Request/response logging
- Error and exception tracking
- Audit trail for compliance
- Model version tracking

### CI/CD Pipeline

**Automated Testing**:
- Unit tests for serving code
- Integration tests for API
- Model validation tests
- Performance regression tests

**Deployment Workflow**:
1. Code push triggers pipeline
2. Run test suite
3. Build Docker image
4. Deploy to staging
5. Run smoke tests
6. Promote to production
7. Monitor deployment

## Outputs

**Generated Files**:
- `notebooks/deployment_setup.ipynb` - Deployment configuration
- `scripts/deploy_model.py` - Deployment automation script
- `scripts/api_server.py` - FastAPI/Flask server code
- `Dockerfile` - Container definition
- `kubernetes/` - K8s manifests (if applicable)
  - `deployment.yaml`
  - `service.yaml`
  - `ingress.yaml`
- `/mnt/artifacts/epoch007-application-development/`
  - `deployment_config.json` - Configuration settings
  - `api_documentation.html` - API docs
  - `monitoring_dashboard.json` - Dashboard config
  - `requirements.txt` - Dependencies

**Logged to MLflow** (`{project}_model` experiment):
- Deployment configuration
- API endpoint URLs
- Monitoring dashboard URLs
- Container image tags
- Deployment timestamp and version

**Reusable Code**: `/mnt/code/src/deployment_utils.py`
- Model loading and initialization
- API route handlers
- Input validation functions
- Monitoring metric collectors
- Drift detection utilities
- Logging configuration

## Cross-Agent Communication

**Reads from Pipeline Context**:
- `epoch005.best_model_path` - Model to deploy
- `epoch005.mlflow_model_uri` - MLflow model URI
- `epoch004.feature_names` - Expected features
- `epoch004.encoding_mappings` - Feature transformations
- `epoch006.production_ready` - Deployment approval
- `epoch006.test_scores` - Expected performance baseline

**Writes to Pipeline Context**:
```json
{
  "epoch007": {
    "deployment": {
      "status": "deployed",
      "platform": "domino",
      "api_endpoint": "https://api.example.com/predict",
      "model_version": "1.0.0",
      "deployment_timestamp": "2025-10-02T10:30:00Z"
    },
    "monitoring": {
      "dashboard_url": "https://monitor.example.com/dashboard",
      "drift_detection_enabled": true,
      "alert_email": "team@example.com",
      "log_retention_days": 90
    },
    "performance": {
      "avg_latency_ms": 48,
      "throughput_per_sec": 2100,
      "availability_percent": 99.9
    },
    "ci_cd": {
      "pipeline_url": "https://github.com/org/repo/actions",
      "auto_retrain_enabled": true,
      "retrain_frequency": "weekly"
    }
  }
}
```

**Used By**:
- **Epoch 008** (App): API endpoint for predictions

## Success Criteria

✅ Model deployed to production environment
✅ REST API serving predictions successfully
✅ API documentation generated
✅ Monitoring dashboard configured
✅ Drift detection active
✅ CI/CD pipeline operational
✅ Performance meets requirements (latency, throughput)
✅ Logging and audit trail configured
✅ Reusable code extracted to `/mnt/code/src/deployment_utils.py`

## Deployment Checklist

- [ ] Load tested model from Epoch 005
- [ ] Import feature engineering pipeline from `/mnt/code/src/`
- [ ] Create API server code (FastAPI/Flask)
- [ ] Implement input validation
- [ ] Add prediction endpoint with error handling
- [ ] Write Dockerfile for containerization
- [ ] Set up monitoring (Evidently, Prometheus, or custom)
- [ ] Configure drift detection thresholds
- [ ] Create CI/CD pipeline (GitHub Actions, GitLab CI, or Jenkins)
- [ ] Deploy to staging environment
- [ ] Run integration tests
- [ ] Deploy to production
- [ ] Verify health checks
- [ ] Set up alerting
- [ ] Log deployment to MLflow (`{project}_model`)
- [ ] Extract reusable code to `/mnt/code/src/deployment_utils.py`
- [ ] Document API usage

## Monitoring Best Practices

**What to Monitor**:
1. **Input Data**: Feature distributions, missing values, outliers
2. **Predictions**: Output distribution, confidence scores
3. **Performance**: Latency, throughput, error rates
4. **Model Accuracy**: Ground truth comparison when available
5. **Infrastructure**: CPU, memory, disk usage

**Alert Thresholds**:
- Feature drift > 0.1 (PSI or KL divergence)
- Prediction drift > 0.15
- Latency p95 > 100ms
- Error rate > 1%
- Accuracy drop > 5% from baseline

## Drift Detection Strategies

**Statistical Tests**:
- **PSI (Population Stability Index)**: Measure distribution shift
- **KS Test (Kolmogorov-Smirnov)**: Compare distributions
- **Chi-Square**: Categorical feature changes
- **Jensen-Shannon Divergence**: Symmetric distribution comparison

**When to Retrain**:
- Significant feature drift detected
- Model performance degraded > 5%
- New data patterns emerge
- Scheduled retraining (weekly/monthly)

## Security Considerations

- [ ] Authentication on API endpoints (API keys, OAuth)
- [ ] Rate limiting to prevent abuse
- [ ] Input sanitization and validation
- [ ] Encryption for data in transit (HTTPS)
- [ ] Secrets management (env vars, vaults)
- [ ] Audit logging for compliance
- [ ] Container security scanning

## Next Steps

Proceed to **Epoch 008: Retrospective** to build interactive Streamlit dashboard and conduct project review.

---

**Ready to start?** Use the MLOps-Engineer-Agent with your deployment platform preference.
