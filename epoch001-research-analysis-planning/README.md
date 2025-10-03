# Epoch 001: Research Analysis and Planning

**Agent**: Business-Analyst-Agent | **MLflow**: `{project}_data` | **Duration**: 2-3 hours

## Purpose

**PROJECT COORDINATOR** - Business Analyst works with you and all other agents to plan the entire ML lifecycle (Epochs 002-008). This establishes the project foundation with comprehensive planning, resource estimates, success criteria, and risk assessment.

## What You'll Do

1. **Requirements Validation**: Read and validate `requirements.md`
   - Identify any ambiguities or unclear requirements
   - Ask user for clarification on critical uncertainties
   - Make recommendations with rationale for unclear items
   - Update requirements.md with Business Analyst review section
2. **Requirements Analysis**: Translate validated business needs into technical specifications
3. **Agent Coordination**: Consult with all agents to plan their epochs
   - Data Wrangler: Data availability and generation strategy
   - Data Scientist: Analytical approach and feature engineering needs
   - Model Developer: Algorithm recommendations and training requirements
   - Model Tester: Validation and compliance requirements
   - MLOps Engineer: Deployment constraints and infrastructure needs
3. **Project Planning**: Create comprehensive plan for Epochs 002-008
4. **Success Criteria**: Define measurable definition of done for each epoch
5. **Resource Estimation**: Estimate time, compute, memory, and cost per epoch
6. **Risk Assessment**: Identify risks with mitigation strategies
7. **Executive Reporting**: Generate summary for stakeholders

## How to Use This Epoch

**Step 1**: Fill out `requirements.md` in this directory with your project requirements

Edit the file and add your requirements under each epoch heading:
```bash
# Open in your favorite editor
nano /mnt/code/epoch001-research-analysis-planning/requirements.md

# Or use Claude Code
"Edit requirements.md and add my project requirements"
```

**Step 2**: Run the Business Analyst

**Via Claude Code Agent**:
```
"Plan [use case] project with comprehensive requirements analysis"
```

**Example Commands**:
- `"Plan credit risk prediction model project with comprehensive requirements analysis"`
- `"Plan patient readmission prediction project with HIPAA compliance requirements"`
- `"Plan fraud detection system project with regulatory compliance assessment"`
- `"Review my requirements in requirements.md and create a comprehensive project plan"`

**What Happens**:
1. Business Analyst reads `requirements.md`
2. Validates all requirements are clear and complete
3. Identifies ambiguities and asks clarification questions
4. Makes recommendations for unclear requirements with rationale
5. Updates `requirements.md` with review section
6. Proceeds with agent coordination and project planning

## Outputs

**Planning Documents** (saved to this epoch directory):
- `requirements.md` - Updated with Business Analyst review, validation, clarifications, and recommendations
- `project_plan.json` - Comprehensive plan with agent assignments for Epochs 002-008
- `success_criteria.json` - Definition of done for each epoch with measurable criteria
- `resource_estimates.json` - Time, compute, memory, and cost estimates per epoch
- `risk_assessment.json` - Identified risks with probability, impact, and mitigations
- `executive_summary.md` - Business objective, approach, timeline, key risks

**Logged to MLflow** (`{project}_data` experiment):
- Business requirements and objectives
- Technology recommendations
- Project timeline estimates
- Risk assessment scores

## Cross-Agent Communication

**Writes to Pipeline Context** (`/mnt/code/.context/pipeline_state.json`):
```json
{
  "project_name": "credit_risk_model",
  "epoch_001_complete": true,
  "epoch_001_review_approved": true,
  "business_requirements": "...",
  "target_metric": "precision",
  "regulatory_frameworks": ["Basel III", "Model Risk Management"],
  "recommended_algorithms": ["XGBoost", "LogisticRegression"],
  "deployment_strategy": "domino_model_api",
  "resource_estimates": {
    "002": {"time_hours": 2, "memory_gb": 4},
    "003": {"time_hours": 3, "memory_gb": 6},
    ...
  }
}
```

**Used By**:
- **Epoch 002-008**: All agents reference plan for their responsibilities
- **Quality Gates**: Prerequisites validation uses completion flags
- **Human Review Gates**: Success criteria used for review checklists

## Quality Gates

**Prerequisites**: None (first epoch)

**Completion Criteria**:
- Project plan created with all epochs detailed
- Success criteria defined for each epoch with measurable thresholds
- Resource estimates provided for each epoch
- Risk assessment completed with mitigation strategies
- Executive summary generated

## Human Review Gate

**MANDATORY REVIEW**: User must review and approve the plan before proceeding to Epoch 002.

**Review Checklist**:
- [ ] Project plan covers all epochs with clear deliverables
- [ ] Success criteria are measurable and realistic
- [ ] Resource estimates align with available capacity
- [ ] Risks are identified with appropriate mitigations
- [ ] Timeline is acceptable
- [ ] Agent assignments make sense

**Actions**:
- **Approve**: Proceed to Epoch 002
- **Request modifications**: Business Analyst revises plan
- **Reject**: Return for major rework

**Approval logged to**: `/mnt/code/.context/approval_log.json`

## Success Criteria

- Project plan with 7 epochs (002-008) detailed
- Success criteria defined for all epochs
- Resource estimates provided with totals
- Risk assessment with at least 3-5 key risks
- Executive summary generated
- **User approval obtained**

## Artifacts Location

```
/mnt/code/epoch001-research-analysis-planning/
├── notebooks/
├── scripts/
├── requirements.md                   # User-filled requirements (updated by BA)
├── project_plan.json                 # Comprehensive plan
├── success_criteria.json             # Definition of done per epoch
├── resource_estimates.json           # Time, compute, memory, cost
├── risk_assessment.json              # Risks with mitigations
├── executive_summary.md              # Stakeholder report
└── README.md
```

## Next Steps

After plan approval:
1. Review gate approval logged
2. Pipeline state updated with `epoch_001_complete: true` and `epoch_001_review_approved: true`
3. Proceed to **Epoch 002: Data Wrangling** based on approved plan
4. Data Wrangler validates prerequisites (Epoch 001 complete)

## Key Features

- **Project Coordination**: Business Analyst leads planning with all agents
- **Comprehensive Planning**: All epochs planned upfront with dependencies
- **Success Criteria**: Measurable targets for each stage
- **Resource Awareness**: Realistic estimates from agent consultation
- **Risk Management**: Proactive identification of blockers
- **Human Oversight**: Explicit plan approval required

---

**Ready to start?** Use the Business-Analyst-Agent with your use case to create the project plan.
