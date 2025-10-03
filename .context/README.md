# Agent Context Directory

This directory contains shared context files used for cross-agent communication during ML pipeline execution.

## Files

- **pipeline_state.json**: Shared state and metadata across all agents in the pipeline
- **{project}_context.json**: Project-specific context files

## Purpose

Agents use these files to:
- Share outputs from previous stages
- Pass metadata between pipeline stages
- Enable error recovery with previous successful states
- Maintain traceability throughout pipeline execution
- Coordinate handoffs between agents

## Structure

Each context file contains:
- Agent outputs and artifacts
- Data schemas and statistics
- Model information and metrics
- Stage completion status
- Timestamps and versioning
- Error states and recovery points

## Usage

Agents automatically read/write to these files during pipeline execution. No manual intervention required.