"""
Agent Context Manager

Shared utility for cross-agent communication and state management.
All agents use this module to read/write pipeline context.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import threading


class PipelineContext:
    """
    Manages shared context across all agents in the ML pipeline.
    Provides thread-safe read/write operations for agent communication.
    """

    _lock = threading.Lock()
    _context_dir = Path("/mnt/code/.context")

    def __init__(self, project_name: str = "default"):
        """
        Initialize pipeline context for a specific project.

        Args:
            project_name: Name of the ML project
        """
        self.project_name = project_name
        self.context_file = self._context_dir / f"{project_name}_pipeline_state.json"
        self._ensure_context_dir()
        self._initialize_context()

    def _ensure_context_dir(self):
        """Create context directory if it doesn't exist."""
        self._context_dir.mkdir(exist_ok=True)

    def _initialize_context(self):
        """Initialize context file if it doesn't exist."""
        if not self.context_file.exists():
            initial_context = {
                "project_name": self.project_name,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "pipeline_version": "1.0.0",
                "stages": {},
                "global_metadata": {},
                "errors": []
            }
            self._write_context(initial_context)

    def _read_context(self) -> Dict[str, Any]:
        """Read context from file with thread safety."""
        with self._lock:
            try:
                with open(self.context_file, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                # Return empty context if file doesn't exist or is corrupted
                return self._get_empty_context()

    def _write_context(self, context: Dict[str, Any]):
        """Write context to file with thread safety."""
        with self._lock:
            context["last_updated"] = datetime.now().isoformat()
            with open(self.context_file, 'w') as f:
                json.dump(context, f, indent=2, default=str)

    def _get_empty_context(self) -> Dict[str, Any]:
        """Return empty context structure."""
        return {
            "project_name": self.project_name,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "pipeline_version": "1.0.0",
            "stages": {},
            "global_metadata": {},
            "errors": []
        }

    def update_stage(self, stage_name: str, stage_data: Dict[str, Any]):
        """
        Update context for a specific pipeline stage.

        Args:
            stage_name: Name of the stage (e.g., 'data_wrangling', 'model_development')
            stage_data: Dictionary containing stage outputs and metadata
        """
        context = self._read_context()

        # Create stage entry if it doesn't exist
        if stage_name not in context["stages"]:
            context["stages"][stage_name] = {
                "started_at": datetime.now().isoformat(),
                "status": "in_progress"
            }

        # Update stage data
        context["stages"][stage_name].update(stage_data)
        context["stages"][stage_name]["updated_at"] = datetime.now().isoformat()

        self._write_context(context)

    def complete_stage(self, stage_name: str, outputs: Dict[str, Any]):
        """
        Mark a stage as completed with its outputs.

        Args:
            stage_name: Name of the completed stage
            outputs: Dictionary containing stage outputs
        """
        stage_data = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "outputs": outputs
        }
        self.update_stage(stage_name, stage_data)

    def fail_stage(self, stage_name: str, error: str):
        """
        Mark a stage as failed with error information.

        Args:
            stage_name: Name of the failed stage
            error: Error message
        """
        context = self._read_context()

        stage_data = {
            "status": "failed",
            "failed_at": datetime.now().isoformat(),
            "error": error
        }
        self.update_stage(stage_name, stage_data)

        # Add to global error log
        context["errors"].append({
            "stage": stage_name,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        self._write_context(context)

    def get_stage_output(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """
        Get outputs from a completed stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Stage outputs if available, None otherwise
        """
        context = self._read_context()
        stage = context["stages"].get(stage_name, {})
        return stage.get("outputs")

    def get_stage_status(self, stage_name: str) -> str:
        """
        Get current status of a stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Stage status ('not_started', 'in_progress', 'completed', 'failed')
        """
        context = self._read_context()
        stage = context["stages"].get(stage_name, {})
        return stage.get("status", "not_started")

    def set_global_metadata(self, key: str, value: Any):
        """
        Set global metadata accessible by all agents.

        Args:
            key: Metadata key
            value: Metadata value
        """
        context = self._read_context()
        context["global_metadata"][key] = value
        self._write_context(context)

    def get_global_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get global metadata.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        context = self._read_context()
        return context["global_metadata"].get(key, default)

    def get_all_stage_outputs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get outputs from all completed stages.

        Returns:
            Dictionary mapping stage names to their outputs
        """
        context = self._read_context()
        outputs = {}
        for stage_name, stage_data in context["stages"].items():
            if stage_data.get("status") == "completed" and "outputs" in stage_data:
                outputs[stage_name] = stage_data["outputs"]
        return outputs

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the entire pipeline state.

        Returns:
            Dictionary containing pipeline summary
        """
        context = self._read_context()

        stages_summary = {}
        for stage_name, stage_data in context["stages"].items():
            stages_summary[stage_name] = {
                "status": stage_data.get("status", "unknown"),
                "started_at": stage_data.get("started_at"),
                "completed_at": stage_data.get("completed_at"),
                "has_outputs": "outputs" in stage_data
            }

        return {
            "project_name": context["project_name"],
            "created_at": context["created_at"],
            "last_updated": context["last_updated"],
            "total_stages": len(context["stages"]),
            "completed_stages": sum(1 for s in context["stages"].values() if s.get("status") == "completed"),
            "failed_stages": sum(1 for s in context["stages"].values() if s.get("status") == "failed"),
            "stages": stages_summary,
            "total_errors": len(context.get("errors", []))
        }

    def clear_context(self):
        """Clear all context data (use with caution)."""
        self._initialize_context()

    def export_context(self, output_path: str):
        """
        Export context to a file.

        Args:
            output_path: Path to export context to
        """
        context = self._read_context()
        with open(output_path, 'w') as f:
            json.dump(context, f, indent=2, default=str)


# Convenience functions for agents
def get_context(project_name: str = "default") -> PipelineContext:
    """
    Get or create pipeline context for a project.

    Args:
        project_name: Name of the project

    Returns:
        PipelineContext instance
    """
    return PipelineContext(project_name)


def quick_update(project_name: str, stage_name: str, outputs: Dict[str, Any]):
    """
    Quick function to update stage outputs.

    Args:
        project_name: Name of the project
        stage_name: Name of the stage
        outputs: Stage outputs
    """
    context = get_context(project_name)
    context.complete_stage(stage_name, outputs)


def quick_read(project_name: str, stage_name: str) -> Optional[Dict[str, Any]]:
    """
    Quick function to read stage outputs.

    Args:
        project_name: Name of the project
        stage_name: Name of the stage

    Returns:
        Stage outputs or None
    """
    context = get_context(project_name)
    return context.get_stage_output(stage_name)