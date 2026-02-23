"""Result models and aggregation for benchmarking."""

import csv
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..models.schemas import (
    ExplanationReport,
    ReplicationResults,
    VerificationReport,
)
from .config import ModelSpec, PaperSpec


class RunArtifacts(BaseModel):
    """Artifacts produced by a single benchmark run."""

    workspace_dir: str = Field(..., description="Path to the run's workspace directory")
    stdout: str = Field(default="", description="Captured stdout")
    stderr: str = Field(default="", description="Captured stderr")
    exit_code: int = Field(default=0, description="Process exit code")
    duration_seconds: float = Field(default=0.0, description="Wall-clock duration")
    replication_results: Optional[ReplicationResults] = Field(
        default=None, description="Parsed replication results (direct for structured, parsed for freestyle)"
    )


class EvaluationResult(BaseModel):
    """Result of evaluating a run with the shared judge."""

    verification_report: VerificationReport = Field(
        ..., description="Verification grades from the judge"
    )
    explanation_report: Optional[ExplanationReport] = Field(
        default=None, description="Explanation of discrepancies"
    )
    overall_grade: str = Field(..., description="Overall grade (A-F)")
    item_grades: dict[str, str] = Field(
        default_factory=dict, description="Per-item grades: item_id -> grade"
    )


class SingleRunResult(BaseModel):
    """Result of a single benchmark run (one model × one paper × one approach)."""

    model: ModelSpec = Field(..., description="Model used")
    paper: PaperSpec = Field(..., description="Paper replicated")
    approach: str = Field(..., description="Approach: 'freestyle' or 'structured'")
    artifacts: RunArtifacts = Field(..., description="Run artifacts")
    evaluation: EvaluationResult = Field(..., description="Judge evaluation")
    duration_seconds: float = Field(default=0.0, description="Total duration including evaluation")


class BenchmarkResults(BaseModel):
    """Aggregated results across all benchmark runs."""

    runs: list[SingleRunResult] = Field(default_factory=list, description="All individual runs")
    summary: dict = Field(default_factory=dict, description="Summary statistics")


class ResultsAggregator:
    """Saves and aggregates benchmark results."""

    @staticmethod
    def save_run(run: SingleRunResult, output_dir: Path) -> None:
        """Save a single run result to JSON."""
        run_dir = output_dir / f"{run.model.model_name}_{run.paper.paper_id}_{run.approach}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "result.json", "w") as f:
            json.dump(run.model_dump(), f, indent=2, default=str)

    @staticmethod
    def save_summary(results: BenchmarkResults, output_dir: Path) -> None:
        """Generate summary CSV and JSON from all runs."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build summary dict
        summary = {}
        for run in results.runs:
            key = f"{run.model.model_name}|{run.paper.paper_id}|{run.approach}"
            summary[key] = {
                "model": run.model.model_name,
                "provider": run.model.provider,
                "paper": run.paper.paper_id,
                "approach": run.approach,
                "overall_grade": run.evaluation.overall_grade,
                "item_grades": run.evaluation.item_grades,
                "duration_seconds": run.duration_seconds,
            }
        results.summary = summary

        # Save JSON
        with open(output_dir / "summary.json", "w") as f:
            json.dump(results.model_dump(), f, indent=2, default=str)

        # Save CSV
        if results.runs:
            csv_path = output_dir / "summary.csv"
            fieldnames = [
                "model", "provider", "paper", "approach",
                "overall_grade", "duration_seconds",
            ]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for run in results.runs:
                    writer.writerow({
                        "model": run.model.model_name,
                        "provider": run.model.provider,
                        "paper": run.paper.paper_id,
                        "approach": run.approach,
                        "overall_grade": run.evaluation.overall_grade,
                        "duration_seconds": round(run.duration_seconds, 1),
                    })
