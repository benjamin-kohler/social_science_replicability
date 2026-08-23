"""Tests for exact production-sample pinning in the paper analysis."""

import json

import pandas as pd
import pytest

from scripts.analyze_i4rep_results import validate_sample_manifest


def _write_manifest(path):
    path.write_text(json.dumps({
        "paper_ids": ["p1", "p2"],
        "approach_model_combinations": ["codex/m1", "opencode/m2"],
        "paper_specific_included_approach_model_combinations": {
            "p2": ["codex/m1"],
        },
        "included_run_count": 3,
    }))


def test_manifest_accepts_exact_included_run_set(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    runs = pd.DataFrame({
        "paper_slug": ["p1", "p1", "p2"],
        "approach": ["codex/m1", "opencode/m2", "codex/m1"],
    })

    validate_sample_manifest(runs, manifest_path)


def test_manifest_rejects_unexpected_run(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    runs = pd.DataFrame({
        "paper_slug": ["p1", "p1", "p2", "p2"],
        "approach": ["codex/m1", "opencode/m2", "codex/m1", "opencode/m2"],
    })

    with pytest.raises(ValueError, match="unexpected"):
        validate_sample_manifest(runs, manifest_path)


def test_manifest_rejects_missing_run(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    runs = pd.DataFrame({
        "paper_slug": ["p1", "p2"],
        "approach": ["codex/m1", "codex/m1"],
    })

    with pytest.raises(ValueError, match="missing"):
        validate_sample_manifest(runs, manifest_path)


def test_manifest_rejects_duplicate_run(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    runs = pd.DataFrame({
        "paper_slug": ["p1", "p1", "p1", "p2"],
        "approach": ["codex/m1", "codex/m1", "opencode/m2", "codex/m1"],
    })

    with pytest.raises(ValueError, match="Duplicate"):
        validate_sample_manifest(runs, manifest_path)
