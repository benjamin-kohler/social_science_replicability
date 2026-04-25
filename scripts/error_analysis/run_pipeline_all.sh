#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd ../.. && pwd)}"

# Optional: activate conda env if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-replicability}" 2>/dev/null || true
fi

# Optional: load .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a && source "$PROJECT_ROOT/.env" && set +a
fi

PAPERS_DIR="${PAPERS_DIR:-$PROJECT_ROOT/data/i4replicate/papers}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/data/i4replicate/results}"
WS_DIR="${WS_DIR:-./explainer_workspaces}"

# Step 00 — prep workspaces (skips existing)
echo "========================================"
echo "STEP 00 — prep workspaces"
echo "========================================"
python3 00_prep_setup.py \
  --papers-dir "$PAPERS_DIR" \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$WS_DIR" \
  --agents claude-code,codex

# Step 01 — trace failures
echo ""
echo "========================================"
echo "STEP 01 — failure tracing"
echo "========================================"
for paper_dir in "$WS_DIR"/*/; do
  paper=$(basename "$paper_dir")
  for agent_dir in "$paper_dir"/*/; do
    agent_label=$(basename "$agent_dir")
    code_dir="$agent_dir/code"
    div_file="$code_dir/divergences.json"

    [[ -f "$div_file" ]] && continue
    [[ ! -d "$code_dir" ]] && echo "SKIP $paper/$agent_label: no code dir" && continue

    # Extract model prefix and approach suffix from agent_label
    # e.g. "claude-opus-4-6_claude-code" -> prefix="claude-opus-4-6" suffix="claude-code"
    # e.g. "gpt-5.3-codex_codex" -> prefix="gpt-5.3-codex" suffix="codex"
    approach_suffix="${agent_label##*_}"
    model_prefix="${agent_label%%_${approach_suffix}}"

    # Find matching VR in results dir
    # Results dirs look like: claude-opus-4-6_10.1257_aer.20210290_claude-code
    vr_path=""
    for rd in "$RESULTS_DIR/$paper"/*; do
      [[ ! -d "$rd" ]] && continue
      rdname=$(basename "$rd")
      [[ "$rdname" == "summaries" ]] && continue
      # Skip archived
      case "$rdname" in *.failed*|*.bak*|*.old*|*.gap*|*.f_*|*.notprod*|*.newdata*|*.datafix*|*.earlystop*) continue ;; esac
      # Match: starts with model_prefix and ends with approach_suffix
      if [[ "$rdname" == "${model_prefix}_"*"_${approach_suffix}" ]]; then
        if [[ -f "$rd/verification_report.json" ]]; then
          vr_path="$rd/verification_report.json"
          break
        fi
      fi
    done

    if [[ -z "$vr_path" ]]; then
      echo "SKIP $paper/$agent_label: no VR"
      continue
    fi

    runner="codex"
    [[ "$agent_label" == *claude* ]] && runner="claude-code"

    echo ""
    echo ">> $paper / $agent_label (step 01)"
    python3 01_trace_failures.py \
      --code-dir "$code_dir" \
      --verification-report "$vr_path" \
      --runner "$runner" \
      --timeout 900 \
      --max-turns 20 || echo "  FAILED: $paper/$agent_label step 01"
  done
done

# Step 02 — error source
echo ""
echo "========================================"
echo "STEP 02 — error source detection"
echo "========================================"
for paper_dir in "$WS_DIR"/*/; do
  paper=$(basename "$paper_dir")
  for agent_dir in "$paper_dir"/*/; do
    agent_label=$(basename "$agent_dir")
    ws="$agent_dir/error_source"
    div_file="$agent_dir/code/divergences.json"
    enriched="$ws/divergences_enriched.json"

    [[ ! -f "$div_file" ]] && continue
    [[ -f "$enriched" ]] && continue

    runner="codex"
    [[ "$agent_label" == *claude* ]] && runner="claude-code"

    echo ""
    echo ">> $paper / $agent_label (step 02)"
    python3 02_detect_error_source.py \
      --comparison "$div_file" \
      --workspace  "$ws" \
      --output     "$enriched" \
      --runner     "$runner" \
      --timeout 600 \
      --max-turns 15 || echo "  FAILED: $paper/$agent_label step 02"
  done
done

# Step 03
echo ""
echo "========================================"
echo "STEP 03 — summary tables"
echo "========================================"
python3 03_summarize_errors.py --output-dir summaries/ || true

# Step 04
echo ""
echo "========================================"
echo "STEP 04 — plots"
echo "========================================"
python3 04_overview_stats.py --output-dir plots/ || true

echo ""
echo "Done."
