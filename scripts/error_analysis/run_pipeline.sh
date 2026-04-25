#!/usr/bin/env bash
# run_pipeline.sh — rerun steps 01-04 for all papers and agents
#
# Usage:
#   bash run_pipeline.sh                                          # run all steps
#   bash run_pipeline.sh --from 02                               # skip step 01, start from 02
#   bash run_pipeline.sh --anthropic-api-key sk-ant-... \
#                        --openai-api-key    sk-...              # pass API keys explicitly
#
# Steps:
#   01  failure tracing  (writes divergences.json per paper/agent)
#   02  error source     (writes divergences_enriched.json per paper/agent)
#   03  LaTeX tables     (auto-discovery, writes summaries/)
#   04  plots            (auto-discovery, writes plots/)

set -euo pipefail
cd "$(dirname "$0")"

FROM_STEP=1
ANTHROPIC_KEY=""
OPENAI_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)               FROM_STEP="$2"; shift 2 ;;
    --from=*)             FROM_STEP="${1#--from=}"; shift ;;
    --anthropic-api-key)  ANTHROPIC_KEY="$2"; shift 2 ;;
    --anthropic-api-key=*)ANTHROPIC_KEY="${1#--anthropic-api-key=}"; shift ;;
    --openai-api-key)     OPENAI_KEY="$2"; shift 2 ;;
    --openai-api-key=*)   OPENAI_KEY="${1#--openai-api-key=}"; shift ;;
    [0-9]*)               FROM_STEP="$1"; shift ;;
    *)                    shift ;;
  esac
done

# Export API keys if provided so all subprocesses inherit them
[[ -n "$ANTHROPIC_KEY" ]] && export ANTHROPIC_API_KEY="$ANTHROPIC_KEY"
[[ -n "$OPENAI_KEY"    ]] && export OPENAI_API_KEY="$OPENAI_KEY"

RESULTS_ROOT="../examples/results"

PAPERS=(
  "10.1257_aer.20190565"
  "10.1257_aer.20210290"
  "10.1257_pol.20180594"
)

AGENTS=("claude" "codex")

agent_result_dir() {   # $1=agent $2=paper
  case "$1" in
    claude) echo "claude-opus-4-6_${2}_claude-code" ;;
    codex)  echo "gpt-5.3-codex_${2}_codex" ;;
  esac
}

agent_runner() {       # $1=agent
  case "$1" in
    claude) echo "claude-code" ;;
    codex)  echo "codex" ;;
  esac
}

# ── Step 01 ─────────────────────────────────────────────────────────────────
if [[ "$FROM_STEP" -le 1 ]]; then
  echo "========================================"
  echo "STEP 01 — failure tracing"
  echo "========================================"
  for paper in "${PAPERS[@]}"; do
    for agent in "${AGENTS[@]}"; do
      code_dir="explainer_workspaces/${paper}/${agent}/code"
      vr_path="${RESULTS_ROOT}/${paper}/$(agent_result_dir "$agent" "$paper")/verification_report.json"
      echo ""
      echo ">> ${paper} / ${agent}"
      python3 01_trace_failures.py \
        --code-dir            "${code_dir}" \
        --verification-report "${vr_path}" \
        --runner "$(agent_runner "$agent")" \
        --rerun
    done
  done
fi

# ── Step 02 ─────────────────────────────────────────────────────────────────
if [[ "$FROM_STEP" -le 2 ]]; then
  echo ""
  echo "========================================"
  echo "STEP 02 — error source detection"
  echo "========================================"
  for paper in "${PAPERS[@]}"; do
    for agent in "${AGENTS[@]}"; do
      ws="explainer_workspaces/${paper}/${agent}/error_source"
      echo ""
      echo ">> ${paper} / ${agent}"
      python3 02_detect_error_source.py \
        --comparison "${ws}/../code/divergences.json" \
        --workspace  "${ws}" \
        --output     "${ws}/divergences_enriched.json" \
        --runner     "$(agent_runner "$agent")" \
        --rerun
    done
  done
fi

# ── Step 03 ─────────────────────────────────────────────────────────────────
if [[ "$FROM_STEP" -le 3 ]]; then
  echo ""
  echo "========================================"
  echo "STEP 03 — LaTeX summary tables"
  echo "========================================"
  python3 03_summarize_errors.py --rerun
fi

# ── Step 04 ─────────────────────────────────────────────────────────────────
if [[ "$FROM_STEP" -le 4 ]]; then
  echo ""
  echo "========================================"
  echo "STEP 04 — plots"
  echo "========================================"
  python3 04_overview_stats.py --rerun
fi

echo ""
echo "Done."
