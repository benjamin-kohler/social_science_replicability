#!/usr/bin/env bash
#
# run_i4rep_batch.sh — Run the replicability benchmark on all i4replication papers.
#
# Usage:
#   # Inside tmux on textlab:
#   tmux new -s i4rep2
#   bash scripts/run_i4rep_batch.sh
#
#   # Override defaults via env vars:
#   PARALLEL=3 APPROACHES="claude-code codex" bash scripts/run_i4rep_batch.sh
#   PAPERS="10.1111_ajps.12599 10.1086_714765" bash scripts/run_i4rep_batch.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — hardcoded to the correct project path on textlab
# ---------------------------------------------------------------------------
PROJECT_ROOT="/data/individual/benjamin/social_science_replicability"
PAPERS_DIR="${PAPERS_DIR:-$PROJECT_ROOT/data/i4replicate/papers}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/data/i4replicate/results}"
CONFIG_DIR="$PROJECT_ROOT/config"
LOG_FILE="${RESULTS_DIR}/batch_run_$(date '+%Y%m%d_%H%M%S').log"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-7200}"
PAPER_FILTER="${PAPERS:-}"
PARALLEL="${PARALLEL:-5}"
ITEM_TYPES="${ITEM_TYPES:-table}"

# Run configurations: "approach:model:provider:api_key_env"
# Each entry is one run per paper. Multiple entries for the same approach
# with different models produce separate runs.
DEFAULT_RUNS="claude-code:claude-opus-4-6:anthropic:ANTHROPIC_API_KEY codex:gpt-5.4:openai:OPENAI_API_KEY swe-agent:gpt-5.4:openai:OPENAI_API_KEY swe-agent:z-ai/glm-5:openrouter:OPENROUTER_API_KEY opencode:gpt-5.4:openai:OPENAI_API_KEY opencode:z-ai/glm-5:openrouter:OPENROUTER_API_KEY"
RUNS="${RUNS:-$DEFAULT_RUNS}"

JUDGE_MODEL="gpt-5-mini"
EXTRACTOR_MODEL="gpt-5-mini"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

if command -v conda &>/dev/null; then
    conda activate replicability 2>/dev/null || true
    # Ensure conda's libstdc++ is found before the system one (needed for sqlite3/ICU)
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [ -d "$HOME/.opencode/bin" ]; then
    export PATH="$HOME/.opencode/bin:$PATH"
fi

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

mkdir -p "$RESULTS_DIR" "$CONFIG_DIR"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

log_separator() {
    log "========================================================================"
}

# ---------------------------------------------------------------------------
# Model lookup for each approach
# ---------------------------------------------------------------------------
# Parse a run spec "approach:model:provider:api_key_env" into variables
parse_run_spec() {
    local spec="$1"
    RUN_APPROACH=$(echo "$spec" | cut -d: -f1)
    RUN_MODEL=$(echo "$spec" | cut -d: -f2)
    RUN_PROVIDER=$(echo "$spec" | cut -d: -f3)
    RUN_API_KEY_ENV=$(echo "$spec" | cut -d: -f4)
}

# Build a unique run ID (used for directory names and deduplication)
run_id() {
    local spec="$1"
    parse_run_spec "$spec"
    local safe_model
    safe_model=$(echo "$RUN_MODEL" | tr '/' '_')
    echo "${safe_model}_${RUN_APPROACH}"
}

# ---------------------------------------------------------------------------
# Generate a per-paper per-approach benchmark config YAML
# ---------------------------------------------------------------------------
generate_config() {
    local paper_slug="$1"
    local run_spec="$2"
    parse_run_spec "$run_spec"
    local approach="$RUN_APPROACH"
    local model_name="$RUN_MODEL"
    local provider="$RUN_PROVIDER"
    local api_key_env="$RUN_API_KEY_ENV"
    local paper_dir="$PAPERS_DIR/$paper_slug"
    local rid
    rid=$(run_id "$run_spec")
    local config_file="$CONFIG_DIR/i4rep_${paper_slug}_${rid}.yaml"

    local api_base_url_line=""
    if [ "$provider" = "openrouter" ]; then
        api_base_url_line="    api_base_url: https://openrouter.ai/api/v1"
    fi

    cat > "$config_file" <<YAML
## Auto-generated config: ${paper_slug} / ${model_name} / ${approach}

models:
  - provider: ${provider}
    model_name: ${model_name}
    api_key_env: ${api_key_env}
${api_base_url_line}
    allow_web_access: false
    approaches:
      - ${approach}

papers:
  - paper_id: "${paper_slug}"
    pdf_path: ${paper_dir}/paper.pdf
    data_path: ${paper_dir}/data
    replication_package_path: ${paper_dir}/replication_package

approaches:
  - ${approach}

judge:
  provider: openai
  model_name: ${JUDGE_MODEL}
  use_vision: true
  comparator_cli_tool: claude-code
  comparator_model: claude-sonnet-4-6
  comparator_timeout: 600

extractor:
  model: ${EXTRACTOR_MODEL}
  use_vision: true

run_explainer: true
explainer_runner_type: codex
explainer_timeout_seconds: 900
explainer_max_turns: 500

explainer_model:
  provider: openai
  model_name: gpt-5.4
  api_key_env: OPENAI_API_KEY

output_dir: ${RESULTS_DIR}/${paper_slug}
opencode_binary: $HOME/.opencode/bin/opencode
claude_code_binary: claude
codex_binary: codex
timeout_seconds: ${TIMEOUT_SECONDS}
allow_web_access: false

item_types:
  - table
YAML

    echo "$config_file"
}

# ---------------------------------------------------------------------------
# Check if a run already has results
# ---------------------------------------------------------------------------
has_results() {
    local paper_slug="$1"
    local run_spec="$2"
    local rid
    rid=$(run_id "$run_spec")
    local results_paper_dir="$RESULTS_DIR/$paper_slug"

    if [ -f "$results_paper_dir/${rid}/verification_report.json" ]; then
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log_separator
    log "i4rep batch benchmark"
    log "  Project:    $PROJECT_ROOT"
    log "  Papers:     $PAPERS_DIR"
    log "  Results:    $RESULTS_DIR"
    log "  Runs:       $(echo $RUNS | wc -w | tr -d ' ') configurations per paper"
    for run_spec in $RUNS; do
        parse_run_spec "$run_spec"
        log "    $RUN_APPROACH ($RUN_MODEL via $RUN_PROVIDER)"
    done
    log "  Judge:      $JUDGE_MODEL"
    log "  Extractor:  $EXTRACTOR_MODEL"
    log "  Timeout:    ${TIMEOUT_SECONDS}s"
    log "  Parallel:   $PARALLEL"
    log_separator

    if [ ! -d "$PAPERS_DIR" ]; then
        log "ERROR: Papers directory not found: $PAPERS_DIR"
        exit 1
    fi

    # Collect papers
    local papers=()
    if [ -n "$PAPER_FILTER" ]; then
        for p in $PAPER_FILTER; do
            if [ -d "$PAPERS_DIR/$p" ]; then
                papers+=("$p")
            else
                log "WARNING: Not found: $PAPERS_DIR/$p"
            fi
        done
    else
        for paper_dir in "$PAPERS_DIR"/*/; do
            [ -d "$paper_dir" ] || continue
            papers+=("$(basename "$paper_dir")")
        done
    fi

    log "Found ${#papers[@]} paper directories"

    # Filter to eligible (has PDF + data)
    local eligible_papers=()
    local skipped_no_pdf=0
    local skipped_no_data=0

    for paper_slug in "${papers[@]}"; do
        local paper_dir="$PAPERS_DIR/$paper_slug"

        if [ ! -f "$paper_dir/paper.pdf" ]; then
            log "  SKIP $paper_slug: no paper.pdf"
            ((skipped_no_pdf++)) || true
            continue
        fi

        local data_count
        data_count=$(find "$paper_dir/data" \( -type f -o -type l \) 2>/dev/null | wc -l | tr -d ' ')
        if [ "$data_count" -eq 0 ]; then
            log "  SKIP $paper_slug: no data files"
            ((skipped_no_data++)) || true
            continue
        fi

        eligible_papers+=("$paper_slug")
    done

    local n_eligible=${#eligible_papers[@]}
    local n_runs_per_paper
    n_runs_per_paper=$(echo $RUNS | wc -w | tr -d ' ')
    log ""
    log "Eligible: $n_eligible papers (skipped: $skipped_no_pdf no PDF, $skipped_no_data no data)"
    log "Total runs planned: $((n_eligible * n_runs_per_paper)) ($n_eligible papers x $n_runs_per_paper runs)"
    log_separator

    if [ "$n_eligible" -eq 0 ]; then
        log "No eligible papers. Exiting."
        exit 0
    fi

    # Worker: run all approaches for one paper
    run_paper() {
        local paper_slug="$1"
        local paper_log="$RESULTS_DIR/$paper_slug/run.log"
        mkdir -p "$RESULTS_DIR/$paper_slug"

        log "START $paper_slug"

        local meta_file="$PAPERS_DIR/$paper_slug/metadata.json"
        if [ -f "$meta_file" ]; then
            local title
            title=$(python -c "import json; print(json.load(open('$meta_file'))['title'][:80])" 2>/dev/null || echo "unknown")
            log "  Title: $title"
        fi

        local paper_completed=0
        local paper_skipped=0
        local paper_failed=0

        for run_spec in $RUNS; do
            local rid
            rid=$(run_id "$run_spec")
            parse_run_spec "$run_spec"
            local approach="$RUN_APPROACH"

            if has_results "$paper_slug" "$run_spec"; then
                log "  SKIP $paper_slug/$rid: results exist"
                ((paper_skipped++)) || true
                continue
            fi

            local config_file
            config_file=$(generate_config "$paper_slug" "$run_spec")
            if [ $? -ne 0 ]; then
                log "  ERROR $paper_slug/$rid: config generation failed"
                ((paper_failed++)) || true
                continue
            fi

            log "  RUN  $paper_slug/$rid at $(date '+%H:%M:%S')..."

            local start_time
            start_time=$(date +%s)

            cd "$PROJECT_ROOT"
            if timeout "$TIMEOUT_SECONDS" python -m src.benchmark_cli \
                --config "$config_file" \
                --approaches "$approach" \
                --papers "$paper_slug" \
                --timeout "$TIMEOUT_SECONDS" \
                --item-types $ITEM_TYPES \
                >> "$paper_log" 2>&1; then
                local duration=$(( $(date +%s) - start_time ))
                log "  DONE $paper_slug/$rid in ${duration}s"
                ((paper_completed++)) || true
            else
                local exit_code=$?
                local duration=$(( $(date +%s) - start_time ))
                if [ $exit_code -eq 124 ]; then
                    log "  TIMEOUT $paper_slug/$rid after ${duration}s"
                else
                    log "  FAILED $paper_slug/$rid (exit $exit_code) after ${duration}s"
                fi
                ((paper_failed++)) || true
            fi

            rm -f "$config_file"
        done

        log "FINISH $paper_slug: done=$paper_completed skip=$paper_skipped fail=$paper_failed"
        echo "$paper_completed $paper_skipped $paper_failed" > "$RESULTS_DIR/$paper_slug/.batch_status"
    }

    # Run papers
    if [ "$PARALLEL" -le 1 ]; then
        for paper_slug in "${eligible_papers[@]}"; do
            run_paper "$paper_slug"
        done
    else
        log "Running with PARALLEL=$PARALLEL"
        local running=0
        local pids=()

        for paper_slug in "${eligible_papers[@]}"; do
            while [ "$running" -ge "$PARALLEL" ]; do
                wait -n 2>/dev/null || true
                running=0
                for pid in "${pids[@]}"; do
                    if kill -0 "$pid" 2>/dev/null; then
                        ((running++)) || true
                    fi
                done
            done

            run_paper "$paper_slug" &
            pids+=($!)
            ((running++)) || true
            log "  Launched $paper_slug (pid ${pids[-1]}, $running/$PARALLEL)"
        done

        log "Waiting for remaining $running jobs..."
        wait
    fi

    # Tally
    local completed_runs=0 skipped_runs=0 failed_runs=0
    for paper_slug in "${eligible_papers[@]}"; do
        local status_file="$RESULTS_DIR/$paper_slug/.batch_status"
        if [ -f "$status_file" ]; then
            read -r c s f < "$status_file"
            completed_runs=$((completed_runs + c))
            skipped_runs=$((skipped_runs + s))
            failed_runs=$((failed_runs + f))
            rm -f "$status_file"
        fi
    done

    log_separator
    log "BATCH COMPLETE"
    log "  Completed:      $completed_runs"
    log "  Skipped:        $skipped_runs"
    log "  Failed/timeout: $failed_runs"
    log_separator
}

main "$@"
