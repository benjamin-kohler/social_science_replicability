#!/usr/bin/env bash
#
# run_postcutoff_batch.sh — Run the replicability benchmark on post-cutoff papers.
#
# Usage:
#   # Inside tmux on textlab:
#   tmux new -s postcutoff
#   bash scripts/run_postcutoff_batch.sh
#
#   # Override defaults via env vars:
#   PARALLEL=3 APPROACHES="claude-code codex" bash scripts/run_postcutoff_batch.sh
#   PAPERS="209827 211322" bash scripts/run_postcutoff_batch.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT="/data/individual/benjamin/social_science_replicability"
PAPERS_DIR="$PROJECT_ROOT/data/postcutoff/papers"
RESULTS_DIR="$PROJECT_ROOT/data/postcutoff/results"
CONFIG_DIR="$PROJECT_ROOT/config"
LOG_FILE="$PROJECT_ROOT/data/postcutoff/batch_run_$(date '+%Y%m%d_%H%M%S').log"

APPROACHES="${APPROACHES:-claude-code codex swe-agent opencode}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-7200}"
PAPER_FILTER="${PAPERS:-}"
PARALLEL="${PARALLEL:-5}"

# Model assignments
MODEL_CLAUDE_CODE="claude-opus-4-6"
MODEL_CODEX="gpt-5.3-codex"
MODEL_SWE_AGENT="gpt-5.2-codex"
MODEL_OPENCODE="gpt-5.2-codex"

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
get_model_for_approach() {
    local approach="$1"
    case "$approach" in
        claude-code) echo "$MODEL_CLAUDE_CODE" ;;
        codex)       echo "$MODEL_CODEX" ;;
        swe-agent)   echo "$MODEL_SWE_AGENT" ;;
        opencode)    echo "$MODEL_OPENCODE" ;;
        *)           echo ""; return 1 ;;
    esac
}

get_provider_for_approach() {
    local approach="$1"
    case "$approach" in
        claude-code) echo "anthropic" ;;
        *)           echo "openai" ;;
    esac
}

get_api_key_env_for_approach() {
    local approach="$1"
    case "$approach" in
        claude-code) echo "ANTHROPIC_API_KEY" ;;
        *)           echo "OPENAI_API_KEY" ;;
    esac
}

# ---------------------------------------------------------------------------
# Generate a per-paper per-approach benchmark config YAML
# ---------------------------------------------------------------------------
generate_config() {
    local paper_slug="$1"
    local approach="$2"
    local paper_dir="$PAPERS_DIR/$paper_slug"
    local config_file="$CONFIG_DIR/postcutoff_${paper_slug}_${approach}.yaml"

    local provider model_name api_key_env
    provider=$(get_provider_for_approach "$approach")
    model_name=$(get_model_for_approach "$approach")
    api_key_env=$(get_api_key_env_for_approach "$approach")

    cat > "$config_file" <<YAML
## Auto-generated config for postcutoff batch: ${paper_slug} / ${approach}

models:
  - provider: ${provider}
    model_name: ${model_name}
    api_key_env: ${api_key_env}
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

extractor:
  model: ${EXTRACTOR_MODEL}
  use_vision: true

run_explainer: true
explainer_runner_type: codex
explainer_model:
  provider: openai
  model_name: gpt-5.3-codex
  api_key_env: OPENAI_API_KEY

output_dir: ${RESULTS_DIR}/${paper_slug}
timeout_seconds: ${TIMEOUT_SECONDS}
allow_web_access: false
YAML

    echo "$config_file"
}

# ---------------------------------------------------------------------------
# Check if a run already has results
# ---------------------------------------------------------------------------
has_results() {
    local paper_slug="$1"
    local approach="$2"
    local results_paper_dir="$RESULTS_DIR/$paper_slug"

    if [ -d "$results_paper_dir" ]; then
        local found
        found=$(find "$results_paper_dir" -path "*${approach}*" -name "verification_report.json" 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            return 0
        fi
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log_separator
    log "Post-cutoff batch benchmark"
    log "  Project:    $PROJECT_ROOT"
    log "  Papers:     $PAPERS_DIR"
    log "  Results:    $RESULTS_DIR"
    log "  Approaches: $APPROACHES"
    log "  Models:     claude-code=$MODEL_CLAUDE_CODE codex=$MODEL_CODEX swe-agent=$MODEL_SWE_AGENT opencode=$MODEL_OPENCODE"
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
        data_count=$(find "$paper_dir/data" -type f 2>/dev/null | wc -l | tr -d ' ')
        if [ "$data_count" -eq 0 ]; then
            log "  SKIP $paper_slug: no data files"
            ((skipped_no_data++)) || true
            continue
        fi

        eligible_papers+=("$paper_slug")
    done

    local n_eligible=${#eligible_papers[@]}
    local n_approaches
    n_approaches=$(echo $APPROACHES | wc -w | tr -d ' ')
    log ""
    log "Eligible: $n_eligible papers (skipped: $skipped_no_pdf no PDF, $skipped_no_data no data)"
    log "Total runs planned: $((n_eligible * n_approaches)) ($n_eligible papers x $n_approaches approaches)"
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

        for approach in $APPROACHES; do
            if has_results "$paper_slug" "$approach"; then
                log "  SKIP $paper_slug/$approach: results exist"
                ((paper_skipped++)) || true
                continue
            fi

            local config_file
            config_file=$(generate_config "$paper_slug" "$approach")
            if [ $? -ne 0 ]; then
                log "  ERROR $paper_slug/$approach: config generation failed"
                ((paper_failed++)) || true
                continue
            fi

            log "  RUN  $paper_slug/$approach at $(date '+%H:%M:%S')..."

            local start_time
            start_time=$(date +%s)

            cd "$PROJECT_ROOT"
            if timeout "$TIMEOUT_SECONDS" python -m src.benchmark_cli \
                --config "$config_file" \
                --approaches "$approach" \
                --papers "$paper_slug" \
                --timeout "$TIMEOUT_SECONDS" \
                >> "$paper_log" 2>&1; then
                local duration=$(( $(date +%s) - start_time ))
                log "  DONE $paper_slug/$approach in ${duration}s"
                ((paper_completed++)) || true
            else
                local exit_code=$?
                local duration=$(( $(date +%s) - start_time ))
                if [ $exit_code -eq 124 ]; then
                    log "  TIMEOUT $paper_slug/$approach after ${duration}s"
                else
                    log "  FAILED $paper_slug/$approach (exit $exit_code) after ${duration}s"
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
