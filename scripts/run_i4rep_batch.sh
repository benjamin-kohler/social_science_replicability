#!/usr/bin/env bash
#
# run_i4rep_batch.sh — Run the replicability benchmark on all i4replication papers.
#
# Runs 3 approaches (claude-code, codex, structured) sequentially on each paper,
# with a 30-minute timeout per approach. Skips papers that already have results.
#
# Usage:
#   # Inside tmux on textlab:
#   tmux new -s i4rep
#   bash scripts/run_i4rep_batch.sh
#
#   # Resume after interruption (will skip completed runs):
#   bash scripts/run_i4rep_batch.sh
#
#   # Only run specific approaches:
#   APPROACHES="claude-code" bash scripts/run_i4rep_batch.sh
#
#   # Only run specific papers:
#   PAPERS="10.1111_ajps.12599 10.1086_714765" bash scripts/run_i4rep_batch.sh
#
#   # Run up to 3 papers in parallel:
#   PARALLEL=3 bash scripts/run_i4rep_batch.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PAPERS_DIR="$PROJECT_ROOT/data/i4replicate/papers"
RESULTS_DIR="$PROJECT_ROOT/data/i4replicate/results"
CONFIG_DIR="$PROJECT_ROOT/config"
LOG_FILE="$PROJECT_ROOT/data/i4replicate/batch_run.log"

# Approaches to run (override with APPROACHES env var)
APPROACHES="${APPROACHES:-claude-code codex swe-agent}"

# Timeout per approach in seconds (2 hours)
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-7200}"

# Filter to specific papers (space-separated DOI slugs, empty = all)
PAPER_FILTER="${PAPERS:-}"

# Number of papers to run in parallel (1 = sequential, default)
PARALLEL="${PARALLEL:-1}"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Source conda if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
fi

# Activate project environment if it exists
if command -v conda &>/dev/null; then
    conda activate replicability 2>/dev/null || conda activate base 2>/dev/null || true
fi

# Load .env for API keys
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Ensure output directories exist
mkdir -p "$RESULTS_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

log_separator() {
    log "========================================================================"
}

# ---------------------------------------------------------------------------
# Generate a per-paper benchmark config YAML
# ---------------------------------------------------------------------------
generate_config() {
    local paper_slug="$1"
    local approach="$2"
    local paper_dir="$PAPERS_DIR/$paper_slug"
    local config_file="$CONFIG_DIR/i4rep_${paper_slug}_${approach}.yaml"

    # Determine model based on approach
    local provider model_name api_key_env
    case "$approach" in
        claude-code)
            provider="anthropic"
            model_name="claude-opus-4-6"
            api_key_env="ANTHROPIC_API_KEY"
            ;;
        codex)
            provider="openai"
            model_name="gpt-5.3-codex"
            api_key_env="OPENAI_API_KEY"
            ;;
        swe-agent)
            provider="openai"
            model_name="gpt-5.3-codex"
            api_key_env="OPENAI_API_KEY"
            ;;
        structured)
            provider="openai"
            model_name="gpt-5.2-codex"
            api_key_env="OPENAI_API_KEY"
            ;;
        freestyle)
            provider="openai"
            model_name="gpt-5.2-codex"
            api_key_env="OPENAI_API_KEY"
            ;;
        *)
            log "ERROR: Unknown approach: $approach"
            return 1
            ;;
    esac

    cat > "$config_file" <<YAML
## Auto-generated config for i4rep batch: ${paper_slug} / ${approach}

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
  model_name: gpt-5-mini
  use_vision: true

extractor:
  model: gpt-5.2
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

    # Check for any result directory matching this approach
    if [ -d "$results_paper_dir" ]; then
        # Look for verification_report.json in any subdirectory matching the approach
        local found
        found=$(find "$results_paper_dir" -path "*${approach}*" -name "verification_report.json" 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            return 0  # has results
        fi
    fi
    return 1  # no results
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
main() {
    log_separator
    log "Starting i4replication batch benchmark"
    log "  Project root: $PROJECT_ROOT"
    log "  Papers dir:   $PAPERS_DIR"
    log "  Results dir:  $RESULTS_DIR"
    log "  Approaches:   $APPROACHES"
    log "  Timeout:      ${TIMEOUT_SECONDS}s per approach"
    log_separator

    # Verify papers directory exists
    if [ ! -d "$PAPERS_DIR" ]; then
        log "ERROR: Papers directory not found: $PAPERS_DIR"
        log "Run setup_i4rep_batch.py first to create paper directories."
        exit 1
    fi

    # Collect papers to process
    local papers=()
    if [ -n "$PAPER_FILTER" ]; then
        for p in $PAPER_FILTER; do
            if [ -d "$PAPERS_DIR/$p" ]; then
                papers+=("$p")
            else
                log "WARNING: Paper directory not found: $PAPERS_DIR/$p"
            fi
        done
    else
        for paper_dir in "$PAPERS_DIR"/*/; do
            [ -d "$paper_dir" ] || continue
            papers+=("$(basename "$paper_dir")")
        done
    fi

    local total_papers=${#papers[@]}
    log "Found $total_papers paper directories"

    # Filter to papers that have both paper.pdf and data/
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

        # Check data dir has at least one file
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
    log ""
    log "Eligible papers: $n_eligible (skipped: $skipped_no_pdf no PDF, $skipped_no_data no data)"
    log ""

    if [ "$n_eligible" -eq 0 ]; then
        log "No eligible papers to process. Exiting."
        exit 0
    fi

    # Count total runs
    local total_runs=0
    local skipped_runs=0
    local completed_runs=0
    local failed_runs=0

    for approach in $APPROACHES; do
        for paper_slug in "${eligible_papers[@]}"; do
            ((total_runs++)) || true
        done
    done
    log "Total runs planned: $total_runs ($n_eligible papers x $(echo $APPROACHES | wc -w | tr -d ' ') approaches)"
    log_separator

    # ---------------------------------------------------------------------------
    # Worker function: run all approaches for one paper
    # ---------------------------------------------------------------------------
    run_paper() {
        local paper_slug="$1"
        local paper_log="$RESULTS_DIR/$paper_slug/run.log"
        mkdir -p "$RESULTS_DIR/$paper_slug"

        log "START $paper_slug"

        # Read metadata if available
        local meta_file="$PAPERS_DIR/$paper_slug/metadata.json"
        if [ -f "$meta_file" ]; then
            local title
            title=$(python3 -c "import json; print(json.load(open('$meta_file'))['title'][:80])" 2>/dev/null || echo "unknown")
            log "  Title: $title"
        fi

        local paper_completed=0
        local paper_skipped=0
        local paper_failed=0

        for approach in $APPROACHES; do
            log "  $paper_slug / $approach"

            # Check for existing results
            if has_results "$paper_slug" "$approach"; then
                log "  SKIP $paper_slug/$approach: results already exist"
                ((paper_skipped++)) || true
                continue
            fi

            # Generate config
            local config_file
            config_file=$(generate_config "$paper_slug" "$approach")
            if [ $? -ne 0 ]; then
                log "  ERROR $paper_slug/$approach: failed to generate config"
                ((paper_failed++)) || true
                continue
            fi

            log "  Starting $paper_slug/$approach at $(date '+%H:%M:%S') with ${TIMEOUT_SECONDS}s timeout..."

            # Run benchmark
            local start_time
            start_time=$(date +%s)

            cd "$PROJECT_ROOT"
            if timeout "$TIMEOUT_SECONDS" python -m src.benchmark_cli \
                --config "$config_file" \
                --approaches "$approach" \
                --papers "$paper_slug" \
                --timeout "$TIMEOUT_SECONDS" \
                >> "$paper_log" 2>&1; then
                local end_time
                end_time=$(date +%s)
                local duration=$((end_time - start_time))
                log "  DONE $paper_slug/$approach in ${duration}s"
                ((paper_completed++)) || true
            else
                local exit_code=$?
                local end_time
                end_time=$(date +%s)
                local duration=$((end_time - start_time))
                if [ $exit_code -eq 124 ]; then
                    log "  TIMEOUT $paper_slug/$approach after ${duration}s"
                else
                    log "  FAILED $paper_slug/$approach (exit $exit_code) after ${duration}s"
                fi
                ((paper_failed++)) || true
            fi

            # Clean up config
            rm -f "$config_file"
        done

        log "FINISH $paper_slug: completed=$paper_completed skipped=$paper_skipped failed=$paper_failed"
        # Write status to a file so the parent can tally results
        echo "$paper_completed $paper_skipped $paper_failed" > "$RESULTS_DIR/$paper_slug/.batch_status"
    }

    # ---------------------------------------------------------------------------
    # Run papers (parallel or sequential)
    # ---------------------------------------------------------------------------
    if [ "$PARALLEL" -le 1 ]; then
        # Sequential mode
        for paper_slug in "${eligible_papers[@]}"; do
            run_paper "$paper_slug"
        done
    else
        log "Running with PARALLEL=$PARALLEL"
        local running=0
        local pids=()

        for paper_slug in "${eligible_papers[@]}"; do
            # Wait if we've hit the parallelism limit
            while [ "$running" -ge "$PARALLEL" ]; do
                # Wait for any child to finish
                wait -n 2>/dev/null || true
                # Recount running jobs
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
            log "  Launched $paper_slug (pid ${pids[-1]}, $running/$PARALLEL slots used)"
        done

        # Wait for all remaining jobs
        log "Waiting for remaining $running jobs to finish..."
        wait
    fi

    # ---------------------------------------------------------------------------
    # Tally results from status files
    # ---------------------------------------------------------------------------
    local completed_runs=0
    local skipped_runs=0
    local failed_runs=0
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

    # Final summary
    log_separator
    log "BATCH COMPLETE"
    log "  Total runs:     $total_runs"
    log "  Completed:      $completed_runs"
    log "  Skipped:        $skipped_runs"
    log "  Failed/timeout: $failed_runs"
    log_separator
}

main "$@"
