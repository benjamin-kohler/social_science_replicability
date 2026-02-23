# Opencode Freestyle vs. LangGraph Structured: A Hybrid Benchmark Strategy

## Overview

This document describes two complementary approaches for benchmarking LLM models on social science paper replication, and why a **hybrid** strategy combining both is more informative than either alone.

## Comparison Table

| Dimension                | Freestyle (Opencode)                       | Structured (LangGraph Pipeline)              |
|--------------------------|--------------------------------------------|----------------------------------------------|
| **What it measures**     | Raw model capability & autonomy            | Model performance within a guided workflow   |
| **Agent structure**      | Single agent, no hand-holding              | 4 specialized agents with defined roles      |
| **Prompt engineering**   | One task prompt; model decides the rest    | Carefully crafted per-agent system prompts   |
| **Error recovery**       | Model must self-correct                    | Pipeline stops on failure with clear errors  |
| **Reproducibility**      | Lower (model may take different paths)     | Higher (deterministic graph execution)       |
| **Code quality**         | Varies widely                              | More consistent (replicator agent patterns)  |
| **Evaluation**           | Shared judge (Verifier + Explainer)        | Same shared judge for fair comparison        |
| **Setup complexity**     | Minimal (just opencode binary)             | Requires full pipeline infrastructure        |
| **Cost per run**         | Single long context call                   | Multiple targeted calls (may be cheaper)     |
| **Best for**             | Comparing raw model intelligence           | Comparing models in production-like settings |

## When to Use Each Approach

### Use Freestyle When:
- Evaluating a model's end-to-end reasoning and coding ability
- Testing whether a model can handle ambiguous instructions
- Comparing models without introducing pipeline-specific advantages
- Quick prototyping before investing in structured pipelines

### Use Structured When:
- Evaluating how well a model follows specific instructions
- Testing performance in a production deployment scenario
- Isolating failures to specific pipeline stages
- Comparing models on individual capabilities (extraction vs. coding vs. verification)

## Hybrid Strategy Rationale

Running **both** approaches on the same set of papers with the same models, evaluated by the same judge, provides the most complete picture:

1. **Disentangle model capability from pipeline quality**: A model that scores well in freestyle but poorly in structured reveals pipeline design issues, not model limitations.

2. **Identify pipeline overhead vs. benefit**: If structured consistently outperforms freestyle, the pipeline adds genuine value. If not, the pipeline may be over-constraining capable models.

3. **Fair comparison via shared judge**: Using a fixed judge model (e.g., GPT-4o) for all evaluations ensures that grade differences reflect actual output quality, not evaluator variance.

4. **Cost-benefit analysis**: Comparing total token usage and wall-clock time across approaches helps optimize the production pipeline.

## Architecture

```
BenchmarkRunner (iterates models x papers x approaches)
  |
  +-- OpencodeRunner (freestyle)  --> artifacts --> ArtifactParser --> ReplicationResults
  |
  +-- StructuredRunner (pipeline) --> artifacts --> ReplicationResults (directly)
                                          |
                                    SharedEvaluator (judge model)
                                    (Verifier + Explainer agents)
                                          |
                                    ResultsAggregator --> JSON + CSV
```

## Interpreting Results

The benchmark produces a summary CSV with one row per (model, paper, approach) combination:

- **Grade A/B across both approaches**: Model is strong; pipeline adds reliability.
- **Grade A structured, C freestyle**: Pipeline compensates for model weakness in planning.
- **Grade A freestyle, C structured**: Pipeline may be constraining the model; investigate agent prompts.
- **Grade D/F across both**: Model is not capable enough for the task; consider a more powerful model.
