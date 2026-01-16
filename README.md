# Code Debugging with RAG + CoT

Automated bug fixing on SWE-bench using RAG (Retrieval-Augmented Generation) and Chain-of-Thought localization.

## Goal

Generate unified diff patches that fix software bugs automatically. The system localizes the buggy file, retrieves relevant context, and generates patches verified against test suites.

## Dataset

**SWE-bench Lite** - Curated subset of real GitHub issues and PRs.

| Split | Instances | Usage |
|-------|-----------|-------|
| Dev | 23 | Development and tuning |
| Test | 300 | Final evaluation |

## Approaches

| Pipeline | Localization | Best For |
|----------|--------------|----------|
| **Agentic** | Multi-strategy (95.5%) | Best accuracy |
| **CoT** | LLM reasoning (87%) | Simple setup |
| **RAG** | Heuristics (~30%) | Clear stacktraces |
| **Baseline** | None | Quick tests |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API key
echo "OPENROUTER_API_KEY=your_key" > .env

# Run Agentic pipeline (recommended)
PYTHONPATH=. python experiments/run_agentic.py --n 5

# Run CoT pipeline
PYTHONPATH=. python experiments/run_cot.py --n 5
```

## Usage

### Run Pipelines

```bash
# Agentic (best accuracy)
PYTHONPATH=. python experiments/run_agentic.py \
    --n 10 \
    --experiment-name my_run

# CoT
PYTHONPATH=. python experiments/run_cot.py \
    --n 10 \
    --experiment-name cot_run

# RAG
PYTHONPATH=. python experiments/run_rag.py \
    --n 10 \
    --experiment-name rag_run

# Baseline
PYTHONPATH=. python experiments/run_baseline.py \
    --n 10 \
    --experiment-name baseline_run
```

### Check Localization Accuracy

```bash
# Compare predicted files vs ground truth
PYTHONPATH=. python tests/swe_lite_lsr_checker.py --approach agentic
PYTHONPATH=. python tests/swe_lite_lsr_checker.py --approach cot
```

### Run SWE-bench Evaluation

```bash
# Convert results to SWE-bench format
python src/verification/convert_results.py results/my_run.json predictions.json

# Run official harness
python -m swebench.harness.run_evaluation \
    --predictions_path predictions.json \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --run_id my_eval \
    --timeout 900
```

### Run Single Instance

```bash
PYTHONPATH=. python experiments/run_agentic.py \
    --instance-id sqlfluff__sqlfluff-1625
```

## File Structure

```
code_debuging_with_rag_cot/
|
+-- experiments/              # Entry points
|   +-- run_agentic.py        # Agentic pipeline runner
|   +-- run_cot.py            # CoT pipeline runner
|   +-- run_rag.py            # RAG pipeline runner
|   +-- run_baseline.py       # Baseline runner
|
+-- src/
|   +-- pipelines/            # Core pipeline implementations
|   |   +-- agentic.py        # Multi-strategy + verification loop
|   |   +-- cot.py            # Chain-of-thought localization
|   |   +-- rag.py            # RAG-based pipeline
|   |   +-- baseline.py       # Simple prompt-only
|   |
|   +-- retrieval/            # Context retrieval
|   |   +-- indexer.py        # HybridRetriever (FAISS + BM25)
|   |   +-- graph.py          # Code dependency graph
|   |   +-- example_retriever.py  # Similar solved bugs
|   |   +-- source_code.py    # Repository cloning
|   |
|   +-- llm/
|   |   +-- provider.py       # OpenRouter API with fallback
|   |
|   +-- verification/
|   |   +-- harness.py        # SWE-bench test runner
|   |   +-- convert_results.py
|   |
|   +-- data/
|   |   +-- swe_bench.py      # Dataset loading
|   |
|   +-- utils/
|       +-- fuzzy_patch.py    # Patch strictification
|
+-- tests/
|   +-- swe_lite_lsr_checker.py   # Localization accuracy
|   +-- check_localization.py     # Result analysis
|
+-- results/                  # Output JSON files
+-- verification_results/     # SWE-bench predictions
+-- repo_cache/               # Cloned repositories
+-- cache/                    # LLM response cache
|
+-- results.md                # Experiment results
+-- architecture.md           # Pipeline diagrams
+-- CLAUDE.md                 # Claude Code instructions
```

## Results

See [results.md](results.md) for detailed experiment results.

| Metric | Value |
|--------|-------|
| Localization (Agentic) | 95.5% (21/23) |
| Localization (CoT) | 87% (20/23) |
| Verified Patches | 4/16 (25%) |

## Architecture

See [architecture.md](architecture.md) for pipeline flow diagrams.

## Configuration

Create `.env`:
```
OPENROUTER_API_KEY=your_key_here
```
