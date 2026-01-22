# Code Debugging with RAG + CoT

Automated bug fixing on SWE-bench using RAG (Retrieval-Augmented Generation) and Chain-of-Thought localization.

## Overview

This system generates unified diff patches to fix software bugs automatically. It localizes the buggy file using multiple strategies, retrieves relevant context, and generates patches that can be verified against test suites.

## Dataset

**SWE-bench Lite** - Curated subset of real GitHub issues and PRs.

| Split | Instances | Usage |
|-------|-----------|-------|
| Dev | 23 | Development and tuning |
| Test | 300 | Final evaluation |

## Pipelines

| Pipeline | Localization | Description |
|----------|--------------|-------------|
| **Agentic** | Multi-strategy (95.5%) | Best accuracy - uses heuristics, graph analysis, and LLM |
| **CoT** | LLM reasoning (87%) | Chain-of-thought file localization |
| **RAG** | Heuristics (~52%) | Stacktrace parsing and symbol search |
| **Baseline** | None | Direct LLM prompt without localization |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
echo "OPENROUTER_API_KEY=your_key" > .env

# Run Agentic pipeline (recommended)
PYTHONPATH=. python experiments/run_agentic.py --n 5

# Run with self-critique (extra LLM validation)
PYTHONPATH=. python experiments/run_agentic.py --n 5 --self-critique
```

## Usage

### Agentic Pipeline (Recommended)

```bash
PYTHONPATH=. python experiments/run_agentic.py \
    --n 10 \
    --model deepseek/deepseek-chat \
    --temperature 0.1 \
    --experiment-name my_run \
    --self-critique  # Optional: enable LLM patch review
```

**Options:**
- `--n`: Number of instances to run
- `--model`: LLM model (default: deepseek/deepseek-chat)
- `--temperature`: Sampling temperature (default: 0.1)
- `--dataset`: Dataset to use (lite/dev)
- `--split`: Dataset split
- `--self-critique`: Enable LLM self-review of generated patches
- `--instance-id`: Run specific instance(s)

### CoT Pipeline

```bash
PYTHONPATH=. python experiments/run_cot.py \
    --n 10 \
    --experiment-name cot_run
```

### RAG Pipeline

```bash
PYTHONPATH=. python experiments/run_rag.py \
    --n 10 \
    --experiment-name rag_run
```

### Baseline Pipeline

```bash
PYTHONPATH=. python experiments/run_baseline.py \
    --n 10 \
    --experiment-name baseline_run
```

### Check Localization Accuracy

```bash
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

## Project Structure

```
code_debuging_with_rag_cot/
├── experiments/              # Entry points
│   ├── run_agentic.py        # Agentic pipeline runner
│   ├── run_cot.py            # CoT pipeline runner
│   ├── run_rag.py            # RAG pipeline runner
│   └── run_baseline.py       # Baseline runner
│
├── src/
│   ├── pipelines/            # Core pipeline implementations
│   │   ├── agentic.py        # Multi-strategy localization + SEARCH/REPLACE
│   │   ├── cot.py            # Chain-of-thought localization
│   │   ├── rag.py            # RAG-based pipeline
│   │   └── baseline.py       # Simple prompt-only
│   │
│   ├── retrieval/            # Context retrieval
│   │   ├── indexer.py        # HybridRetriever (FAISS + BM25)
│   │   ├── graph.py          # Code dependency graph
│   │   ├── example_retriever.py  # Similar solved bugs
│   │   ├── repo_retriever.py # Repository file embedding
│   │   └── source_code.py    # Repository cloning
│   │
│   ├── validation/           # Patch validation
│   │   └── pass1_validator.py # Static checks + LLM self-critique
│   │
│   ├── llm/
│   │   └── provider.py       # OpenRouter API with fallback
│   │
│   ├── verification/
│   │   ├── harness.py        # SWE-bench test runner
│   │   └── convert_results.py # Results format converter
│   │
│   ├── data/
│   │   └── swe_bench.py      # Dataset loading
│   │
│   └── utils/
│       └── fuzzy_patch.py    # Patch application utilities
│
├── tests/
│   ├── swe_lite_lsr_checker.py   # Localization accuracy checker
│   └── check_localization.py     # Result analysis
│
├── results/                  # Output JSON files
├── repo_cache/               # Cloned repositories
└── cache/                    # LLM response cache
```

## Results

| Metric | Value |
|--------|-------|
| Localization (Agentic) | 95.5% (21/22) |
| Localization (CoT) | 87% (20/23) |
| Localization (Baseline) | 82.6% (19/23) |

## Validation Checks

The pass@1 validator performs these static checks (no test execution):

1. **Diff quality** - Size, deletions, keyword relevance
2. **Syntax validation** - Python AST parsing
3. **Balanced brackets** - Parentheses, brackets, braces
4. **Mass deletion detection** - Prevents accidental code removal
5. **Structure preservation** - Functions/classes not deleted
6. **Debug code detection** - No print, pdb, breakpoint
7. **Defensive coding** - No bare except, getattr(..., None)
8. **Indentation consistency** - No mixed tabs/spaces
9. **Import preservation** - Critical imports not removed
10. **LLM self-critique** - Optional code review (--self-critique)

## Configuration

Create `.env` file:
```
OPENROUTER_API_KEY=your_key_here
OPENROUTER_API_KEY1=backup_key_1  # Optional fallback
OPENROUTER_API_KEY2=backup_key_2  # Optional fallback
```

## Models

Default model: `deepseek/deepseek-chat`

Fallback models (automatic):
- `qwen/qwen-2.5-coder-32b-instruct`
- `deepseek/deepseek-coder`
- `meta-llama/llama-3.1-70b-instruct`
