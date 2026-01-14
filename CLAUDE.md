# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) + Chain-of-Thought (CoT) pipeline for automated bug fixing on the SWE-bench benchmark. The system takes bug reports, localizes the relevant file, retrieves documentation and similar solved examples, then generates unified diff patches.

## Commands

### Run experiments
```bash
# Run on a single instance
PYTHONPATH=. python experiments/run_cot.py --instance-id mwaskom__seaborn-3010

# Run on multiple instances
PYTHONPATH=. python experiments/run_cot.py --n 5 --experiment-name my_run

# Run with different model
PYTHONPATH=. python experiments/run_cot.py --model deepseek/deepseek-chat --n 10
```

### Official SWE-bench evaluation
```bash
# Convert results to SWE-bench format
python analysis/convert_results.py results/my_run.json predictions.json

# Run evaluation
python -m swebench.harness.run_evaluation \
    --predictions_path predictions.json \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --run_id my_run_eval
```

## Architecture

### Pipeline Flow
1. **Data Loading** (`src/data/swe_bench.py`): Loads bug instances from HuggingFace datasets (SWE-bench_Lite, SWE-bench_Verified)
2. **Repository Cloning** (`src/retrieval/source_code.py`): RepoManager clones repos at specific commits to `repo_cache/`
3. **File Localization** (`src/pipelines/cot.py`): CoT prompting identifies the primary file to edit
4. **Context Retrieval** (`src/retrieval/`): HybridRetriever (FAISS + BM25) fetches relevant docs and ExampleRetriever finds similar solved bugs
5. **Patch Generation**: LLM generates SEARCH/REPLACE blocks
6. **Verification** (`src/verification/harness.py`): Runs tests to verify patches, with a rectification loop on failure
7. **Patch Strictification** (`src/utils/fuzzy_patch.py`): Converts fuzzy patches to strict git diffs

### Key Components

**Pipelines** (`src/pipelines/`):
- `CoTPipeline`: Main pipeline with CoT localization + verification loop (recommended)
- `RAGPipeline`: Multi-strategy file finding (stacktrace, module paths, grep)
- `BaselinePipeline`: Simple prompt-only baseline

**Retrieval** (`src/retrieval/`):
- `HybridRetriever`: Combines semantic (FAISS) and lexical (BM25) search with cross-encoder reranking
- `ExampleRetriever`: Retrieves similar solved bugs from SWE-bench training set
- `CodeGraph`: Builds import dependency graphs for context expansion

**LLM** (`src/llm/provider.py`): OpenRouter API client with disk caching, multiple API key fallback, and model fallback (deepseek → qwen → llama)

### Data Classes
- `SWEBenchInstance`: Bug report with `instance_id`, `repo`, `base_commit`, `problem_statement`, `patch`
- `PipelineResult`: Output with `generated_patch`, `ground_truth_patch`, `success`
- `Document`: Retrieval document with `content`, `title`, `library`, `source`

## Configuration

Create `.env` with:
```
OPENROUTER_API_KEY=your_key_here
```

Results are saved to `results/` as JSON. Cloned repositories are cached in `repo_cache/`.