# Code Debugging with RAG

A RAG (Retrieval-Augmented Generation) pipeline for automated bug fixing on the SWE-bench Lite benchmark.

## Overview

This project uses an LLM to generate patches that fix software bugs. It retrieves relevant documentation and code context from the target repository to help the LLM produce accurate fixes.

**Pipeline Steps:**
1. Load a bug report from SWE-bench Lite
2. Clone the repository at the correct commit
3. Identify the file that needs to be edited (using stacktraces, module paths, or grep)
4. Retrieve relevant documentation from the repo
5. Send the context + bug report to the LLM
6. Generate a unified diff patch
7. Validate syntax and convert to strict git format

---

## 1. Installation

```bash
# Clone the repository
git clone <https://github.com/Habetyan/code_debugging_swe.git>
cd code_debuging_with_rag_cot

# Install dependencies
pip install -r requirements.txt

# Additional dependencies for RAG
pip install sentence-transformers faiss-cpu rank-bm25
```

## 2. Configuration

Create a `.env` file with your API key:
```bash
OPENROUTER_API_KEY=your_key_here
```

---

## 3. Running the Pipeline

### Run on a single bug instance
```bash
python experiments/run_rag.py --instance-id mwaskom__seaborn-3010
```

### Run on multiple instances
```bash
python experiments/run_rag.py --num-instances 5 --experiment-name my_run
```

### Run on specific instances (comma-separated)
```bash
python experiments/run_rag.py --instance-id django__django-11099,scikit-learn__scikit-learn-10297
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--instance-id` | Specific instance ID(s) to run | None |
| `--num-instances`, `-n` | Number of random instances | 1 |
| `--model`, `-m` | LLM model to use | `deepseek/deepseek-chat` |
| `--experiment-name`, `-e` | Name for output file | `rag_best` |

---

## 4. Analyze Results

After running, check the generated patches:
```bash
python analysis/analyze.py --run-id my_run
```

This shows statistics about patch generation success rates.

---
### TO-DO
## 5. Official SWE-bench to-do

To verify if patches actually fix the bugs we will test on the official SWE-bench later.

### Install SWE-bench
```bash
pip install swe-bench
```

### Convert results to SWE-bench format
```bash
python analysis/convert_results.py --input results/my_run.json --output predictions.json
```

### Run official evaluation
```bash
python -m swebench.harness.run_evaluation \
    --predictions_path predictions.json \
    --swe_bench_tasks princeton-nlp/SWE-bench_Lite \
    --log_dir ./eval_logs \
    --testbed ./testbed \
    --skip_existing \
    --timeout 9000 \
    --verbose
```

This will:
- Clone each repository
- Apply your generated patch
- Run the test suite
- Report pass/fail for each instance

---

## Project Structure

```
code_debuging_with_rag_cot/
├── experiments/
│   ├── run_rag.py          # Main script to run the RAG pipeline
│   └── run_baseline.py     # Baseline without RAG 
├── src/
│   ├── data/               # SWE-bench data loading
│   ├── llm/                # LLM provider abstraction
│   ├── pipelines/
│   │   ├── baseline.py     # Simple prompt-only pipeline
│   │   └── rag.py          # Full RAG pipeline with retrieval
│   ├── retrieval/
│   │   ├── corpus.py       # Document storage
│   │   ├── indexer.py      # Hybrid retrieval (FAISS + BM25)
│   │   ├── graph.py        # Code dependency graph
│   │   └── source_code.py  # Repository management
│   │   └── example_retriever.py  # Example retriever
│   ├── evaluation/         # Metrics and experiment runner
│   └── utils/              # Fuzzy patching utilities
├── analysis/               # Result analysis scripts
├── results/                # Output JSON files
└── repo_cache/             # Cloned repositories (auto-created)
```
