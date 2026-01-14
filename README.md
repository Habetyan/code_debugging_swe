# Code Debugging with RAG + CoT

A RAG (Retrieval-Augmented Generation) pipeline with Chain-of-Thought (CoT) localization for automated bug fixing on the SWE-bench Lite benchmark.

## Overview

This project uses an LLM to generate patches that fix software bugs. It combines:
1. **CoT Localization**: Accurately identifying the file to edit.
2. **RAG**: Retrieving relevant documentation and similar solved bugs.

**Pipeline Steps:**
1. Load a bug report from SWE-bench Lite
2. Clone the repository at the correct commit
3. **Localize File**: Use CoT to find the primary file to edit.
4. Retrieve relevant documentation and examples.
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

The main command uses `run_cot.py` for the best performance.

### Run on a single bug instance
```bash
PYTHONPATH=. python experiments/run_cot.py --instance-id mwaskom__seaborn-3010
```

### Run on multiple instances
```bash
PYTHONPATH=. python experiments/run_cot.py --n 5 --experiment-name my_run
```

### Run on specific instances (comma-separated)
```bash
PYTHONPATH=. python experiments/run_cot.py --instance-id django__django-11099,scikit-learn__scikit-learn-10297
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--instance-id` | Specific instance ID(s) to run | None |
| `--n`, `-n` | Number of random instances | 10 |
| `--model`, `-m` | LLM model to use | `deepseek/deepseek-chat` |
| `--experiment-name`, `-e` | Name for output file | `cot_run` |
| `--dataset` | SWE-bench dataset variant | `lite` |

### Available SWE-bench Datasets
```bash
# SWE-bench Lite (300 instances, curated for quality)
PYTHONPATH=. python experiments/run_cot.py --dataset lite --n 10

# SWE-bench Dev (smaller dev set for quick testing)
PYTHONPATH=. python experiments/run_cot.py --dataset dev --n 5
```

**Dataset options:** `lite` (default, 300 instances) | `dev` (smaller test set)

---

## 4. Analyze Results

To check localization accuracy (CoT vs Ground Truth):
```bash
python analysis/check_localization.py results/my_run.json
```

---

## 5. Official SWE-bench Evaluation

To verify if patches actually fix the bugs:

### Convert results to SWE-bench format
```bash
python analysis/convert_results.py results/my_run.json predictions.json
```

### Run official evaluation
```bash
python -m swebench.harness.run_evaluation \
    --predictions_path predictions.json \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --run_id my_run_eval \
    --timeout 900
```

---

## Project Structure

```
code_debuging_with_rag_cot/
├── experiments/
│   ├── run_cot.py          # Main CoT+RAG experiment runner (Use this!)
│   ├── run_rag.py          # Baseline RAG runner
│   └── run_baseline.py     # Simple baseline 
├── src/
│   ├── data/               # SWE-bench data loading
│   ├── llm/                # LLM provider abstraction
│   ├── pipelines/
│   │   ├── cot.py          # Chain-of-Thought pipeline (New)
│   │   ├── baseline.py     # Simple prompt-only pipeline
│   │   └── rag.py          # Standard RAG pipeline
│   ├── retrieval/
│   │   ├── corpus.py       # Document storage
│   │   ├── indexer.py      # Hybrid retrieval (FAISS + BM25)
│   │   ├── graph.py        # Code dependency graph
│   │   └── source_code.py  # Repository management
│   │   └── example_retriever.py  # Example retriever
│   ├── evaluation/         # Metrics and experiment runner
│   └── utils/              # Fuzzy patching utilities
├── analysis/               # Analysis scripts
│   ├── check_localization.py # Localization accuracy checker
│   ├── convert_results.py    # Convert to SWE-bench format
└── results/                # Output JSON files
```
