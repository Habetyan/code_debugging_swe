# Code Debugging with RAG 

### 1. Run the RAG Agent
To attempt to fix a specific bug (e.g., `mwaskom__seaborn-3010`):
```bash
python experiments/run_rag.py --instance-id mwaskom__seaborn-3010
```

To run on a random subset of 5 instances:
```bash
python experiments/run_rag.py --num-instances 5 --experiment-name my_test_run
```

### 2. Analyze Results
After the run completes, use the `run_id` (printed at the end) to inspect the details:
```bash
python analysis/analyze.py --run-id my_test_run
```
