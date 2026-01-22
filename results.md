# Experiment Results

## Dataset

**SWE-bench Lite Dev Split**: 23 instances from various Python repositories including:
- sqlfluff/sqlfluff
- marshmallow-code/marshmallow
- pylint-dev/astroid
- pvlib/pvlib-python
- pydicom/pydicom
- pyvista/pyvista

---

## Localization Accuracy

File localization determines which source file needs to be modified to fix the bug.

**Model:** `meta-llama/llama-3.1-8b-instruct`

| Approach | Accuracy | Correct | Wrong | Time/Instance | LLM Calls |
|----------|----------|---------|-------|---------------|-----------|
| **Agentic (Multi-Heuristic)** | **100%** | 23 | 0 | 4.5s | 138 |
| CoT (Chain-of-Thought) | 56.5% | 13 | 10 | 5.3s | 23 |
| RAG (Embedding Only) | 52.2% | 12 | 11 | 5.6s | 0 |

### Agentic Localization Strategies
1. Stacktrace file extraction
2. Error message grep search
3. Test file stem matching (test_X.py -> X.py)
4. Rule/class identifier search
5. Explicit file path mentions
6. Module path conversion
7. Code graph expansion (imports)
8. Keyword scoring with test names

## Patch Generation Results

### Verified Patches

| Instance ID | Repository | File Modified |
|-------------|------------|---------------|
| sqlfluff__sqlfluff-1625 | sqlfluff/sqlfluff | src/sqlfluff/rules/L031.py |
| sqlfluff__sqlfluff-2419 | sqlfluff/sqlfluff | src/sqlfluff/rules/L060.py |
| marshmallow-code__marshmallow-1359 | marshmallow-code/marshmallow | src/marshmallow/fields.py |
| marshmallow-code__marshmallow-1343 | marshmallow-code/marshmallow | src/marshmallow/schema.py |

---

## Patch Generation Results

Pass@1 Validation here is not swe offical benchmark tool!

**Models Used:**
- Localization: `meta-llama/llama-3.1-8b-instruct`
- Patch Generation: `deepseek/deepseek-chat`

| Metric | Count | Rate |
|--------|-------|------|
| Total Instances | 23 | - |
| Patches Generated | 22 | 95.7% |
| Pass@1 Validation | 21 | 91.3% |
| **Harness Verified** | **4** | **17.4%** |

---

## Improvement History

### Pass@1 Validation Improvement

| Experiment | Pass@1 Validation | Change |
|------------|-------------------|--------|
| `agentic_lite_dev.json` (original) | 10/23 (43.5%) | baseline |
| `agentic_lite_dev_v2.json` (prompt fixes) | 21/23 (91.3%) | +47.8% |
| `agentic_dev_v3.json` (larger RAG limits) | 22/23 (95.7%) | +4.4% |

