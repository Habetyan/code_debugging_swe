# Experiment Results

## Dataset

**SWE-bench Lite Dev Split**: 23 instances from various Python repositories:
- sqlfluff/sqlfluff
- marshmallow-code/marshmallow
- pylint-dev/astroid
- pvlib/pvlib-python
- pydicom/pydicom
- pyvista/pyvista

---

## Localization Results

**Model:** `meta-llama/llama-3.1-8b-instruct`

| Approach | Accuracy | Correct | Wrong | Time/Instance | LLM Calls |
|----------|----------|---------|-------|---------------|-----------|
| **Agentic (Multi-Heuristic)** | **100%** | 23 | 0 | 4.5s | 178 |
| CoT (Chain-of-Thought) | 56.5% | 13 | 10 | 5.3s | 23 |
| RAG (Embedding Only) | 52.2% | 12 | 11 | 5.6s | 0 |

---

## Patch Generation Results

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

**Total improvement: 43.5% → 95.7% (+52.2%)**

### Key Changes That Worked

1. **Stronger prompts** with explicit "NEVER delete" rules
2. **Validation checks** for control flow and parameter removal
3. **Larger RAG truncation** (400→5000 chars for problems, 2000→10000 for patches)

Only 1 instance (`pydicom__pydicom-1413`) failed - the LLM couldn't generate a fix at all.

---

## Verified Patches (Passed Official SWE-bench Tests)

| Instance ID | Repository | File Modified | Fix Description |
|-------------|------------|---------------|-----------------|
| marshmallow-code__marshmallow-1343 | marshmallow-code/marshmallow | src/marshmallow/schema.py | Fixed null data handling in schema validation |
| marshmallow-code__marshmallow-1359 | marshmallow-code/marshmallow | src/marshmallow/fields.py | Fixed DateTime field binding to nested schemas |
| sqlfluff__sqlfluff-1625 | sqlfluff/sqlfluff | src/sqlfluff/rules/L031.py | Fixed L031 rule for single-table queries |
| sqlfluff__sqlfluff-2419 | sqlfluff/sqlfluff | src/sqlfluff/rules/L060.py | Replace IFNULL/NVL with COALESCE |

---

## All Instances Status

| Instance ID | Localized | Patch | Verified |
|-------------|-----------|-------|----------|
| marshmallow-code__marshmallow-1343 | ✓ | ✓ | ✓ |
| marshmallow-code__marshmallow-1359 | ✓ | ✓ | ✓ |
| sqlfluff__sqlfluff-1625 | ✓ | ✓ | ✓ |
| sqlfluff__sqlfluff-2419 | ✓ | ✓ | ✓ |
| sqlfluff__sqlfluff-1733 | ✓ | ✓ | ✗ |
| pydicom__pydicom-1139 | ✓ | ✓ | ✗ |
| sqlfluff__sqlfluff-1517 | ✓ | ✓ | ✗ |
| sqlfluff__sqlfluff-1763 | ✓ | ✓ | ✗ |
| pylint-dev__astroid-1333 | ✓ | ✓ | ✗ |
| pylint-dev__astroid-1196 | ✓ | ✓ | ✗ |
| pylint-dev__astroid-1268 | ✓ | ✓ | ✗ |
| pylint-dev__astroid-1866 | ✓ | ✓ | ✗ |
| pylint-dev__astroid-1978 | ✓ | ✓ | ✗ |
| pvlib__pvlib-python-1606 | ✓ | ✓ | ✗ |
| pvlib__pvlib-python-1707 | ✓ | ✓ | ✗ |
| pvlib__pvlib-python-1854 | ✓ | ✓ | ✗ |
| pvlib__pvlib-python-1154 | ✓ | ✓ | ✗ |
| pvlib__pvlib-python-1072 | ✓ | ✓ | ✗ |
| pydicom__pydicom-901 | ✓ | ✓ | ✗ |
| pydicom__pydicom-1256 | ✓ | ✓ | ✗ |
| pydicom__pydicom-1694 | ✓ | ✓ | ✗ |
| pyvista__pyvista-4315 | ✓ | ✓ | ✗ |
| pydicom__pydicom-1413 | ✓ | ✗ | ✗ |
