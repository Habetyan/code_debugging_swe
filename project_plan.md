# Project Plan: Chain-of-Thought Retrieval-Augmented Code Generation for Automated Program Repair

## Project Title
**CoT-RACG: Reasoning-Driven Retrieval for Automated Bug Fixing on SWE-bench**

---

## 1. Dataset & Benchmark Selection

### Primary Evaluation Dataset: SWE-bench-Lite
**Choice Rationale:**
- 300 carefully curated real-world bug instances from popular Python repositories
- Manageable size for thorough experimentation within project timeline
- Each instance includes: issue description, buggy code, test cases, and ground-truth fixes
- Execution-based evaluation (tests must pass) aligns with functional correctness requirements
- Represents realistic debugging scenarios developers face

**Initial Working Subset:**
- Start with 50 instances stratified by:
  - Repository diversity (Django, scikit-learn, matplotlib, requests, etc.)
  - Bug difficulty (estimated by test complexity)
  - Bug type (logic errors, API misuse, edge cases, type errors)
- Expand to full 300 instances for final evaluation

**Fallback Option:**
- If SWE-bench-Lite proves too challenging, use SWE-bench-verified (higher quality annotations)
- Can further filter to specific repositories or bug types

### Retrieval Corpus: CodeRAG-Bench + Domain Documentation

**CodeRAG-Bench Components:**
- Code snippets with documentation
- Programming examples across multiple languages (focus on Python)
- API usage patterns
- Common debugging patterns

**Supplementary Sources:**
- Official Python documentation (scraped and indexed)
- Library-specific documentation for SWE-bench repositories:
  - Django documentation
  - scikit-learn API reference
  - matplotlib examples
  - pandas documentation
  - requests library docs
- Stack Overflow high-quality Q&A pairs (filtered by votes/acceptance)
- GitHub issue-resolution pairs from similar repositories

**Corpus Size Target:**
- 15,000-25,000 documents total
- Indexed for both semantic and keyword search
- Organized by: language, library, problem type, code vs documentation

---

## 2. System Architecture Components

### Core Pipeline: Three Variants for Comparison

#### Variant 1: Baseline (No Retrieval)
**Purpose:** Establish LLM's inherent code repair capability

**Flow:**
1. Input: Bug report + error trace + code context
2. Direct prompt: "Fix this bug"
3. Generate patch
4. Execute and evaluate

**Metrics to Track:**
- Pass@1, Pass@3, Pass@5
- Syntax error rate
- Execution time
- Error categories

#### Variant 2: Direct RAG (Standard Retrieval)
**Purpose:** Measure basic retrieval benefit

**Flow:**
1. Input: Bug report + error trace
2. Embed entire bug description
3. Retrieve top-k similar documents (k=5,10)
4. Prompt: Bug + Retrieved Context → Generate fix
5. Execute and evaluate

**Retrieval Methods to Implement:**
- Embedding-based (CodeBERT or all-MiniLM-L6-v2)
- Keyword-based (BM25)
- Hybrid (weighted combination)

#### Variant 3: CoT-RAG (Your Innovation)
**Purpose:** Test reasoning-driven retrieval hypothesis

**Flow:**
1. **Diagnosis Phase:**
   - Prompt: "Analyze this bug trace step-by-step. What is the root cause?"
   - LLM generates structured diagnosis:
     - Error type classification
     - Affected components
     - Likely cause hypothesis
     - Required knowledge to fix

2. **Query Formulation Phase:**
   - Input: Diagnosis + Original bug
   - Prompt: "Based on this diagnosis, generate specific search queries for documentation/examples"
   - LLM generates 2-3 targeted queries
   - Examples:
     - "pandas DataFrame merge duplicate column handling"
     - "Django admin list_display None value error"

3. **Targeted Retrieval Phase:**
   - Use formulated queries (not raw error messages)
   - Retrieve top-k documents per query
   - Merge and deduplicate results
   - Expected: Higher precision retrieval

4. **Repair Generation Phase:**
   - Prompt includes:
     - Original bug report
     - Diagnosis explanation
     - Retrieved documentation
     - Code context
   - Generate patch with reasoning

5. **Execute and Evaluate**

---

## 3. Implementation Details

### Retrieval System Setup

**Indexing Pipeline:**
1. **Document Collection:**
   - Scrape/download all corpus sources
   - Clean and preprocess (remove boilerplate, ads, navigation)
   - Extract code blocks separately from prose
   - Add metadata (source, library, date, type)

2. **Preprocessing:**
   - Tokenize for keyword search
   - Generate embeddings using sentence-transformers
   - Create document summaries/titles
   - Split long documents into chunks (512 tokens max)

3. **Index Construction:**
   - **Embedding Index:** FAISS with cosine similarity
   - **Keyword Index:** BM25 implementation (Whoosh or custom)
   - Store both with document mapping

**Retrieval Implementation:**
- Support multiple strategies: embedding-only, keyword-only, hybrid
- Implement query expansion for CoT queries
- Re-ranking based on multiple signals
- Return top-k with confidence scores

### LLM Provider Strategy

**Primary Models (Free Tier):**
- NVIDIA NIM: Llama-CodeLlama-70B, DeepSeek-Coder
- Groq: Llama-3-70B (very fast inference)
- Together.AI: CodeLlama models

**Fallback:**
- Quantized local models (DeepSeek-Coder-6.7B, CodeLlama-7B)
- Run on CPU with acceptable latency for experimentation

**Provider Management:**
- Implement rotation on rate limits
- Cache all LLM responses (prompt → response mapping)
- Track costs per experiment
- Exponential backoff on failures

### Execution Environment

**Docker Setup:**
- Create base images for common Python versions
- Pre-install common libraries (pandas, numpy, django, etc.)
- Resource limits: 4GB RAM, 2 CPU cores, 60s timeout
- Network isolation (no external calls during test)
- Fresh container per test instance

**Evaluation Process:**
1. Parse generated patch
2. Apply to repository code
3. Run test suite
4. Capture: pass/fail, execution time, error messages
5. Destroy container
6. Log results

---

## 4. Experimental Design

### Experiment 1: Baseline Establishment
**Goal:** Understand LLM's raw capability

**Setup:**
- 50-instance test set
- 3 generation attempts per instance (for Pass@3)
- Temperature: 0.7
- No retrieval

**Measurements:**
- Pass@1, Pass@3, Pass@5
- Error distribution
- Average tokens in generated patches

### Experiment 2: Direct RAG Evaluation
**Goal:** Measure standard retrieval impact

**Variables to Test:**
- Retrieval method: Embedding vs BM25 vs Hybrid
- Number of documents: k=3,5,10
- Document type: Code-only vs Docs-only vs Mixed

**Setup:**
- Same 50 instances
- 3 attempts per configuration
- Compare against baseline

**Measurements:**
- Performance delta vs baseline
- Retrieval precision (manual annotation on sample)
- Context window utilization

### Experiment 3: CoT-RAG Evaluation
**Goal:** Test reasoning-driven retrieval

**Setup:**
- Same 50 instances
- Full 4-step CoT pipeline
- 3 attempts per instance

**Measurements:**
- Performance vs both baselines
- Diagnosis quality (manual evaluation of 20 samples)
- Query quality (relevance to actual fix)
- Retrieved document relevance (precision improvement)

**Qualitative Analysis:**
- Manually inspect 20 cases where CoT succeeded but Direct RAG failed
- Identify patterns in diagnosis quality
- Document reasoning traces

### Experiment 4: Ablation Studies

**What to Ablate:**

**A. CoT Depth:**
- 1-step: Diagnosis only, then direct retrieval
- 2-step: Diagnosis + Query formulation
- 3-step: Full pipeline
- Measure marginal benefit of each step

**B. Retrieval Strategy in CoT:**
- Use CoT queries with different retrievers
- Test if improved queries help all retrieval methods

**C. Context Size:**
- Vary retrieved documents: 1, 3, 5, 10, 15
- Find optimal point for CoT pipeline

**D. Diagnosis Format:**
- Structured (JSON) vs Natural language
- With/without explicit error classification

**Setup:**
- 30 instances per ablation
- Control all other variables
- Statistical significance testing (paired t-test)

### Experiment 5: Error Analysis (Critical Component)

**Categorization Framework:**

**By Error Type:**
- Syntax/compilation errors
- Import/dependency errors  
- Logic errors (wrong algorithm)
- Incomplete fixes (partial solution)
- Over-fixes (changed too much)
- API misuse (wrong parameters/functions)
- Context mismatch (retrieval brought wrong info)

**By Pipeline Stage:**
- Where did CoT fail? (Diagnosis, Query, Retrieval, Generation)
- Where did Direct RAG fail?

**Comparative Analysis:**
- Which errors does CoT reduce vs Direct RAG?
- Which errors persist across all methods?
- New errors introduced by CoT?

**Process:**
- Categorize all 50 test instances
- Calculate error distribution per method
- Deep-dive case studies on 15-20 interesting failures
- Identify patterns and propose solutions

---

## 5. Metrics & Evaluation

### Primary Metrics (Project Requirement)

**Functional Correctness:**
- **Pass@k:** Percentage of instances solved within k attempts
- Calculate for k=1,3,5
- Report with confidence intervals

**Execution-Based:**
- Test pass rate (% of tests passing)
- Partial correctness (some tests pass)
- Compilation rate (syntactically valid)

### Novel Metrics (CoT-Specific)

**Reasoning Validity:**
- Diagnosis accuracy: Is root cause correctly identified? (manual eval on sample)
- Query relevance: Do queries match actual needed information? (manual eval)
- Retrieval precision: % of retrieved docs actually relevant to fix (manual eval)

**Process Metrics:**
- Average retrieval precision: CoT vs Direct
- Context efficiency: Relevant tokens / Total tokens in context
- Diagnosis-fix alignment: Does fix address diagnosed issue?

### Cost-Benefit Metrics

**Latency:**
- Time to first fix
- Breakdown: Diagnosis time, Retrieval time, Generation time
- Total pipeline time per instance

**API Costs:**
- LLM calls per instance
- Tokens consumed per pipeline
- Estimated cost per successful fix

**Resource Usage:**
- Memory consumption
- Execution time distribution

---

## 6. Stretch Goals Implementation

### Stretch 1: Iterative Refinement Loop

**Design:**
```
Loop (max 3 iterations):
  1. Generate fix using CoT pipeline
  2. Execute tests
  3. If success: STOP
  4. If failure:
     - Extract failure information
     - Feed back to diagnosis module
     - Include: previous attempt, why it failed, new error trace
     - Re-diagnose with enriched context
     - Re-formulate query
     - Retrieve new/additional documents
     - Generate refined fix
```

**Measurements:**
- Success rate by iteration number
- Diminishing returns analysis
- Cost per iteration
- Types of bugs that benefit from iteration

**Expected Outcome:**
- Some bugs unsolvable in 1 shot become solvable
- Measure improvement: 5-15% additional success

### Stretch 2: Self-Correction via Document Grading

**Implementation:**
```
After retrieval, before generation:
  For each retrieved document:
    Prompt: "Given this diagnosis: [X]
             Is this document relevant? Score 0-10 and explain."
    LLM grades document
  
  Filter: Keep only documents scoring ≥ 7
  Proceed to generation with filtered set
```

**Measurements:**
- Retrieval precision before/after filtering
- Generation quality improvement
- False negative rate (good docs filtered out)
- Cost overhead vs benefit

**Hypothesis:**
- Reducing noise improves generation quality
- LLM can judge relevance better than similarity scores

### Stretch 3: Multi-Strategy Comparison

**Retrieval Strategies to Compare:**
- Pure embedding (CodeBERT)
- Pure keyword (BM25)
- Hybrid (0.5 embedding + 0.5 keyword)
- Re-ranked hybrid (retrieve 20, re-rank to top 5)
- CoT-enhanced versions of each

**Setup:**
- Run each on same 50 instances
- Plot performance vs retrieval precision
- Identify best strategy per bug type

### Stretch 4: Scaling to Full Dataset

**Plan:**
- After validating on 50 instances
- Scale to full SWE-bench-Lite (300 instances)
- Batch processing with checkpoint/resume
- Parallel execution where possible

**Challenges:**
- API rate limits (manage via queuing)
- Execution time (run overnight/distributed)
- Cost management (budget for ~1000-3000 API calls)

---

## 7. Deliverables Breakdown

### Code Repository Structure

```
cot-racg-apr/
├── data/
│   ├── swe_bench/           # SWE-bench instances
│   ├── corpus/              # Retrieval corpus
│   └── processed/           # Preprocessed data
├── src/
│   ├── retrieval/           # Indexing & search
│   ├── pipelines/           # Baseline, RAG, CoT-RAG
│   ├── evaluation/          # Docker execution, metrics
│   ├── llm/                 # Provider management
│   └── utils/               # Helpers
├── experiments/
│   ├── configs/             # Experiment configurations
│   ├── results/             # Raw results
│   └── analysis/            # Analysis notebooks
├── docker/
│   └── Dockerfile           # Execution environment
├── docs/
│   ├── setup.md             # Setup instructions
│   └── api.md               # Code documentation
├── notebooks/
│   └── error_analysis.ipynb # Interactive analysis
├── requirements.txt
└── README.md
```

### Working Code

**Must Include:**
1. **Data Loading:** Scripts to download and preprocess SWE-bench
2. **Indexing:** Build retrieval corpus indexes
3. **Pipelines:** Implementations of all 3 variants
4. **Evaluation:** Docker setup and test execution
5. **Experiments:** Runnable scripts for all experiments
6. **Analysis:** Jupyter notebooks for metrics and visualization

**Quality Standards:**
- Modular, well-documented code
- Configuration files (not hardcoded parameters)
- Logging throughout pipeline
- Error handling and recovery
- Reproducible with single command

### Evaluation Pipeline

**Components:**
1. **Test Harness:** Automates running all experiments
2. **Docker Manager:** Handles container lifecycle
3. **Metrics Calculator:** Computes all metrics from results
4. **Statistical Analysis:** Significance tests, confidence intervals

**Outputs:**
- Per-instance results (JSON)
- Aggregate metrics (CSV)
- Comparison tables
- Visualization plots

### Written Report (15-20 pages)

**Structure:**

**1. Introduction (2 pages)**
- Motivation: Why automated program repair matters
- Problem: Limitations of current RAG for code
- Hypothesis: Reasoning before retrieval improves precision
- Contributions summary

**2. Related Work (2 pages)**
- Automated Program Repair literature
- RAG for code generation
- Chain-of-thought reasoning in LLMs
- SWE-bench and CodeRAG-Bench benchmarks

**3. Methodology (4 pages)**
- System architecture
- Three pipeline variants detailed
- CoT design rationale
- Retrieval implementation
- Evaluation setup

**4. Experimental Design (2 pages)**
- Datasets and metrics
- Experiment configurations
- Ablation study design
- Evaluation protocol

**5. Results (4 pages)**
- Quantitative results: All metrics, comparison tables
- Ablation study findings
- Stretch goal results (if completed)
- Statistical significance

**6. Error Analysis (3 pages)**
- Error categorization and distribution
- Case studies: Success and failure examples
- Comparative analysis: CoT vs Direct RAG failures
- Insights and patterns

**7. Discussion (2 pages)**
- Why does CoT help (or not)?
- Trade-offs: Latency vs accuracy
- Limitations of approach
- Practical implications

**8. Conclusion & Future Work (1 page)**
- Summary of findings
- Recommendations
- Future research directions

**Appendix:**
- Implementation details
- Full experimental configurations
- Additional results tables
- Example prompts

### Presentation/Demo (15 minutes)

**Slide Deck Structure:**
1. **Title & Motivation** (1 slide)
2. **Problem & Approach** (2 slides)
   - Current RAG limitations
   - CoT-RAG solution
3. **System Architecture** (2 slides)
   - Pipeline diagram
   - CoT workflow visualization
4. **Experimental Setup** (1 slide)
5. **Results** (3 slides)
   - Performance comparison
   - Ablation insights
   - Error analysis highlights
6. **Case Study** (2 slides)
   - One success story walkthrough
   - Show diagnosis → query → retrieval → fix
7. **Conclusions** (1 slide)
8. **Demo** (Live or video, 3 minutes)

**Demo Options:**
- Live: Run CoT pipeline on new bug, show reasoning trace
- Video: Pre-recorded execution with voiceover
- Interactive: Gradio/Streamlit interface showing pipeline steps

---

## 8. Timeline & Milestones

### Phase 1: Setup & Baseline (Week 1)
- Download SWE-bench-Lite, create 50-instance subset
- Set up Docker evaluation environment
- Implement baseline (no retrieval) pipeline
- Run baseline experiments
- **Deliverable:** Baseline results, working evaluation harness

### Phase 2: Direct RAG Implementation (Week 2)
- Build retrieval corpus from CodeRAG-Bench + docs
- Implement indexing (embedding + keyword)
- Build Direct RAG pipeline
- Run experiments (retrieval method variants)
- Compare to baseline
- **Deliverable:** Direct RAG results, retrieval system

### Phase 3: CoT-RAG Development (Week 3)
- Implement diagnosis module
- Implement query formulation
- Integrate with retrieval
- Build full CoT pipeline
- Run CoT experiments
- Conduct ablation studies
- **Deliverable:** CoT-RAG results, ablation findings

### Phase 4: Analysis & Stretch Goals (Week 4)
- Comprehensive error analysis
- Implement 1-2 stretch goals (prioritize iterative refinement)
- Scale to larger subset or full dataset
- Generate all visualizations
- Begin report writing
- **Deliverable:** Complete experimental results

### Phase 5: Documentation & Presentation (Week 5)
- Finalize report
- Create presentation slides
- Prepare demo
- Code cleanup and documentation
- **Deliverable:** All final deliverables

---

## 9. Success Criteria

### Minimum Viable Project (Must Achieve)
✓ Working system on SWE-bench-Lite subset (50+ instances)
✓ All three pipelines implemented and evaluated
✓ Execution-based metrics calculated (Pass@k)
✓ Error categorization completed
✓ Comparison showing whether CoT improves retrieval/generation
✓ Comprehensive documentation and report

### Target Success (Expected)
✓ All minimum criteria met
✓ Statistical significance demonstrated (if improvement exists)
✓ Deep error analysis with actionable insights
✓ At least 1 stretch goal completed
✓ Scale to 100-150 instances
✓ Publication-quality report and presentation

### Exceptional Success (Stretch)
✓ All target criteria met
✓ Multiple stretch goals completed
✓ Full SWE-bench-Lite evaluation (300 instances)
✓ Novel insights about reasoning-driven retrieval
✓ Reusable framework released as open-source
✓ Results competitive with or exceeding published baselines

---

## 10. Risk Mitigation

### Technical Risks

**Risk: API rate limits prevent experimentation**
- Mitigation: Multi-provider setup, local model fallback, batch processing with delays

**Risk: Docker execution too complex**
- Mitigation: Start with subprocess execution, migrate to Docker incrementally

**Risk: Retrieval corpus too large to process**
- Mitigation: Start with smaller corpus (5K docs), scale if needed

**Risk: LLM generates invalid patches consistently**
- Mitigation: Improve prompts, add syntax validation, use better models

**Risk: Results show no improvement from CoT**
- Mitigation: Still valuable negative result, analyze why, adjust approach

### Project Management Risks

**Risk: Scope too ambitious**
- Mitigation: Prioritize core requirements, stretch goals are optional

**Risk: Debugging takes too long**
- Mitigation: Start with very small dataset (10 instances), validate pipeline early

**Risk: Infrastructure failures**
- Mitigation: Regular backups, checkpointing, version control

---

## 11. Alignment with Course Requirements

### Fulfills Project C Requirements

**From Rubric:**
✓ **Engineering (30%):** Three complete pipelines, modular code, Docker evaluation
✓ **Research Rigor (20%):** Ablation studies, baseline comparisons, error analysis
✓ **Report (20%):** Comprehensive write-up with all required sections
✓ **Presentation (10%):** Clear demo of CoT reasoning process
✓ **Innovation (20%):** Novel CoT-RAG approach, stretch goals

**Specific Requirements Met:**
✓ Uses CodeRAG-Bench (as retrieval corpus)
✓ Implements multiple retrieval methods
✓ Baseline vs RAG comparison experiments
✓ Execution-based evaluation (Pass@k)
✓ Failure mode analysis

**Novel Contribution:**
✓ Chain-of-thought for automated debugging (not in original project spec)
✓ Reasoning-driven retrieval vs similarity-driven
✓ Diagnosis validity as new metric

---

This plan provides a complete, executable roadmap that:
- Combines Project C requirements with your CoT innovation
- Uses SWE-bench-Lite as practical evaluation benchmark
- Leverages CodeRAG-Bench for retrieval corpus
- Includes all required components plus research contributions
- Remains feasible with free resources
- Delivers meaningful results regardless of outcome