"""
SWE-bench Lite Dataset Loader.
Provides utilities to load and filter the SWE-bench dataset.
"""
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
import random


@dataclass
class SWEBenchInstance:
    """Represents a single bug instance from SWE-bench."""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    patch: str  # Ground truth patch
    test_patch: str
    version: str
    fail_to_pass: str
    pass_to_pass: str
    environment_setup_commit: str
    
    @property
    def repo_name(self) -> str:
        """Extract repository name (e.g., 'django' from 'django/django')."""
        return self.repo.split('/')[-1] if '/' in self.repo else self.repo


def load_swe_bench_dataset(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    instance_ids: Optional[list[str]] = None
) -> list[SWEBenchInstance]:
    """
    Load any SWE-bench dataset from Hugging Face.
    
    Args:
        dataset_name: HuggingFace dataset name. Options:
            - "princeton-nlp/SWE-bench_Lite" (300 instances, test only)
            - "princeton-nlp/SWE-bench_Verified" (500 instances, test only)
            - "princeton-nlp/SWE-bench" (full dataset, has train/test splits)
        split: Dataset split ("test", "train", "dev")
        
    Returns:
        List of SWEBenchInstance objects
    """
    print(f"Loading dataset: {dataset_name} [{split}]...")
    dataset = load_dataset(dataset_name, split=split)
    
    # Filter by instance_ids if provided
    if instance_ids:
        dataset = dataset.filter(lambda x: x['instance_id'] in instance_ids)
        
    instances = []
    for item in dataset:
        instances.append(SWEBenchInstance(
            instance_id=item['instance_id'],
            repo=item['repo'],
            base_commit=item['base_commit'],
            problem_statement=item['problem_statement'],
            hints_text=item['hints_text'] or "",
            patch=item['patch'],
            test_patch=item['test_patch'],
            version=item.get('version', ''),
            fail_to_pass=item.get('FAIL_TO_PASS', ''),
            pass_to_pass=item.get('PASS_TO_PASS', ''),
            environment_setup_commit=item.get('environment_setup_commit', item['base_commit'])
        ))
    
    print(f"Loaded {len(instances)} instances.")
    return instances


def load_swe_bench_verified() -> list[SWEBenchInstance]:
    """Load SWE-bench Verified (clean eval set, 500 instances)."""
    return load_swe_bench_dataset("princeton-nlp/SWE-bench_Verified", "test")


def load_swe_bench_lite(split: str = "test") -> list[SWEBenchInstance]:
    """Load SWE-bench Lite (legacy, 300 instances)."""
    return load_swe_bench_dataset("princeton-nlp/SWE-bench_Lite", split)


def load_swe_bench_dev() -> list[SWEBenchInstance]:
    """Load SWE-bench Full dev split (225 instances)."""
    return load_swe_bench_dataset("princeton-nlp/SWE-bench", "dev")


def load_swe_bench_train(exclude_ids: set[str] = None) -> list[SWEBenchInstance]:
    """
    Load the full SWE-bench training set for RAG corpus.
    Optionally exclude specific instance IDs to prevent data leakage.
    
    Args:
        exclude_ids: Set of instance IDs to exclude (e.g., eval set IDs)
    """
    instances = load_swe_bench_dataset("princeton-nlp/SWE-bench", "train")
    
    if exclude_ids:
        before = len(instances)
        instances = [i for i in instances if i.instance_id not in exclude_ids]
        print(f"Filtered out {before - len(instances)} instances to prevent leakage.")
    
    return instances


def create_stratified_subset(
    instances: list[SWEBenchInstance],
    n: int = 50,
    seed: int = 42
) -> list[SWEBenchInstance]:
    """
    Create a stratified subset of instances across different repositories.
    
    Args:
        instances: Full list of SWE-bench instances
        n: Number of instances to select
        seed: Random seed for reproducibility
        
    Returns:
        Stratified subset of instances
    """
    random.seed(seed)
    
    # Group by repository
    repo_groups: dict[str, list[SWEBenchInstance]] = {}
    for inst in instances:
        repo = inst.repo
        if repo not in repo_groups:
            repo_groups[repo] = []
        repo_groups[repo].append(inst)
    
    # Calculate instances per repo (proportional)
    total = len(instances)
    subset = []
    
    for repo, repo_instances in repo_groups.items():
        # Proportional selection
        count = max(1, int(n * len(repo_instances) / total))
        selected = random.sample(repo_instances, min(count, len(repo_instances)))
        subset.extend(selected)
    
    # Trim or pad to exact n
    random.shuffle(subset)
    if len(subset) > n:
        subset = subset[:n]
    elif len(subset) < n:
        # Add more from remaining instances
        remaining = [i for i in instances if i not in subset]
        random.shuffle(remaining)
        subset.extend(remaining[:n - len(subset)])
    
    return subset


def get_instance_by_id(
    instances: list[SWEBenchInstance],
    instance_id: str
) -> Optional[SWEBenchInstance]:
    """Find an instance by its ID."""
    for inst in instances:
        if inst.instance_id == instance_id:
            return inst
    return None
