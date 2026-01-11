"""
SWE-bench-Lite Data Loader

Downloads and processes the SWE-bench-Lite dataset from Hugging Face.
Provides utilities for creating stratified subsets for experiments.
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
    
    @property
    def repo_name(self) -> str:
        """Extract repository name (e.g., 'django' from 'django/django')."""
        return self.repo.split('/')[-1] if '/' in self.repo else self.repo


def load_swe_bench_lite(split: str = "test") -> list[SWEBenchInstance]:
    """
    Load SWE-bench-Lite dataset from Hugging Face.
    
    Args:
        split: Dataset split to load ("test" for evaluation)
        
    Returns:
        List of SWEBenchInstance objects
    """
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    
    instances = []
    for item in dataset:
        instances.append(SWEBenchInstance(
            instance_id=item["instance_id"],
            repo=item["repo"],
            base_commit=item["base_commit"],
            problem_statement=item["problem_statement"],
            hints_text=item.get("hints_text", ""),
            patch=item["patch"],
            test_patch=item["test_patch"],
        ))
    
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
