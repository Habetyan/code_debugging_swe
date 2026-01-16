import re
import sys
import os
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class PatchInfo:
    file_path: str
    before_code: str
    after_code: str

def parse_unified_diff(patch: str) -> List[PatchInfo]:
    if not patch or not patch.strip():
        return []

    patches = []
    current_file = None
    hunks = []

    lines = patch.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith('--- a/'):
            if current_file and hunks:
                before, after = reconstruct_code_from_hunks(hunks)
                patches.append(PatchInfo(current_file, before, after))
            current_file = line[6:].strip()
            hunks = []
            i += 1
            continue

        if line.startswith('+++ b/'):
            i += 1
            continue

        if line.startswith('@@'):
            hunk_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].startswith('@@') and not lines[i].startswith('--- '):
                hunk_lines.append(lines[i])
                i += 1
            hunks.append(hunk_lines)
            continue

        i += 1

    if current_file and hunks:
        before, after = reconstruct_code_from_hunks(hunks)
        patches.append(PatchInfo(current_file, before, after))

    return patches

def reconstruct_code_from_hunks(hunks: List[List[str]]) -> Tuple[str, str]:
    before_lines = []
    after_lines = []

    for hunk in hunks:
        for line in hunk[1:]:
            if line.startswith('-'):
                before_lines.append(line[1:])
            elif line.startswith('+'):
                after_lines.append(line[1:])
            elif line.startswith(' '):
                before_lines.append(line[1:])
                after_lines.append(line[1:])
            elif not line.startswith('\\'):
                before_lines.append(line)
                after_lines.append(line)

    return '\n'.join(before_lines), '\n'.join(after_lines)

def get_full_context_diff(patch: str) -> List[Dict]:
    if not patch or not patch.strip():
        return []

    results = []
    current_file = None
    before_lines = []
    after_lines = []

    lines = patch.split('\n')
    in_hunk = False

    for line in lines:
        if line.startswith('--- a/'):
            if current_file:
                results.append({
                    'file': current_file,
                    'before': '\n'.join(before_lines),
                    'after': '\n'.join(after_lines)
                })
            current_file = line[6:].strip()
            before_lines = []
            after_lines = []
            in_hunk = False
        elif line.startswith('+++ b/'):
            continue
        elif line.startswith('@@'):
            in_hunk = True
            match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
            if match:
                before_lines.append(f"[Line {match.group(1)}]")
                after_lines.append(f"[Line {match.group(2)}]")
        elif in_hunk:
            if line.startswith('-'):
                before_lines.append(line[1:])
            elif line.startswith('+'):
                after_lines.append(line[1:])
            elif line.startswith(' '):
                before_lines.append(line[1:])
                after_lines.append(line[1:])
            elif not line.startswith('\\'):
                before_lines.append(line)
                after_lines.append(line)

    if current_file:
        results.append({
            'file': current_file,
            'before': '\n'.join(before_lines),
            'after': '\n'.join(after_lines)
        })

    return results

_dataset_cache = {}

def get_swe_bench_dataset(dataset_name: str, split: str) -> dict:
    cache_key = f"{dataset_name}_{split}"
    if cache_key not in _dataset_cache:
        try:
            from datasets import load_dataset
            dataset = load_dataset(dataset_name, split=split)
            _dataset_cache[cache_key] = {item['instance_id']: item for item in dataset}
        except Exception as e:
            _dataset_cache[cache_key] = {}
    return _dataset_cache[cache_key]

def load_swe_bench_instance(instance_id: str) -> Optional[dict]:
    datasets_to_try = [
        ("princeton-nlp/SWE-bench_Lite", "test"),
        ("princeton-nlp/SWE-bench", "dev"),
        ("princeton-nlp/SWE-bench_Verified", "test"),
    ]

    for dataset_name, split in datasets_to_try:
        try:
            dataset_dict = get_swe_bench_dataset(dataset_name, split)
            if instance_id in dataset_dict:
                item = dataset_dict[instance_id]
                return {
                    'instance_id': item['instance_id'],
                    'repo': item['repo'],
                    'base_commit': item['base_commit'],
                    'problem_statement': item['problem_statement'],
                    'hints_text': item.get('hints_text', ''),
                    'patch': item['patch'],
                    'test_patch': item['test_patch'],
                    'fail_to_pass': item.get('FAIL_TO_PASS', '[]'),
                    'pass_to_pass': item.get('PASS_TO_PASS', '[]'),
                    'environment_setup_commit': item.get('environment_setup_commit', item['base_commit']),
                    'version': item.get('version', '')
                }
        except:
            continue
    return None

def run_agentic_inference(instance_data: dict) -> dict:
    try:
        from src.pipelines.agentic import AgenticPipeline
        from src.llm.provider import LLMProvider
        from src.data.swe_bench import SWEBenchInstance

        llm = LLMProvider()
        pipeline = AgenticPipeline(llm_provider=llm, use_harness=False)

        instance = SWEBenchInstance(
            instance_id=instance_data.get('instance_id', ''),
            repo=instance_data.get('repo', ''),
            base_commit=instance_data.get('base_commit', ''),
            problem_statement=instance_data.get('problem_statement', ''),
            hints_text=instance_data.get('hints_text', ''),
            patch=instance_data.get('patch', ''),
            test_patch=instance_data.get('test_patch', ''),
            fail_to_pass=instance_data.get('fail_to_pass', '[]'),
            pass_to_pass=instance_data.get('pass_to_pass', '[]'),
            environment_setup_commit=instance_data.get('environment_setup_commit', ''),
            version=instance_data.get('version', '')
        )

        result = pipeline.run(instance)

        return {
            'success': result.success,
            'generated_patch': result.generated_patch,
            'error': result.error
        }
    except Exception as e:
        return {
            'success': False,
            'generated_patch': '',
            'error': str(e)
        }
