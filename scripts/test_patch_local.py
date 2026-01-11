import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import subprocess
import tempfile
import shutil
from pathlib import Path
import argparse


def clone_repo(repo: str, commit: str, dest: str) -> bool:
    """Clone a repository to a specific commit."""
    try:
        # Clone shallow
        subprocess.run(
            ["git", "clone", "--depth", "100", f"https://github.com/{repo}.git", dest],
            check=True,
            capture_output=True,
            timeout=120,
        )
        # Checkout specific commit
        subprocess.run(
            ["git", "checkout", commit],
            cwd=dest,
            check=True,
            capture_output=True,
        )
        return True
    except Exception as e:
        print(f"Failed to clone: {e}")
        return False


def apply_patch(patch: str, repo_dir: str) -> tuple[bool, str]:
    """Try to apply a patch to a repository."""
    # Write patch to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
        f.write(patch)
        patch_file = f.name
    
    try:
        # Try git apply
        result = subprocess.run(
            ["git", "apply", "--check", patch_file],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            # Actually apply
            subprocess.run(
                ["git", "apply", patch_file],
                cwd=repo_dir,
                check=True,
                capture_output=True,
            )
            return True, "Patch applied successfully"
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)
    finally:
        Path(patch_file).unlink()


def load_predictions(path: str) -> list[dict]:
    """Load predictions from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description="Test patch application locally")
    parser.add_argument("predictions", help="Path to predictions JSON file")
    parser.add_argument("--instance", "-i", help="Specific instance ID to test")
    
    args = parser.parse_args()
    
    predictions = load_predictions(args.predictions)
    
    print(f"Loaded {len(predictions)} predictions")
    print("=" * 60)
    
    for pred in predictions:
        instance_id = pred["instance_id"]
        
        if args.instance and args.instance != instance_id:
            continue
        
        patch = pred["model_patch"]
        
        print(f"\n🔍 Testing: {instance_id}")
        print("-" * 40)
        
        # Check patch format first
        if not patch or not patch.strip():
            print("Empty patch")
            continue
        
        if not patch.startswith("---"):
            print("Invalid patch format (doesn't start with ---)")
            continue
        
        lines = patch.split('\n')
        has_hunk = any(l.startswith('@@') for l in lines)
        has_minus = any(l.startswith('-') and not l.startswith('---') for l in lines)
        has_plus = any(l.startswith('+') and not l.startswith('+++') for l in lines)
        
        print(f"Has hunk header: {has_hunk}")
        print(f"Has minus lines: {has_minus}")
        print(f"Has plus lines: {has_plus}")
        print(f"Ends with newline: {patch.endswith(chr(10))}")
        
        if has_hunk and (has_minus or has_plus):
            print("Patch format looks valid!")
        else:
            print("Patch may be incomplete")
        
        # Show first few lines
        print("\n Patch preview:")
        for i, line in enumerate(lines[:10]):
            print(f"   {i+1}: {line[:70]}")
        if len(lines) > 10:
            print(f"   ... ({len(lines) - 10} more lines)")
    
    print("\n" + "=" * 60)
    print("Format check complete!")


if __name__ == "__main__":
    main()
