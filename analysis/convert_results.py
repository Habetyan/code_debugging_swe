import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Convert results to predictions format")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("output_file", help="Path to output predictions JSON file")
    parser.add_argument("--attempt", type=int, default=1, help="Attempt number to use (1-indexed)")
    args = parser.parse_args()
    
    try:
        with open(args.results_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load results: {e}")
        sys.exit(1)
        
    predictions = []
    model_name = data.get("config", {}).get("model", "unknown")
    
    print(f"Converting {len(data['instances'])} instances (Attempt {args.attempt})...")
    
    for inst in data['instances']:
        instance_id = inst['instance_id']
        attempts = inst.get('attempts', [])
        
        # Get specified attempt
        target_attempt = None
        for attempt in attempts:
            if attempt['attempt'] == args.attempt:
                target_attempt = attempt
                break
        
        if target_attempt and target_attempt.get('generated_patch'):
            predictions.append({
                "instance_id": instance_id,
                "model_patch": target_attempt['generated_patch'],
                "model_name_or_path": model_name
            })
        else:
            print(f"[WARN] No patch for {instance_id} attempt {args.attempt}")
            predictions.append({
                "instance_id": instance_id,
                "model_patch": "",
                "model_name_or_path": model_name
            })
            
    with open(args.output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
        
    print(f"[SUCCESS] Saved {len(predictions)} predictions to {args.output_file}")

if __name__ == "__main__":
    main()
