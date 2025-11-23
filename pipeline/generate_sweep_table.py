import json
import sys

def generate_table(results_file="sweep_results.json"):
    try:
        with open(results_file, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {results_file} not found. Run sweep first.")
        return
    
    print("| Config | Model | Optimizer | LR | Batch Size | Test Accuracy | Time (min) |")
    print("|--------|-------|-----------|----|-----------|---------------|------------|")
    
    for r in results:
        config = r.get("config", 0)
        params = r.get("params", {})
        test_acc = r.get("test_accuracy", 0.0)
        time_min = r.get("training_time", 0.0)
        
        model = params.get("model_name", "unknown")
        optimizer = params.get("optimizer", "unknown")
        lr = params.get("learning_rate", 0.0)
        batch_size = params.get("batch_size", 0)
        
        print(f"| {config} | {model} | {optimizer} | {lr} | {batch_size} | {test_acc:.2f}% | {time_min:.1f} |")

    successful = [r for r in results if r.get("test_accuracy", 0.0) > 70.0]
    print(f"\n**Total configurations: {len(results)}**")
    print(f"**Configurations with >70% test accuracy: {len(successful)}**")

if __name__ == "__main__":
    results_file = sys.argv[1] if len(sys.argv) > 1 else "sweep_results.json"
    generate_table(results_file)

