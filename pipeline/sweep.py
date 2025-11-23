import yaml
import argparse
from train import train_pipeline


def run_custom_sweep(config_file="sweep_configs.yaml"):
    with open(config_file, "r") as f:
        configs = yaml.safe_load(f)
    
    results = []
    
    for i, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Running configuration {i+1}/{len(configs)}")
        print(f"{'='*60}")
        
        cfg["experiment_name"] = f"config_{i+1}"
        cfg["use_wandb"] = False
        
        try:
            result = train_pipeline(cfg)
            if isinstance(result, dict):
                test_acc = result["test_accuracy"]
                training_time = result["training_time_minutes"]
            else:
                test_acc = result
                training_time = 0.0
            
            results.append({
                "config": i+1,
                "test_accuracy": test_acc,
                "training_time": training_time,
                "params": cfg
            })
            print(f"Configuration {i+1} completed: Test Accuracy {test_acc:.2f}% | Time: {training_time:.1f} min")
        except Exception as e:
            print(f"Configuration {i+1} failed: {e}")
            results.append({
                "config": i+1,
                "test_accuracy": 0.0,
                "training_time": 0.0,
                "error": str(e),
                "params": cfg
            })

    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    successful = [r for r in results if r.get("test_accuracy", 0.0) > 70.0]
    print(f"Total configurations: {len(results)}")
    print(f"Configurations with >70% test accuracy: {len(successful)}")
    print(f"\nTop configurations:")
    sorted_results = sorted(successful, key=lambda x: x.get("test_accuracy", 0.0), reverse=True)
    for r in sorted_results[:10]:
        test_acc = r.get("test_accuracy", 0.0)
        time_min = r.get("training_time", 0.0)
        print(f"  Config {r['config']}: Test Acc {test_acc:.2f}% | Time: {time_min:.1f} min")
        print(f"    Model: {r['params'].get('model_name')}, "
              f"LR: {r['params'].get('learning_rate')}, "
              f"BS: {r['params'].get('batch_size')}, "
              f"Opt: {r['params'].get('optimizer')}")

    import json
    results_file = "sweep_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep using custom script + Tensorboard"
    )
    parser.add_argument("--config", type=str, default="sweep_configs.yaml",
                       help="Config file for custom sweep")
    args = parser.parse_args()
    
    run_custom_sweep(args.config)

