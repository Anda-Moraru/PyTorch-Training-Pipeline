import argparse
import yaml

from train import train_pipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Training Pipeline - Configurable via CLI or YAML config"
    )
    # Config file
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to YAML configuration file")
    
    # Dataset and model
    parser.add_argument("--dataset", type=str, 
                       choices=["mnist", "cifar10", "cifar100", "oxfordpets"],
                       help="Override dataset")
    parser.add_argument("--model_name", type=str,
                       choices=["resnet18", "resnet50", "resnest14d", "resnest26d", "mlp"],
                       help="Override model name")

    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--weight_decay", type=float, help="Override weight decay")
    parser.add_argument("--optimizer", type=str,
                       choices=["sgd", "adam", "adamw", "sam", "muon"],
                       help="Override optimizer")
    parser.add_argument("--scheduler", type=str,
                       choices=["steplr", "reduceonplateau", "none"],
                       help="Override scheduler")
    
    # Features
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Wandb logging")
    parser.add_argument("--use_cutout", action="store_true", help="Enable Cutout augmentation")
    parser.add_argument("--use_autoaugment", action="store_true", help="Enable AutoAugment")
    
    # Device
    parser.add_argument("--device", type=str,
                       choices=["auto", "cuda", "cpu", "mps"],
                       help="Override device")
    
    return parser.parse_args()


def load_config(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.model_name is not None:
        cfg["model_name"] = args.model_name
    if args.num_epochs is not None:
        cfg["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    if args.optimizer is not None:
        cfg["optimizer"] = args.optimizer
    if args.scheduler is not None:
        cfg["scheduler"] = args.scheduler
    if args.device is not None:
        cfg["device"] = args.device

    if args.use_amp:
        cfg["use_amp"] = True
    if args.use_wandb:
        cfg["use_wandb"] = True
    if args.use_cutout:
        cfg["use_cutout"] = True
    if args.use_autoaugment:
        cfg["use_autoaugment"] = True

    return cfg


def main():
    args = parse_args()
    cfg = load_config(args)
    train_pipeline(cfg)


if __name__ == "__main__":
    main()
