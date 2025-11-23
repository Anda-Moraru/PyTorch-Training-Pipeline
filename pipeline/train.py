import time
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import cross_entropy

from data import get_dataloaders, get_num_classes
from models import get_model
from optimizers import get_optimizer, get_scheduler, SAM
from utils import BatchSizeScheduler, recreate_dataloader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def accuracy(outputs, targets):
    preds = outputs.argmax(1)
    return (preds == targets).float().mean().item()


def train_pipeline(cfg):
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])
    
    print(f"Using device: {device}")

    use_wandb = cfg.get("use_wandb", False) and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=cfg.get("wandb_project", "pytorch-training"),
            name=cfg.get("experiment_name", "experiment"),
            config=cfg
        )

    # data
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    num_classes = get_num_classes(cfg["dataset"])

    # model
    use_pretrained = cfg.get("use_pretrained", False)
    model = get_model(cfg["model_name"], num_classes=num_classes, pretrained=use_pretrained).to(device)
    
    if use_pretrained:
        print(f"Using pretrained model (ImageNet weights)")

    if cfg.get("compile_model", False) and hasattr(torch, "compile") and device.type == "cuda":
        print("Compiling model for efficiency...")
        model = torch.compile(model, mode="reduce-overhead")
    elif cfg.get("compile_model", False) and device.type == "cpu":
        print("Model compilation skipped on CPU (requires C++ compiler on Windows)")

    label_smoothing = float(cfg.get("label_smoothing", 0.0))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = get_optimizer(cfg, model.parameters())
    scheduler = get_scheduler(cfg, optimizer)

    use_sam = isinstance(optimizer, SAM)

    batch_scheduler = BatchSizeScheduler(
        initial_batch_size=cfg["batch_size"],
        strategy=cfg.get("batch_size_scheduler", "none")
    )
    train_dataset = train_loader.dataset

    # logging
    base_log_dir = cfg.get("log_dir", "./runs")
    experiment_name = cfg.get("experiment_name", "experiment")
    # timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_log_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"TensorBoard logs: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)
    best_acc = 0.0
    best_train_acc = 0.0
    epochs_no_improve = 0
    patience = int(cfg["early_stopping_patience"])

    if device.type == "cuda" and cfg.get("use_amp", False):
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    num_epochs = int(cfg["num_epochs"])
    total_time = 0.0
    
    for epoch in range(1, num_epochs + 1):
        if batch_scheduler.strategy != "none":
            batch_scheduler.step(epoch)
            new_batch_size = batch_scheduler.get_batch_size()
            if new_batch_size != train_loader.batch_size:
                print(f"Updating batch size to {new_batch_size}")
                train_loader = recreate_dataloader(
                    train_dataset, new_batch_size, cfg["num_workers"], shuffle=True
                )

        model.train()
        train_loss, train_acc = 0.0, 0.0

        t0 = time.time()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            use_amp = (device.type == "cuda" and cfg.get("use_amp", False))
            
            if use_sam:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                optimizer.first_step(zero_grad=True)
                
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # metrics
            train_acc += accuracy(outputs, targets)
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_acc = 100.0 * train_acc / len(train_loader)
        

        if train_acc > best_train_acc:
            best_train_acc = train_acc
        

        if train_acc > best_train_acc:
            best_train_acc = train_acc

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.inference_mode():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_acc += accuracy(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_acc / len(val_loader)

        if scheduler is not None:
            if cfg["scheduler"].lower() == "reduceonplateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()

        dt = time.time() - t0
        total_time += dt

        current_lr = optimizer.param_groups[0]["lr"]
        
        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"Train Loss {train_loss:.3f} Acc {train_acc:.2f}% | "
            f"Val Loss {val_loss:.3f} Acc {val_acc:.2f}% | "
            f"LR {current_lr:.6f} | {dt:.1f}s"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        writer.add_scalar("Time/epoch", dt, epoch)

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr,
                "epoch_time": dt,
            })

        # early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            save_dir = cfg.get("save_dir", "./checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, os.path.join(save_dir, "best_model.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()
    if use_wandb:
        wandb.finish()

    print("\nEvaluating on test set (final accuracy)...")
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    
    with torch.inference_mode():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            test_acc += accuracy(outputs, targets)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    test_acc = 100.0 * test_acc / len(test_loader)

    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Best Training Accuracy: {best_train_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%  <-- This is the accuracy for 'Achieve 79% accuracy'")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"{'='*60}\n")

    return {
        "test_accuracy": test_acc,
        "best_val_accuracy": best_acc,
        "best_train_accuracy": best_train_acc,
        "training_time_seconds": total_time,
        "training_time_minutes": total_time / 60
    }
