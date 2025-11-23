import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple


class Cutout:
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)
        for _ in range(self.n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        mask = mask.expand_as(img)
        img = img * mask
        return img


def get_transforms(dataset: str, train: bool, use_cutout: bool = False, use_autoaugment: bool = False):
    if dataset.lower() in ["cifar10", "cifar100"]:
        if train:
            aug_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            
            if use_autoaugment:
                aug_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
            
            aug_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761),
                ),
            ])
            
            if use_cutout:
                aug_list.append(Cutout(n_holes=1, length=16))
            
            return transforms.Compose(aug_list)
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761),
                ),
            ])
    elif dataset.lower() == "mnist":
        tf = [transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))]
        return transforms.Compose(tf)
    elif dataset.lower() == "oxfordpets":
        if train:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_num_classes(dataset: str) -> int:
    d = dataset.lower()
    if d == "mnist":
        return 10
    if d == "cifar10":
        return 10
    if d == "cifar100":
        return 100
    if d == "oxfordpets":
        return 37
    raise ValueError(f"Unknown dataset: {dataset}")


def get_dataloaders(cfg, batch_size=None) -> Tuple[DataLoader, DataLoader, DataLoader]:

    dataset = cfg["dataset"].lower()
    root = cfg["data_root"]
    batch_size = batch_size if batch_size is not None else cfg["batch_size"]
    num_workers = cfg["num_workers"]
    use_cutout = cfg.get("use_cutout", False)
    use_autoaugment = cfg.get("use_autoaugment", False)
    val_split = cfg.get("val_split", 0.1)

    if dataset == "mnist":
        train_set = datasets.MNIST(
            root, train=True, download=True,
            transform=get_transforms(dataset, train=True, use_cutout=use_cutout, use_autoaugment=use_autoaugment)
        )
        test_set = datasets.MNIST(
            root, train=False, download=True,
            transform=get_transforms(dataset, train=False, use_cutout=False, use_autoaugment=False)
        )
    elif dataset == "cifar10":
        train_set = datasets.CIFAR10(
            root, train=True, download=True,
            transform=get_transforms(dataset, train=True, use_cutout=use_cutout, use_autoaugment=use_autoaugment)
        )
        test_set = datasets.CIFAR10(
            root, train=False, download=True,
            transform=get_transforms(dataset, train=False, use_cutout=False, use_autoaugment=False)
        )
    elif dataset == "cifar100":
        train_set = datasets.CIFAR100(
            root, train=True, download=True,
            transform=get_transforms(dataset, train=True, use_cutout=use_cutout, use_autoaugment=use_autoaugment)
        )
        test_set = datasets.CIFAR100(
            root, train=False, download=True,
            transform=get_transforms(dataset, train=False, use_cutout=False, use_autoaugment=False)
        )
    elif dataset == "oxfordpets":
        from torchvision.datasets import OxfordIIITPet
        train_set = OxfordIIITPet(
            root, split="trainval", download=True,
            transform=get_transforms(dataset, train=True, use_cutout=use_cutout, use_autoaugment=use_autoaugment),
            target_types="category",
        )
        test_set = OxfordIIITPet(
            root, split="test", download=True,
            transform=get_transforms(dataset, train=False, use_cutout=False, use_autoaugment=False),
            target_types="category",
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    total_size = len(train_set)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(cfg.get("seed", 42))
    train_subset, val_subset = random_split(train_set, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    return train_loader, val_loader, test_loader
