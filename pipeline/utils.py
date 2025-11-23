import torch
from torch.utils.data import DataLoader


class BatchSizeScheduler:
    
    def __init__(self, initial_batch_size: int, strategy: str = "none"):
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.strategy = strategy.lower()
        self.epoch = 0
        
    def step(self, epoch: int):
        self.epoch = epoch
        
        if self.strategy == "double_at_half":
            if epoch == 1:
                self.current_batch_size = self.initial_batch_size
            elif epoch % 2 == 0:
                self.current_batch_size = min(
                    self.current_batch_size * 2,
                    self.initial_batch_size * 4
                )
        elif self.strategy == "linear_increase":
            self.current_batch_size = int(
                self.initial_batch_size * (1 + 0.1 * epoch)
            )
        elif self.strategy == "none":
            pass
        else:
            raise ValueError(f"Unknown batch size scheduler strategy: {self.strategy}")
    
    def get_batch_size(self) -> int:
        return self.current_batch_size


def recreate_dataloader(dataset, batch_size: int, num_workers: int, shuffle: bool = True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

