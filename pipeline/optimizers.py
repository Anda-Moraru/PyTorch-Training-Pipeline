import torch
from torch import optim


class SAM(optim.Optimizer):

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def get_optimizer(cfg, model_params):
    opt_name = cfg["optimizer"].lower()
    lr = float(cfg["learning_rate"])
    wd = float(cfg["weight_decay"])
    momentum = float(cfg.get("momentum", 0.9))
    rho = float(cfg.get("sam_rho", 0.05))  # For SAM

    if opt_name == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=momentum,
                         weight_decay=wd, nesterov=True)
    if opt_name == "adam":
        return optim.Adam(model_params, lr=lr, weight_decay=wd)
    if opt_name == "adamw":
        return optim.AdamW(model_params, lr=lr, weight_decay=wd)
    if opt_name == "sam":
        base_opt = lambda params, **kwargs: optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=True, **kwargs
        )
        return SAM(model_params, base_opt, rho=rho)
    if opt_name == "muon":
        return optim.AdamW(model_params, lr=lr, weight_decay=wd, betas=(0.9, 0.999))

    raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(cfg, optimizer):
    sched_name = cfg["scheduler"].lower()
    if sched_name == "none":
        return None

    if sched_name == "steplr":
        step_size = int(cfg["step_size"])
        gamma = float(cfg["gamma"])
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if sched_name == "reduceonplateau":
        patience = int(cfg["plateau_patience"])
        factor = float(cfg["plateau_factor"])
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience
        )
    
    raise ValueError(f"Unknown scheduler: {sched_name}")
