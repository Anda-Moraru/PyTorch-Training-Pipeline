import torch.nn as nn
import timm


class MLP(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, input_size: int = 32):
        super().__init__()
        if input_size == 32:
            input_dim = in_channels * 32 * 32  # CIFAR
        elif input_size == 28:
            input_dim = in_channels * 28 * 28  # MNIST
        else:
            input_dim = in_channels * input_size * input_size
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def get_model(model_name: str, num_classes: int, in_channels: int = 3, pretrained: bool = False):

    name = model_name.lower()
    
    # MLP model
    if name == "mlp":
        return MLP(num_classes=num_classes, in_channels=in_channels, input_size=32)
    
    # timm models
    if name == "resnet18":
        model = timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes)
        return model
    
    if name == "resnet50":
        model = timm.create_model("resnet50", pretrained=pretrained, num_classes=num_classes)
        return model
    
    if name == "resnest14d":
        model = timm.create_model("resnest14d", pretrained=pretrained, num_classes=num_classes)
        return model
    
    if name == "resnest26d":
        model = timm.create_model("resnest26d", pretrained=pretrained, num_classes=num_classes)
        return model

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Supported models: resnet18, resnet50, resnest14d, resnest26d, mlp"
    )
