import torch
import torch.nn as nn
import torchvision.models as models


class FractureModel(nn.Module):
    """
    Multi-task PyTorch model for bone fracture detection and classification.
    Uses a ResNet-50 backbone with two custom output heads:

    1. severity_head: 4-class classification (hairline, simple, displaced, comminuted)
       Structure: Linear(2048, 512) -> ReLU -> Dropout(0.4) -> Linear(512, 4)

    2. bone_head: 6-class classification (distal_radius, clavicle, ankle, femur, humerus, metatarsal)
       Structure: Linear(2048, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 6)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load the ResNet-50 backbone
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            print("[*] Initializing ResNet-50 backbone with pretrained weights.")
        else:
            weights = None
            print("[*] Initializing ResNet-50 backbone with random weights.")

        self.backbone = models.resnet50(weights=weights)

        # Replace the original fully connected classifier layer with Identity
        # to expose the 2048-dimensional global pooling features.
        self.backbone.fc = nn.Identity()

        # Output Head 1: severity_head (4-class classifier)
        self.severity_head = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(512, 4))

        # Output Head 2: bone_head (6-class classifier)
        self.bone_head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(256, 6))

        # Freeze ResNet layers for Stage 1:
        # Freeze all layers except layer3, layer4, and fc (which is Identity)
        self._freeze_stage1()

    def _freeze_stage1(self):
        """Freezes all layers except layer3, layer4, and the output heads."""
        # First freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze layer3 and layer4 of ResNet backbone
        for name, child in self.backbone.named_children():
            if name in ["layer3", "layer4"]:
                for param in child.parameters():
                    param.requires_grad = True

        # Unfreeze the custom heads
        for param in self.severity_head.parameters():
            param.requires_grad = True
        for param in self.bone_head.parameters():
            param.requires_grad = True

        print("[*] Stage 1 Freezing Complete: ResNet backbone layers below layer3 are frozen.")

    def unfreeze_all(self):
        """Unfreezes all parameters of the model (backbone + heads) for Stage 2 fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        print("[*] Stage 2 Unfreezing Complete: All layers unfrozen for end-to-end fine-tuning.")

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Preprocessed X-ray image batch of shape (B, 3, 224, 224).
        Returns:
            tuple: (severity_logits, bone_logits)
        """
        # Exposes global features of shape (B, 2048) after AvgPool
        features = self.backbone(x)

        severity_logits = self.severity_head(features)
        bone_logits = self.bone_head(features)

        return severity_logits, bone_logits


class Stage2FractureModel(nn.Module):
    """
    Unified Stage 2 Model with two output heads:
    1. fracture_head: 2 classes (not_fractured, fractured)
    2. region_head: 5 classes (hand, leg, hip, shoulder, unknown)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load the ResNet-50 backbone
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            print("[*] Initializing ResNet-50 backbone with pretrained weights.")
        else:
            weights = None
            print("[*] Initializing ResNet-50 backbone with random weights.")

        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Identity()

        # Output Head 1: fracture_head (binary classifier)
        self.fracture_head = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(512, 2))

        # Output Head 2: region_head (5-class classifier)
        self.region_head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(256, 5))

    def unfreeze_all(self):
        """Unfreezes all parameters of the model for end-to-end Stage 2 fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        print("[*] Stage 2 Unfreezing Complete: All layers unfrozen for end-to-end fine-tuning.")

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Returns:
            tuple: (fracture_logits, region_logits)
        """
        features = self.backbone(x)
        fracture_logits = self.fracture_head(features)
        region_logits = self.region_head(features)
        return fracture_logits, region_logits
