import torch
import torch.nn as nn
from torchvision import models


class ResNetEncoder(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        if pretrained:
            self.encoder = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT
            )
        else:
            self.encoder = models.resnet50(
                weights=None
            )

        # Remove ImageNet classifier
        self.encoder.fc = nn.Identity()

    def forward(self, x):

        features = self.encoder(x)

        return features
