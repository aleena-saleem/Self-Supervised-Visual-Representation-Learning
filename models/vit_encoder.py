import torch
import torch.nn as nn
from torchvision import models


class ViTEncoder(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        if pretrained:

            self.encoder = models.vit_b_16(
                weights=models.ViT_B_16_Weights.DEFAULT
            )

        else:

            self.encoder = models.vit_b_16(
                weights=None
            )

        self.feature_dim = self.encoder.heads.head.in_features

        self.encoder.heads = nn.Identity()


    def forward(self, x):

        features = self.encoder(x)

        return features


    def freeze(self):

        for param in self.parameters():
            param.requires_grad = False


    def unfreeze(self):

        for param in self.parameters():
            param.requires_grad = True
