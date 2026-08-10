
import torch
import torch.nn as nn

from models.resnet_encoder import ResNetEncoder
from models.projection_head import ProjectionHead


class SimCLR(nn.Module):

    def __init__(
        self,
        pretrained_encoder=True,
        projection_dim=128
    ):

        super().__init__()

        # Encoder
        self.encoder = ResNetEncoder(
            pretrained=pretrained_encoder
        )

        # Projection head
        self.projector = ProjectionHead(
            input_dim=2048,
            hidden_dim=512,
            output_dim=projection_dim
        )

    def forward(self, x):

        # Representation from encoder
        h = self.encoder(x)

        # Projection for contrastive learning
        z = self.projector(h)

        return h, z
