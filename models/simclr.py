import torch
import torch.nn as nn

from models.resnet_encoder import ResNetEncoder
from models.vit_encoder import ViTEncoder
from models.projection_head import ProjectionHead


class SimCLR(nn.Module):

    def __init__(
        self,
        encoder_type="resnet",
        pretrained_encoder=True,
        projection_dim=128
    ):

        super().__init__()

        if encoder_type == "resnet":

            self.encoder = ResNetEncoder(
                pretrained=pretrained_encoder
            )

            input_dim = 2048

        elif encoder_type == "vit":

            self.encoder = ViTEncoder(
                pretrained=pretrained_encoder
            )

            input_dim = self.encoder.feature_dim

        else:

            raise ValueError(
                "encoder_type must be 'resnet' or 'vit'"
            )

        self.projector = ProjectionHead(
            input_dim=input_dim,
            hidden_dim=512,
            output_dim=projection_dim
        )

    def forward(self, x):

        h = self.encoder(x)

        z = self.projector(h)

        return h, z
