
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=512,
        output_dim=128
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim)

        )

    def forward(self, x):

        return self.network(x)
