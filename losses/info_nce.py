
import torch
import torch.nn.functional as F


class InfoNCELoss(torch.nn.Module):

    def __init__(self, temperature=0.5):

        super().__init__()

        self.temperature = temperature

    def forward(self, z1, z2):

        batch_size = z1.size(0)

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Combine both views
        z = torch.cat([z1, z2], dim=0)

        # Pairwise similarity
        similarity_matrix = torch.matmul(z, z.T)

        # Temperature scaling
        logits = similarity_matrix / self.temperature

        # Remove self-similarity
        mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=z.device
        )

        logits = logits.masked_fill(
            mask,
            float("-inf")
        )

        # Positive-pair labels
        labels = torch.cat([
            torch.arange(
                batch_size,
                2 * batch_size,
                device=z.device
            ),
            torch.arange(
                0,
                batch_size,
                device=z.device
            )
        ])

        # Contrastive loss
        loss = F.cross_entropy(
            logits,
            labels,
            reduction="mean"
        )

        return loss
