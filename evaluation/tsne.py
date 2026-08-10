
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def compute_tsne(
    features,
    perplexity=30,
    random_state=42
):
    """
    Convert high-dimensional features
    into a 2-dimensional t-SNE representation.
    """

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state
    )

    embeddings_2d = tsne.fit_transform(
        features
    )

    return embeddings_2d


def plot_tsne(
    embeddings_2d,
    labels,
    save_path=None
):
    """
    Plot 2-D t-SNE embeddings.
    """

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        s=20
    )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("SimCLR Feature Representations")

    plt.colorbar(
        scatter,
        label="Class"
    )

    plt.tight_layout()

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=200
        )

    plt.show()
