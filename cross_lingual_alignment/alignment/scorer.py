import torch
import torch.nn.functional as F


class ProcrustesSimilarityScorer:
    """Alignment score based on Procrustes distance.

    Finds the optimal rotation aligning X to Y, returns 1 - distance/2 ∈ [0, 1].
    Score of 1 means perfect alignment; 0 means maximally misaligned.
    """

    def score(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Args:
            X: (N, d) source language representations (mean-pooled, L2-normalized)
            Y: (N, d) target language representations

        Returns:
            float alignment score in [0, 1]
        """
        X = self._center_and_normalize(X)
        Y = self._center_and_normalize(Y)

        # SVD of Y^T @ X
        M = Y.T @ X  # (d, d)
        U, _, Vt = torch.linalg.svd(M)
        W = U @ Vt  # optimal rotation

        # Procrustes distance: ||X - YW||_F, max value is 2 after normalization
        dist = torch.linalg.norm(X - Y @ W, ord="fro").item()
        return float(1.0 - dist / 2.0)

    @staticmethod
    def _center_and_normalize(X):
        X = X - X.mean(dim=0)
        frob = torch.linalg.norm(X, ord="fro")
        return X / frob.clamp(min=1e-9)


class CosineSimilarityScorer:
    """Alignment score as mean cosine similarity of matched translation pairs."""

    def score(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Args:
            X: (N, d) source representations (already L2-normalized or not)
            Y: (N, d) target representations

        Returns:
            float mean cosine similarity of diagonal pairs, in [-1, 1] (typically [0, 1])
        """
        X_norm = F.normalize(X, dim=-1)
        Y_norm = F.normalize(Y, dim=-1)
        # Diagonal of the cosine similarity matrix = matched pair similarities
        cos_sims = (X_norm * Y_norm).sum(dim=-1)
        return float(cos_sims.mean().item())


class CKAScorer:
    """Alignment score via linear Centered Kernel Alignment.

    Uses efficient linear form: ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F).
    Returns value in [0, 1].
    """

    def score(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Args:
            X: (N, d) source representations
            Y: (N, d) target representations

        Returns:
            float CKA value in [0, 1]
        """
        X = X - X.mean(dim=0)
        Y = Y - Y.mean(dim=0)

        numerator = torch.linalg.norm(Y.T @ X, ord="fro") ** 2
        denom = (
            torch.linalg.norm(X.T @ X, ord="fro")
            * torch.linalg.norm(Y.T @ Y, ord="fro")
        )
        return float((numerator / denom.clamp(min=1e-9)).item())


def get_scorer(scorer_type: str):
    scorers = {
        "procrustes": ProcrustesSimilarityScorer(),
        "cosine": CosineSimilarityScorer(),
        "cka": CKAScorer(),
    }
    if scorer_type not in scorers:
        raise ValueError(f"Unknown scorer: {scorer_type}. Choose from {list(scorers)}")
    return scorers[scorer_type]
