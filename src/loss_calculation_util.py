from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


########## Calculate DTW similarity ##########
def compute_trend_similarity(
        trend_i: list,
        trend_j: list,
        normalize: bool = False,
        alpha: float = 0.2
) -> float:
    """
    Compute DTW based similarity between two 1-Dimensional time series sequences.
    """

    def z_normalize(series: np.ndarray) -> np.ndarray:
        if len(series) == 0:
            return series
        mean = np.mean(series)
        std = np.std(series)
        return (series - mean) / std if std > 0 else series - mean

    # Convert list to numpy arrays
    series_i = np.array(trend_i, dtype=np.float32)
    series_j = np.array(trend_j, dtype=np.float32)
    # Check if the series are empty
    if series_i.size == 0 or series_j.size == 0:
        return 0.0  # Return zero similarity for empty series

    # Normalize if needed
    if normalize:
        series_i = z_normalize(series_i)
        series_j = z_normalize(series_j)

    scalar_dist = lambda a, b: abs(float(np.asarray(a).item()) - float(np.asarray(b).item()))
    # Compute DTW distance
    distance, _ = fastdtw(series_i, series_j, dist=scalar_dist)

    # Convert distance to similarity
    similarity = np.exp(-alpha * distance)
    return similarity


def compute_movie_description_similarity(descriptions: list[str], encoder: SentenceTransformer) -> np.ndarray:
    """
    Compute the similarity between movie descriptions. Directly use cosine similarity. And returns a similarity matrix.
    """
    # Encode the descriptions
    with torch.inference_mode():
        embeddings = encoder.encode(
            descriptions,
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    # Compute cosine similarity and form it into a matrix
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, -np.inf)
    return similarity_matrix


def compute_score_similarity(
        pred_score_i: float,
        pred_score_j: float,
        prev_trend_i: list,
        prev_trend_j: list,
        epsilon: float = 1e-8,
        sigma: float = 0.2,
) -> float:
    """
    Compute the similarity between two predicted scores based on their previous trends.

    Will return a scalar value representing the similarity.
    """
    if pred_score_i == -1 or pred_score_j == -1:
        return 0.0

    last_score_i = prev_trend_i[-1] if len(prev_trend_i) > 0 else 0.0
    last_score_j = prev_trend_j[-1] if len(prev_trend_j) > 0 else 0.0

    # Calculate rate of change
    rate_i = (pred_score_i - last_score_i) / (last_score_i + epsilon)
    rate_j = (pred_score_j - last_score_j) / (last_score_j + epsilon)

    # Gaussian similarity
    rate_diff = np.abs(rate_i - rate_j)
    similarity = np.exp(-rate_diff ** 2 / (2 * (sigma ** 2)))

    return float(similarity)
