import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_precision_at_k(inferenced_df: pd.DataFrame, K: int) -> float:
    inferenced_df = inferenced_df.copy()  # Avoid modifying the original DataFrame

    ##### Construct binary classification columns for top K items #####
    topk_df = inferenced_df.sort_values(by='predict_popularity_score', ascending=False).head(K)
    # Define relevance using Rate > 0
    num_relevant_in_topk = (topk_df["Rate"] > 0).sum()
    precision = num_relevant_in_topk / K
    return precision


def compute_recall_at_k(inferenced_df: pd.DataFrame, K: int) -> float:
    inferenced_df = inferenced_df.copy()  # Avoid modifying the original DataFrame

    ##### Construct binary classification columns for top K items #####
    relevant_items = set(inferenced_df[inferenced_df["Rate"] >= 3]["movieid"])
    topk_items = set(
        inferenced_df.sort_values(by='predict_popularity_score', ascending=False).head(K)["movieid"]
    )
    # Hits
    num_hits = len(relevant_items & topk_items)
    recall = num_hits / max(1, len(relevant_items))  # Avoid division by zero
    return recall


def compute_hit_ratio_at_k(inferenced_df: pd.DataFrame, K: int) -> float:
    inferenced_df = inferenced_df.copy()  # Avoid modifying the original DataFrame

    return 1.0 if (inferenced_df[
                       "Rate"] > 0).any() else 0.0  # change to 0 so that if one user has watched a movie, we see it as hit.


def compute_f1_at_k(inferenced_df: pd.DataFrame, K: int) -> float:
    inferenced_df = inferenced_df.copy()  # Avoid modifying the original DataFrame

    ##### Construct binary classification columns for top K items #####
    relevant_items = set(inferenced_df[inferenced_df["Rate"] >= 3]["movieid"])
    topk_items = set(
        inferenced_df.sort_values("predict_popularity_score", ascending=False)
        .head(K)["movieid"]
    )
    num_hits = len(topk_items & relevant_items)
    precision = num_hits / K if K > 0 else 0
    recall = num_hits / max(1, len(relevant_items))  # Avoid division by zero
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1
