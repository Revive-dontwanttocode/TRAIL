"""
Evaluate the model performance based on the provided evaluation results.

The metric formulas are the shared ones of this project in `src/evaluation`,
and every user gets the same recommendation list without removing the
items the user has seen, which is the evaluation protocol of this project.

Author: Zhixi Chen
Date: 2025-06-18
"""


import os

import numpy as np
import pandas as pd

from src.common.config import project_path, write_json
from src.evaluation.metrics import metrics

from .data_prepare_utils import resolve_trend


def clean_inferenced_dataframe(inferenced_df: pd.DataFrame) -> pd.DataFrame:
    inferenced_df = inferenced_df.copy()
    # Remove rows with NaN values in 'predict_popularity_score'
    inferenced_df["predict_popularity_score"] = pd.to_numeric(inferenced_df["predict_popularity_score"], errors='coerce')
    cleaned_inferenced_df = inferenced_df.dropna(subset=["predict_popularity_score"])
    # Convert 'predict_popularity_score' to numeric
    cleaned_inferenced_df["predict_popularity_score"] = pd.to_numeric(cleaned_inferenced_df["predict_popularity_score"], errors='coerce')
    cleaned_inferenced_df["predict_popularity_score"] = cleaned_inferenced_df["predict_popularity_score"].clip(lower=0)
    return cleaned_inferenced_df


def construct_last_popularity_dataframe(inferenced_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the training-free control: the predicted score of an item is its last observed popularity score.
    An item without historical popularity trend gets 0.
    """
    def last_popularity_score(historical_popularity_trends):
        trend_values = resolve_trend(historical_popularity_trends) if isinstance(historical_popularity_trends, str) else []
        return trend_values[-1] if trend_values else 0.0

    last_popularity_df = inferenced_df.copy()
    last_popularity_df["predict_popularity_score"] = last_popularity_df["historical_popularity_trends"].apply(last_popularity_score)
    return last_popularity_df


def sort_by_predicted_score(inferenced_df: pd.DataFrame, item_column: str) -> pd.DataFrame:
    # the items with the same predicted score are ordered by their item id, like the other models of this project
    return inferenced_df.sort_values(by=['predict_popularity_score', item_column], ascending=[False, True])


def save_ranking(inferenced_df: pd.DataFrame, output_dir, item_column: str):
    """
    Save the item ranking and the predicted scores, so that the metrics can be checked later.
    """
    global_topk_df = sort_by_predicted_score(inferenced_df, item_column)
    items = global_topk_df[item_column].tolist()
    scores = global_topk_df["predict_popularity_score"].astype(float).tolist()
    ranking = []
    for rank, (item, score) in enumerate(zip(items, scores), start=1):
        ranking.append({"rank": rank, item_column: item, "predict_popularity_score": score})
    write_json(os.path.join(output_dir, "ranking.json"), ranking)
    np.savez_compressed(os.path.join(output_dir, "scores.npz"), items=np.asarray(items), scores=np.asarray(scores))


def userwise_metric_calculation(
    inferenced_df: pd.DataFrame,
    user_interact_df: pd.DataFrame,
    top_k_list: list,
    item_column: str,
    user_column: str,
    interacted_item_column: str,
    output_dir,
) -> dict:
    """
    Compute the recommendation metrics of the users in `user_interact_df`.

    Args:
        inferenced_df (pd.DataFrame): The cleaned inference results with the item id and `predict_popularity_score`.
        user_interact_df (pd.DataFrame): The interactions of the prediction period, one row for one user and one item.
        top_k_list (list): The cutoffs of the metrics.
    Returns:
        dict: The metrics averaged over the users.
    """
    user_interact_df = user_interact_df.copy()
    inferenced_df = inferenced_df.copy()

    global_topk_df = sort_by_predicted_score(inferenced_df, item_column)
    user_interact_df[interacted_item_column] = user_interact_df[interacted_item_column].astype(str).str.strip()
    global_topk_items = global_topk_df[item_column].astype(str).str.strip().tolist()[:max(top_k_list)]

    user_ids = []
    user_targets = []
    for user_id, user_df in user_interact_df.groupby(user_column):
        user_ids.append(user_id)
        user_targets.append(set(user_df[interacted_item_column]))

    # all the users get the same recommendation list
    user_rankings = [global_topk_items for _ in user_ids]
    user_metrics = metrics(user_targets, user_rankings, top_k_list)
    np.savez_compressed(os.path.join(output_dir, "rankings.npz"), users=np.asarray(user_ids), rankings=np.asarray(user_rankings))
    user_metrics["users"] = len(user_ids)
    return user_metrics


def save_inference_results(result_df: pd.DataFrame, output_dir, item_column: str, evaluation_config: dict) -> dict:
    """
    Save the inferenced dataframe, its ranking and, if the user interactions are given, its recommendation metrics.

    Returns:
        dict: The numbers of candidates and the metrics, which are recorded in `results.json`.
    """
    os.makedirs(output_dir, exist_ok=True)
    result_df.to_csv(os.path.join(output_dir, "inference_result.csv"), index=False)
    cleaned_inferenced_df = clean_inferenced_dataframe(result_df)
    summary = {
        "candidates": len(result_df),
        # the candidates without a numeric predicted score are not ranked
        "failed_candidates": len(result_df) - len(cleaned_inferenced_df),
        "metrics": None,
    }

    save_ranking(cleaned_inferenced_df, output_dir, item_column)
    if evaluation_config["user_interaction_path"]:
        user_interact_df = pd.read_csv(
            project_path(evaluation_config["user_interaction_path"]),
            dtype={evaluation_config["item_column"]: str},
        )
        summary["metrics"] = userwise_metric_calculation(
            cleaned_inferenced_df,
            user_interact_df,
            top_k_list=evaluation_config["top_k_list"],
            item_column=item_column,
            user_column=evaluation_config["user_column"],
            interacted_item_column=evaluation_config["item_column"],
            output_dir=output_dir,
        )
    return summary
