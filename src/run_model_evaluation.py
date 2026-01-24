import pandas as pd
import matplotlib.pyplot as plt
import argparse

from tqdm import tqdm
from datetime import datetime

from metric_measurement_order_utils import compute_ndcg_at_k
from metric_measurement_pointwise_utils import compute_mae_error, compute_mse_metric, compute_r2_score
from metric_measurement_topk_utils import compute_f1_at_k, compute_hit_ratio_at_k, compute_precision_at_k, \
    compute_recall_at_k
from metric_measurement_set_utils import compute_jaccard_distance, compute_kendall_tau_distance, \
    compute_jaccard_similarity


def clean_inferenced_dataframe(inferenced_df: pd.DataFrame) -> pd.DataFrame:
    inferenced_df = inferenced_df.copy()
    # Remove rows with NaN values in 'predict_popularity_score'
    inferenced_df["predict_popularity_score"] = pd.to_numeric(inferenced_df["predict_popularity_score"],
                                                              errors='coerce')
    cleaned_inferenced_df = inferenced_df.dropna(subset=["predict_popularity_score"])
    # Convert 'predict_popularity_score' to numeric
    cleaned_inferenced_df["predict_popularity_score"] = pd.to_numeric(cleaned_inferenced_df["predict_popularity_score"],
                                                                      errors='coerce')
    cleaned_inferenced_df["predict_popularity_score"] = cleaned_inferenced_df["predict_popularity_score"].clip(lower=0)
    return cleaned_inferenced_df


def monthly_metric_calculation(inferenced_df, top_k_list) -> pd.DataFrame:
    monthly_metrics = []
    for month, month_df in inferenced_df.groupby('predict_timestamp'):
        for k in top_k_list:
            if len(month_df) < k:
                print(f"Skipping month {month} with insufficient data for k={k}.")
                continue
            metrics_dict = {
                "month": month,
                "k": k,
                "ndcg_at_k": compute_ndcg_at_k(month_df, k),
                "mae_error": compute_mae_error(month_df),
                "mse_error": compute_mse_metric(month_df)[0],
                "rmse_error": compute_mse_metric(month_df)[1],
                "r2_score": compute_r2_score(month_df),
                "f1_at_k": compute_f1_at_k(month_df, k),
                "hit_ratio_at_k": compute_hit_ratio_at_k(month_df, k),
                "precision_at_k": compute_precision_at_k(month_df, k),
                "recall_at_k": compute_recall_at_k(month_df, k)
            }
            monthly_metrics.append(metrics_dict)

    return pd.DataFrame(monthly_metrics)


def userwise_metric_calculation(inferenced_df: pd.DataFrame, user_interact_df, top_k_list) -> pd.DataFrame:
    user_interact_df = user_interact_df.copy()
    inferenced_df = inferenced_df.copy()
    user_interact_df["MovieID"] = user_interact_df["MovieID"].astype(str).str.strip()
    inferenced_df["movieid"] = inferenced_df["movieid"].astype(str).str.strip()

    # Convert Time column to datetime
    user_interact_df["Time"] = user_interact_df["Time"].astype(int)

    # Only use ratings BEFORE the prediction month as history
    predicted_month_timestamp = "2019-11-01 00:00:00"
    end_month_timestamp = "2019-12-01 00:00:00"
    dt_obj = datetime.strptime(predicted_month_timestamp, "%Y-%m-%d %H:%M:%S")
    predicted_month_unix_timestamp = dt_obj.timestamp()
    end_month_timestamp = datetime.strptime(end_month_timestamp, "%Y-%m-%d %H:%M:%S").timestamp()
    user_history_df = user_interact_df[user_interact_df["Time"] < predicted_month_unix_timestamp]
    user_current_df = user_interact_df[
        (user_interact_df["Time"] >= predicted_month_unix_timestamp) &
        (user_interact_df["Time"] < end_month_timestamp)
    ]

    global_topk_df = inferenced_df.sort_values(by='predict_popularity_score', ascending=False)
    user_metrics = []
    for user_id, user_df in tqdm(user_current_df.groupby('UserID'), total=user_current_df['UserID'].nunique(),
                                 desc="Userwise Eval"):
        # Get watched movie IDs for the user
        user_df_history = user_history_df[user_history_df["UserID"] == user_id]
        seen_movie_ids = set(user_df_history["MovieID"])

        user_rate_map = dict(zip(user_df["MovieID"], user_df["Rate"]))

        topk_movies = global_topk_df[~global_topk_df["movieid"].isin(seen_movie_ids)].copy()
        topk_movies["Rate"] = topk_movies["movieid"].map(user_rate_map).fillna(0)

        for k in top_k_list:
            eval_df = topk_movies.head(k)
            metrics_dict = {
                "user_id": user_id,
                "k": k,
                "precision_at_k": compute_precision_at_k(eval_df, k),
                "recall_at_k": compute_recall_at_k(eval_df, k),
                "f1_at_k": compute_f1_at_k(eval_df, k),
                "hit_ratio_at_k": compute_hit_ratio_at_k(eval_df, k),
                "ndcg_at_k": compute_ndcg_at_k(eval_df, user_df, k),
            }
            user_metrics.append(metrics_dict)
    # Compute pointwise metrics
    pointwise_metrics = {
        "mae_error": compute_mae_error(inferenced_df),
        "mse_error": compute_mse_metric(inferenced_df)[0],
        "rmse_error": compute_mse_metric(inferenced_df)[1],
        "r2_score": compute_r2_score(inferenced_df)
    }

    print(pointwise_metrics)
    return pd.DataFrame(user_metrics)


def amazon_userwise_metric_calculation(inferenced_df: pd.DataFrame, user_interact_df, top_k_list) -> pd.DataFrame:
    user_interact_df = user_interact_df.copy()
    inferenced_df = inferenced_df.copy()

    user_interact_df["item"] = user_interact_df["item"].astype(str).str.strip()
    inferenced_df["movieid"] = inferenced_df["movieid"].astype(str).str.strip()

    # Convert Time column to datetime
    user_interact_df["time"] = user_interact_df["time"].astype(int)

    # Only use ratings BEFORE the prediction month as history
    predicted_month = "2014-06-01 00:00:00"
    end_month = "2014-07-01 00:00:00"
    predicted_month_unix = int(datetime.strptime(predicted_month, "%Y-%m-%d %H:%M:%S").timestamp())
    end_month_unix = int(datetime.strptime(end_month, "%Y-%m-%d %H:%M:%S").timestamp())

    user_history_df = user_interact_df[user_interact_df["time"] < predicted_month_unix]
    user_current_df = user_interact_df[
        (user_interact_df["time"] >= predicted_month_unix) &
        (user_interact_df["time"] <= end_month_unix)
        ]

    global_topk_df = (
        inferenced_df
        .assign(movieid=lambda df: df["movieid"].astype(str))
        .sort_values(by='predict_popularity_score', ascending=False)
    )

    user_metrics = []
    for user_id, user_df in tqdm(user_current_df.groupby('user'), total=user_current_df['user'].nunique(),
                                 desc="Userwise Eval"):
        user_df_history = user_history_df[user_history_df["user"] == user_id]
        seen_item_ids = set(user_df_history["item"])

        user_df_local = user_df.copy()
        user_df_local["movieid"] = user_df_local["item"].astype(str)
        user_df_local["rating"] = pd.to_numeric(user_df_local["rating"], errors="coerce").fillna(0.0)
        user_rate_map = dict(zip(user_df_local["movieid"], user_df_local["rating"]))
        topk_movies = global_topk_df[~global_topk_df["movieid"].isin(seen_item_ids)].copy()
        topk_movies["Rate"] = topk_movies["movieid"].map(user_rate_map).fillna(0.0)

        for k in top_k_list:
            eval_df = topk_movies.head(k)
            metrics_dict = {
                "user_id": user_id,
                "k": k,
                "hit_ratio_at_k": compute_hit_ratio_at_k(eval_df, k),
                "ndcg_at_k": compute_ndcg_at_k(eval_df, user_df_local, k, "gt"),
            }
            user_metrics.append(metrics_dict)
    # Compute pointwise metrics
    pointwise_metrics = {
        "mae_error": compute_mae_error(inferenced_df),
        "mse_error": compute_mse_metric(inferenced_df)[0],
        "rmse_error": compute_mse_metric(inferenced_df)[1],
        "r2_score": compute_r2_score(inferenced_df)
    }

    print(pointwise_metrics)
    return pd.DataFrame(user_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation for the DeepSeek MovieRecLLM model.")
    parser.add_argument("--path", type=str, help="The filename of dataset.")
    parser.add_argument("--mode", type=str, choices=["douban", "baby", "beauty"], default="douban",
                        help="The evaluation mode of the model.")
    args = parser.parse_args()

    inferenced_df_file_name = f''

    if args.mode == "douban":
        # Load the inferenced results DataFrame
        inferenced_df = pd.read_csv(inferenced_df_file_name, encoding='latin1')
        top_k_list = [5, 10, 15, 20]

        # Clean the DataFrame
        cleaned_inferenced_df = clean_inferenced_dataframe(inferenced_df)

        user_interact_df = pd.read_csv("")
        # Map user ratings
        user_interact_df["Rate"] = user_interact_df["Rate"].astype(float)
        user_interact_df.loc[user_interact_df['Rate'] < 0, 'Rate'] = 1
        print("Starting evaluation...")
        user_metrics_df = userwise_metric_calculation(cleaned_inferenced_df, user_interact_df, top_k_list=top_k_list)
        metric_columns = [
            'precision_at_k', 'recall_at_k', 'f1_at_k', 'hit_ratio_at_k',
            'ndcg_at_k',
        ]
        macro_avg = user_metrics_df.groupby('k')[metric_columns].mean().reset_index()
        print(macro_avg[['k', 'hit_ratio_at_k', 'ndcg_at_k']])
        macro_avg.to_csv("1_douban_recommendation_metrics.csv", index=False)

        # Check ground truth top-K and inferenced top-K
        df1 = cleaned_inferenced_df.sort_values(by='predict_popularity_score', ascending=False)
        df2 = cleaned_inferenced_df.sort_values(by='ground_truth_popularity_score', ascending=False)
        df1_top = df1.head(20).copy().reset_index(drop=True)
        df2_top = df2.head(20).copy().reset_index(drop=True)

        df1_top["topK_predicted_rank"] = df1_top.index + 1
        df2_top["topK_ground_truth_rank"] = df2_top.index + 1

        setwise_metrics_dict_list = []
        for k in top_k_list:
            predicted_pop_top_k = df1.head(k)["movieid"].tolist()
            ground_truth_pop_top_k = df2.head(k)["movieid"].tolist()
            # calculate the setwise metrics
            jaccard_dist = compute_jaccard_similarity(set(predicted_pop_top_k), set(ground_truth_pop_top_k))
            setwise_metrics_dict = {
                "k": k,
                "jaccard_similarity": jaccard_dist
            }
            setwise_metrics_dict_list.append(setwise_metrics_dict)

        setwise_df = pd.DataFrame(setwise_metrics_dict_list)
        print(setwise_df)
        setwise_df.to_csv("1_setwise_metrics.csv", index=False)

        predicted_pop_top_k = df1.head(20)
        ground_truth_pop_top_k = df2.head(20)
        intersect_ids = set(predicted_pop_top_k["movieid"]) & set(ground_truth_pop_top_k["movieid"])

        results = []
        for _id in intersect_ids:
            pred_local_rank = df1_top[df1_top["movieid"] == _id]["topK_predicted_rank"].values[0]
            gt_local_rank = df2_top[df2_top["movieid"] == _id]["topK_ground_truth_rank"].values[0]
            pred_score = df1_top[df1_top["movieid"] == _id]["predict_popularity_score"].values[0]
            gt_score = df2_top[df2_top["movieid"] == _id]["ground_truth_popularity_score"].values[0]
            results.append({
                "movieid": _id,
                "predict_popularity_rank": pred_local_rank if pred_local_rank else '',
                "ground_truth_rank": gt_local_rank,
                "predict_popularity_score": pred_score,
                "ground_truth_popularity_score": gt_score
            })

        intersect_df = pd.DataFrame(results).sort_values(by="predict_popularity_rank")
        print(intersect_df)
        intersect_df.to_csv("1_ranking.csv", index=False)

    if args.mode in ["baby", "beauty"]:
        # Load the inferenced results DataFrame
        inferenced_df = pd.read_csv(inferenced_df_file_name, encoding='latin1')
        inferenced_df = inferenced_df.rename(columns={"item_id": "movieid"})
        top_k_list = [5, 10, 15, 20]

        # Clean the DataFrame
        cleaned_inferenced_df = clean_inferenced_dataframe(inferenced_df)
        if args.mode == "baby":
            user_interact_df = pd.read_csv("")
        if args.mode == "beauty":
            user_interact_df = pd.read_csv("")

        # Map user ratings
        user_interact_df["Rate"] = user_interact_df["rating"].astype(float)
        user_metrics_df = amazon_userwise_metric_calculation(
            cleaned_inferenced_df, user_interact_df, top_k_list=top_k_list
        )
        metric_columns = [
            'hit_ratio_at_k', 'ndcg_at_k',
        ]

        user_metrics_df.to_csv("user_metrics.csv")
        print(user_metrics_df.groupby('k').size())
        print(user_metrics_df.groupby('k')[["hit_ratio_at_k", "ndcg_at_k"]].sum())
        macro_avg = user_metrics_df.groupby('k')[metric_columns].mean().reset_index()
        macro_avg.to_csv(f"1_amazon_{args.mode}_evaluation_metrics.csv", index=False)
        print(macro_avg)

        df1 = cleaned_inferenced_df.sort_values(by='predict_popularity_score', ascending=False)
        df2 = cleaned_inferenced_df.sort_values(by='ground_truth_popularity_score', ascending=False)

        setwise_metrics_dict_list = []
        for k in top_k_list:
            predicted_pop_top_k = df1.head(k)["movieid"].tolist()
            ground_truth_pop_top_k = df2.head(k)["movieid"].tolist()
            # calculate the setwise metrics
            jaccard_dist = compute_jaccard_similarity(set(predicted_pop_top_k), set(ground_truth_pop_top_k))
            setwise_metrics_dict = {
                "k": k,
                "jaccard_similarity": jaccard_dist
            }
            setwise_metrics_dict_list.append(setwise_metrics_dict)
        setwise_df = pd.DataFrame(setwise_metrics_dict_list)
        print(setwise_df)
        df1_top = df1.head(20).copy().reset_index(drop=True)
        df2_top = df2.head(20).copy().reset_index(drop=True)

        print(df1_top[['movieid', 'predict_popularity_score', 'ground_truth_popularity_score']])
        print(df2_top[['movieid', 'predict_popularity_score', 'ground_truth_popularity_score']])

        predicted_pop_top_k = df1.head(20)
        ground_truth_pop_top_k = df2.head(20)
        intersect_ids = set(predicted_pop_top_k["movieid"]) & set(ground_truth_pop_top_k["movieid"])
        df1_top["topK_predicted_rank"] = df1_top.index + 1
        df2_top["topK_ground_truth_rank"] = df2_top.index + 1
        results = []
        for _id in intersect_ids:
            pred_local_rank = df1_top[df1_top["movieid"] == _id]["topK_predicted_rank"].values[0]
            gt_local_rank = df2_top[df2_top["movieid"] == _id]["topK_ground_truth_rank"].values[0]
            pred_score = df1_top[df1_top["movieid"] == _id]["predict_popularity_score"].values[0]
            gt_score = df2_top[df2_top["movieid"] == _id]["ground_truth_popularity_score"].values[0]
            results.append({
                "movieid": _id,
                "predict_popularity_rank": pred_local_rank if pred_local_rank else '',
                "ground_truth_rank": gt_local_rank,
                "predict_popularity_score": pred_score,
                "ground_truth_popularity_score": gt_score
            })

        intersect_df = pd.DataFrame(results).sort_values(by="predict_popularity_rank")
        print(intersect_df)