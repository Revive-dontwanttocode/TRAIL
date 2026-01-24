import pandas as pd
import numpy as np


def compute_ndcg_at_k(inferenced_df: pd.DataFrame, user_interact_df: pd.DataFrame, K: int, mode="pred") -> float:
    """
    Compute NDCG@K for one single user.
    """
    inferenced_df = inferenced_df.copy()
    user_interact_df = user_interact_df.copy()

    # unify the column name
    if 'movieid' not in user_interact_df.columns and 'MovieID' in user_interact_df.columns:
        user_interact_df = user_interact_df.rename(columns={'MovieID': 'movieid'})

    # align the data types
    if 'movieid' in user_interact_df.columns:
        user_interact_df['movieid'] = user_interact_df['movieid'].astype(int)

    inferenced_df['movieid'] = inferenced_df['movieid'].astype(int)
    df = pd.merge(
        inferenced_df[[
            'movieid',
            'predict_popularity_score',
            'ground_truth_popularity_score']],
        user_interact_df[['movieid', 'Rate']],
        on='movieid',
        how='outer'
    )
    df['predict_popularity_score'] = df['predict_popularity_score'].fillna(-np.inf)
    df['ground_truth_popularity_score'] = df['ground_truth_popularity_score'].fillna(-np.inf)
    df['Rate'] = df['Rate'].fillna(0.0).astype(float)
    rel = (df['Rate'] > 0).astype(float).to_numpy()

    df_dcg = df.sort_values(by=['predict_popularity_score', 'movieid'],
                            ascending=[False, True])
    m = min(K, len(df_dcg))
    rel_dcg = rel[df_dcg.index][:K]
    discounts = np.log2(np.arange(2, K + 2))
    dcg = float(np.sum(rel_dcg / discounts)) if m > 0 else 0.0

    df_idcg = df.sort_values(by=['Rate', 'movieid'], ascending=[False, True])
    rel_idcg = (df_idcg['Rate'] > 0).astype(float).to_numpy()[:K]
    idcg = float(np.sum(rel_idcg / discounts)) if m > 0 else 0.0
    if idcg != 0.0:
        res = (dcg / idcg) / len(user_interact_df)
    else:
        res = 0.0
    return res
