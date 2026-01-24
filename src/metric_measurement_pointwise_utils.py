import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_mse_metric(inferenced_df: pd.DataFrame) -> tuple:
    """
    Compute Mean Squared Error for the given dataframe containing inferenced results.

    Mean Squared Error (MSE) is calculated as the average of the squared differences:
    """
    y_true = inferenced_df['ground_truth_popularity_score']
    y_pred = inferenced_df['predict_popularity_score']

    # if one element is none, set it to 0
    y_true = y_true.fillna(0)
    y_pred = y_pred.fillna(0)

    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    return mse, rmse


def compute_mae_error(inferenced_df: pd.DataFrame) -> float:
    """
    Compute Mean Absolute Error for the given dataframe containing inferenced results.

    Mean Absolute Error (MAE) is calculated as the average of the absolute differences:
    """
    y_true = inferenced_df['ground_truth_popularity_score']
    y_pred = inferenced_df['predict_popularity_score']

    # if one element is none, set it to 0
    y_true = y_true.fillna(0)
    y_pred = y_pred.fillna(0)

    mae = mean_absolute_error(y_true, y_pred)
    return mae


def compute_r2_score(inferenced_df: pd.DataFrame) -> float:
    """
    Compute R-squared score for the given dataframe containing inference results.

    R-squared score is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.
    """
    y_true = inferenced_df['ground_truth_popularity_score']
    y_pred = inferenced_df['predict_popularity_score']

    # if one element is none, set it to 0
    y_true = y_true.fillna(0)
    y_pred = y_pred.fillna(0)

    r2 = r2_score(y_true, y_pred)
    return r2
