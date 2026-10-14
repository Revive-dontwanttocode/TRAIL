"""
Teacher-forced token metrics for causal language models.
"""

import numpy as np


def causal_lm_token_metrics(predicted_token_ids, labels):
    """Compare next-token predictions, excluding targets masked with -100."""
    predictions = np.asarray(predicted_token_ids)
    labels = np.asarray(labels)
    if predictions.ndim != 2 or predictions.shape != labels.shape:
        raise ValueError("Predicted token IDs and labels must have matching [batch, sequence] shapes.")

    # A causal LM's logits at t predict the label at t+1, as in its CE loss.
    predictions = predictions[:, :-1]
    targets = labels[:, 1:]
    valid = targets != -100
    supervised_tokens = int(valid.sum())
    if supervised_tokens == 0:
        raise ValueError("Token accuracy requires at least one supervised next-token target.")

    return {
        "token_accuracy": float(np.mean(predictions[valid] == targets[valid])),
        "supervised_tokens": supervised_tokens,
    }
