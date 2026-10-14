"""
Ranking metrics shared by every model: HR, Reference NDCG, Recall, Precision.

Pure functions of targets and rankings. No model, data loading or torch here.
Reference NDCG is the PARE author convention: a user's NDCG divided by their target count.
"""

import math


def ranking_metrics(relevant, ranking, k):
    """Metrics@k for one user."""
    relevant = set(relevant)
    if not relevant or k < 1:
        raise ValueError("Ranking metrics require positives and k >= 1")
    hits = [int(item in relevant) for item in ranking[:k]]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return {"hr": float(any(hits)), "ndcg": dcg / ideal,
            "recall": sum(hits) / len(relevant), "precision": sum(hits) / k}


def metrics(targets, rankings, cutoffs):
    """User-averaged metrics at every cutoff, plus the number of users with a hit."""
    if not targets or len(targets) != len(rankings):
        raise ValueError("Targets/rankings must contain the same nonzero number of users")
    result, hit_users = {}, {}
    for k in cutoffs:
        standard, reference, recall, precision, hits = [], [], [], [], 0
        for relevant, ranking in zip(targets, rankings):
            if not relevant or len(ranking) < k or len(set(ranking)) != len(ranking):
                raise ValueError("Nonempty targets and a unique ranking covering every cutoff are required")
            gains = [1.0 / math.log2(j + 2) for j, i in enumerate(ranking[:k]) if i in relevant]
            ndcg = math.fsum(gains) / math.fsum(1.0 / math.log2(j + 2) for j in range(min(k, len(relevant))))
            hits += bool(gains)
            standard.append(ndcg)
            reference.append(ndcg / len(relevant))
            recall.append(len(gains) / len(relevant))
            precision.append(len(gains) / k)
        hit_users[f"@{k}"] = hits
        result.update({f"hr@{k}": hits / len(targets), f"ndcg@{k}": math.fsum(standard) / len(targets),
                       f"reference_ndcg@{k}": math.fsum(reference) / len(targets),
                       f"recall@{k}": math.fsum(recall) / len(targets),
                       f"precision@{k}": math.fsum(precision) / len(targets)})
    return dict(result, hit_users=hit_users)
