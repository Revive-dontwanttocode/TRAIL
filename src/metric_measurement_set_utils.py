from scipy.stats import kendalltau


def compute_jaccard_distance(set1: set, set2: set) -> float:
    """
    Calculate the Jaccard distance between two sets.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return 1.0 - len(intersection) / len(union)


def compute_jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculate the Jaccard similarity between two sets.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)


def compute_kendall_tau_distance(list1: list, list2: list) -> float:
    """
    Calculate the Kendall tau distance between two lists.
    """
    # Compute Kendall tau distance
    tau, _ = kendalltau(list1, list2)
    return (1.0 - tau) / 2.0
