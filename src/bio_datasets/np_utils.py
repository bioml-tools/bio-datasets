from typing import List

import numpy as np


def map_categories_to_indices(arr: np.ndarray, categories: List[str]) -> np.ndarray:
    """
    Map categories to indices.

    Args:
        arr (np.ndarray): The input array of categories.
        categories (List[str]): The list of categories to map to indices.

    Returns:
        np.ndarray: An array of indices corresponding to the categories.
    """
    unique_categories, indices = np.unique(arr, return_inverse=True)
    category_indices = np.array([categories.index(i) for i in unique_categories])
    return category_indices[indices]


def unique_topk(arr: np.ndarray, k: int, return_counts: bool = False) -> np.ndarray:
    """Return the k largest unique values in the array."""
    unique_vals, unique_counts = np.unique(
        arr, return_counts=True
    )  # might already be sorted?
    assert len(unique_vals) >= k, "Not enough unique values to return top k"
    highest_count_indices = np.argsort(unique_vals)[-k:]
    if return_counts:
        return unique_vals[highest_count_indices], unique_counts[highest_count_indices]
    return unique_vals[highest_count_indices]
