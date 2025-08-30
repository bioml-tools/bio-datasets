from typing import List

import numpy as np


def map_categories_to_indices(arr: np.ndarray, categories: List[str]) -> np.ndarray:
    """Map categories to indices.

    Args:
        arr (np.ndarray): The input array of categories.
        categories (List[str]): The list of categories to map to indices.

    Returns:
        np.ndarray: An array of indices corresponding to the categories.
    """
    unique_categories, indices = np.unique(arr, return_inverse=True)
    category_indices = np.array([categories.index(i) for i in unique_categories])
    return category_indices[indices]
