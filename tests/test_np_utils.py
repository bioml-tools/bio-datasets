import numpy as np
import pytest

from bio_datasets.np_utils import map_categories_to_indices, unique_topk


def test_map_categories_to_indices():
    categories = ["A", "B", "C"]
    arr = np.array(["A", "B", "C", "B", "C", "A"])
    assert np.all(
        map_categories_to_indices(arr, categories) == np.array([0, 1, 2, 1, 2, 0])
    )


@pytest.mark.parametrize(
    "arr,expected_arr",
    [
        (np.array([1, 2, 2, 3, 3, 3, 5, 8, 9]), np.array([8, 9])),
        (np.array([3, 2, 2, 1, 1, 1, 4, 4, 4, 4]), np.array([3, 4])),
    ],
)
def test_numpy_topk(arr, expected_arr):
    tk = unique_topk(arr, k=2)
    assert np.all(tk == expected_arr)
