from typing import Text
import numpy as np


def train_test_split(L: int, train_ratio: float=0.8):
    """Split a dataset, return train_indices and test_indices."""
    indices = np.arange(L)
    np.random.shuffle(indices)
    indices = indices.tolist()

    train_size = int(len(indices) * train_ratio)
    return indices[:train_size], indices[train_size:]
