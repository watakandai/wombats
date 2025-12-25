from typing import Hashable, List, Tuple, Iterable
import numpy as np


def compute_experts_feature(n_features: int, trajectories: np.ndarray) -> np.ndarray:
    """
    A function that computes the probability of each state feature.
    In the grid world case, the state is represented as a cell.
    In case of learning a formal language, the state could be a word.

    Arguments
    =========
    n_features: int
        Number of features
    trajectories: np.ndarray
        A list of trajectories over n steps. Each element is a feature

    Returns
    =======
    normalized_one_hot_trajectories: np.ndarray
        Probabilities over the features. Higher the probability, more frequently the expert has visited
    """
    if isinstance(trajectories, List):
        trajectories = np.array(trajectories)
    if not isinstance(trajectories, np.ndarray):
        raise TypeError('trajctories must be a np.ndarray')
    if trajectories.ndim != 2:
        raise ValueError('trajctories must be a np.ndarray of dimension 2')

    n_trajectories, n_steps = trajectories.shape
    print('n_trajectories: %i' % (n_trajectories))
    print('n_steps: %i' % (n_steps))

    def one_hot_encoder(array):
        ncols = n_features
        out = np.zeros((array.size, ncols))
        out[np.arange(array.size), array.ravel()] = 1
        out.shape = array.shape + (ncols,)
        return out

    one_hot_trajectories = one_hot_encoder(trajectories)
    assert one_hot_trajectories.shape == (n_trajectories, n_steps, n_features)

    normalized_one_hot_trajectories = one_hot_trajectories.sum(axis=(0, 1)) / n_trajectories
    assert normalized_one_hot_trajectories.shape == (n_features,)

    return normalized_one_hot_trajectories
