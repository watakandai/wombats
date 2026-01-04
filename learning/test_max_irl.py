import numpy as np
import pytest
from typing import Tuple, List
import warnings

Cell = Tuple[int, int]

from .max_irl import compute_experts_feature

class Grid:
    def __init__(self, n_grid: int):
        self.n_grid = n_grid
        self.n_state = n_grid * n_grid
        self.start = (0, 0)
        self.goal = (n_grid-1, n_grid-1)

    def check_cell(self, cell: Cell) -> bool:
        if len(cell) != 2:
            warnings.warn('Cell must be a tuple of size 2')
            return False

        for i in (0, 1):
            if cell[i] < 0 or self.n_grid-1 < cell[i]:
                warnings.warn(f"Cel must be within the range of [1, {self.n_grid}]")
                return False
        return True

    def set_start(self, start: Cell) -> None:
        if self.check_cell(start):
            self.start = start

    def set_goal(self, goal: Cell) -> None:
        if self.check_cell(goal):
            self.goal = goal

    def to_index(self, cell: Cell) -> int:
        return int(np.ravel_multi_index(cell, (self.n_grid, self.n_grid)))

    def to_cell(self, index: int) -> Cell:
        res = np.unravel_index(index, (self.n_grid, self.n_grid))
        return (int(res[0]), int(res[1]))

    def to_indices(self, cells: List[Cell]) -> List[int]:
        return list(map(self.to_index, cells))

    def to_cells(self, indices: List[int]) -> List[Cell]:
        return list(map(self.to_cell, indices))

    def sample_indices(self, n_step: int) -> List[int]:
        pass

    def sample_cells(self, n_step: int) -> List[Cell]:
        pass


def test_compute_experts_cell():
    """
    Test whether it can take trajectories of cell states.
    """
    # Precondition
    n_grid = 3
    grid = Grid(n_grid)
    n_features = grid.n_state

    trajectories = [
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
    ]

    # Convert cell trajectories to index trajectories
    trajectories_idx = np.array([grid.to_indices(traj) for traj in trajectories])

    # Under Test
    normalized_one_hot_trajectories = compute_experts_feature(n_features, trajectories_idx)

    # Postcondition
    assert normalized_one_hot_trajectories.shape == (n_features, )


def test_compute_experts_strings():
    """
    Test whether it can take trajectories of strings.
    """
    # Precondition
    trajectories = [
        ['empty_red_open', 'empty_red_open', 'empty_red_open', 'empty_red_open', 'empty_red_open', 'empty_red_open', 'floor_green_open'],
        ['empty_red_open', 'empty_red_open', 'empty_red_open', 'empty_red_open', 'carpet_yellow_open', 'empty_red_open', 'floor_green_open']
    ]
    n_features = len(set([item for traj in trajectories for item in traj]))
    # Assign unique index number to each string
    unique_elements = sorted(set(item for traj in trajectories for item in traj))
    mapping = {val: i for i, val in enumerate(unique_elements)}
    trajectories = np.array([[mapping[item] for item in traj] for traj in trajectories])

    # Under Test
    normalized_one_hot_trajectories = compute_experts_feature(n_features, trajectories)

    # Postcondition
    assert normalized_one_hot_trajectories.shape == (n_features, )
