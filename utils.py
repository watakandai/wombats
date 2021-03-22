import heapq
import os
import numpy as np
from wombats.systems.minigrid import GYM_MONITOR_LOG_DIR_NAME


def get_experiment_paths(EXPERIMENT_NAME: str):

    # setting all paths
    EXPERIMENTS_BASE = 'experiments'
    EXPERIMENT_DIR = os.path.join(EXPERIMENTS_BASE, EXPERIMENT_NAME)

    LEARNER_DATA_DIR_NAME = 'flexfringe_data'
    LEARNER_DATA_DIR = os.path.join(EXPERIMENT_DIR, LEARNER_DATA_DIR_NAME)
    LEARNING_TRAIN_DATA_NAME = EXPERIMENT_NAME + '_train'
    LEARNING_TEST_DATA_NAME = EXPERIMENT_NAME + '_test'
    LEARNING_TRAIN_DATA_REL_FILEPATH = os.path.join(LEARNER_DATA_DIR_NAME,
                                                    LEARNING_TRAIN_DATA_NAME)
    LEARNING_TEST_DATA_REL_FILEPATH = os.path.join(LEARNER_DATA_DIR_NAME,
                                                   LEARNING_TEST_DATA_NAME)

    ANALYSIS_DATA_DIR_NAME = 'analysis'
    ANALYSIS_DATA_DIR = os.path.join(EXPERIMENT_DIR, ANALYSIS_DATA_DIR_NAME)

    LIB_BASE_NAME = 'wombats'
    LIB_CONFIG_NAME = 'config'
    LIB_CONFIG_DIR = os.path.join(LIB_BASE_NAME, LIB_CONFIG_NAME)

    PDFA_MODEL_CONFIG_FILE = os.path.join(LIB_CONFIG_DIR,
                                          'PDFA_' + EXPERIMENT_NAME + '.yaml')
    DFA_MODEL_CONFIG_FILE = os.path.join(LIB_CONFIG_DIR,
                                         'DFA_' + EXPERIMENT_NAME + '.yaml')
    TS_MODEL_CONFIG_FILE = os.path.join(LIB_CONFIG_DIR,
                                        'TS_' + EXPERIMENT_NAME + '.yaml')

    GYM_MONITOR_LOG_DIR = os.path.join(EXPERIMENT_DIR,
                                       GYM_MONITOR_LOG_DIR_NAME)

    # put it all into a dictionary so we don't need to change interface for
    # new path I/O
    path_data = {
        'EXPERIMENT_DIR': EXPERIMENT_DIR,
        'PDFA_MODEL_CONFIG_FILE': PDFA_MODEL_CONFIG_FILE,
        'DFA_MODEL_CONFIG_FILE': DFA_MODEL_CONFIG_FILE,
        'TS_MODEL_CONFIG_FILE': TS_MODEL_CONFIG_FILE,
        'GYM_MONITOR_LOG_DIR': GYM_MONITOR_LOG_DIR,
        'LEARNER_DATA_DIR': LEARNER_DATA_DIR,
        'LEARNING_TRAIN_DATA_REL_FILEPATH': LEARNING_TRAIN_DATA_REL_FILEPATH,
        'LEARNING_TEST_DATA_REL_FILEPATH': LEARNING_TEST_DATA_REL_FILEPATH,
        'ANALYSIS_DATA_DIR': ANALYSIS_DATA_DIR}

    return path_data


class MaxHeapObj(object):
    """
    Overrides the comparison, so you can create a max heap easily
    See https://stackoverflow.com/a/40455775
    """

    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val > other.val

    def __eq__(self, other):
        return self.val == other.val

    def __str__(self):
        return str(self.val)


class MinHeap(object):
    """
    A nice class-based interface to the heapq library
    See https://stackoverflow.com/a/40455775
    """

    def __init__(self):
        self.h = []

    def heappush(self, x):
        heapq.heappush(self.h, x)

    def heappop(self):
        return heapq.heappop(self.h)

    def __getitem__(self, i):
        return self.h[i]

    def __len__(self):
        return len(self.h)


class MaxHeap(MinHeap):
    """
    A nice class-based interface to create a max heap, using the heapq lib.
    See https://stackoverflow.com/a/40455775
    """

    def heappush(self, x):
        heapq.heappush(self.h, MaxHeapObj(x))

    def heappop(self):
        return heapq.heappop(self.h).val

    def __getitem__(self, i):
        return self.h[i].val

def logx(x, base=2):
    return np.asscalar(np.log(x) / np.log(base))

def xlogx(x, **kwargs):
    return ylogx(x, x, **kwargs)

def xlogy(x, y, **kwargs):
    if isinstance(x, float) and x == 0.0:
        return 0.0
    if isinstance(x, int) and x == 0:
        return 0

    return x * logx(y,  **kwargs)

def ylogx(x, y,  **kwargs):
    if isinstance(y, float) and y == 0.0:
        return 0.0
    if isinstance(y, int) and y == 0:
        return 0

    return y * logx(x,  **kwargs)
