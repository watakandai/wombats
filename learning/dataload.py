import copy
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Callable
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

from wombats.automaton.pdfa import PDFA
from wombats.automaton.product import Product
from wombats.automaton.types import Symbols, Probabilities


def split_dict(functions: List[Callable],
               dictionary: Dict) -> List[Dict]:
    """
    This function analyzes each function's possible arguments,
    and automatically splits kwargs (a dict) into a list of dict.
    """
    # First analyze all possible arguments for each function
    varnames_list = []
    for func in functions:
        varnames = func.__code__.co_varnames
        varnames_list.append(varnames)

    # Split the kwargs / dictionary into a list of dict
    split_dictionary = []
    for varnames in varnames_list:
        partial_dictionary = {}
        for k, v in dictionary.items():
            if k in varnames:
                partial_dictionary.update({k:v})
        split_dictionary.append(partial_dictionary)

    return split_dictionary


class Dataset:
    """
    This data class contains a generated dataset.
    """
    def __init__(self,
                 X_train: List[Symbols],
                 X_test: List[Symbols] = None,
                 y_test: Probabilities = None,
                 params: Dict = {},
                 specification: PDFA = None):
        """
        Dataset is consisted of main data X alongside with
        a model and parameters used to generate the data.

        :param X_train:         A list of traces for training
        :param X_test:          A list of traces for testing
        :param y_test:          A list of traces for testing
        :param params:          A dict of parameters
        :param specification:   A PDFA Specification
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.params = params
        self.specification = specification


class TracesGenerator():
    """
    This class has 2 responsibilities
        1. Data Generation
            It is responsible to generate possibly ALL data we want to analyze
        2. Data Selection
            Once data is generated, a subset of data can be selected
    """

    def __init__(self,
                 specification: Union[PDFA, Product]) -> None:
        """
        Given a PDFA specification, this class generates a set
        of multiple datasets

        :param specification:   A PDFA specification
        """
        # private variables
        self._specification = specification
        self._max_num_sample = None
        self._df = None

    def generate_datasets(self,
                          param_grid: Dict[str, List] = None,
                          get_help: bool = False,
                          use_subset: bool = True,
                          verbose: int = 0,
                          test_size: float = 0.2,
                          **kwargs) -> List[Dataset]:
        """
        Generate the largest No. of samples
        Will retrieve subset of the samples from the generated samples

        :param param_grid:      Variable parameters to pass to
                                PDFA.generate_traces() function
        :param get_help:        Get help on how to use this function
        :param use_subset:      Whether to use subset of the data
                                sampled from max num sample If true,
                                it would be faster
        :param **kwargs:        Constant parameters to pass

        :returns None:
        """
        # --help
        if get_help or not isinstance(param_grid, Dict):
            print('Please provide a parameter grid that is going '\
                + 'to be parsed to PDFA.generate_traces() function')
            return

        # Error Checks
        required_keys = ['num_samples', 'N']
        if any(k not in param_grid for k in required_keys) or \
                any(len(param_grid[k]) == 0 for k in required_keys):
            raise ValueError("Please provide at least 1 element for {}"
                             .format(required_keys))

        param_grids = self.__split_param_grids(param_grid, use_subset)

        rows = []
        for sample_gen_params in param_grids[0]:
            # Generate samples
            traces, _, probs = \
                self._specification.generate_traces(
                    **sample_gen_params, verbose=verbose)

            self._specification.refit_prob_dist(traces)

            # Select subset
            for choose_subset_params in param_grids[1]:
                if not use_subset:
                    d = {'num_samples':
                        sample_gen_params['num_samples']}
                    choose_subset_params = \
                        choose_subset_params.update(d)

                X, _ = self.choose_subset_from_samples(
                     traces, probs, **choose_subset_params)

                test_params = choose_subset_params.copy()
                test_params.update({'num_samples':
                    test_size * test_params['num_samples']})

                X_test, y_test = self.choose_subset_from_samples(
                    traces, probs, select_method='random',
                    **test_params)

                # Store the parameters and resulting data into a pandas dataframe
                # print(sample_gen_params, choose_subset_params)
                params = {**sample_gen_params, **choose_subset_params}

                dataset = Dataset(
                    X_train=X,
                    X_test=X_test,
                    y_test=y_test,
                    params=params,
                    specification=copy.deepcopy(self._specification))

                rows.append({'dataset': dataset, **params})

        self._df = pd.DataFrame(rows)
        return self._df['dataset'].values.tolist()

    def __split_param_grids(self, param_grid,
                            use_subset: bool = True):
        """
        Split parameters between multiple functions

        :param param_grid:      Variable parameters to pass to
                                PDFA.generate_traces() function
        :param use_subset:      Whether to use subset of the data
                                sampled from max num sample If true,
                                it would be faster

        :returns:               A list of split parameters
        """

        # Keep num_samples away from param_grid before splitting
        # param_grid and kwargs
        num_samples = param_grid['num_samples']
        num_samples = set([int(num) for num in num_samples])
        self._max_num_sample = max(num_samples)
        del param_grid['num_samples']

        # Split variables and constants to parse to two functions
        steps = [self._specification.generate_traces,
                 self.choose_subset_from_samples]
        split_params = split_dict(steps, param_grid)

        if use_subset:
            # Only generate samples of maximum num_samples
            split_params[0].update({'num_samples': [self._max_num_sample]})
            # and select subsets of the generated samples
            split_params[1].update({'num_samples': num_samples})
        else:
            # Generate samples for all given num_samples
            split_params[0].update({'num_samples': num_samples})

        sample_gen_param_grids = list(ParameterGrid(split_params[0]))
        choose_subset_param_grids = list(ParameterGrid(split_params[1]))

        return [sample_gen_param_grids, choose_subset_param_grids]

    def choose_subset_from_samples(self,
                                   traces: List[Symbols],
                                   probs: Probabilities,
                                   num_samples: int,
                                   select_method='order') -> np.ndarray:
        """
        Given traces, this function chooses a subset from the set
        randomly or in order.

        :param traces:              A list of traces
        :param num_samples:         A number of samples to select
        :param select_method:       Random or in order

        :return:                    A subset of traces
        """
        if probs and len(traces) != len(probs):
            msg = 'Length of traces and probs must be same'
            raise ValueError(msg)

        # check errors
        max_num_sample = int(len(traces))
        num_samples = int(num_samples)

        if max_num_sample < num_samples:
            num_samples = max_num_sample
            msg = f'Length of traces {max_num_sample} must be' + \
                  f'larger than num_samples {num_samples}'
            warnings.warn(msg)

        # choose subset depending on the selected method
        if select_method == 'order':
            indices = np.arange(num_samples).astype(int)
        elif select_method == 'random':
            indices = np.random.randint(max_num_sample, size=num_samples)
        else:
            raise ValueError("method must be either 'order' or 'random'. ")

        return np.array(traces, dtype=object)[indices].tolist(),\
            np.array(probs, dtype=object)[indices].tolist()

    def select_datasets(self, param_grid: Dict[str, Any],
                        **kwargs) -> pd.DataFrame:
        """
        Provide a list of possible parameters,
        loop over those possible combinations,
        and return a nested list in the same
        order as the params were provided.

        :param params:              A dictionary of parameters
        :param selected_method:     Select Method for sampling

        :return:                    pd.DataFrame
        """
        if self._df is None:
            self.generate_datasets(param_grid, **kwargs)

        param_grid = list(ParameterGrid(param_grid))
        dfs = []

        for param_dict in param_grid:
            query = ' and '.join([f'{k} == {repr(v)}' for k, v in param_dict.items()])
            selected_df = self._df.query(query)
            dfs.append(selected_df)

        return pd.concat(dfs)['dataset'].values.tolist()

    # Getters

    @property
    def max_num_sample(self) -> int:
        return self._max_num_sample

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df
