import copy
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Callable
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

from wombats.automaton.pdfa import PDFA
from wombats.automaton.product import Product
from wombats.automaton.types import (
    Symbols,
    Probabilities,
    GeneratedTraceData
)


def split_dict(functions: List[Callable], dictionary: Dict) -> List[Dict]:
    # First extract all possible variable names
    varnames_list = []
    for func in functions:
        varnames = func.__code__.co_varnames
        varnames_list.append(varnames)

    split_dictionary = []
    for varnames in varnames_list:
        partial_dictionary = {}
        for k, v in dictionary.items():
            if k in varnames:
                partial_dictionary.update({k:v})
        split_dictionary.append(partial_dictionary)

    return split_dictionary


class Dataset:
    def __init__(self,
                 X_train: Symbols,
                 X_test: Symbols,
                 y_train: Probabilities,
                 y_test: Probabilities,
                 length_train: List[int],
                 length_test: List[int],
                 output_directory: str = './'):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.length_train = length_train
        self.length_test = length_test
        self.output_directory = output_directory
        self._params = None
        self._pdfa = None

    def set_params(self, params):
        self._params = params

    def set_pdfa(self, pdfa):
        self._pdfa = pdfa

    @property
    def params(self):
        if self._params is None:
            raise ValueError('params are not set')
        return self._params

    @property
    def pdfa(self):
        if self._pdfa is None:
            raise ValueError('pdfa is not set')
        return self._pdfa


class PDFADataLoader():
    """
    This class has 2 possibilities
        1. Data Generation
            It is responsible to generate possibly ALL data we want to analyze
        2. Data Load (Selection)
            It is responsible to pick one/multiple sets of data from the
            generated data
    """

    def __init__(self,
                 specification: Union[PDFA, Product],
                 output_directory: str = './') -> None:
        """
        Provide a PDFA specification,
        generate a set of multiple datasets,
        and export to the output_directory
        """
        # private variables
        self._specification = specification
        self._output_directory = output_directory
        self._df = None

    def generate_datasets(self,
                          param_grid: Dict[str, List] = None,
                          get_help: bool = False,
                          use_subset_of_max_num_sample: bool = True,
                          verbose: int = 0,
                          **kwargs) -> None:
        """
        Generate the largest No. of samples
        Will retrieve subset of the samples from the generated samples

        :param param_grid:                      Variable parameters to pass to
                                                PDFA.generate_traces() function
        :param get_help:                        Get help on how to use this
                                                function
        :param use_subset_of_max_num_sample:    Whether to use subset of the data
                                                sampled from max num sample
                                                If true, it would be faster
        :param **kwargs:                        Constant parameters to pass

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

        # Given parameters X, generate traces Y
        # and save both of them as a row of a pandas dataframe
        param_keys = list(param_grid.keys())

        # Keep num_samples away from param_grid before splitting
        # param_grid and kwargs
        num_samples = param_grid['num_samples']
        self._max_num_sample = max(num_samples)
        del param_grid['num_samples']

        # Split variables and constants to parse to two functions
        steps = [self._specification.generate_traces,
                 self.choose_subset_from_samples]
        split_params = split_dict(steps, param_grid)
        split_kwargs = split_dict(steps, kwargs)

        if use_subset_of_max_num_sample:
            # Only generate samples of maximum num_samples
            split_params[0].update({'num_samples': [self._max_num_sample]})
            # and select subsets of the generated samples
            split_params[1].update({'num_samples': num_samples})
        else:
            # Generate samples for all given num_samples
            split_params[0].update({'num_samples': num_samples})

        sample_gen_param_grids = list(ParameterGrid(split_params[0]))
        choose_subset_param_grids = list(ParameterGrid(split_params[1]))

        datasets = []
        for sample_gen_params in sample_gen_param_grids:
            # Generate samples
            (traces,
             trace_lengths,
             trace_probs) = self._specification.generate_traces(
                **sample_gen_params,
                **split_kwargs[0],
                verbose=verbose)

            # WARNING: traces sampled from PDFA do not represent the
            # probability distribution correctly
            # Therefore, it requires recomputing and updating the prob weights
            pdfa_updated_prob = copy.deepcopy(self._specification)
            pdfa_updated_prob.recompute_prob_dist(traces)
            trace_probs = [pdfa_updated_prob.score(trace) for trace in traces]

            max_num_sample = int(len(traces))

            # Select subset
            for choose_subset_params in choose_subset_param_grids:
                if not use_subset_of_max_num_sample:
                    # Copy the same num_samples from the
                    # params used to generate samples
                    d = {'num_samples': sample_gen_params['num_samples']}
                    choose_subset_params = choose_subset_params.update(d)

                dataset = self.choose_subset_from_samples(
                    traces,
                    trace_lengths,
                    trace_probs,
                    **choose_subset_params,
                    **split_kwargs[1])

                dataset.set_pdfa(pdfa_updated_prob)

                if max_num_sample < choose_subset_params['num_samples']:
                    actual_num_samples = max_num_sample
                else:
                    actual_num_samples = choose_subset_params['num_samples']

                # Store the parameters and resulting data into a pandas dataframe
                dictionary = {**sample_gen_params, **choose_subset_params}
                dictionary = {**dictionary, **{'dataset': dataset,
                                               'actual_num_samples': actual_num_samples}}

                datasets.append(dictionary)

        self._df = pd.DataFrame(datasets)

    def choose_subset_from_samples(self,
                                   traces: Symbols,
                                   trace_lengths: List[int],
                                   trace_probs: Probabilities,
                                   num_samples: int,
                                   select_method='order',
                                   test_size: float = 0.2) -> Dataset:

        # check errors
        max_num_sample = len(traces)

        if max_num_sample < num_samples:
            num_samples = max_num_sample
            msg = f'Length of traces {max_num_sample} must be' + \
                  f'larger than num_samples {num_samples}'
            warnings.warn(msg)

        # choose subset depending on the selected method
        if select_method == 'order':
            indices = np.arange(num_samples).astype(int)
        elif select_method == 'random':
            indices = np.random.choice(max_num_sample, num_samples).astype(int)
        else:
            raise ValueError("method must be either 'order' or 'random'. ")

        partial_traces = np.array(traces, dtype=object)[indices]
        partial_trace_probs = np.array(trace_probs)[indices]
        partial_trace_lengths = np.array(trace_lengths)[indices]

        (X_train, X_test,
         y_train, y_test,
         l_train, l_test) = train_test_split(partial_traces,
                                             partial_trace_probs,
                                             partial_trace_lengths,
                                             test_size=test_size)

        return Dataset(X_train, X_test,
                       y_train, y_test,
                       l_train, l_test,
                       self._output_directory)

    def select_datasets(self, param_grid: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """
        Provide a list of possible parameters,
        loop over those possible combinations,
        and return a nested list in the same
        order as the params were provided.

        :param params:              A dictionary of parameters
        :param selected_method:     Select Method for sampling

        :return:                    Dataset
        """
        if self._df is None:
            self.generate_datasets(param_grid, **kwargs)

        # required_keys = ['num_samples', 'N']
        # if any(k not in param_grid for k in required_keys):
        #     raise ValueError("params should include {}".format(required_keys))

        param_grid = list(ParameterGrid(param_grid))

        df = pd.DataFrame(columns=self._df.columns)

        dfs = []
        for param_dict in param_grid:
            query = ' and '.join([f'{k} == {repr(v)}' for k, v in param_dict.items()])
            selected_df = self._df.query(query)
            dfs.append(selected_df)

        df = pd.concat(dfs)
        dicts = df.to_dict('records')

        datasets = []
        for d in dicts:
            dataset = d.pop('dataset')
            dataset.set_params(d)
            datasets.append(dataset)

        return datasets

    # Getters

    @property
    def max_num_sample(self) -> int:
        return self._max_num_sample

    @property
    def output_directory(self) -> str:
        return self._output_directory

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df
