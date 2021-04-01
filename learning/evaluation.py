import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Any
import matplotlib.pyplot as plt

from wombats.automaton.pdfa import PDFA
from wombats.automaton.product import Product
from wombats.automaton.types import Symbols
from .training import GridSearch, SpecificationEstimator
from .metrics import EvaluationFunctions, TestDataset


class Evaluator:
    """
    This class evaluates all of the learned automaton on all metrics
    """
    def __init__(self,
                 eval_func: EvaluationFunctions,
                 output_directory: str = './'):
        """
        :param eval_func:           A evaluation function
        :param output_directory:    An output directory for plots
        """

        self.eval_func = eval_func
        self.metrics = eval_func.metrics

        self._output_directory = output_directory
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        self.param_names = None
        """A list of parameter names"""

        self.test_dataset = None
        """Test Dataset to evaluate on"""

        self._df = None
        """pd.DataFrame to store experiment & evaluation results"""

    def evaluate(self,
            estimator: Union[GridSearch, SpecificationEstimator],
            specification: Union[PDFA, Product] = None,
            test_traces: Symbols = None,
            test_size: float = 0.2,
            double_test_size: bool = True,
            **kwargs) -> None:
        """
        Evaluate all experiments

        :param estimators:          A fitted estimator
        :param specification:       A PDFA specification
        :param test_traces:         If given, it is used to evaluate
                                    PDFAs
        :param test_size:           If nothing is given, test_size is
                                    used to evalute PDFAs
        :param double_test_size:    Double the test size by sampling
                                    from est. Automaton

        :return: None
        """

        # if not isinstance(estimators,
        #                   Union[GridSearch, SpecificationEstimator]):
        #     msg = 'estimators has to be either of GridSearch or ' + \
        #           'SpecificationEstimator'
        #     raise ValueError(msg)

        if isinstance(estimator, SpecificationEstimator):
            estimators = [estimator]
        else:
            estimators = estimator.estimators

        if specification is None and \
            any([e.specification == None for e in estimators]):
            msg = 'Ground True Automaton must be given'
            raise ValueError(msg)

        if self.param_names is None:
            self.param_names = self.__concat_all_params(estimators[0]).keys()

        if self.test_dataset is None:
            self.__generate_test_dataset(estimators,
                                         specification,
                                         test_traces,
                                         test_size,
                                         double_test_size)

        self.__evaluate(estimators)

    def __concat_all_params(self,
            estimator: SpecificationEstimator) -> Dict:
        """
        :param estimator:       A fitted estimator
        :return:               None
        """
        return {**estimator.params,
                **estimator.dataset.params}

    def __generate_test_dataset(self,
            estimators: List[SpecificationEstimator],
            specification: Union[PDFA, Product] = None,
            test_traces: Symbols = None,
            test_size: float = 0.2,
            double_test_size: bool = True,
            max_resamples: int = 10,
            verbose: int = 0):
        """
        Generate test dataset from the fitted estimator

        :param estimators:      A list of fitted estimators
        :param specification:   A PDFA or Product automaton

        :return:                None
        """
        eval_spec = specification if specification else None
        Xest = None

        self.test_dataset = []
        for estimator in estimators:

            # Choose specification
            if eval_spec is None or eval_spec != specification:
                eval_spec = estimator.specification

            X_train = estimator.dataset.X_train

            # If given a test traces to evaluate, use it
            if test_traces:
                X_true = test_traces
                y_true = eval_spec.scores(X_true)
                X_train = test_traces
                test_size = 1
            # If not, but there exists training set in the dataset
            else:
                X_true = estimator.dataset.X_test
                y_true = estimator.dataset.y_test

            num_samples = test_size * len(X_train)
            N = max(list(map(len, X_train)))

            # Else, generate new samples from a specification PDFA
            if X_true is None or y_true is None:
                X_true, _, y_true = eval_spec.generate_traces(
                    num_samples, N, max_resamples, verbose=verbose)

            ## Additionally, sample traces from learned automaton
            if double_test_size and estimator.pdfa is not None:
                X_est, _, _ = estimator.pdfa.generate_traces(
                    num_samples, N, max_resamples, verbose=verbose)
                y_est = eval_spec.scores(X_est)

            testdata = TestDataset(X_true, y_true, X_est, y_est)
            self.test_dataset.append(testdata)

    def __evaluate(self,
            estimators: List[SpecificationEstimator]):
        """
        Evaluate estimators

        :param estimators:      A list of fitted estimators
        :param specification:   A PDFA or Product automaton

        :return:                None
        """

        rows = []
        for i, estimator in enumerate(estimators):

            # Generate Test Data based on the arguments
            # Keep data consistent across same data params
            test_dataset = self.test_dataset[i]

            # Evaluate for each metric
            results = self.eval_func(estimator, test_dataset)

            params = self.__concat_all_params(estimator)

            # Store as a dictionary
            row = {'estimator': estimator,
                   **params, **results}

            rows.append(row)

        self._df = pd.DataFrame(rows)

    def __make_dir_from_params(self, params: Dict):
        """
        :param params:      A dict of parameters

        :return filedir:    A created directory
        """
        # Convert to a list of param strings
        params_strs = [f'{k}{str(v)}' for k, v in params.items()]
        # Concatenate all param strings into one string
        params_str = '_'.join(params_strs)
        # Use the concatenated string as a directory name
        filedir = os.path.join(self._output_directory, params_str)
        # Create a new directory if it doesn't already exist
        Path(filedir).mkdir(parents=True, exist_ok=True)

        return filedir

    def plot(self, x: str = None,
             selected_params: Dict = None,
             filename: str = None,
             ext="png",
             dpi=300,
             clean_legend=True,
             ones_as_title=True,
             **kwargs) -> None:
        """
        Plot the performances of each method on the y axis against
        a value on the x axis (generally, No. of samples).

        :param x:                   A measure on the x axis
        :param selected_params:     A dict of params to select from
                                    the experiments for plotting
        :param filename:            A filename for exporting plots
        :param ext:                 Extension of the exporting image
        :param dpi:                 DPI of the exporting image
        """
        df = self.select_df(selected_params)

        n_uniques = df[self.param_names].nunique()
        if x is None:
            # Let x be a variable with the most number of options
            x = n_uniques.idxmax()

        # Find Left keys in param_names
        used_params = [x] + list(selected_params.keys())
        left_params = list(set(self.param_names) - set(used_params))

        # Omit variables with a single unique value
        ones = list(n_uniques[left_params][n_uniques == 1].index)
        left_params = list(n_uniques[left_params][n_uniques != 1].index)

        # Use variables with unique value as a title
        d = {o: df[o].unique()[0] for o in ones}
        d.update(selected_params)
        title = ', '.join([f'{k}={v}' for k,v in d.items()])

        # Prepare exporting directory
        filedir = self.__make_dir_from_params(selected_params)

        for metric in self.metrics.keys():

            print(f'x={x}, legend={left_params}, y={metric}')

             # Create a pivot table
            df_ = df.pivot(index=x, columns=left_params, values=metric)

            # # Rename ("c1", "c2") -> "c1, c2" for cleaner legend
            if clean_legend and isinstance(df_.columns, pd.MultiIndex):
                flat_tuple = lambda t: ', '.join(map(str, t))
                df_.set_axis(labels=df_.columns.map(flat_tuple),
                             axis='columns', inplace=True)

            # Plot data
            ax = df_.plot(**kwargs)

            # Set Labels
            ax.set_xlabel('No of Samples')
            ax.set_ylabel(metric)
            plt.tight_layout()

            if ones_as_title and len(ones) > 0:
                ax.set_title(title)

            # Export as an image
            filepath = os.path.join(
                filedir, metric.replace(" ", "") + '.' + ext)
            plt.savefig(filepath, format=ext, dpi=dpi)

    def select_df(self,
                   selected_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Returns a new DF that match the selected_params

        :param selected_params:     A dict of params to select from
                                    the experiments for plotting

        :return:                    pd.DataFrame
        """
        if selected_params:
            query = ' and '.join(
                [f'{k} == {repr(v)}' for k, v in selected_params.items()])
            df = self._df.query(query)
        else:
            selected_params = {}
            df = self._df
        return df

    def draw(self, selected_params: Dict[str, Any],
             filename: str = None) -> None:
        """
        Draw pdfas with the selected params

        :param selected_params:     A dict of params to select from
                                    the experiments for plotting
        :param filename:            A filename for exporting plots
        """
        pdfas = self.pdfas(selected_params)

        if pdfas is None:
            print('Failed to obtain a PDFA due to the failure in the learning process')
            return

        for pdfa in pdfas:
            pdfa.draw(filename)

    def estimators(self, selected_params: Dict[str, Any]) -> List[SpecificationEstimator]:
        """
        return estimators according to the selected params

        :param selected_params:     A dict of params to select from
                                    the experiments for plotting

        :return:                    A SpecificationEstimator
        """
        df = self.select_df(selected_params)

        if len(df) == 0:
            print('There is no such Estimator that satisfies the given parameters')
            return

        # return df.to_dict(orient='records')
        return df['estimator'].values.tolist()

    def pdfas(self, selected_params: Dict[str, Any]) -> List[PDFA]:
        """
        return pdfa according to the selected params

        :param selected_params:     A dict of params to select from
                                    the experiments for plotting

        :return:                    A PDFA or a list of PDFAs
        """
        estimators = self.estimators(selected_params)
        return [e.pdfa for e in estimators]

