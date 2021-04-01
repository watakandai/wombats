"""
This script contains classes to train estimators in batch
"""

# Author: Kandai Watanabe <kandai.wata@gmail.com>

import copy
from sklearn.pipeline import Pipeline
from typing import List, Dict, Union
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid

from wombats.automaton import PDFA
from wombats.learning.dataload import Dataset
from wombats.automaton.types import Symbols
from .estimator import DummyEstimator, SpecificationEstimator


class GridSearch(BaseEstimator):
    """
    Train all possible combinations of (estimators, parameters).
    It copies the feature of of sklearn's GridSearchCV.

    See below for how to use Pipeline with GridSearchCV.
    https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py

    ex)
        # Instantiate a dummy estimator
        estimators = [('clf', DummyEstimator())]
        pipeline = Pipeline(steps=estimators)

        # Specify all possible parameter sets
        parameters = [{
            'clf': [
                Vanilla(specification),
                Postprocess(specification, safe_specification),
                GreedyPreprocess(specification, safe_specification),
                Preprocess(specification, safe_specification)],
            'clf__p': [0.6, 1, 2, 5]}]

        # Run all trainings
        clf = GridSearch(pipeline, parameters)
        clf.fit(X, y)
    """

    key = 'clf'

    def __init__(self, estimators: List[SpecificationEstimator],
                 estimator_params: Union[List, Dict],
                 output_directory: str = './'):
        """
        :param estimators:          An estimator that inherits sklearn's
                                    BaseEstimator.
        :param estimator_params:    A (nested) dictionary of parameters.
        """

        self.estimator_params = estimator_params
        self.output_directory = output_directory

        # Create sklearn ParameterGrid
        params = self.__append_estimators_to_params(
            estimators, estimator_params)
        self._param_grid = ParameterGrid(params)

        # Create a Pipeline Estimator
        steps = [(self.key, DummyEstimator())]
        self.pipeline = Pipeline(steps=steps)

        self.estimators = []
        """A list of Estimators"""

    def __append_estimators_to_params(self,
            estimators: List[SpecificationEstimator],
            params: Union[List, Dict]) -> Dict:
        """
        Append estimators as 'clf' to a dict of params

        :param estimators:          An estimator that inherits sklearn's
                                    BaseEstimator.
        :param params:              A (nested) dictionary of parameters.

        :return:                    An augmented dict of parameters
        """
        is_dict = False
        if isinstance(params, Dict):
            is_dict = True
            params = [params]

        new_param_list = []

        for ps in params:
            new_params = {self.key: estimators}
            for k, v in ps.items():
                newk = self.key + '__' + k
                new_params.update({newk: v})
            new_param_list.append(new_params)

        if is_dict:
            return new_param_list[0]

        return new_param_list

    def fit(self, X: Union[Dataset, List[Symbols]], y=None,
            **kwargs) -> None:
        """
        Run inference only on one dataset (a list of traces)

        :param X:               A list of traces or a dataset
        :param y:               Never be provided. Just being consistent
                                with the base class
        :param specification:   A PDFA Specification


        :return:        None
        """
        if isinstance(X, Dataset):
            dataset = X
        else:
            dataset = Dataset(X)

        self.batchfit([dataset])

    def batchfit(self, datasets: List[Dataset], y=None,
                 **kwargs) -> None:
        """
        Run all possible combinations of (estimators, parameters)

        :param datasets:    A list of datasets
        :param y:           Never be provided. Just being consistent
                            with the base class

        :return:        None
        """
        if not isinstance(datasets, List) or len(datasets) == 0:
            raise ValueError('datasets must be a list of Dataset')

        for dataset in datasets:
            for params in self._param_grid:

                # TODO: Must be refactored once we fix prob dist bias
                self.pipeline.set_params(**params)
                self.pipeline.set_params(**{
                    self.key+'__dataset': dataset,
                    self.key+'__specification': dataset.specification,
                    self.key+'__output_directory': self.output_directory})

                # Run Automata Inference
                self.pipeline.fit(dataset.X_train)

                # The learned estimator is stored in parameters['clf']
                estimator = copy.deepcopy(params[self.key])

                self.estimators.append(estimator)
