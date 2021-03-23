import os
import copy
import warnings
import numpy as np
from typing import List, Dict, Any, Union
from networkx.drawing.nx_pydot import read_dot
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid

from abc import ABCMeta, abstractmethod
from wombats.automaton import SafetyDFA, PDFA, Product
from wombats.automaton import active_automata
from .external_tools import FlexfringeInterface
from wombats.learning.dataload import Dataset


class DummyEstimator(BaseEstimator):
    """
    This class is a dummy estimator for defining a sklearn pipeline.
    The actual estimators will be provided to a GridSearch
    as a list of parameters and replaced with the dummy
    estimator during the fitting process.
    """
    def fit(self, dataset, y=None, **kwargs): pass
    def score(self, X): pass


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
        parameters = [{'clf': [VanillaSL(),
                                PostprocessSL(),
                                GreedyPreprocessSL(),
                                PreprocessSL],
                        'clf__specification': specification,
                        'clf__safe_specification': safe_specification,
                        'clf__p': [0.6, 1, 2, 5]}]

        # Run all trainings
        clf = GridSearch(pipeline, parameters)
        clf.fit(X, y)
    """

    def __init__(self, estimator: BaseEstimator, param_grid: Dict):
        """
        :param estimator:           An estimator that inherits sklearn's
                                    BaseEstimator.
        :param param_grid:          A (nested) dictionary of parameters.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self._candidate_params = ParameterGrid(param_grid)
        self._experiments = []

    def fit(self, datasets: Union[Dataset, List[Dataset]],
            y=None, **kwargs):
        """
        Run all possible combinations of (estimators, parameters)

        :param datasets:        A Dataset or a list of Dataset
        :param y:               Never be provided. Just being consistent
                                with the base class

        :return:                None
        """

        if not isinstance(datasets, (Dataset, List)):
            raise ValueError('dataset must be either type \
                              of Dataset or List of Dataset')

        if isinstance(datasets, Dataset):
            datasets = [datasets]

        last_step = -1
        key = 0
        estimator_key = self.estimator.steps[last_step][key]

        for dataset in datasets:
            for parameters in self._candidate_params:

                # Recursively set parameters
                self.estimator.set_params(**parameters)
                self.estimator.fit(dataset)
                # parameters[estimator_key] is the actual learned estimator
                actual_estimator = copy.deepcopy(parameters[estimator_key])
                old_kwargs = actual_estimator.kwargs
                new_kwargs = actual_estimator.get_param_kwargs()
                hyperparams = dict(set(new_kwargs.items()) - set(old_kwargs.items()))
                hyperparams.update({estimator_key: actual_estimator.__class__.__name__})

                experiment = Experiment(actual_estimator, hyperparams, dataset)
                self._experiments.append(experiment)

    @property
    def experiments(self):
        """
        :return:            A list of Experiment
        """
        if len(self._experiments) == 0:
            raise ValueError('No experiment has been run')
        return self._experiments


class SpecificationLearning(BaseEstimator, metaclass=ABCMeta):
    """
    This is a base class for specification (Automata) learning
    type of estimator.
    """

    def __init__(self, specification: Union[PDFA, Product] = None,
                 safe_specification: SafetyDFA = None,
                 **kwargs):
        """
        :param specification:           A PDFA or Product Automaton
        :param safe_specification:      A SafetyDFA
        """
        self.specification = specification
        self.safe_specification = safe_specification

        self.kwargs = {
            'h': 'kldistance',
            'd': 'kl_data',
            'n': '2',
            'x': '0',
            'w': '0',
            'b': '1',
            'f': '1',
            'I': '0',
            't': '1',
            'l': '0',
            'q': '0',
            'y': '0',
            'T': '0',
            'p': '1'}

        for k, v in self.kwargs.items():
            setattr(self, k, v)

        # learned object
        self.pdfa = None
        self.fitted = False

        self.X_test = None
        self.y_test = None

        # metrics
        self.elapsed_time = 0.0

    def get_param_kwargs(self):
        """
        Update kwargs for passing to FlexfringeInterface
        Assumption: options' lengths must be 1 alphabet
        """
        kwargs = {}
        for k, v in self.__dict__.items():
            if len(k) == 1: # Assumption
                kwargs.update({k: v})
        return kwargs

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        :param deep:            If True, will return the parameters for
                                this estimator and contained subobjects that are estimators.
        :return:                Parameter names mapped to their values.
        """
        out = dict()
        for key in self.__dict__.keys():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value

        return out

    @abstractmethod
    def fit(self, dataset: Dataset, y=None, **kwargs) -> None:
        """
        :param dataset:         A Dataset
        :param y:               Never be provided. Just being consistent
                                with the base class

        :return PDFA: Learned PDFA
        """
        raise NotImplementedError('')

    def _preprocess(self, dataset: Dataset) -> str:
        """
        :param dataset:         A Dataset

        :return:                A training data file output
                                for flexfringe
        """

        if self.specification is None or \
           self.safe_specification is None:
           msg = 'Please provide specification and safe_specification'
           raise ValueError(msg)

        # Split dataset into train and test sets

        output_directory = dataset.output_directory
        traces_train = dataset.X_train

        train_data_file = self.specification.write_traces_to_file(
            traces_train,
            alphabet_size=self.specification.alphabet_size,
            file=os.path.join('flexfringe_data', 'train'))

        return train_data_file


class Vanilla(SpecificationLearning):
    def fit(self, dataset: Dataset,
            y=None, n_trial=2, **kwargs) -> None:
        """
        :param dataset:         A Dataset
        :param y:               Never be provided. Just being consistent
                                with the base class

        :return PDFA: Learned PDFA
        """
        train_data_file = self._preprocess(dataset)

        # Learn PDFA specification from train sets
        flexfringe = FlexfringeInterface(output_directory=dataset.output_directory)

        kwargs = self.get_param_kwargs()
        success = False
        i_trial = 0

        while not success and i_trial != n_trial:
            try:
                data = flexfringe.infer_model(training_file=train_data_file,
                                            record_time=True, **kwargs)

                fdfa = active_automata.get(automaton_type='FDFA',
                                        graph_data=flexfringe,
                                        graph_data_format='learning_interface')
                pdfa = active_automata.get(automaton_type='PDFA',
                                        graph_data=fdfa,
                                        graph_data_format='fdfa_object',
                                        merge_sinks=True)
                success = True
            except:
                pdfa = None
                i_trial += 1

        # store learned pdfa
        self.pdfa = pdfa
        self.fitted = True
        self.elapsed_time = flexfringe.average_elapsed_time


class Postprocess(SpecificationLearning):
    def fit(self, dataset: Dataset,
            y=None, n_trial=2, **kwargs) -> None:
        """
        :param dataset:         A Dataset
        :param y:               Never be provided. Just being consistent
                                with the base class

        :return PDFA: Learned PDFA
        """
        train_data_file = self._preprocess(dataset)

        flexfringe = FlexfringeInterface(output_directory=dataset.output_directory)

        kwargs = self.get_param_kwargs()
        success = False
        i_trial = 0

        while not success and i_trial != n_trial:
            try:
                data = flexfringe.infer_model(training_file=train_data_file,
                                            record_time=True, **kwargs)

                fdfa = active_automata.get(automaton_type='FDFA',
                                        graph_data=flexfringe,
                                        graph_data_format='learning_interface')

                specification = active_automata.get(automaton_type='PDFA',
                                                    graph_data=fdfa,
                                                    graph_data_format='fdfa_object',
                                                    merge_sinks=True)

                # Postprocessing Safety
                pdfa = active_automata.get(automaton_type='PDFA',
                                           graph_data=(specification,
                                                       self.safe_specification),
                                           graph_data_format='existing_objects',
                                           normalize_trans_probabilities=True,
                                           delete_sinks=False)
                success = True
            except:
                pdfa = None
                i_trial += 1

        self.pdfa = pdfa
        self.fitted = True
        self.elapsed_time = flexfringe.average_elapsed_time


class GreedyPreprocess(SpecificationLearning):
    def fit(self, dataset: Dataset,
            y=None, n_trial=2, **kwargs) -> None:
        """
        :param dataset:         A Dataset
        :param y:               Never be provided. Just being consistent
                                with the base class

        :return PDFA: Learned PDFA
        """
        train_data_file = self._preprocess(dataset)

        flexfringe = FlexfringeInterface(output_directory=dataset.output_directory)

        # Embedding Safety during the learning process
        kwargs = self.get_param_kwargs()
        kwargs.update({'C': '1'})
        success = False
        i_trial = 0

        while not success and i_trial != n_trial:
            try:
                data = flexfringe.infer_model(training_file=train_data_file,
                                            S=self.safe_specification.graph_data_file,
                                            record_time=True,
                                            **kwargs)

                fdfa = active_automata.get(automaton_type='FDFA',
                                        graph_data=flexfringe,
                                        graph_data_format='learning_interface')

                specification = active_automata.get(automaton_type='PDFA',
                                                    graph_data=fdfa,
                                                    graph_data_format='fdfa_object',
                                                    merge_sinks=True)
                success = True
            except:
                specification = None
                i_trial += 1

        self.pdfa = specification
        self.fitted = True
        self.elapsed_time = flexfringe.average_elapsed_time


class Preprocess(SpecificationLearning):
    def fit(self, dataset: Dataset,
            y=None, n_trial=2, **kwargs) -> None:
        """
        :param dataset:         A Dataset
        :param y:               Never be provided. Just being consistent
                                with the base class

        :return PDFA: Learned PDFA
        """
        train_data_file = self._preprocess(dataset)

        flexfringe = FlexfringeInterface(output_directory=dataset.output_directory)

        # Embedding Safety during the learning process
        kwargs = self.get_param_kwargs()
        kwargs.update({'C': '0'}) # C = Choose safetyAlgorithmNum: 0->Preprocessing
        success = False
        i_trial = 0

        while not success and i_trial != n_trial:
            try:
                data = flexfringe.infer_model(training_file=train_data_file,
                                            S=self.safe_specification.graph_data_file,
                                            record_time=True,
                                            **kwargs)
                print('loaded flexfringe')
                fdfa = active_automata.get(automaton_type='FDFA',
                                        graph_data=flexfringe,
                                        graph_data_format='learning_interface')

                print('loaded fdfa')
                specification = active_automata.get(automaton_type='PDFA',
                                                    graph_data=fdfa,
                                                    graph_data_format='fdfa_object',
                                                    merge_sinks=True)
                success = True
            except Exception as e:
                print(e)
                msg = f'Cannot train a model properly'
                warnings.warn(msg)
                # raise e
                specification = None
                i_trial += 1

        self.pdfa = specification
        self.elapsed_time = flexfringe.average_elapsed_time


class TargetSpecification(SpecificationLearning):
    def fit(self, dataset: Dataset,
            y=None, n_trial=2, **kwargs) -> None:
        """
        :param dataset:         A Dataset
        :param y:               Never be provided. Just being consistent
                                with the base class

        :return PDFA: Learned PDFA
        """
        self.pdfa = dataset.pdfa
        self.elapsed_time = 0.0


class Experiment:
    """
    This data class contains the result of an experiment.
    This class will be modified in later versions.
    """
    def __init__(self, estimator: SpecificationLearning,
                       parameters: Dict[str, Any],
                       dataset: Dataset = None):
        """
        :param estimator:           A Specification Learning Estimator
        :param parameters:          A dict of parameters
        :param dataset:             A dataset used for training the
                                    estimator
        """
        self.dataset = dataset
        self.parameters = parameters
        self.estimator = estimator
        self.__double_testdata()

    def __double_testdata(self):
        """
        This function generates traces from the learned automaton,
        such that the No. of traces are equal to that of the original
        test set.
        """
        num_samples = len(self.dataset.X_test)
        N = self.dataset.params['N']
        try:
            traces, trace_lengths, _ = self.estimator.pdfa.generate_traces(
                num_samples, N)
            trace_probs = [self.dataset.pdfa.score(t) for t in traces]

            self.estimator.X_test = traces
            self.estimator.y_test = trace_probs
        except Exception as e:
            print(e)
            return

    @property
    def X_test(self):
        """
        :return:            Symbols (A list of traces)
        """
        X_test = list(self.dataset.X_test)
        if self.estimator.X_test:
            return X_test + self.estimator.X_test
        return X_test

    @property
    def y_test(self):
        """
        :return:            Probabilities
                            (A list of trace probabilities)
        """
        y_test = list(self.dataset.y_test)
        if self.estimator.y_test:
            return y_test + self.estimator.y_test
        return y_test
