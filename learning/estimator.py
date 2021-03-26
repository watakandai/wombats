"""
This script contains estimators for solving automata inference problem.
The classes should only depend on data.
Additionally, you can choose to provide specification & parameters used
to generate the data.
"""

# Author: Kandai Watanabe <kandai.wata@gmail.com>

import os
import copy
import warnings
from typing import List, Dict, Union
from sklearn.base import BaseEstimator

from abc import ABCMeta, abstractmethod
from wombats.automaton import Automaton, SafetyDFA, PDFA, Product
from wombats.automaton import active_automata
from .external_tools import FlexfringeInterface
from wombats.automaton.types import Symbols, Probabilities
from .dataload import Dataset



class DummyEstimator(BaseEstimator):
    """
    This class is a dummy estimator for defining a sklearn pipeline.
    The actual estimators will be provided to a GridSearch
    as a list of parameters and replaced with the dummy
    estimator during the fitting process.
    """
    def fit(self, X, y=None, **kwargs): pass
    def score(self, X) -> List[float]: return len(X)*[0.0]


class SpecificationEstimator(BaseEstimator, metaclass=ABCMeta):
    """
    This is the base class for specification (Automata) learning
    estimator.
    """

    key = 'clf'

    def __init__(self, specification: Union[PDFA, Product] = None,
                 output_directory: str = './'):
        """
        :param specification:           A PDFA or Product Automaton.
                                        Specification does not have to
                                        be provided, but if it did, then
                                        it makes the computation
                                        slightly faster.
        :param output_directory:        An output directory for
                                        exporting data for flexfringe
        """
        self.specification = specification
        self.output_directory = output_directory

        self.default_params = {
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
            'T': '1',
            'p': '1'}

        for k, v in self.default_params.items():
             setattr(self, k, v)

        self.flexfringe = None
        """FlexfringeInterface"""

        self.params = {self.key: self.__class__.__name__}
        """Parameters that were modified by the user"""

        self.pdfa = None
        """An estimate PDFA from provided data"""

        self.dataset = None
        """Dataset used to train this estimator"""

    def get_param_kwargs(self) -> Dict:
        """
        Update kwargs for passing to FlexfringeInterface
        Assumption: options' lengths must be 1 alphabet
        """
        kwargs = {}
        for key in self.default_params.keys():
            val = getattr(self, key)
            kwargs.update({key: val})
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

    def set_params(self, **params) -> None:
        """
        Set parameters for this estimator

        :param **params:        A dict of params

        :return:                None
        """
        for key, val in params.items():
            if key in self.default_params.keys():
                self.params.update({key: val})
        super().set_params(**params)

    @abstractmethod
    def fit(self, X: List[Symbols], y=None, **kwargs) -> None:
        """
        :param X:           A list of traces datasets
        :param y:           Never be provided. Just being consistent
                            with the base class

        :return PDFA:       A learned PDFA
        """
        raise NotImplementedError('')

    def predict(self, X: List[Symbols]) -> Probabilities:
        """
        Run prediction on the dataset X to compute the scores y

        :param X:       A list of traces

        :return y:      A list of probabilities
        """
        if self.predict is None:
            raise ReferenceError('Not fitted yet')
        return self.pdfa.scores(X)

    def _preprocess(self, X: List[Symbols]) -> str:
        """
        Preprocess step before running inference

        :param X:       A list of traces

        :return:        A filename of the exported training data
        """
        if self.specification is None:
            symbols = [symbol for trace in X for symbol in trace]
            alphabet_size = len(set(symbols))
        else:
            alphabet_size = self.specification.alphabet_size

        self.flexfringe = FlexfringeInterface(
            output_directory=self.output_directory)

        train_data_file = Automaton.write_traces_to_file(X,
            alphabet_size=alphabet_size,
            file=os.path.join('flexfringe_data', 'train.abbadingo'))

        return train_data_file

    def _postprocess(self, X: List[Symbols],
                     pdfa: Union[PDFA, Product]) -> None:
        """
        Postprocess step after running inference

        :param X:           A list of traces
        :param pdfa:        A learned PDFA
        :param flexfringe:  An interface to flexfringe library

        :return:            None
        """
        self.pdfa = pdfa

        if self.dataset is None:
            self.dataset = Dataset(
                Xtrain=X, specification=self.specification)


class Vanilla(SpecificationEstimator):
    def fit(self, X: List[Symbols], y=None, n_trial: int =2,
            **kwargs) -> None:
        """
        :param X:           A list of traces or a list of datasets
        :param y:           Never be provided. Just being consistent
                            with the base class
        :param n_trial:     A number of trials to run inference

        :return PDFA: Learned PDFA
        """
        train_data_file = self._preprocess(X)

        kwargs = self.get_param_kwargs()
        success = False
        i_trial = 0

        while not success and i_trial != n_trial:
            try:
                data = self.flexfringe.infer_model(
                    training_file=train_data_file,
                    record_time=True, **kwargs)

                fdfa = active_automata.get(
                    automaton_type='FDFA',
                    graph_data=self.flexfringe,
                    graph_data_format='learning_interface')

                pdfa = active_automata.get(
                    automaton_type='PDFA',
                    graph_data=fdfa,
                    graph_data_format='fdfa_object',
                    merge_sinks=True)

                success = True
            except:
                pdfa = None
                i_trial += 1

        self._postprocess(X, pdfa)


class Postprocess(SpecificationEstimator):
    def __init__(self, safe_specification: SafetyDFA = None, **kwargs):
        """
        :param safe_specification:      A SafetyDFA
        """
        super().__init__(**kwargs)
        self.safe_specification = safe_specification

    def fit(self, X: List[Symbols], y=None, n_trial: int =2,
            **kwargs) -> None:
        """
        :param X:           A list of traces or a list of datasets
        :param y:           Never be provided. Just being consistent
                            with the base class
        :param n_trial:     A number of trials to run inference

        :return PDFA: Learned PDFA
        """

        train_data_file = self._preprocess(X)

        kwargs = self.get_param_kwargs()
        success = False
        i_trial = 0

        while not success and i_trial != n_trial:
            try:
                data = self.flexfringe.infer_model(
                    training_file=train_data_file,
                    record_time=True, **kwargs)

                fdfa = active_automata.get(
                    automaton_type='FDFA',
                    graph_data=self.flexfringe,
                    graph_data_format='learning_interface')

                pdfa = active_automata.get(
                    automaton_type='PDFA',
                    graph_data=fdfa,
                    graph_data_format='fdfa_object',
                    merge_sinks=True)

                # Postprocessing Safety
                if self.safe_specification:
                    pdfa = active_automata.get(
                        automaton_type='PDFA',
                        graph_data=(pdfa, self.safe_specification),
                        graph_data_format='existing_objects',
                        normalize_trans_probabilities=True,
                        delete_sinks=False)

                success = True
            except:
                pdfa = None
                i_trial += 1

        self._postprocess(X, pdfa)


class GreedyPreprocess(SpecificationEstimator):
    def __init__(self, safe_specification: SafetyDFA = None, **kwargs):
        """
        :param safe_specification:      A SafetyDFA
        """
        super().__init__(**kwargs)
        self.safe_specification = safe_specification

    def fit(self, X: List[Symbols], y=None, n_trial: int =2,
            **kwargs) -> None:
        """
        :param X:           A list of traces or a list of datasets
        :param y:           Never be provided. Just being consistent
                            with the base class
        :param n_trial:     A number of trials to run inference

        :return PDFA: Learned PDFA
        """

        train_data_file = self._preprocess(X)

        # Embedding Safety during the learning process
        kwargs = self.get_param_kwargs()
        kwargs.update({'C': '1'})
        success = False
        i_trial = 0


        while not success and i_trial != n_trial:
            try:
                if self.safe_specification:
                    data = self.flexfringe.infer_model(
                        training_file=train_data_file,
                        S=self.safe_specification.graph_data_file,
                        record_time=True, **kwargs)
                else:
                    data = self.flexfringe.infer_model(
                        training_file=train_data_file,
                        record_time=True,
                        **kwargs)

                fdfa = active_automata.get(
                    automaton_type='FDFA',
                    graph_data=self.flexfringe,
                    graph_data_format='learning_interface')

                pdfa = active_automata.get(
                    automaton_type='PDFA',
                    graph_data=fdfa,
                    graph_data_format='fdfa_object',
                    merge_sinks=True)

                success = True
            except:
                pdfa = None
                i_trial += 1

        self._postprocess(X, pdfa)


class Preprocess(SpecificationEstimator):
    def __init__(self, safe_specification: SafetyDFA = None, **kwargs):
        """
        :param safe_specification:      A SafetyDFA
        """
        super().__init__(**kwargs)
        self.safe_specification = safe_specification

    def fit(self, X: List[Symbols], y=None, n_trial: int =2,
            **kwargs) -> None:
        """
        :param X:           A list of traces or a list of datasets
        :param y:           Never be provided. Just being consistent
                            with the base class
        :param n_trial:     A number of trials to run inference

        :return PDFA: Learned PDFA
        """

        train_data_file = self._preprocess(X)

        # Embedding Safety during the learning process
        kwargs = self.get_param_kwargs()
        kwargs.update({'C': '0'}) # C = Choose safetyAlgorithmNum: 0->Preprocessing
        success = False
        i_trial = 0

        while not success and i_trial != n_trial:
            try:
                if self.safe_specification:
                    data = self.flexfringe.infer_model(training_file=train_data_file,
                                                S=self.safe_specification.graph_data_file,
                                                record_time=True,
                                                **kwargs)
                else:
                    data = self.flexfringe.infer_model(training_file=train_data_file,
                                                record_time=True,
                                                **kwargs)

                fdfa = active_automata.get(
                    automaton_type='FDFA',
                    graph_data=self.flexfringe,
                    graph_data_format='learning_interface')

                pdfa = active_automata.get(
                    automaton_type='PDFA',
                    graph_data=fdfa,
                    graph_data_format='fdfa_object',
                    merge_sinks=True)

                success = True
            except Exception as e:
                print(e)
                msg = f'Cannot train a model properly'
                warnings.warn(msg)
                pdfa = None
                i_trial += 1

        self._postprocess(X, pdfa)


class TargetSpecification(SpecificationEstimator):
    def fit(self, X: List[Symbols], y=None, n_trial: int =2,
            **kwargs) -> None:
        """
        :param X:           A list of traces or a list of datasets
        :param y:           Never be provided. Just being consistent
                            with the base class
        :param n_trial:     A number of trials to run inference

        :return PDFA: Learned PDFA
        """
        self._postprocess(X, self.specification)
