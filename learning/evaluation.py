import pandas as pd
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod
from .training import BatchEstimator, Experiment


class EvaluationFunction(metaclass=ABCMeta):
    def __init__(self):
        # Pass parameters or some bool switches
        pass

    @abstractmethod
    def __call__(self, experiment: Experiment, **kwargs):
        raise NotImplementedError('Evaluation Function must be implemented')


class L2Norm(EvaluationFunction):
    def __call__(self, experiment):
        if experiment.estimator.pdfa is None:
            return None

        return experiment.estimator.pdfa.average_norm(
            experiment.X_test,
            experiment.y_test)


class L1Norm(EvaluationFunction):
    def __call__(self, experiment):
        if experiment.estimator.pdfa is None:
            return None

        return experiment.estimator.pdfa.average_norm(
            experiment.X_test,
            experiment.y_test, n=1)

class Perplexity(EvaluationFunction):
    def __call__(self, experiment, **kwargs):
        if experiment.estimator.pdfa is None:
            return None

        print(len(experiment.X_test), len(experiment.y_test))
        print(experiment.estimator.__class__.__name__)
        print(experiment.estimator.pdfa.average_norm(
            experiment.X_test,
            experiment.y_test))

        n_trace = len(experiment.y_test)

        return experiment.estimator.pdfa.perplexity(
            experiment.X_test,
            experiment.y_test,
            ignore_reject_trace=False,
            **kwargs) / n_trace


class KLDivergence(EvaluationFunction):
    def __call__(self, experiment):
        if experiment.estimator.pdfa is None:
            return None

        n_trace = len(experiment.y_test)

        return experiment.estimator.pdfa.kldivergence(
            experiment.X_test,
            experiment.y_test,
        ) / n_trace


class NoOfNodes(EvaluationFunction):
    def __call__(self, experiment):
        if experiment.estimator.pdfa is None:
            return None
        return len(list(experiment.estimator.pdfa.nodes.keys()))


class NoOfEdges(EvaluationFunction):
    def __call__(self, experiment):
        if experiment.estimator.pdfa is None:
            return None
        return len(list(experiment.estimator.pdfa.edges.keys()))


class ShortestPathLength(EvaluationFunction):
    def __call__(self, experiment):
        if experiment.estimator.pdfa is None:
            return None
        MPS, MPS_prob, _ = experiment.estimator.pdfa.most_probable_string(
            disable_pbar=True)
        return len(MPS)


class ElapsedTime(EvaluationFunction):
    def __call__(self, experiment):
        return experiment.estimator.elapsed_time


class Safe(EvaluationFunction):
    def __call__(self, experiment):
        if experiment.estimator.pdfa is None:
            return int(False)
        return int(experiment.estimator.safe_specification.is_safe(
            experiment.estimator.pdfa))


class BatchEvaluationFunction(EvaluationFunction):
    """
    This class should be a stand alone class from estimators

    Receiving PDFA
    """
    def __init__(self, metrics: Dict = None,
                       additional_metrics: Dict = None):
        if metrics:
            self.metrics = metrics
        else:
            self.metrics = {
                'KL Divergence': KLDivergence(),
                'L2 Norm of Trace Probabilities': L2Norm(),
                'L1 Norm of Trace Probabilities': L1Norm(),
                'No. of Nodes': NoOfNodes(),
                'Elapsed Time': ElapsedTime(),
                'Safe': Safe()}

        if additional_metrics:
            self.metrics.update(additional_metrics)

    def __call__(self, experiment: Experiment):
        """
        Pass an Experiment to receive its evaluation
        """
        evaluations = {}
        for name, eval_func in self.metrics.items():
            evaluations[name] = eval_func(experiment)

        return evaluations


class ComparativeEvaluation(metaclass=ABCMeta):
    def __init__(self, experiments: List[Experiment] = None,
                       eval_func: EvaluationFunction = None):
        self._experiments = experiments
        self._df = None

        if eval_func:
            self.eval_func = eval_func
        else:
            self.eval_func = BatchEvaluationFunction()

        self._evaluate(experiments)

    @classmethod
    def evaluate(cls, experiments: List[Experiment]):
        if len(experiments) < 2:
            raise ValueError('No. of given experiments are less than 2')

    def _evaluate(self, experiments: List[Experiment] = None):
        if experiments:
            self._experiments = experiments

        if self._experiments is None:
            raise ValueError('No experiment is provided')

        if len(self._experiments) < 2:
            raise ValueError('No. of Experiments greater more than 1')

        # Create a db table
        variables = self._read_variables(self._experiments[0])
        columns = list(variables.keys()) + list(self.eval_func.metrics.keys())

        dicts = []
        for experiment in self._experiments:
            evaluation = self.eval_func(experiment)
            variables = self._read_variables(experiment)
            dictionary = {**evaluation, **variables, **{'pdfa': experiment.estimator.pdfa}}
            dicts.append(dictionary)

        self._df = pd.DataFrame(dicts)

    def _read_variables(self, experiment) -> Dict:
        """
        Read all possible variables from an experiment data
        """
        return {**experiment.dataset.params, **experiment.parameters}

    def plot(self, x: str, selected_params: Dict = None, filename: str = None, **kwargs):
        if selected_params:
            query = ' and '.join(
                [f'{k} == {repr(v)}' for k, v in selected_params.items()])
            df = self._df.query(query)
        else:
            df = self._df

        for metric in self.eval_func.metrics.keys():
            if metric == 'L2 Norm of Trace Probabilities':
                logy = False
            else:
                logy = False
            df_ = df.pivot(index=x, columns='clf', values=metric)
            df_.plot(title=metric, logy=logy, **kwargs)

    def draw(self, selected_params: Dict, filename: str = None):
        pdfa = self.pdfa(selected_params)

        if pdfa is None:
            print('Failed to obtain a PDFA due to the failure in the learning process')
            return

        pdfa.draw(filename)

    def pdfa(self, selected_params: Dict):
        """return pdfa"""
        query = ' and '.join(
            [f'{k} == {repr(v)}' for k, v in selected_params.items()])
        df = self._df.query(query)

        if len(df) == 0:
            print('There is no such PDFA that satisfies the given parameters')
            return

        if len(df) != 1:
            print('There are several choices of PDFAs. Please provide more parameters')
            print(df)
            return

        return df['pdfa'].values[0]
