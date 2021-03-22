import os
import pandas as pd
import numpy as np
from pathlib import Path
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

        n_trace = len(experiment.y_test)

        return experiment.estimator.pdfa.perplexity(
            experiment.X_test,
            experiment.y_test,
            ignore_reject_trace=False,
            **kwargs) / n_trace


class ForwardKLWithEstDist(EvaluationFunction):
    def __call__(self, experiment):
        if experiment.estimator.pdfa is None:
            return None

        n_trace = len(experiment.y_test)

        return experiment.estimator.pdfa.kldivergence(
            experiment.estimator.X_test,
            experiment.estimator.y_test,
        ) / n_trace


class ReverseKLWithTargetDist(EvaluationFunction):
    def __call__(self, experiment):
        if experiment.estimator.pdfa is None:
            return None

        n_trace = len(experiment.y_test)

        return experiment.estimator.pdfa.reverse_kldivergence(
            experiment.dataset.X_test,
            experiment.dataset.y_test,
        ) / n_trace


class ForwardKLWithTargetDist(EvaluationFunction):
    def __call__(self, experiment):
        if experiment.estimator.pdfa is None:
            return None

        n_trace = len(experiment.y_test)

        return experiment.estimator.pdfa.kldivergence(
            experiment.dataset.X_test,
            experiment.dataset.y_test,
        ) / n_trace

class ReverseKLWithEstDist(EvaluationFunction):
    def __call__(self, experiment):
        if experiment.estimator.pdfa is None:
            return None

        n_trace = len(experiment.y_test)

        return experiment.estimator.pdfa.reverse_kldivergence(
            experiment.estimator.X_test,
            experiment.estimator.y_test,
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
            return None
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
                'Average True Forward KL': ForwardKLWithTargetDist(),
                'Average True Reverse KL': ReverseKLWithEstDist(),
                'Average False Forward KL': ForwardKLWithEstDist(),
                'Average False Reverse KL': ReverseKLWithTargetDist(),
                'L2 Norm of Trace Probabilities': L2Norm(),
                'L1 Norm of Trace Probabilities': L1Norm(),
                'No. of Nodes': NoOfNodes(),
                'No. of Edges': NoOfEdges(),
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
                       eval_func: EvaluationFunction = None,
                       output_directory: str = './'):
        self._experiments = experiments
        self._df = None
        self._output_directory = output_directory
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        if eval_func:
            self.eval_func = eval_func
        else:
            self.eval_func = BatchEvaluationFunction()

        self._evaluate(experiments)

    @classmethod
    def evaluate(cls, experiments: List[Experiment]):
        if len(experiments) < 1:
            raise ValueError('No. of given experiments are less than 2')

    def _evaluate(self, experiments: List[Experiment] = None):
        if experiments:
            self._experiments = experiments

        if self._experiments is None:
            raise ValueError('No experiment is provided')

        if len(self._experiments) < 1:
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

        self._df = pd.DataFrame(dicts, dtype=object)

    def _read_variables(self, experiment) -> Dict:
        """
        Read all possible variables from an experiment data
        """
        return {**experiment.dataset.params, **experiment.parameters}

    def plot(self, x: str, selected_params: Dict = None, filename: str = None, columns=['clf', 'p'], save=True, ext="png", dpi=300, **kwargs):
        if selected_params:
            query = ' and '.join(
                [f'{k} == {repr(v)}' for k, v in selected_params.items()])
            df = self._df.query(query)
        else:
            df = self._df

        # if 'p' in selected_params.keys():
        #     columns = ['clf']

        params_str = '_'.join([f'{k}{str(v)}' for k, v in selected_params.items()])
        filedir = os.path.join(self._output_directory, params_str)
        Path(filedir).mkdir(parents=True, exist_ok=True)

        for metric in self.eval_func.metrics.keys():
            df_ = df.pivot(index=x, columns=columns, values=metric)
            # print(df_.columns.levels[0])
            # print(type(df_.columns.levels[0][0]))
            # print(df_.columns.levels[1])
            # print(type(df_.columns.levels[1][0]))
            # print(type(df_.columns.levels[1][1]))
            ax = df_.set_axis(df_.columns.map(', '.join), axis=1, inplace=False).plot(**kwargs)
            ax.set_xlabel('No. of Samples')
            ax.set_ylabel(metric)
            filepath = os.path.join(filedir, metric.replace(" ", "") + '.' + ext)
            plt.tight_layout()
            plt.savefig(filepath, format=ext, dpi=dpi)

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
