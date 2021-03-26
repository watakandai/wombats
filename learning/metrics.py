from typing import List, Dict
from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error

from wombats.automaton.dfa import SafetyDFA
from wombats.automaton.types import Symbols, Probabilities
from .training import SpecificationEstimator


class TestDataset:
    def __init__(self,
                 X_true: List[Symbols],
                 y_true: Probabilities,
                 X_est: List[Symbols] = None,
                 y_est: List[Symbols] = None):
        """
        :param X_true:      Traces sampled from a PDFA specification.
                            X_true = Specification.generate_traces()

        :param y_true:      Trace probabilities computed by the PDFA
                            specification, given X_true.
                            y_true = Specification(X_true)

        :param X_est:       Traces sampled from an estimated PDFA.
                            X_est = PDFA.generate_traces()

        :param y_est:       Trace probabilities computed by the PDFA
                            specification, given X_est.
                            y_est = Specification(X_est)
        """
        self.X_true = X_true
        self.y_true = y_true
        self.X_est = X_est
        self.y_est = y_est

    @property
    def X(self):
        if self.X_est is None:
            print('Skipping X_est')
            return self.X_true

        return self.X_true + self.X_est

    @property
    def y(self):
        if self.y_est is None:
            return self.y_true

        return self.y_true + self.y_est


class EvaluationFunction(metaclass=ABCMeta):
    """
    This is a base class for evaluating a learned automaton
    based on a metric
    """
    def __init__(self, metric_name=''):
        self.metric_name = metric_name

    @abstractmethod
    def __call__(self, estimator: SpecificationEstimator,
                 testdata: TestDataset, **kwargs) -> Dict[str, float]:
        """
        :param estimator:       A fitted estimator
        :param test_dataset:    A test dataset

        :return:                    A float value
        """
        msg = 'Evaluation Function must be implemented'
        raise NotImplementedError(msg)


class L2Norm(EvaluationFunction):
    def __init__(self, metric_name='L2 Norm of Trace Probabilities'):
        super().__init__(metric_name)

    def __call__(self, estimator: SpecificationEstimator,
                 testdata: TestDataset, **kwargs) -> Dict[str, float]:
        """
        Compute the L2 Norm of the probability error between
        the estimated value and the ground truth

        :param estimator:       A fitted estimator
        :param test_dataset:    A test dataset

        :return:                    A float value
        """
        if estimator.pdfa is None:
            return None

        y_true = testdata.y
        y_pred = estimator.predict(testdata.X)

        metric = mean_squared_error(y_true, y_pred)

        return {self.metric_name: metric}


class L1Norm(EvaluationFunction):
    def __init__(self, metric_name='L1 Norm of Trace Probabilities'):
        super().__init__(metric_name)

    def __call__(self, estimator: SpecificationEstimator,
                 testdata: TestDataset, **kwargs) -> Dict[str, float]:
        """
        Compute the L1 Norm of the probability error between
        the estimated value and the ground truth

        :param estimator:       A fitted estimator
        :param test_dataset:    A test dataset


        :return:                    A float value
        """
        if estimator.pdfa is None:
            return None

        y_true = testdata.y
        y_pred = estimator.predict(testdata.X)

        metric = mean_absolute_error(y_true, y_pred)

        return {self.metric_name: metric}


class Perplexity(EvaluationFunction):
    def __init__(self, metric_name='Perplexity'):
        super().__init__(metric_name)

    def __call__(self, estimator: SpecificationEstimator,
                 testdata: TestDataset, **kwargs) -> Dict[str, float]:
        """
        Compute the perplexity between
        the estimated probability dist and the ground truth

        :param estimator:       A fitted estimator
        :param test_dataset:    A test dataset

        :return:                    A float value
        """
        if estimator.pdfa is None:
            return None

        n_trace = len(testdata.y)

        metric = estimator.pdfa.perplexity(
            testdata.X,
            testdata.y) / n_trace

        return {self.metric_name: metric}


class ForwardKLWithEstDist(EvaluationFunction):
    def __init__(self, metric_name='Average False Forward KL'):
        super().__init__(metric_name)

    def __call__(self, estimator: SpecificationEstimator,
                 testdata: TestDataset, **kwargs) -> Dict[str, float]:
        """
        Compute the Forward KL divergence between
        the estimated probability dist and the ground truth,
        using the samples generated from the estimated automaton

        :param estimator:       A fitted estimator
        :param test_dataset:    A test dataset

        :return:                    A float value
        """
        if estimator.pdfa is None:
            return None

        if testdata.y_est is None:
            return None

        n_trace = len(testdata.y_est)

        metric = estimator.pdfa.kldivergence(
            testdata.X_est,
            testdata.y_est) / n_trace

        return {self.metric_name: metric}


class ReverseKLWithTargetDist(EvaluationFunction):
    def __init__(self, metric_name='Average False Reverse KL'):
        super().__init__(metric_name)

    def __call__(self, estimator: SpecificationEstimator,
                 testdata: TestDataset, **kwargs) -> Dict[str, float]:
        """
        Compute the Reverse KL divergence between
        the estimated probability dist and the ground truth,
        using the samples generated from the target automaton

        :param estimator:       A fitted estimator
        :param test_dataset:    A test dataset

        :return:                    A float value
        """
        if estimator.pdfa is None:
            return None

        n_trace = len(testdata.y_true)

        metric = estimator.pdfa.reverse_kldivergence(
            testdata.X_true,
            testdata.y_true) / n_trace

        return {self.metric_name: metric}


class ForwardKLWithTargetDist(EvaluationFunction):
    def __init__(self, metric_name='Average True Forward KL'):
        super().__init__(metric_name)

    def __call__(self, estimator: SpecificationEstimator,
                 testdata: TestDataset, **kwargs) -> Dict[str, float]:
        """
        Compute the Forward KL divergence between
        the estimated probability dist and the ground truth,
        using the samples generated from the target automaton

        :param estimator:       A fitted estimator
        :param test_dataset:    A test dataset

        :return:                    A float value
        """
        if estimator.pdfa is None:
            return None

        n_trace = len(testdata.y_true)

        metric = estimator.pdfa.kldivergence(
            testdata.X_true,
            testdata.y_true) / n_trace

        return {self.metric_name: metric}


class ReverseKLWithEstDist(EvaluationFunction):
    def __init__(self, metric_name='Average True Reverse KL'):
        super().__init__(metric_name)

    def __call__(self, estimator: SpecificationEstimator,
                 testdata: TestDataset, **kwargs) -> Dict[str, float]:
        """
        Compute the Reverse KL divergence between
        the estimated probability dist and the ground truth,
        using the samples generated from the estimated automaton

        :param estimator:       A fitted estimator
        :param test_dataset:    A test dataset

        :return:                    A float value
        """
        if estimator.pdfa is None:
            return None

        if testdata.y_est is None:
            return None

        n_trace = len(testdata.y_est)

        metric = estimator.pdfa.reverse_kldivergence(
            testdata.X_est,
            testdata.y_est) / n_trace

        return {self.metric_name: metric}


class NoOfNodes(EvaluationFunction):
    def __init__(self, metric_name='No of Nodes'):
        super().__init__(metric_name)

    def __call__(self, estimator: SpecificationEstimator,
                 testdata: TestDataset, **kwargs) -> Dict[str, float]:
        """
        Compute the number of nodes of the estimated automaton

        :param estimator:       A fitted estimator
        :param test_dataset:    A test dataset

        :return:                    A float value
        """
        if estimator.pdfa is None:
            return None

        metric = len(list(estimator.pdfa.nodes.keys()))

        return {self.metric_name: metric}


class NoOfEdges(EvaluationFunction):
    def __init__(self, metric_name='No of Edges'):
        super().__init__(metric_name)

    def __call__(self, estimator: SpecificationEstimator,
                 testdata: TestDataset, **kwargs) -> Dict[str, float]:
        """
        Compute the number of edges of the estimated automaton

        :param estimator:       A fitted estimator
        :param test_dataset:    A test dataset

        :return:                    A float value
        """
        if estimator.pdfa is None:
            return None

        metric = len(list(estimator.pdfa.edges.keys()))

        return {self.metric_name: metric}


class ShortestPathLength(EvaluationFunction):
    def __init__(self, metric_name='Shortest Path Length'):
        super().__init__(metric_name)

    def __call__(self, estimator: SpecificationEstimator,
                 testdata: TestDataset, **kwargs) -> Dict[str, float]:
        """
        Compute the shortest path on the estimated automaton

        :param estimator:       A fitted estimator
        :param test_dataset:    A test dataset

        :return:                    A float value
        """
        if estimator.pdfa is None:
            return None

        MPS, MPS_prob, _ = \
            estimator.pdfa.most_probable_string(disable_pbar=True)

        metric = len(MPS)

        return {self.metric_name: metric}


class ElapsedTime(EvaluationFunction):
    def __init__(self, metric_name='Average Elapsed Time'):
        super().__init__(metric_name)

    def __call__(self, estimator: SpecificationEstimator,
                 testdata: TestDataset, **kwargs) -> Dict[str, float]:
        """
        Compute the average elapsed time took to estimate
        the learned automaton

        :param estimator:       A fitted estimator
        :param test_dataset:    A test dataset

        :return:                    A float value
        """
        metric = estimator.flexfringe.average_elapsed_time

        return {self.metric_name: metric}


class Safe(EvaluationFunction):
    def __init__(self, safe_specification: SafetyDFA,
                 metric_name='Safe'):
        super().__init__(metric_name)
        self.safe_specification = safe_specification

    def __call__(self, estimator: SpecificationEstimator,
                 testdata: TestDataset, **kwargs) -> Dict[str, float]:
        """
        Check whether the estimated automaton avoids
        the violation of the safety specification

        :param estimator:       A fitted estimator
        :param test_dataset:    A test dataset

        :return:                    A float value
        """
        if estimator.pdfa is None:
            return {self.metric_name: int(False)}

        metric = int(self.safe_specification.is_safe(estimator.pdfa))

        return {self.metric_name: metric}


class EvaluationFunctions(EvaluationFunction, metaclass=ABCMeta):
    def __init__(self, metrics: Dict[str, EvaluationFunction],
                 **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics
        """A dict of metric names to its evaluation function"""

    def __call__(self, estimator: SpecificationEstimator,
                 testdata: TestDataset, **kwargs) -> Dict[str, float]:
        """
        :param estimator:       A fitted estimator
        :param test_dataset:    A test dataset

        :return:                A dict of results
        """
        results = {}
        for name, func in self.metrics.items():
            results[name] = list(func(estimator, testdata).values())[0]

        return results


class DefaultEvaluationFunctions(EvaluationFunctions):
    def __init__(self, safety_specification: SafetyDFA):
        """
        :param safety_specification:            Safety DFA if needed
        """
        metrics = {
            'Average True Forward KL': ForwardKLWithTargetDist(),
            'Average True Reverse KL': ReverseKLWithEstDist(),
            'Average False Forward KL': ForwardKLWithEstDist(),
            'Average False Reverse KL': ReverseKLWithTargetDist(),
            'L1 Norm of Trace Probabilities': L1Norm(),
            'L2 Norm of Trace Probabilities': L2Norm(),
            'No of Nodes': NoOfNodes(),
            'No of Edges': NoOfEdges(),
            'Elapsed Time': ElapsedTime(),
            'Safe': Safe(safety_specification)}
        super().__init__(metrics)

