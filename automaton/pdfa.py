# 3rd-party packages
import multiprocessing
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.random import RandomState
from joblib import Parallel, delayed
from collections.abc import Iterable
from typing import List, Hashable, Tuple, Callable

# local packages
from wombats.factory.builder import Builder
from .base import Automaton, NXNodeList, NXEdgeList
from .fdfa import FDFA

# needed for pred_method type hint annotations
Trans = (str, int, float)
Categorical_data = (List[float], List[int], List[Hashable])

# needed for multi-threaded sampling routine
NUM_CORES = multiprocessing.cpu_count()


def check_predict_method(prediction_function: Callable):
    """
    decorator to check an enumerated typestring for prediction method.
    pred_method:  The pred_method string to check. one of: {'sample',
    'max_prob'}

    :type       prediction_function:  function handle to check. Must have
                                      keyword argument: 'pred_method'
    :param      prediction_function:  the function to decorate

    :raises     ValueError: raises if:
                                - pred_method is not a keyword argument
                                - pred_method is not one of allowed methods
    """

    def checker(*args, **kwargs):

        # checking if the decorator has been applied to an appropriate function
        print(args, kwargs)
        if 'pred_method' not in kwargs:
            f_name = prediction_function.__name__
            msg = f'"pred_method" is not a kwarg of {f_name}'
            raise ValueError(msg)

        pred_method = kwargs['pred_method']

        # checking for the enumerated types
        allowed_methods = ['max_prob', 'sample']

        if pred_method not in allowed_methods:
            msg = f'pred_method: "{pred_method}" must be one of allowed ' + \
                  f'methods: {allowed_methods}'
            raise ValueError(msg)

        return prediction_function(*args, **kwargs)

    return checker


class PDFA(Automaton):
    """
    This class describes a probabilistic deterministic finite automaton (pdfa).

    built on networkx, so inherits node and edge data structure definitions

    inherits some of its api from the NLTK LM API

    Node Attributes
    -----------------
        - final_probability: final state probability for the node
        - trans_distribution: a sampled-able function to select the next state
          and emitted symbol
        - is_accepting: a boolean flag determining whether the pdfa considers
          the node accepting

    Edge Properties
    -----------------
        - symbol: the symbol value emitted when the edge is traversed
        - probability: the probability of selecting this edge for traversal
    """

    def __init__(self, nodes: NXNodeList, edge_list: NXEdgeList,
                 alphabet_size: int, num_states: int, start_state,
                 beta: float = 0.95,
                 smooth_transitions: bool = False,
                 is_stochastic: bool = False,
                 final_transition_sym: str = -1) -> 'PDFA':
        """
        Constructs a new instance of a PDFA object.

        :param      nodes:                 node list as expected by
                                           networkx.add_nodes_from()
        :type       nodes:                 list of tuples: (node label, node,
                                           attribute dict)
        :param      edges:                 edge list as expected by
                                           networkx.add_edges_from()
        :type       edges:                 list of tuples: (src node label,
                                           dest node label, edge attribute
                                           dict)
        :param      alphabet_size:         number of symbols in fdfa alphabet
        :type       alphabet_size:         Int
        :param      num_states:            number of states in automaton state
                                           space
        :type       num_states:            Int
        :param      start_state:           unique start state string label of
                                           fdfa
        :type       start_state:           same type as PDFA.nodes node object
        :param      beta:                  the final state probability needed
                                           for a state to accept. Not used for
                                           PDFA (default None)
        :type       beta:                  Float
        :param      smooth_transitions:    whether to smooth the symbol
                                           transitions distributions
        :param      is_stochastic:         the transitions are
                                           non-probabilistic, so we are going
                                           to assign a uniform distribution
                                           over all symbols for the purpose of
                                           generation
        :param      final_transition_sym:  representation of the empty string /
                                           symbol (a.k.a. lambda) (default -1)
        :type       final_transition_sym:  same type as PDFA.edges symbol
                                           property
        """

        # need to start with a fully initialized automaton
        super().__init__(nodes, edge_list, alphabet_size, num_states,
                         start_state, smooth_transitions,
                         is_stochastic, final_transition_sym=-1)

        self._transition_map = {}
        """keep a map of start state label and symbol to destination state"""

        self._beta = beta
        """the final state probability needed for a state to accept"""

        self._smoothing_amount = 0.00001
        """probability mass to re-assign to unseen symbols at each node"""

        self._initialize_node_edge_properties(
            final_weight_key='final_probability',
            can_have_accepting_nodes=True,
            edge_weight_key='probability')

    def generate_traces(self, num_samples: int, N: int) -> (List[List[int]],
                                                            List[int],
                                                            List[float]):
        """
        generates num_samples random traces from the pdfa

        :param      num_samples:  The number of trace samples to generate
        :type       num_samples:  scalar int
        :param      N:            maximum length of trace
        :type       N:            scalar integer

        :returns:   list of sampled traces,
                    list of the associated trace lengths,
                    list of the associated trace probabilities
        :rtype:     tuple(list(list(int)), list(int), list(float))
        """

        start_state = self.start_state

        # make sure the num_samples is an int, so you don't have to wrap shit
        # in an 'int()' every time...
        num_samples = int(num_samples)

        iters = range(0, num_samples)
        results = Parallel(n_jobs=NUM_CORES, verbose=1)(
            delayed(self.generate_trace)(start_state, N) for i in iters)

        samples, trace_lengths, trace_probs = zip(*results)

        return samples, trace_lengths, trace_probs

    def write_traces_to_file(self, traces: List[List[int]], num_samples: int,
                             trace_lengths: List[int], f_name: str) -> None:
        """
        Writes trace samples to a file in the abbadingo format for use in
        grammatical inference tools like flexfringe

        :param      traces:         The traces to write to a file
        :type       traces:         List[List[int]]
        :param      num_samples:    The number sampled traces
        :type       num_samples:    int
        :param      trace_lengths:  list of sampled trace lengths
        :type       trace_lengths:  List[int]
        :param      f_name:         The file name to write to
        :type       f_name:         str
        """

        # make sure the num_samples is an int, so you don't have to wrap shit
        # in an 'int()' every time...
        num_samples = int(num_samples)

        with open(f_name, 'w+') as f:

            # need the header to be:
            # number_of_training_samples size_of_alphabet
            f.write(str(num_samples) + ' ' + str(self._alphabet_size) + '\n')

            for trace, trace_length in zip(traces, trace_lengths):
                f.write(self._get_abbadingo_string(trace, trace_length,
                                                   is_pos_example=True))

    @staticmethod
    def convert_states_edges(nodes: dict, edges: dict) -> (NXNodeList,
                                                           NXEdgeList):
        """
        Converts node and edges data from a manually specified YAML config file
        to the format needed by:
            - networkx.add_nodes_from()
            - networkx.add_edges_from()

        :param      nodes:  dict of node objects to be converted
        :type       nodes:  dict of node label to node properties
        :param      edges:  dictionary adj. list to be converted
        :type       edges:  dict of src node label to dict of dest label to
                            edge properties

        :returns:   properly formated node and edge list containers
        :rtype:     tuple: ( nodes - list of tuples: (node label, node
                    attribute dict), edges - list of tuples: (src node label,
                    dest node label, edge attribute dict) )
        """

        # need to convert the configuration adjacency list given in the config
        # to an edge list given as a 3-tuple of (source, dest, edgeAttrDict)
        edge_list = []
        for source_node, dest_edges_data in edges.items():

            # don't need to add any edges if there is no edge data
            if dest_edges_data is None:
                continue

            for dest_node in dest_edges_data:

                symbols = dest_edges_data[dest_node]['symbols']
                probabilities = dest_edges_data[dest_node]['probabilities']

                for symbol, probability in zip(symbols, probabilities):

                    edge_data = {'symbol': symbol, 'probability': probability}
                    newEdge = (source_node, dest_node, edge_data)
                    edge_list.append(newEdge)

        # best convention is to convert dict_items to a list, even though both
        # are iterable
        converted_nodes = list(nodes.items())

        return converted_nodes, edge_list

    def predict(self, symbols: List[int],
                pred_method: str = 'max_prob') -> int:
        """
        predicts the next symbol conditioned on the given previous symbols

        :param      symbols:      The previously observed emitted symbols
        :type       symbols:      list of symbols strings
        :param      pred_method:  The method used to choose the next
                                  state. see _choose_next_state for details on
                                  how each pred_method is implemented.
                                  {'sample', 'max_prob'} (default 'max_prob')
        :type       pred_method:  str

        :returns:   the most probable next symbol in the sequence
        :rtype:     str
        """

        # simulating the state trajectory under the given sequence
        state = self.start_state

        for symbol in symbols:
            state, _ = self._get_next_state(state, symbol)

        # now making the next state prediction based on the "causal" model
        # state induced by the deterministic sequence governed by the
        # observed symbols
        _, next_symbol, _ = self._choose_next_state(state)

        return next_symbol

    def generate_trace(self, start_state: Hashable, N: int,
                       random_state: RandomState = None) -> (List[int], int,
                                                             float):
        """
        Generates a trace from the pdfa starting from start_state

        :param      start_state:   the state label to start sampling traces
                                   from
        :type       start_state:   Hashable
        :param      N:             maximum length of trace
        :type       N:             scalar integer
        :param      random_state:  The np.random.RandomState() seed parameter
                                   for sampling from the state transition
                                   distribution. Defaulting to None causes the
                                   seed to reset. (default None)
        :type       random_state:  {None, int, array_like}

        :returns:   the sequence of symbols emitted, the length of the trace,
                    the probability of the trace in the language of the pdfa
        :rtype:     tuple(List[int], integer, float)
        """

        curr_state = start_state
        length_of_trace = 1
        trace_prob = 1.0

        (next_state,
         next_symbol,
         trans_probability) = self._choose_next_state(curr_state, random_state)

        sampled_trace = [next_symbol]
        curr_state = next_state
        at_terminal_state = next_symbol == self._final_transition_sym
        trace_prob *= trans_probability

        while (not at_terminal_state and length_of_trace <= N):
            (next_state,
             next_symbol,
             trans_probability) = self._choose_next_state(curr_state,
                                                          random_state)

            if next_symbol == self._final_transition_sym:
                break

            sampled_trace.append(next_symbol)
            length_of_trace += 1
            curr_state = next_state
            trace_prob *= trans_probability

        return sampled_trace, length_of_trace, trace_prob

    def plot_node_trans_dist(self, curr_state: Hashable) -> None:
        """!
        Plots the transition pmf at the given curr_state / node.

        :param      curr_state:  state to display its transition distribution
        :type       curr_state:  Hashable
        """

        trans_dist = self._get_node_data(curr_state, 'trans_distribution')

        fig, ax = plt.subplots(1, 1)
        ax.plot(trans_dist.xk, trans_dist.pmf(trans_dist.xk), 'ro',
                ms=12, mec='r')
        ax.vlines(trans_dist.xk, 0, trans_dist.pmf(trans_dist.xk),
                  colors='r', lw=4)
        plt.show()

    def score(self, trace: List[int]) -> float:
        """
        Calculates the given trace's probability in the language of the PDFA.

        PDFA is a language model (LM) in this case:
            ==> score = P_{PDFA LM}(trace)

        :param      trace:  The trace
        :type       trace:  List[int]

        :returns:   The trace probability.
        :rtype:     float
        """

        curr_state = self.start_state
        trace_prob = 1.0

        for symbol in trace:
            next_state, trans_probability = self._get_next_state(curr_state,
                                                                 symbol)

            trace_prob *= trans_probability
            curr_state = next_state

        return trace_prob

    def logscore(self, trace: List[int], base: float = 2.0) -> float:
        """
        computes the log of the score (sequence probability) of the given trace
        in the language of the PDFA

        :param      trace:  The sequence of symbols to compute the log score of
        :type       trace:  List[int]
        :param      base:   The log base. Commonly set to 2 in classic
                            information theory literature (default 2.0)
        :type       base:   float

        :returns:   log of the probability - NOT log odds
        :rtype:     float
        """

        score = self.score(trace)

        return np.asscalar(np.log(score) / np.log(base))

    def cross_entropy_approx(self, trace: List[int],
                             base: float = 2.0) -> float:
        """
        computes approximate cross-entropy of the given trace in the language
        of the PDFA

        Here, we are using the Shannon-McMillian-Breiman theorem to define
        the cross-entropy of the trace, given that we sampled the trace from
        the actual target distribution and we are evaluating it in the PDFA LM.
        Then, as a PDFA is a stationary ergodic stochastic process model, we
        can calculate the cross-entropy as (eq. 3.49 from SLP ch3):

            trace ~ target
            H(target, model) = lim {(- 1 / n) * log(P_{model}(trace))}
                             n -> inf

        where:

            H(target) <= H(target, model)

        The finite-length approximation to the cross-entropy is then given by
        (eq. 3.51 from SLP ch3):

            H(trace) = (- 1 / N) log(P_{model}(trace))

        References:
        NLTK.lm.api
        Speech and Language Processing (SLP), 3 ed., Ch3
        (https://web.stanford.edu/~jurafsky/slp3/3.pdf)

        :param      trace:  The sequence of symbols to evaluate
        :type       trace:  List[int]
        :param      base:   The log base. Commonly set to 2 in classic
                            information theory literature (default 2.0)
        :type       base:   float

        :returns:   the approximate cross-entropy of the given trace
        :rtype:     float
        """

        N = len(trace)

        return (-1.0 / N) * self.logscore(trace, base)

    def perplexity_approx(self, trace: List[int], base: float = 2.0) -> float:
        """
        computes approximate perplexity of the given trace in the language of
        the PDFA

        The approximate perplexity is based on computing the approximate
        cross-entropy (cross_entropy_approximate) (eq. 3.52 of SLP).

        References:
        NLTK.lm.api
        Speech and Language Processing (SLP), 3 ed., Ch3
        (https://web.stanford.edu/~jurafsky/slp3/3.pdf)

        :param      trace:  The sequence of symbols to evaluate
        :type       trace:  List[int]
        :param      base:   The log base used for log probability calculations
                            of the approximate cross-entropy underpinning the
                            perplexity. Commonly set to 2 in classic
                            information theory literature (default 2.0)
        :type       base:   float

        :returns:   the approximate perplexity of the given trace
        :rtype:     float
        """

        return base ** self.cross_entropy_approx(trace, base)

    def cross_entropy(self, traces: List[List[int]],
                      actual_trace_probs: List[float],
                      base: float = 2.0) -> float:
        """
        computes actual cross-entropy of the given traces in the language of
        the PDFA on the given actual trace probabilities

        References:
        Speech and Language Processing (SLP), 3 ed., Ch3
        (https://web.stanford.edu/~jurafsky/slp3/3.pdf)

        :param      traces:              The list of sequence of symbols to
                                         evaluate the model's actual cross
                                         entropy on.
        :type       traces:              List[int]
        :param      actual_trace_probs:  The actual probability of each trace
                                         in the target language distribution
        :type       actual_trace_probs:  List[float]
        :param      base:                The log base. Commonly set to 2 in
                                         classic information theory literature
                                         (default 2.0)
        :type       base:                float

        :returns:   the actual cross-entropy of the given trace
        :rtype:     float
        """

        cross_entropy_sum = 0.0

        for target_prob, trace in zip(actual_trace_probs, traces):
            cross_entropy_sum += target_prob * self.logscore(trace, base)

        N = len(actual_trace_probs)

        return (-1.0 / N) * cross_entropy_sum

    def perplexity(self, traces: List[List[int]],
                   actual_trace_probs: List[float],
                   base: float = 2.0) -> float:
        """
        computes actual perplexity of the given traces in the language of
        the PDFA on the given actual trace probabilities

        References:
        Speech and Language Processing (SLP), 3 ed., Ch3
        (https://web.stanford.edu/~jurafsky/slp3/3.pdf)

        :param      traces:              The list of sequence of symbols to
                                         evaluate the model's actual cross
                                         entropy on.
        :type       traces:              List[int]
        :param      actual_trace_probs:  The actual probability of each trace
                                         in the target language distribution
        :type       actual_trace_probs:  List[float]
        :param      base:                The log base. Commonly set to 2 in
                                         classic information theory literature
                                         (default 2.0)
        :type       base:                float

        :returns:   the actual cross-entropy of the given trace
        :rtype:     float
        """

        return base ** self.cross_entropy(traces, actual_trace_probs, base)

    def predictive_accuracy(self, test_traces: List[List[int]],
                            pred_method: str = 'max_prob') -> float:
        """
        compares the model's predictions to the actual values of the next
        symbol and returns the ratio of correct predictions.

        :param      test_traces:  The traces to compute predictive accuracy for
        :type       test_traces:  List
        :param      pred_method:  The method used to choose the next state.
                                  see _choose_next_state for details on how
                                  each pred_method is implemented.
                                  {'sample', 'max_prob'} (default 'max_prob')
        :type       pred_method:       str

        :returns:   predictive accuracy ratio of the model on the given traces
        :rtype:     float from [0, 1]
        """

        N = len(test_traces)
        num_correct_predictions = 0

        for trace in test_traces:

            observations = trace[:-1]
            actual_symbol = trace[-1]

            # check the predictive capability when conditioned on all but the
            # last symbol
            predicted_symbol = self.predict(observations, pred_method)

            if predicted_symbol == actual_symbol:
                num_correct_predictions += 1

        return num_correct_predictions / N

    @classmethod
    def _fdfa_to_pdfa_data(cls, fdfa: FDFA) -> Tuple[NXNodeList, NXEdgeList]:
        """
        convert fdfa nodes and edges to pdfa nodes and edges

        :param      cls:   The class instance reference (not an object instance
                           reference)
        :type       cls:   PDFA class data reference
        :param      fdfa:  The fdfa to convert to a PDFA
        :type       fdfa:  FDFA

        :returns:   nodes, edges lists with all data initialized for creation
                    of pdfa from networkx.add_nodes_from() and
                    networkx.add_edges_from()
        :rtype:     list of tuples: (node label, node, attribute dict),
                    list of tuples: (src node label, dest node label,
                                     edge attribute dict)
        """

        fdfa_nodes = fdfa.nodes(data=True)
        pdfa_nodes = []
        pdfa_edges = []

        # converting final state frequencies to final state probabilities
        for curr_node, curr_node_data in fdfa_nodes:

            # the final probability is just how often the execution ends at the
            # curr_node divided by the all of sum of frequencies over all
            # possible transitions from that node
            final_freq = fdfa._get_node_data(curr_node, 'final_frequency')
            out_freq = fdfa._get_node_data(curr_node, 'out_frequency')
            self_freq = fdfa._get_node_data(curr_node, 'self_frequency')
            number_of_choices = final_freq + out_freq + self_freq
            new_final_probability = final_freq / number_of_choices

            new_node_data = {'final_probability': new_final_probability,
                             'trans_distribution': None,
                             'is_accepting': None}
            pdfa_nodes.append((curr_node, new_node_data))

            # converting transition frequencies to transition probabilities
            #
            # the edge transition probability is the edge's frequency divided
            # by the the number of time you either ended or transitioned out
            # of the that node
            for node_post in fdfa.successors(curr_node):

                curr_edges_out = fdfa.get_edge_data(curr_node, node_post)

                for _, curr_out_edge_data in curr_edges_out.items():

                    edge_freq = curr_out_edge_data['frequency']
                    symbol = curr_out_edge_data['symbol']
                    trans_probability = edge_freq / number_of_choices
                    new_edge_data = {'symbol': symbol,
                                     'probability': trans_probability}

                    new_edge = (curr_node,
                                node_post,
                                new_edge_data)

                    pdfa_edges.append(new_edge)

        return pdfa_nodes, pdfa_edges

    def _get_next_state(self, curr_state: Hashable,
                        symbol: str) -> Tuple[Hashable, float]:
        """
        Gets the next state given the current state and the "input" symbol.

        :param      curr_state:  The curr state
        :type       curr_state:  Hashable
        :param      symbol:      The input symbol
        :type       symbol:      str

        :returns:   (The next state label, the transition probability)
        :rtype:     (Hashable, float)
        """

        trans_distribution = self._get_node_data(curr_state,
                                                 'trans_distribution')
        possible_symbols = trans_distribution.xk
        probabilities = trans_distribution.pk

        if symbol not in possible_symbols:
            msg = ('given symbol ({}) is not found in the '
                   'curr_state\'s ({}) '
                   'transition distribution').format(symbol, curr_state)
            raise ValueError(msg)

        symbol_idx = np.where(possible_symbols == symbol)
        num_matched_symbols = len(symbol_idx)
        if num_matched_symbols != 1:
            msg = ('given symbol ({}) is found multiple times in '
                   'curr_state\'s ({}) '
                   'transition distribution').format(symbol, curr_state)
            raise ValueError(msg)

        # stored in numpy array, so we just want the float probability value
        symbol_probability = np.asscalar(probabilities[symbol_idx])

        if symbol_probability == 0.0:
            msg = ('symbol ({}) has zero probability of transition in the '
                   'pdfa from curr_state: {}').format(symbol, curr_state)
            raise ValueError(msg)

        next_state = self._transition_map[(curr_state, symbol)]

        return next_state, symbol_probability

    def _set_state_acceptance(self, curr_state: Hashable, beta: float) -> None:
        """
        Sets the state acceptance property for the given state.

        If curr_state's final_probability >= beta, then the state accepts

        :param      curr_state:  The current state's node label
        :type       curr_state:  Hashable
        :param      beta:        The cut point final state probability
                                 acceptance parameter for the PDFA
        :type       beta:        float
        """

        curr_final_prob = self._get_node_data(curr_state, 'final_probability')

        if curr_final_prob >= self._beta:
            state_accepts = True
        else:
            state_accepts = False

        self._set_node_data(curr_state, 'is_accepting', state_accepts)

    def _smooth_categorical(self, curr_state: Hashable,
                            edge_probs: List[float],
                            edge_symbols: List[int],
                            edge_dests: List[Hashable]) -> Categorical_data:
        """
        Applies Laplace smoothing to the given categorical state-symbol
        distribution

        :param      curr_state:    The current state label for which to smooth
                                   the distribution
        :type       curr_state:    Hashable
        :param      edge_probs:    The transition probability values for each
                                   edge
        :type       edge_probs:    list of floats
        :param      edge_symbols:  The emitted symbols for each edge
        :type       edge_symbols:  list of integer symbols
        :param      edge_dests:    The labels of the destination states under
                                   each symbol at the curr_state
        :type       edge_dests:    label

        :returns:   The smoothed version of edge_probs, edge_symbols,
                    edge_dests
        :rtype:     Tuple(list(float), list(int), list(node_label_type))
        """

        all_possible_trans = [idx for idx, prob in enumerate(edge_probs) if
                              prob > 0.0]
        num_orig_samples = len(all_possible_trans)

        # here we add in the missing transition probabilities as just very
        # unlikely self-loops
        num_of_missing_transitions = 0
        new_edge_probs, new_edge_dests, new_edge_symbols = [], [], []
        for i in range(self._alphabet_size):
            if i not in edge_symbols:

                num_of_missing_transitions += 1
                new_edge_probs.append(self._smoothing_amount)
                new_edge_dests.append(curr_state)
                new_edge_symbols.append(i)

        # now, we need to remove the smoothed probability mass from the
        # original transition distribution
        num_added_symbols = len(new_edge_symbols)
        added_prob_mass = self._smoothing_amount * num_added_symbols
        smoothing_per_orig_trans = added_prob_mass / num_orig_samples

        for trans_idx in all_possible_trans:
            edge_probs[trans_idx] -= smoothing_per_orig_trans

        # combining the new transitions with the smoothed, original
        # distribution to get the final smoothed distribution
        edge_probs += new_edge_probs
        edge_dests += new_edge_dests
        edge_symbols += new_edge_symbols

        return edge_probs, edge_dests, edge_symbols

    def _choose_next_state(self, curr_state: Hashable,
                           random_state: {None, int, Iterable}=None,
                           pred_method: str = 'sample') -> Trans:
        """
        Chooses the next state based on curr_state's transition distribution

        :param      curr_state:    The current state label
        :type       curr_state:    Hashable
        :param      random_state:  The np.random.RandomState() seed parameter
                                   for sampling from the state transition
                                   distribution. Defaulting to None causes the
                                   seed to reset. (default None)
        :type       random_state:  {None, int, Iterable}
        :param      pred_method:   The method used to choose the next state:
                                   'sample':
                                   sample from the transition
                                   distribution of the casual state of the PDFA
                                   (the state the machine is left in after the
                                   sequence of observations). makes
                                   non-deterministic predictions.
                                   'max_prob':
                                   like many language models, the selection of
                                   the next state s_{t+1}, and thus the next
                                   emitted symbol, conditioned on the set of
                                   observation symbols O_t = {o_1, ..., o_t}
                                   is:
                                   s_{t+1} = argmax_{s'}P(s' | s_t, O_t)
                                   makes deterministic predictions.
                                   {'sample', 'max_prob'} (default 'max_prob')
        :type       pred_method:        str

        :returns:   The next state's label, the symbol emitted by changing
                    states, the probability of this transition occurring
        :rtype:     tuple(string, int, float)
        """

        trans_dist = self.nodes[curr_state]['trans_distribution']

        # critical step for use with parallelized libraries. This must be reset
        # before sampling, as otherwise each of the threads is using the same
        # seed, and we get lots of duplicated strings
        trans_dist.random_state = RandomState(random_state)

        # sampling an action (symbol) from the state-action distribution at
        # curr_state
        next_symbol = trans_dist.rvs(size=1)[0]

        if next_symbol == self._final_transition_sym:
            trans_probability = 1.0
            return curr_state, next_symbol, trans_probability
        else:
            next_state, trans_probability = self._get_next_state(curr_state,
                                                                 next_symbol)
            return next_state, next_symbol, trans_probability

    def _get_abbadingo_string(self, trace: List[int], trace_length: int,
                              is_pos_example: bool) -> str:
        """
        Returns the Abbadingo (sigh) formatted string given a trace string and
        the label for the trace

        :param      trace:           The trace string to represent in Abbadingo
        :type       trace:           List[int]
        :param      trace_length:    The trace length
        :type       trace_length:    integer
        :param      is_pos_example:  Indicates if the trace is a positive
                                     example of the pdfa
        :type       is_pos_example:  boolean

        :returns:   The abbadingo formatted string for the given trace
        :rtype:     string
        """
        trace = ' '.join(str(x) for x in trace)

        trace_label = {False: '0', True: '1'}[is_pos_example]
        return trace_label + ' ' + str(trace_length) + ' ' + trace + '\n'


class PDFABuilder(Builder):
    """
    Implements the generic automaton builder class for PDFA objects
    """

    def __init__(self) -> 'PDFABuilder':
        """
        Constructs a new instance of the PDFABuilder
        """

        # need to call the super class constructor to gain its properties
        Builder.__init__(self)

        # keep these properties so we don't re-initialize unless underlying
        # data changes
        self.nodes = None
        self.edges = None

    def __call__(self, graph_data: {str, FDFA},
                 graph_data_format: str = 'yaml') -> PDFA:
        """
        Returns an initialized PDFA instance given the graph_data

        graph_data and graph_data_format must match

        :param      graph_data:         The object containing graph data
        :type       graph_data:         {str, FDFA}
        :param      graph_data_format:  The graph data file format.
                                        (default 'yaml')
                                        {'yaml', 'fdfa_object'}
        :type       graph_data_format:  string

        :returns:   instance of an initialized PDFA object
        :rtype:     PDFA

        :raises     ValueError:         checks if graph_data and
                                        graph_data_format have a
                                        compatible data loader
        """

        if graph_data_format == 'yaml':
            self._instance = self._from_yaml(graph_data)
        elif graph_data_format == 'fdfa_object':
            self._instance = self._from_fdfa(graph_data)
        else:
            msg = 'graph_data_format ({}) must be one of: "yaml", ' + \
                  '"fdfa_object"'.format(graph_data_format)
            raise ValueError(msg)

        return self._instance

    def _from_yaml(self, graph_data_file: str) -> PDFA:
        """
        Returns an instance of a PDFA from the .yaml graph_data_file

        Only reads the config data once, otherwise just returns the built
        object

        :param      graph_data_file:  The graph configuration file name
        :type       graph_data_file:  filename path string

        :returns:   instance of an initialized PDFA object
        :rtype:     PDFA

        :raises     ValueError:       checks if graph_data_file's ext is YAML
        """

        _, file_extension = os.path.splitext(graph_data_file)

        allowed_exts = ['.yaml', '.yml']
        if file_extension in allowed_exts:
            config_data = self.load_YAML_config_data(graph_data_file)
        else:
            msg = 'graph_data_file ({}) is not a ({}) file'
            raise ValueError(msg.format(graph_data_file, allowed_exts))

        nodes_have_changed = (self.nodes != config_data['nodes'])
        edges_have_changed = (self.edges != config_data['edges'])
        no_instance_loaded_yet = (self._instance is None)

        if no_instance_loaded_yet or nodes_have_changed or edges_have_changed:

            # nodes and edge_list must be in the format needed by:
            #   - networkx.add_nodes_from()
            #   - networkx.add_edges_from()
            states, edges = PDFA.convert_states_edges(config_data['nodes'],
                                                      config_data['edges'])

            # saving these so we can just return initialized instances if the
            # underlying data has not changed
            self.nodes = states
            self.edges = edges

            self._instance = PDFA(
                nodes=states,
                edge_list=edges,
                beta=config_data['beta'],
                alphabet_size=config_data['alphabet_size'],
                num_states=config_data['num_states'],
                final_transition_sym=config_data['final_transition_sym'],
                start_state=config_data['start_state'],
                smooth_transitions=True,
                is_stochastic=True)

            return self._instance

    def _from_fdfa(self, fdfa: FDFA) -> PDFA:
        """
        Returns an instance of a PDFA from an instance of FDFA

        :param      fdfa:  initialized fdfa instance to convert to a pdfa
        :type       fdfa:  FDFA

        :returns:   instance of an initialized PDFA object
        :rtype:     PDFA
        """

        nodes, edges = PDFA._fdfa_to_pdfa_data(fdfa)

        # saving these so we can just return initialized instances if the
        # underlying data has not changed
        self.nodes = nodes
        self.edges = edges

        self._instance = PDFA(
            nodes=nodes,
            edge_list=edges,
            # just choose a default value, FDFAs have no notion of acceptance
            # this at the moment
            beta=0.95,
            alphabet_size=fdfa._alphabet_size,
            num_states=fdfa._num_states,
            final_transition_sym=fdfa._final_transition_sym,
            start_state=fdfa.start_state)

        return self._instance
