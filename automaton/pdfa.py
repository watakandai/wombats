# 3rd-party packages
import numpy as np
import os
from typing import List, Hashable, Tuple, Callable
from bidict import bidict

# local packages
from wombats.factory.builder import Builder
from .base import (Automaton, NXNodeList, NXEdgeList, Node, Symbol, Symbols,
                   Probabilities)
from .fdfa import FDFA


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

    def __init__(self,
                 nodes: NXNodeList,
                 edges: NXEdgeList,
                 symbol_display_map: bidict,
                 alphabet_size: int,
                 num_states: int,
                 start_state,
                 beta: float = 0.95,
                 smooth_transitions: bool = True,
                 final_transition_sym: Hashable = -1) -> 'PDFA':
        """
        Constructs a new instance of a PDFA object.

        :param      nodes:                 node list as expected by
                                           networkx.add_nodes_from() list of
                                           tuples: (node label, node, attribute
                                           dict)
        :param      edges:                 edge list as expected by
                                           networkx.add_edges_from() list of
                                           tuples: (src node label, dest node
                                           label, edge attribute dict)
        :param      symbol_display_map:    bidirectional mapping of hashable
                                           symbols, to a unique integer index
                                           in the symbol map. Needed to
                                           translate between the indices in the
                                           transition distribution and the
                                           hashable representation which is
                                           meaningful to the user
        :param      alphabet_size:         number of symbols in fdfa alphabet
        :param      num_states:            number of states in automaton state
                                           space
        :param      start_state:           unique start state string label of
                                           fdfa
        :param      beta:                  the final state probability needed
                                           for a state to accept. Not used for
                                           PDFA (default None)
        :param      smooth_transitions:    whether to smooth the symbol
                                           transitions distributions
        :param      final_transition_sym:  representation of the empty string /
                                           symbol (a.k.a. lambda) (default -1)
        """

        self._beta = beta
        """the final state probability needed for a state to accept"""

        self._smoothing_amount = 0.00001
        """probability mass to re-assign to unseen symbols at each node"""

        # need to start with a fully initialized automaton
        super().__init__(nodes, edges, symbol_display_map,
                         alphabet_size, num_states, start_state,
                         final_transition_sym=final_transition_sym,
                         smooth_transitions=True,
                         is_stochastic=True,
                         final_weight_key='final_probability',
                         can_have_accepting_nodes=True,
                         edge_weight_key='probability')

    def write_traces_to_file(self, traces: List[Symbols], num_samples: int,
                             trace_lengths: List[int], f_name: str) -> None:
        """
        Writes trace samples to a file in the abbadingo format for use in
        grammatical inference tools like flexfringe

        :param      traces:         The traces to write to a file
        :param      num_samples:    The number sampled traces
        :param      trace_lengths:  list of sampled trace lengths
        :param      f_name:         The file name to write to
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

    def predict(self, symbols: Symbols,
                pred_method: str = 'max_prob') -> Symbol:
        """
        predicts the next symbol conditioned on the given previous symbols

        :param      symbols:      The previously observed emitted symbols
        :param      pred_method:  The method used to choose the next state. see
                                  _choose_next_state for details on how each
                                  pred_method is implemented.
                                  {'sample', 'max_prob'} (default 'max_prob')

        :returns:   the most probable next symbol in the sequence
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

    def score(self, trace: Symbols) -> float:
        """
        Calculates the given trace's probability in the language of the PDFA.

        PDFA is a language model (LM) in this case:
            ==> score = P_{PDFA LM}(trace)

        :param      trace:  The trace

        :returns:   The trace probability.
        """

        curr_state = self.start_state
        trace_prob = 1.0

        for symbol in trace:
            next_state, trans_probability = self._get_next_state(curr_state,
                                                                 symbol)

            trace_prob *= trans_probability
            curr_state = next_state

        return trace_prob

    def logscore(self, trace: Symbols, base: float = 2.0) -> float:
        """
        computes the log of the score (sequence probability) of the given trace
        in the language of the PDFA

        :param      trace:  The sequence of symbols to compute the log score of
        :param      base:   The log base. Commonly set to 2 in classic
                            information theory literature (default 2.0)

        :returns:   log of the probability - NOT log odds
        """

        score = self.score(trace)

        return np.asscalar(np.log(score) / np.log(base))

    def cross_entropy_approx(self, trace: Symbols,
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
        :param      base:   The log base. Commonly set to 2 in classic
                            information theory literature (default 2.0)

        :returns:   the approximate cross-entropy of the given trace
        """

        N = len(trace)

        return (-1.0 / N) * self.logscore(trace, base)

    def perplexity_approx(self, trace: Symbols, base: float = 2.0) -> float:
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
        :param      base:   The log base used for log probability calculations
                            of the approximate cross-entropy underpinning the
                            perplexity. Commonly set to 2 in classic
                            information theory literature (default 2.0)

        :returns:   the approximate perplexity of the given trace
        """

        return base ** self.cross_entropy_approx(trace, base)

    def cross_entropy(self, traces: List[Symbols],
                      actual_trace_probs: Probabilities,
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
        :param      actual_trace_probs:  The actual probability of each trace
                                         in the target language distribution
        :param      base:                The log base. Commonly set to 2 in
                                         classic information theory literature
                                         (default 2.0)

        :returns:   the actual cross-entropy of the given trace
        """

        cross_entropy_sum = 0.0

        for target_prob, trace in zip(actual_trace_probs, traces):
            cross_entropy_sum += target_prob * self.logscore(trace, base)

        N = len(actual_trace_probs)

        return (-1.0 / N) * cross_entropy_sum

    def perplexity(self, traces: List[Symbols],
                   actual_trace_probs: Probabilities,
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
        :param      actual_trace_probs:  The actual probability of each trace
                                         in the target language distribution
        :param      base:                The log base. Commonly set to 2 in
                                         classic information theory literature
                                         (default 2.0)

        :returns:   the actual cross-entropy of the given trace
        """

        return base ** self.cross_entropy(traces, actual_trace_probs, base)

    def predictive_accuracy(self, test_traces: List[Symbols],
                            pred_method: str = 'max_prob') -> float:
        """
        compares the model's predictions to the actual values of the next
        symbol and returns the ratio of correct predictions.

        :param      test_traces:  The traces to compute predictive accuracy for
        :param      pred_method:  The method used to choose the next state.
                                  see _choose_next_state for details on how
                                  each pred_method is implemented.
                                  {'sample', 'max_prob'} (default 'max_prob')

        :returns:   predictive accuracy ratio ([0 -> 1]) of the model on the
                    given traces
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

    def _set_state_acceptance(self, curr_state: Node) -> None:
        """
        Sets the state acceptance property for the given state.

        If curr_state's final_probability >= self._beta, then the state accepts

        :param      curr_state:  The current state's node label
        """

        curr_final_prob = self._get_node_data(curr_state, 'final_probability')

        if curr_final_prob >= self._beta:
            state_accepts = True
        else:
            state_accepts = False

        self._set_node_data(curr_state, 'is_accepting', state_accepts)

    def _get_abbadingo_string(self, trace: Symbols, trace_length: int,
                              is_pos_example: bool) -> str:
        """
        Returns the Abbadingo (sigh) formatted string given a trace string and
        the label for the trace

        :param      trace:           The trace string to represent in Abbadingo
        :param      trace_length:    The trace length
        :param      is_pos_example:  Indicates if the trace is a positive
                                     example of the pdfa

        :returns:   The abbadingo formatted string for the given trace
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

        :param      graph_data:         The variable specifying graph data
        :param      graph_data_format:  The graph data file format.
                                        {'yaml', 'fdfa_object'}

        :returns:   instance of an initialized PDFA object

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

        :returns:   instance of an initialized PDFA object

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

            # nodes and edges must be in the format needed by:
            #   - networkx.add_nodes_from()
            #   - networkx.add_edges_from()
            final_transition_sym = config_data['final_transition_sym']
            (symbol_display_map,
             states,
             edges) = Automaton._convert_states_edges(config_data['nodes'],
                                                      config_data['edges'],
                                                      final_transition_sym,
                                                      is_stochastic=True)

            # saving these so we can just return initialized instances if the
            # underlying data has not changed
            self.nodes = states
            self.edges = edges

            self._instance = PDFA(
                nodes=states,
                edges=edges,
                symbol_display_map=symbol_display_map,
                beta=config_data['beta'],
                alphabet_size=config_data['alphabet_size'],
                num_states=config_data['num_states'],
                final_transition_sym=final_transition_sym,
                start_state=config_data['start_state'])

            return self._instance

    def _from_fdfa(self, fdfa: FDFA) -> PDFA:
        """
        Returns an instance of a PDFA from an instance of FDFA

        :param      fdfa:  initialized fdfa instance to convert to a pdfa

        :returns:   instance of an initialized PDFA object
        """

        nodes, edges = PDFA._fdfa_to_pdfa_data(fdfa)

        # saving these so we can just return initialized instances if the
        # underlying data has not changed
        self.nodes = nodes
        self.edges = edges

        self._instance = PDFA(
            nodes=nodes,
            edges=edges,
            symbol_display_map=fdfa._symbol_display_map,
            # just choose a default value, FDFAs have no notion of acceptance
            # this at the moment
            beta=0.95,
            alphabet_size=fdfa._alphabet_size,
            num_states=fdfa._num_states,
            final_transition_sym=fdfa._final_transition_sym,
            start_state=fdfa.start_state)

        return self._instance
