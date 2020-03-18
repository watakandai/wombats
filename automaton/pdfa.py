# 3rd-party packages
import multiprocessing
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.random import RandomState
from scipy.stats import rv_discrete
from joblib import Parallel, delayed
from collections.abc import Iterable
from typing import Union, List, Hashable, Tuple

# local packages
from wombats.factory.builder import Builder
from .stochastic_automaton import StochasticAutomaton, NXNodeList, NXEdgeList
from .fdfa import FDFA

# needed for method type hint annotations
Trans = (str, int, float)

# needed for multi-threaded sampling routine
NUM_CORES = multiprocessing.cpu_count()


class PDFA(StochasticAutomaton):
    """
    This class describes a probabilistic deterministic finite automaton (pdfa).

    built on networkx, so inherits node and edge data structure definitions

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
                 beta: float=0.95, final_transition_sym=-1) -> 'PDFA':
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
        :param      final_transition_sym:  representation of the empty string /
                                           symbol (a.k.a. lambda) (default -1)
        :type       final_transition_sym:  same type as PDFA.edges symbol
                                           property
        """

        # need to start with a fully initialized automaton
        super().__init__(nodes, edge_list, alphabet_size, num_states,
                         start_state, beta=0.95, final_transition_sym=-1)

        self._transition_map = {}
        """keep a map of start state label and symbol to destination state"""

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

    def compute_trace_probability(self, trace: List[int]) -> float:
        """
        Calculates the given trace's probability in the language of the PDFA.

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

    def generate_trace(self, start_state: Hashable, N: int,
                        random_state: RandomState=None) -> (List[int], int,
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
        print(trace_prob)
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
        symbols = trans_distribution.xk
        probabilities = trans_distribution.pk

        if symbol not in symbols:
            msg = ('given symbol ({}) is not found in the ' +
                   'curr_state\'s ({}) ' +
                   'transition distribution').format(symbol, curr_state)
            raise ValueError(msg)

        symbol_idx = np.where(symbols == symbol)
        num_matched_symbols = len(symbol_idx)
        if num_matched_symbols != 1:
            msg = ('given symbol ({}) is found multiple times in ' +
                   'curr_state\'s ({}) ' +
                   'transition distribution').format(symbol, curr_state)
            raise ValueError(msg)

        symbol_probability = probabilities[symbol_idx]

        if symbol_probability == 0.0:
            msg = ('symbol ({}) has zero probability of transition in the ' +
                   'pdfa from curr_state: {}').format(symbol, curr_state)
            raise ValueError(msg)

        next_state = self._transition_map[(curr_state, symbol)]

        return next_state, symbol_probability

    def _compute_node_data_properties(self) -> None:
        """
        Calculates the properties for each node.

        currently calculated properties:
            - 'is_accepting'
            - 'trans_distribution'
        """

        for node in self.nodes:

            # beta-acceptance property shouldn't change after load in
            self._set_state_acceptance(node, self._beta)

            # if we compute this once, we can sample from each distribution
            (self.nodes[node]['trans_distribution'],
             new_trans_map_entries) = \
                self._set_state_transition_dist(node, self.edges)

            # need to merge the newly computed transition map at node to the
            # existing map
            #
            # for a PDFA, a given start state and symbol must have a
            # deterministic transition
            for key in new_trans_map_entries.keys():
                if key in self._transition_map:
                    curr_state = key[0]
                    symbol = key[1]
                    msg = ('duplicate transition from state {} '
                           'under symbol {} found - transition must be '
                           'deterministic').format(curr_state, symbol)
                    raise ValueError(msg)

            self._transition_map = {**self._transition_map,
                                    **new_trans_map_entries}

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

    def _set_state_transition_dist(self, curr_state: Hashable,
                                   edges: NXEdgeList) -> (rv_discrete, dict):
        """
        Computes a static state transition distribution for given state

        :param      curr_state:  The current state label
        :type       curr_state:  Hashable
        :param      edges:       The networkx edge list
        :type       edges:       list

        :returns:   (a function to sample the discrete state transition
                    distribution, the mapping from (start state, symbol) ->
                    edge_dests
        :rtype:     tuple(stats.rv_discrete object, dict)
        """

        edge_data = edges([curr_state], data=True)

        edge_dests = [edge[1] for edge in edge_data]
        edge_symbols = [edge[2]['symbol'] for edge in edge_data]

        # need to add final state probability to discrete rv dist
        edge_probs = [edge[2]['probability'] for edge in edge_data]

        curr_final_state_prob = self._get_node_data(curr_state,
                                                    'final_probability')

        # adding the final-state sequence end transition to the distribution
        edge_probs.append(curr_final_state_prob)
        edge_dests.append(curr_state)
        edge_symbols.append(self._final_transition_sym)

        next_symbol_dist = rv_discrete(name='transition',
                                       values=(edge_symbols, edge_probs))

        # creating the mapping from (start state, symbol) -> edge_dests
        state_symbol_keys = list(zip([curr_state] * len(edge_symbols),
                                     edge_symbols))
        transition_map = dict(zip(state_symbol_keys, edge_dests))

        return next_symbol_dist, transition_map

    def _choose_next_state(self, curr_state: Hashable,
                           random_state: {None, int, Iterable}=None) -> Trans:
        """
        Chooses the next state based on curr_state's transition distribution

        :param      curr_state:    The current state label
        :type       curr_state:    Hashable
        :param      random_state:  The np.random.RandomState() seed parameter
                                   for sampling from the state transition
                                   distribution. Defaulting to None causes the
                                   seed to reset. (default None)
        :type       random_state:  {None, int, Iterable}

        :returns:   The next state's label, the symbol emitted by changing
                    states, the probability of this transition occurring
        :rtype:     tuple(string, int, float)

        :raises     ValueError:    if more than one non-zero probability
                                   transition from curr_state under a given
                                   symbol exists
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
                 graph_data_format: str='yaml') -> PDFA:
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
                start_state=config_data['start_state'])

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
            beta=fdfa._beta,
            alphabet_size=fdfa._alphabet_size,
            num_states=fdfa._num_states,
            final_transition_sym=fdfa._final_transition_sym,
            start_state=fdfa.start_state)

        return self._instance
