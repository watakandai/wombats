# 3rd-party packages
import graphviz as gv
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import collections
import multiprocessing
from numpy.random import RandomState
from joblib import Parallel, delayed
from abc import ABCMeta, abstractmethod
from scipy.stats import rv_discrete
from networkx.drawing.nx_pydot import to_pydot
from IPython.display import display
from pydot import Dot
from typing import Hashable, List, Tuple, Iterable
from bidict import bidict

# needed for multi-threaded sampling routine
NUM_CORES = multiprocessing.cpu_count()

# define these type defs for method annotation type hints
NXNodeList = List[Tuple[Hashable, dict]]
NXEdgeList = List[Tuple[Hashable, Hashable, dict]]

Node = Hashable
Observation = Hashable
Symbol = Hashable
Weight = int
Probability = float

Nodes = Iterable[Node]
Observations = Iterable[Observation]
Symbols = Iterable[Symbol]
Weights = Iterable[Weight]
Probabilities = Iterable[Probability]

Categorical_data = (Weights, Nodes, Symbols)
Sampled_Trans_Data = (Node, Symbol, Probability)


class Automaton(nx.MultiDiGraph, metaclass=ABCMeta):
    """
    This class describes a automaton with stochastic transition

    built on networkx, so inherits node and edge data structure definitions

    Node Attributes
    -----------------
        - final_probability: final state probability for the node
        - trans_distribution: a sampled-able function to select the next state
          and emitted symbol
        - is_accepting: a boolean flag determining whether the automaton
          considers the node accepting

    Edge Properties
    -----------------
        - symbol: the symbol value emitted when the edge is traversed
        - probability: the probability of selecting this edge for traversal
    """

    def __init__(self,
                 nodes: NXNodeList,
                 edge_list: NXEdgeList,
                 symbol_display_map: bidict,
                 alphabet_size: int,
                 num_states: int,
                 start_state: Hashable,
                 smooth_transitions: bool,
                 is_stochastic: bool,
                 final_transition_sym: Hashable = -1,
                 final_weight_key: str = None,
                 state_observation_key: str = None,
                 can_have_accepting_nodes: bool = True,
                 edge_weight_key: str = None) -> 'Automaton':
        """
        Constructs a new instance of an Automaton object.

        :param      nodes:                     node list as expected by
                                               networkx.add_nodes_from()
        :param      edge_list:                 edge list as expected by
                                               networkx.add_edges_from()
        :param      symbol_display_map:        bidirectional mapping of
                                               hashable symbols, to a unique
                                               integer index in the symbol map.
                                               Needed to translate between the
                                               indices in the transition
                                               distribution and the hashable
                                               representation which is
                                               meaningful to the user
        :param      alphabet_size:             number of symbols in automaton
        :param      num_states:                number of states in automaton
                                               state space
        :param      start_state:               unique start state string label
                                               of automaton
        :param      smooth_transitions:        whether to smooth the symbol
                                               transitions distributions
        :param      is_stochastic:             the transitions are
                                               non-probabilistic, so we are
                                               going to assign a uniform
                                               distribution over all symbols
                                               for the purpose of generation
        :param      final_transition_sym:      representation of the empty
                                               string / symbol (a.k.a. lambda)
                                               (default -1)
        :param      final_weight_key:          key in the automaton's node data
                                               corresponding to the weight /
                                               probability of ending in that
                                               node.
                                               If None, don't include this info
                                               in the display of the automaton.
                                               (default None)
        :param      state_observation_key:     The key in each node's data
                                               dict for state observations.
                                               If None, don't include this info
                                               in the display of the automaton
                                               (default None)
        :param      can_have_accepting_nodes:  Indicates if the automata can
                                               have accepting nodes
                                               (default True)
        :param      edge_weight_key:           The key in each edge's data
                                               dict for edge weight / prob.
                                               If None, don't include this info
                                               in the display of the automaton
                                               (default None)
        """

        self._transition_map = {}
        """keep a map of start state label and symbol to destination state"""

        self._symbol_display_map = symbol_display_map

        self._alphabet_size = alphabet_size
        """number of symbols in automaton alphabet"""

        self._num_states = num_states
        """number of states in automaton state space"""

        self._final_transition_sym = final_transition_sym
        """representation of the empty string / symbol (a.k.a. lambda)"""

        self.start_state = start_state
        """unique start state string label of pdfa"""

        self._is_stochastic = is_stochastic
        """whether symbol probabilities are given for string generation"""

        self._use_smoothing = smooth_transitions
        """whether or not to smooth the input transition distributions"""

        # need to start with a fully initialized networkx digraph
        super().__init__()

        self.add_nodes_from(nodes)
        self.add_edges_from(edge_list)

        self._initialize_node_edge_properties(
            state_observation_key=state_observation_key,
            final_weight_key=final_weight_key,
            can_have_accepting_nodes=can_have_accepting_nodes,
            edge_weight_key=edge_weight_key)

    def disp_edges(self, graph: {None, nx.MultiDiGraph}=None) -> None:
        """
        Prints each edge in the graph in an edge-list tuple format

        :param      graph:  The graph to access. Default = None => use instance
        :type       graph:  {None, nx.MultiDiGraph}
        """

        if graph is None:
            graph = self

        for node, neighbors in graph.adj.items():
            for neighbor, edges in neighbors.items():
                for edge_number, edge_data in edges.items():

                    print(node, neighbor, edge_data)

    def disp_nodes(self, graph: {None, nx.MultiDiGraph}=None) -> None:
        """
        Prints each node's data view

        :param      graph:  The graph to access. Default = None => use instance
        :type       graph:  {None, nx.MultiDiGraph}
        """

        if graph is None:
            graph = self

        for node in graph.nodes(data=True):
            print(node)

    def draw_IPython(self) -> None:
        """
        Draws the pdfa structure in a way compatible with a jupyter / IPython
        notebook
        """

        graph = self._get_pydot_representation()

        dot_string = graph.to_string()
        display(gv.Source(dot_string))

    def plot_node_trans_dist(self, curr_state: Node) -> None:
        """
        Plots the transition pmf at the given curr_state / node.

        :param      curr_state:  state to display its transition distribution
        :type       curr_state:  Hashable
        """

        trans_dist = self._get_node_data(curr_state, 'trans_distribution')
        symbols = self._convert_symbol_idxs(trans_dist.xk)

        fig, ax = plt.subplots(1, 1)
        ax.plot(symbols, trans_dist.pmf(trans_dist.xk), 'ro',
                ms=12, mec='r')
        ax.vlines(symbols, 0, trans_dist.pmf(trans_dist.xk),
                  colors='r', lw=4)
        plt.show()

    def generate_traces(self, num_samples: int, N: int) -> (List[Symbols],
                                                            List[int],
                                                            Probabilities):
        """
        generates num_samples random traces from the pdfa

        :param      num_samples:  The number of trace samples to generate
        :param      N:            maximum length of trace

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

    def generate_trace(self, start_state: Node, N: int,
                       random_state: {None, int, Iterable}=None) -> (Symbols,
                                                                     int,
                                                                     Probability):
        """
        Generates a trace from the pdfa starting from start_state

        :param      start_state:   the state label to start sampling traces
                                   from
        :param      N:             maximum length of trace
        :param      random_state:  The np.random.RandomState() seed parameter
                                   for sampling from the state transition
                                   distribution. Defaulting to None causes the
                                   seed to reset. (default None)

        :returns:   the sequence of symbols emitted, the length of the trace,
                    the probability of the trace in the language of the pdfa
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

            curr_state = next_state
            trace_prob *= trans_probability

            if next_symbol == self._final_transition_sym:
                break

            sampled_trace.append(next_symbol)
            length_of_trace += 1

        return sampled_trace, length_of_trace, trace_prob

    def _choose_next_state(self, curr_state: Node,
                           random_state: {None, int, Iterable}=None,
                           pred_method: str = 'sample') -> Sampled_Trans_Data:
        """
        Chooses the next state based on curr_state's transition distribution

        :param      curr_state:    The current state label
        :type       curr_state:    Hashable
        :param      random_state:  The np.random.RandomState() seed parameter
                                   for sampling from the state transition
                                   distribution. Defaulting to None causes the
                                   seed to reset. (default None)
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

        :returns:   The next state's label, the symbol emitted by changing
                    states, the probability of this transition occurring
        """

        trans_dist = self.nodes[curr_state]['trans_distribution']

        # critical step for use with parallelized libraries. This must be reset
        # before sampling, as otherwise each of the threads is using the same
        # seed, and we get lots of duplicated strings
        trans_dist.random_state = RandomState(random_state)

        # sampling an action (symbol) from the state-action distribution at
        # curr_state
        next_symbol_idx = trans_dist.rvs(size=1)[0]
        next_symbol = self._convert_symbol_idxs(next_symbol_idx)

        next_state, trans_probability = self._get_next_state(curr_state,
                                                             next_symbol)
        return next_state, next_symbol, trans_probability

    def _get_next_state(self, curr_state: Node,
                        symbol: Symbol) -> Tuple[Node, Probability]:
        """
        Gets the next state given the current state and the "input" symbol.

        :param      curr_state:  The curr state
        :param      symbol:      The input symbol

        :returns:   (The next state label, the transition probability)

        :raises     ValueError:  symbol not in curr_state's transition function
        :raises     ValueError:  duplicate symbol in curr_state's transition
                                 function
        :raises     ValueError:  symbol has 0 prob. in curr_state's transition
                                 function
        """

        trans_distribution = self._get_node_data(curr_state,
                                                 'trans_distribution')
        possible_symbols = self._convert_symbol_idxs(trans_distribution.xk)
        probabilities = trans_distribution.pk

        if symbol not in possible_symbols:
            msg = ('given symbol ({}) is not found in the '
                   'curr_state\'s ({}) '
                   'transition distribution').format(symbol, curr_state)
            raise ValueError(msg)

        symbol_idx = [i for i, val in enumerate(possible_symbols)
                      if val == symbol]
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

    @staticmethod
    def _convert_states_edges(nodes: dict, edges: dict,
                              final_transition_sym,
                              is_stochastic: bool) -> (bidict,
                                                       NXNodeList, NXEdgeList):
        """
        Converts node and edges data from a manually specified YAML config file
        to the format needed by:
            - networkx.add_nodes_from()
            - networkx.add_edges_from()

        :param      nodes:                 dict of node objects to be converted
        :param      edges:                 dictionary adj. list to be converted
        :param      final_transition_sym:  representation of the empty string /
                                           symbol (a.k.a. lambda)
        :param      is_stochastic:         the transitions are
                                           non-probabilistic, so we are going
                                           to assign a uniform distribution
                                           over all symbols for the purpose of
                                           generation

        :returns:   mapping to display symbols according to their
                    index in the transition distributions,
                    properly formated node and edge list containers
        :rtype:     tuple:
                    (symbol_display_map - bidirectional mapping of hashable
                                          symbols, to a unique integer index in
                                          the symbol map.
                     nodes - list of tuples:
                     (node label, node attribute dict),
                     edges - list of tuples:
                     (src node label, dest node label, edge attribute dict))
        """

        # need to convert the configuration adjacency list given in the config
        # to an edge list given as a 3-tuple of (source, dest, edgeAttrDict)
        edge_list = []
        symbol_count = 0
        symbol_display_map = bidict({})
        for source_node, dest_edges_data in edges.items():

            # don't need to add any edges if there is no edge data
            if dest_edges_data is None:
                continue

            for dest_node in dest_edges_data:

                symbols = dest_edges_data[dest_node]['symbols']
                if is_stochastic:
                    probabilities = dest_edges_data[dest_node]['probabilities']

                for symbol_idx, symbol in enumerate(symbols):

                    # need to store new symbols in a map for display
                    if symbol not in symbol_display_map:
                        symbol_count += 1
                        symbol_display_map[symbol] = symbol_count

                    edge_data = {'symbol': symbol}

                    if is_stochastic:
                        probability = probabilities[symbol_idx]
                        edge_data['probability'] = probability

                    newEdge = (source_node, dest_node, edge_data)
                    edge_list.append(newEdge)

        # best convention is to convert dict_items to a list, even though both
        # are iterable
        converted_nodes = list(nodes.items())

        # we need to add the empty / final symbol to the display map
        # for completeness
        symbol_display_map[final_transition_sym] = final_transition_sym

        return symbol_display_map, converted_nodes, edge_list

    @abstractmethod
    def _set_state_acceptance(self, curr_state: Node) -> None:
        """
        Sets the state acceptance property for the given state.

        Abstract method - must be overridden by subclass
        """

        raise NotImplementedError

    def _initialize_node_edge_properties(self, final_weight_key: str = None,
                                         state_observation_key: str = None,
                                         can_have_accepting_nodes: bool = True,
                                         edge_weight_key: str = None) -> None:
        """
        Initializes the node and edge data properties correctly for a pdfa.

        :param      final_weight_key:          key in the automaton's node data
                                               corresponding to the weight /
                                               probability of ending in that
                                               node.
                                               If None, don't include this info
                                               in the display of the automaton.
                                               (default None)
        :param      state_observation_key:     The key in each node's data
                                               dict for state observations.
                                               If None, don't include this info
                                               in the display of the automaton
                                               (default None)
        :param      can_have_accepting_nodes:  Indicates if the automata can
                                               have accepting nodes
                                               (default True)
        :param      edge_weight_key:           The key in each edge's data
                                               dict for edge weight / prob.
                                               If None, don't include this info
                                               in the display of the automaton
                                               (default None)
        """

        # do batch computations at initialization, as these shouldn't
        # frequently change
        for node in self.nodes:
            self._compute_node_data_properties(node)

        self._set_node_labels(final_weight_key, state_observation_key,
                              can_have_accepting_nodes)
        self._set_edge_labels(edge_weight_key)

    def _compute_node_data_properties(self, node: Node) -> None:
        """
        Base method for calculating the properties for the given node.

        currently calculated properties:
            - 'is_accepting'
            - 'trans_distribution'

        :param      node:        The node to calculate properties for

        :returns:   Nothing

        :raises     ValueError:  checks for non-deterministic transitions
        """

        # acceptance property shouldn't change after load in
        self._set_state_acceptance(node)

        # if we compute this once, we can sample from each distribution
        (self.nodes[node]['trans_distribution'],
         new_trans_map_entries) = \
            self._set_state_transition_dist(node, self.edges,
                                            stochastic=self._is_stochastic,
                                            smooth=self._use_smoothing)

        # need to merge the newly computed transition map at node to the
        # existing map
        #
        # for a automaton, a given start state and symbol must have a
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

    def _set_state_transition_dist(self, curr_state: Node,
                                   edges: NXEdgeList,
                                   stochastic: bool,
                                   smooth: bool) -> (rv_discrete, dict):
        """
        Computes a static state transition distribution for given state

        :param      curr_state:      The current state label
        :param      edges:           The networkx edge list
        :param      deterministic:   the transitions are non-probabilistic, so
                                     we are going to assign a uniform
                                     distribution over all symbols for the
                                     purpose of generation
        :param      smooth:          turn transition smoothing on / off

        :returns:   (a function to sample the discrete state transition
                    distribution, the mapping from (start state, symbol) ->
                    edge_dests
        :rtype:     tuple(stats.rv_discrete object, dict)
        """

        edge_data = edges([curr_state], data=True)

        edge_dests = [edge[1] for edge in edge_data]

        # need to convert the hashable symbols to their integer indices for
        # creating the categorical distribution, which only works with
        # integers
        original_edge_symbols = [edge[2]['symbol'] for edge in edge_data]
        edge_symbols = [self._symbol_display_map[symbol] for symbol in
                        original_edge_symbols]

        if stochastic:
            # need to add final state probability to discrete rv dist
            edge_probs = [edge[2]['probability'] for edge in edge_data]

            curr_final_state_prob = self._get_node_data(curr_state,
                                                        'final_probability')

            # adding the final-state sequence end transition to the
            # distribution
            edge_probs.append(curr_final_state_prob)
            edge_dests.append(curr_state)
            edge_symbols.append(self._final_transition_sym)

            # need to smooth to better generalize and not have infinite
            # perplexity on unknown symbols in the alphabet
            if smooth:
                (edge_probs,
                 edge_dests,
                 edge_symbols) = self._smooth_categorical(curr_state,
                                                          edge_probs,
                                                          edge_symbols,
                                                          edge_dests)
        else:
            # using a uniform distribution to not bias the sampling of symbols
            # in a deterministic that does not actually have edge
            # probabilities
            num_symbols = len(edge_symbols)
            is_final_state = num_symbols == 0
            if is_final_state:
                edge_probs = [1.0]
                edge_dests.append(curr_state)
                edge_symbols.append(self._final_transition_sym)
            else:
                edge_probs = [1.0 / num_symbols for symbol in edge_symbols]

        next_symbol_dist = rv_discrete(name='transition',
                                       values=(edge_symbols, edge_probs))

        # creating the mapping from (start state, symbol) -> edge_dests
        disp_edge_symbols = self._convert_symbol_idxs(edge_symbols)
        state_symbol_keys = list(zip([curr_state] * len(disp_edge_symbols),
                                     disp_edge_symbols))
        transition_map = dict(zip(state_symbol_keys, edge_dests))

        return next_symbol_dist, transition_map

    def _complete_transition(self, curr_state: Node,
                             edge_probs: Probabilities,
                             edge_symbols: Symbols,
                             prob_to_add: Probability = None,
                             dest_state: Node = None) -> Categorical_data:
        """
        computes missing transitions from the current state

        :param      curr_state:    The current state label for which to smooth
                                   the distribution
        :param      edge_probs:    The transition probability values for each
                                   edge
        :param      edge_symbols:  The emitted symbols for each edge
        :param      prob_to_add:   The probability mass to add to each missing
                                   symbol
                                   (default self._smoothing_amount)
        :param      dest_state:    The destination state label for the missing
                                   transitions.
                                   (default curr_state)

        :returns:   Missing transitions added edge_probs, edge_dests, and
                    edge_symbols
        """

        if prob_to_add is None:
            prob_to_add = self._smoothing_amount

        # here we add in the missing transition probabilities as just very
        # unlikely self-loops
        num_of_missing_transitions = 0
        new_edge_probs, new_edge_dests, new_edge_symbols = [], [], []
        all_symbols_idxs = list(self._symbol_display_map.inv.keys())

        for symbol in all_symbols_idxs:
            if symbol not in edge_symbols:

                num_of_missing_transitions += 1
                new_edge_probs.append(prob_to_add)

                if dest_state is None:
                    new_edge_dests.append(curr_state)
                else:
                    new_edge_dests.append(dest_state)

                new_edge_symbols.append(symbol)

        return new_edge_probs, new_edge_dests, new_edge_symbols

    def _smooth_categorical(self, curr_state: Node,
                            edge_probs: Probabilities,
                            edge_symbols: Symbols,
                            edge_dests: Nodes) -> Categorical_data:
        """
        Applies Laplace smoothing to the given categorical state-symbol
        distribution

        :param      curr_state:    The current state label for which to smooth
                                   the distribution
        :param      edge_probs:    The transition probability values for each
                                   edge
        :param      edge_symbols:  The emitted symbols for each edge
        :param      edge_dests:    The labels of the destination states under
                                   each symbol at the curr_state

        :returns:   The smoothed version of edge_probs, edge_dests,
                    and edge_symbols

        """

        (new_edge_probs,
         new_edge_dests,
         new_edge_symbols) = self._complete_transition(curr_state,
                                                       edge_probs,
                                                       edge_symbols)

        all_possible_trans = [idx for idx, prob in enumerate(edge_probs) if
                              prob > 0.0]
        num_orig_samples = len(all_possible_trans)

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

    def _convert_symbol_idxs(self, integer_symbols: {List[int], int}) -> List:
        """
        Convert an iterable container of integer representations of automaton
        symbols to their readable, user-meaningful form.

        :param      integer_symbols:  The integer symbol(s) to convert

        :returns:   a list of displayable automaton symbols corresponding to
                    the inputted integer symbols

        :raises     ValueError:       all given symbol indices must be ints
        """

        display_symbols = []

        # need to do type-checking / polymorphism handling here
        if not isinstance(integer_symbols, collections.Iterable):
            if np.issubdtype(integer_symbols, np.integer):
                return self._symbol_display_map.inv[integer_symbols]
            else:
                msg = f'symbol index ({integer_symbols}) is not an int'
                raise ValueError(msg)
        else:

            all_ints = all(np.issubdtype(type(sym), np.integer) for sym in
                           integer_symbols)
            if not all_ints:
                msg = f'not all symbol indices ({integer_symbols}) are ints'
                raise ValueError(msg)

        for integer_symbol in integer_symbols:
            converted_symbol = self._symbol_display_map.inv[integer_symbol]
            display_symbols.append(converted_symbol)

        return display_symbols

    def _get_pydot_representation(self) -> Dot:
        """
        converts the networkx graph to pydot and sets graphviz graph attributes

        :returns:   The pydot Dot data structure representation.
        :rtype:     pydot.Dot
        """

        graph = to_pydot(self)
        graph.set_splines(True)
        graph.set_nodesep(0.5)
        graph.set_sep('+25,25')
        graph.set_ratio(1)

        return graph

    def _set_node_labels(self, final_weight_key: str,
                         state_observation_key: str,
                         can_have_accepting_nodes: bool,
                         graph: {None, nx.MultiDiGraph}=None) -> None:
        """
        Sets each node's label property for use in graphviz output

        :param      final_weight_key:          key in the automaton's node data
                                               corresponding to the weight /
                                               probability of ending in that
                                               node
        :param      state_observation_key:     The state observation key
        :param      can_have_accepting_nodes:  Indicates if the automata can
                                               have accepting nodes
        :param      graph:                     The graph to access. Default =
                                               None => use instance (default
                                               None)
        :type       graph:                     {None, nx.MultiDiGraph}
        :type       final_weight_key:          string
        :type       can_have_accepting_nodes:  boolean
        """

        if graph is None:
            graph = self

        label_dict = {}

        for node_name, node_data in graph.nodes.data():

            if final_weight_key is not None:
                weight = node_data[final_weight_key]
                final_prob_string = edge_weight_to_string(weight)
                node_dot_label_string = node_name + ': ' + final_prob_string
            else:
                node_dot_label_string = node_name

            graphviz_node_label = {'label': node_dot_label_string,
                                   'fillcolor': 'gray80',
                                   'style': 'filled'}

            if state_observation_key is not None:
                obs_label = node_obs_to_str(node_data[state_observation_key])
                external_label = '{' + obs_label + '}'
                graphviz_node_label['xlabel'] = external_label

            is_start_state = (node_name == self.start_state)

            if can_have_accepting_nodes and node_data['is_accepting']:
                graphviz_node_label.update({'peripheries': 2})
                graphviz_node_label.update({'fillcolor': 'tomato1'})

            if is_start_state:
                graphviz_node_label.update({'shape': 'box'})
                graphviz_node_label.update({'fillcolor': 'royalblue1'})

            label_dict[node_name] = graphviz_node_label

        nx.set_node_attributes(graph, label_dict)

    def _set_edge_labels(self, edge_weight_key: str = None,
                         graph: {None, nx.MultiDiGraph}=None) -> None:
        """
        Sets each edge's label property for use in graphviz output

        :param      edge_weight_key:  The edge data's "weight" key
        :param      graph:            The graph to access. Default = None =>
                                      use instance (default None)
        """

        if graph is None:
            graph = self

        # this needs to be a mapping from edges (node label tuples) to a
        # dictionary of attributes
        label_dict = {}

        for u, v, key, data in graph.edges(data=True, keys=True):

            if edge_weight_key is not None:
                wt_str = edge_weight_to_string(data[edge_weight_key])
                edge_label_string = str(data['symbol']) + ': ' + wt_str
            else:
                edge_label_string = str(data['symbol'])

            new_label_property = {'label': edge_label_string,
                                  'fontcolor': 'blue'}
            node_identifier = (u, v, key)

            label_dict[node_identifier] = new_label_property

        nx.set_edge_attributes(graph, label_dict)

    def _get_node_data(self, node_label: Node, data_key: str,
                       graph: {None, nx.MultiDiGraph}=None):
        """
        Gets the node's data_key data from the graph

        :param      node_label:  The node label
        :param      data_key:    The desired node data's key name
        :param      graph:       The graph to access. Default = None => use
                                 instance (default None)

        :returns:   The node data associated with the node_label and data_key
        :rtype:     type of self.nodes.data()[node_label][data_key]
        """

        if graph is None:
            graph = self

        node_data = graph.nodes.data()

        return node_data[node_label][data_key]

    def _set_node_data(self, node_label: Node, data_key: str, data,
                       graph: {None, nx.MultiDiGraph}=None) -> None:
        """
        Sets the node's data_key data from the graph

        :param      node_label:  The node label
        :param      data_key:    The desired node data's key name
        :param      data:        The data to associate with data_key
        :param      graph:       The graph to access. Default = None => use
                                 instance (default None)
        """

        if graph is None:
            graph = self

        node_data = graph.nodes.data()
        node_data[node_label][data_key] = data


def node_obs_to_str(obs: Observation) -> str:
    """
    returns a node observation label as an appropriately formatted string

    :param      obs:  The node observation label

    :returns:   properly formatted observation label string
    """
    if isinstance(obs, int):
        obs_str = '{obs:d}'.format(obs=obs)
    elif isinstance(obs, str):
        obs_str = obs

    return obs_str


def edge_weight_to_string(weight: {int, float}) -> str:
    """
    returns a numeric edge weight as an appropriately formatted string

    :param      weight:  The edge weight to convert to string.
    :type       weight:  int or float

    :returns:   properly formatted weight string
    :rtype:     string
    """
    if isinstance(weight, int):
        wt_str = '{weight:d}'.format(weight=weight)
    elif isinstance(weight, float):
        wt_str = '{weight:.{digits}f}'.format(weight=weight,
                                              digits=2)

    return wt_str
