# 3rd-party packages
import numpy as np
import os
import queue
import warnings
from typing import List, Callable
from bidict import bidict

# local packages
from wombats.factory.builder import Builder
from .types import (NXNodeList, NXEdgeList, Node, Symbol, Symbols,
                    Probabilities)
from .base import Automaton, SMOOTHING_AMOUNT
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


class DFA(Automaton):
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
    :param      alphabet_size:         number of symbols in pdfa alphabet
    :param      num_states:            number of states in automaton state
                                       space
    :param      start_state:           unique start state string label of
                                       pdfa
    :param      smooth_transitions:    whether to smooth the symbol
                                       transitions distributions
    :param      smoothing_amount:      probability mass to re-assign to
                                       unseen symbols at each node
    :param      final_transition_sym:  representation of the termination
                                       symbol. If not given, will default
                                       to base class default.
    :param      empty_transition_sym:  representation of the empty symbol
                                       (a.k.a. lambda). If not given, will
                                       default to base class default.
    :param      beta:                  the final state probability needed
                                       for a state to accept.
    :param      merge_sinks:           whether to combine all states
                                       together that have no outgoing
                                       edges
    """

    def __init__(self,
                 nodes: NXNodeList,
                 edges: NXEdgeList,
                 symbol_display_map: bidict,
                 alphabet_size: int,
                 num_states: int,
                 start_state: Node,
                 smooth_transitions: bool,
                 smoothing_amount: float = SMOOTHING_AMOUNT,
                 final_transition_sym: {Symbol, None}=None,
                 empty_transition_sym: {Symbol, None}=None,
                 beta: float = 0.95,
                 merge_sinks: bool = False,
                 graph_data_file: str = None) -> 'DFA':

        self._beta = beta
        """the final state probability needed for a state to accept"""

        self.graph_data_file = graph_data_file
        """filepath to the yaml file if exists"""

        # need to start with a fully initialized automaton
        super().__init__(nodes, edges, symbol_display_map,
                         alphabet_size, num_states, start_state,
                         smooth_transitions=smooth_transitions,
                         is_stochastic=False,
                         is_sampleable=True,
                         num_obs=None,
                         final_transition_sym=final_transition_sym,
                         empty_transition_sym=empty_transition_sym,
                         can_have_accepting_nodes=True,
                         smoothing_amount=smoothing_amount,
                         merge_sinks=merge_sinks)

    def _set_state_acceptance(self, curr_state: Node) -> None:
        """
        Sets the state acceptance property for the given state.

        If curr_state's final_probability >= self._beta, then the state accepts

        :param      curr_state:  The current state's node label
        """

        is_accepting = self._get_node_data(curr_state, 'is_accepting')
        self._set_node_data(curr_state, 'is_accepting', is_accepting)

    def satisfy_formula(self, formula, sigma, symbols):
        """
        Convert a boolean formula string to a boolean formula
        and evaluate if sigma satisfies the given formula

        For example,
            "!water & carpet" -> "not water and carpet"
        """
        for symbol in symbols:
            exec(symbol + ' = False')
        exec(sigma + ' = True')
        boolean_formula = formula\
            .replace('!', 'not ')\
            .replace('&', 'and')\
            .replace('|', 'or')
        return eval(boolean_formula)

    def is_safe(self, pdfa):
        """
        Checks if the pdfa is safe

        :param      pdfa:               A PDFA

        :returns:   True if safe else False
        """
        # naming to follow written algorithm
        C = pdfa
        S = self

        symbols = set()
        # Visit all nodes and edges to find all symbols
        for node in S.nodes:
            for dest_state, edges in S[node].items():
                for edge_key, edge_data in edges.items():
                    partial_symbols = DFA._extract_symbols_from_formula(
                        edge_data['symbol'])
                    symbols |= partial_symbols

        # Initial State
        qc_init = C.start_state
        qs_init = S.start_state
        init_prod_state = self._get_product_state_label(qc_init, qs_init)

        # Prepare dicts to store graph info
        nodes = {}
        edges = {}

        # Start Search from the initial state
        search_queue = queue.Queue()
        search_queue.put((qc_init, qs_init))
        visited = set()
        visited.add((qc_init, qs_init))

        nodes_to_delete = []
        inverted_edges = {}

        while not search_queue.empty():

            # pop a node
            qc, qs = search_queue.get()

            # Search each edge in Cosafety PDFA
            cosafe_sigmas = []
            for cosafe_edges in C[qc].values():
                for cosafe_edge in cosafe_edges.values():
                    cosafe_sigmas.append(cosafe_edge['symbol'])

            formulas = []
            for safe_edges in S[qs].values():
                for safe_edge in safe_edges.values():
                    formulas.append(safe_edge['symbol'])

            # Check if each edge in Cosafety PDFA satisfies
            # Safety Specification
            for cosafe_sigma in cosafe_sigmas:

                valids = []

                for formula in formulas:
                    # Check if the "sigma" satisfies the "formula"
                    valid = S.satisfy_formula(formula, cosafe_sigma, symbols)
                    valids.append(valid)

                    # No need for searching for invalid transition
                    if not valid:
                        continue

                    # Next state
                    qs_prime, _ = S._get_next_state(qs, formula)
                    qc_prime, _ = C._get_next_state(qc, cosafe_sigma)

                    prod_dest_state = (qc_prime, qs_prime)
                    if prod_dest_state not in visited:
                        visited.add(prod_dest_state)
                        search_queue.put(prod_dest_state)

                if not any(valids):
                    return False

        return True

    @staticmethod
    def _extract_symbols_from_formula(formula):
        """
                symbols = DFA._extract_symbols_from_formula(formula)
                seen_symbols |= symbols
        """
        symbols = formula.replace('!', '')\
            .replace(' &', ',')\
            .replace('|', ',')\
            .split(', ')
        return set(symbols)

    def _get_product_state_label(self, dynamical_system_state: Node,
                                 specification_state: Node) -> Node:
        """
        Computes the combined product state label

        :param      dynamical_system_state:  The dynamical system state label
        :param      specification_state:     The specification state label

        :returns:   The product state label.
        """

        if type(dynamical_system_state) != str:
            dynamical_system_state = str(dynamical_system_state)

        if type(specification_state) != str:
            specification_state = str(specification_state)

        return dynamical_system_state + ', ' + specification_state


class DFABuilder(Builder):
    """
    Implements the generic automaton builder class for DFA objects
    """

    def __init__(self) -> 'DFABuilder':
        """
        Constructs a new instance of the DFABuilder
        """

        # need to call the super class constructor to gain its properties
        Builder.__init__(self)

        # keep these properties so we don't re-initialize unless underlying
        # data changes
        self.nodes = None
        self.edges = None

    def __call__(self, graph_data: {str, FDFA},
                 graph_data_format: str = 'yaml',
                 **kwargs: dict) -> DFA:
        """
        Returns an initialized DFA instance given the graph_data

        graph_data and graph_data_format must match

        :param      graph_data:         The variable specifying graph data
        :param      graph_data_format:  The graph data file format.
                                        {'yaml', 'fdfa_object'}
        :param      kwargs:             The keywords arguments to the specific
                                        constructors

        :returns:   instance of an initialized DFA object

        :raises     ValueError:         checks if graph_data and
                                        graph_data_format have a compatible
                                        data loader
        """

        if graph_data_format == 'yaml':
            self._instance = self._from_yaml(graph_data, **kwargs)
        else:
            msg = 'graph_data_format ({}) must be one of: "yaml", '.format(
                graph_data_format)
            raise ValueError(msg)

        return self._instance

    def _from_yaml(self, graph_data_file: str) -> DFA:
        """
        Returns an instance of a DFA from the .yaml graph_data_file

        Only reads the config data once, otherwise just returns the built
        object

        :param      graph_data_file:  The graph configuration file name

        :returns:   instance of an initialized DFA object

        :raises     ValueError:       checks if graph_data_file's ext is YAML
        """

        _, file_extension = os.path.splitext(graph_data_file)

        allowed_exts = ['.yaml', '.yml']
        if file_extension in allowed_exts:
            config_data = self.load_YAML_config_data(graph_data_file)
            config_data['graph_data_file'] = graph_data_file
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
            empty_transition_sym = config_data['empty_transition_sym']
            (symbol_display_map,
            states,
            edges) = Automaton._convert_states_edges(config_data['nodes'],
                                                    config_data['edges'],
                                                    final_transition_sym,
                                                    empty_transition_sym,
                                                    is_stochastic=False)
            config_data['symbol_display_map'] = symbol_display_map
            config_data['nodes'] = states
            config_data['edges'] = edges

            # saving these so we can just return initialized instances if the
            # underlying data has not changed
            self.nodes = states
            self.edges = edges

            instance = DFA(**config_data)

            return instance

        return self._instance
