import copy
import itertools
from typing import Tuple
from bidict import bidict

# local packages
from wombats.factory.builder import Builder
from .transition_system import TransitionSystem
from .pdfa import PDFA
from .base import (Automaton, NXNodeList, NXEdgeList, Node, Probability,
                   Observation, Symbol)

# define these type defs for method annotation type hints
TS_Trans_Data = Tuple[Node, Observation]

SPEC_VIOLATING_STATE = 'q_v'


class Product(Automaton):

    """
    Describes a product automaton between a specification automaton
    and a dynamics automaton.

    You can use this class to compose the two automaton together and then find
    a controller for the dynamical system that satisfies the specification

    :param      nodes:                 node list as expected by
                                       networkx.add_nodes_from() (node
                                       label, node attribute dict)
    :param      edges:                 edge list as expected by
                                       networkx.add_edges_from() (src node
                                       label, dest node label, edge
                                       attribute dict)
    :param      alphabet_size:         number of symbols in system alphabet
    :param      num_states:            number of states in automaton state
                                       space
    :param      num_obs:               number of observation symbols
    :param      start_state:           unique start state string label of
                                       system
    :param      final_transition_sym:  representation of the empty string /
                                       symbol (a.k.a. lambda) (default -1)
    """

    def __init__(self,
                 nodes: NXNodeList,
                 edges: NXEdgeList,
                 symbol_display_map: bidict,
                 alphabet_size: int,
                 num_states: int,
                 num_obs: int,
                 start_state,
                 final_transition_sym=-1) -> 'Product':
        """
        Constructs a new instance of an Product automaton object.


        """

        # need to start with a fully initialized automaton
        super().__init__(nodes, edges, symbol_display_map,
                         alphabet_size, num_states, start_state,
                         final_transition_sym=final_transition_sym,
                         smooth_transitions=False,
                         is_stochastic=False,
                         state_observation_key='observation',
                         can_have_accepting_nodes=False,
                         edge_weight_key=None)

    @classmethod
    def _complete_specification(cls, specification: PDFA) -> PDFA:
        """
        processes the automaton and makes sure each state has a transition for
        each symbol

        completed nodes will be sent to "violating", cyclic state with
        uniform probability over all symbols, as producing the missing symbols
        is impossible given the language defined by the specification

        :param      specification:  The specification to complete

        :returns:   the completed version of the specification
        """

        # first need to define and add the "violating" state to the
        # specification's underlying graph
        violating_state = SPEC_VIOLATING_STATE
        violating_state_props = {'final_probability': 0.00,
                                 'trans_distribution': None,
                                 'is_accepting': None}
        specification.add_node(violating_state, **violating_state_props)

        specification._initialize_node_edge_properties(
            can_have_accepting_nodes=True,
            final_weight_key='final_probability',
            edge_weight_key='probability',
            should_complete=True,
            violating_state=violating_state,
            complete='violate')

        return specification

    @classmethod
    def _augment_initial_state(cls, dynamical_system: TransitionSystem,
                               specification: PDFA) -> TransitionSystem:
        """
        Adds an initialization state to the dynamical system to maintain
        language distributional similarity with the specification

        :param      dynamical_system:  The dynamical system to augment
        :param      specification:     The specification to take a product with

        :returns:   The transition system with a new initialization state added
        """

        initialization_state = 'x_init'

        spec_empty_symbol = specification._empty_transition_sym
        initialization_state_props = {'observation': spec_empty_symbol}
        if spec_empty_symbol not in dynamical_system.observations:
            dynamical_system.observations.add(spec_empty_symbol)
            dynamical_system._num_obs += 1

        dynamical_system.add_node(initialization_state,
                                  **initialization_state_props)

        # can choose any symbol to be the initialization symbol, doesn't matter
        initialization_symbol = list(dynamical_system.symbols)[0]
        initialization_edge_props = {'symbol': initialization_symbol,
                                     'probability': 1.00}
        dynamical_system.add_edge(initialization_state,
                                  dynamical_system.start_state,
                                  **initialization_edge_props)

        dynamical_system._initialize_node_edge_properties(
            can_have_accepting_nodes=False,
            state_observation_key='observation',
            should_complete=False)

        return dynamical_system

    @classmethod
    def _compute_product(cls, dynamical_system: TransitionSystem,
                         specification: PDFA) -> dict:
        """
        Calculates the product automaton given pre-processed automata

        :param      dynamical_system:  The dynamical system
        :param      specification:     The specification

        :returns:   The initialized product automaton configuration data
        """

        # naming to follow written algorithm
        T = dynamical_system
        A = specification
        X = dynamical_system.state_labels
        Sigma = dynamical_system.symbols
        Q = specification.state_labels

        # set of all POSSIBLE product states
        P = itertools.product(X, Q)
        all_possible_prod_trans = itertools.product(P, Sigma)

        nodes = {}
        edges = {}

        for ((x, q), sigma) in all_possible_prod_trans:

            dyn_trans = (x, sigma)
            dynamically_compatible = dyn_trans in T._transition_map

            if dynamically_compatible:
                x_prime = T._transition_map[dyn_trans]
                o_x_prime = T.observe(x_prime)
                spec_trans = (q, o_x_prime)
                specification_compatible = spec_trans in A._transition_map

                if specification_compatible:
                    # violating SCC has uniform transition probability over all
                    # possible control symbols
                    in_violating_SCC = q == SPEC_VIOLATING_STATE
                    if in_violating_SCC:
                        q_prime, _ = A._get_next_state(q, o_x_prime)
                        trans_prob = 1.0 / T._alphabet_size
                    else:
                        q_prime, trans_prob = A._get_next_state(q, o_x_prime)

                    q_final_prob = A._get_next_state(q,
                                                     A._final_transition_sym)
                    q_prime_final_prob = A._get_next_state(
                        q_prime,
                        A._final_transition_sym)
                    o_x = T.observe(x)

                    (nodes,
                     edges,
                     prod_src_state,
                     proc_dest_state) = \
                        cls._add_product_edge(
                            nodes, edges,
                            x_src=x, x_dest=x_prime,
                            q_src=q, q_dest=q_prime,
                            q_src_final_prob=q_final_prob,
                            q_dest_final_prob=q_prime_final_prob,
                            observation_src=o_x,
                            observation_dest=o_x_prime,
                            sigma=sigma,
                            trans_prob=trans_prob)

    @classmethod
    def _get_product_state_label(cls, dynamical_system_state: Node,
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

        return dynamical_system_state + specification_state

    @classmethod
    def _add_product_node(cls, nodes: dict, x: Node, q: Node,
                          q_final_prob: Probability,
                          observation: Observation) -> Tuple[dict, Node]:
        """
        Adds a newly identified product state to the nodes dict w/ needed data

        :param      nodes:         dict of nodes to build the product out of.
                                   must be in the format needed by
                                   networkx.add_nodes_from()
        :param      x:             state label in the dynamical system
        :param      q:             state label in the specification
        :param      q_final_prob:  the probability of terminating at q in the
                                   specification
        :param      observation:   The observation emitted by the dynamical
                                   system / product at the dynamical
                                   system state (x)

        :returns:   nodes dict populated with all of the given data, and
                    the label of the newly added product state
        """

        prod_state_data = {'final_probability': q_final_prob,
                           'trans_distribution': None,
                           'is_accepting': None,
                           'observation': observation}
        prod_state = cls._get_product_state_label(x, q)
        nodes[prod_state] = prod_state_data

        return nodes, prod_state

    @classmethod
    def _add_product_edge(cls, nodes: dict, edges: dict,
                          x_src: Node, x_dest: Node,
                          q_src: Node, q_dest: Node,
                          q_src_final_prob: Probability,
                          q_dest_final_prob: Probability,
                          observation_src: Observation,
                          observation_dest: Observation,
                          sigma: Symbol,
                          trans_prob: Probability) -> Tuple[dict, dict,
                                                            Node, Node]:
        """
        Adds a newly identified product edge to the nodes & edges dicts

        :param      nodes:              dict of nodes to build the product out
                                        of. Must be in the format needed by
                                        networkx.add_nodes_from()
        :param      edges:              dict of edges to build the product out
                                        of. Must be in the format needed by
                                        networkx.add_edges_from()
        :param      x_src:              source product edge's dynamical system
                                        state
        :param      x_dest:             dest. product edge's dynamical system
                                        state
        :param      q_src:              source product edge's specification
                                        state
        :param      q_dest:             dest. product edge's specification
                                        state
        :param      q_src_final_prob:   the probability of terminating at q_src
                                        in the specification
        :param      q_dest_final_prob:  the probability of terminating at
                                        q_dest in the specification
        :param      observation_src:    The observation emitted by the
                                        dynamical system / product at the
                                        source dynamical system state (x_src)
        :param      observation_dest:   The observation emitted by the
                                        dynamical system / product at the
                                        dest. dynamical system state (x_dest)
        :param      sigma:              dynamical system control input symbol
                                        enabling the product edge
        :param      trans_prob:         The product edge's transition prob.

        :returns:   nodes dict populated w/ all the given data for src & dest
                    edges dict populated w/ all the given data,
                    the label of the newly added source product state,
                    the label of the newly added product state
        """

        nodes, prod_src = cls._add_product_node(nodes, x_src, q_src,
                                                q_src_final_prob,
                                                observation_src)
        nodes, prod_dest = cls._add_product_node(nodes, x_dest, q_dest,
                                                 q_dest_final_prob,
                                                 observation_dest)

        prod_edge_data = {'symbols': [sigma],
                          'probabilities': [trans_prob]}
        prod_edge = {prod_dest: prod_edge_data}
        edges[prod_src] = prod_edge

        return nodes, edges, prod_src, prod_dest

    def _set_state_acceptance(self, curr_state: Node) -> None:
        """
        Sets the state acceptance property for the given state.

        TS doesn't accept anything, so this just passes
        """
        pass


class ProductBuilder(Builder):
    """
    Implements the generic automaton builder class for Product objects
    """

    def __init__(self) -> 'ProductBuilder':
        """
        Constructs a new instance of the ProductBuilder
        """

        # need to call the super class constructor to gain its properties
        Builder.__init__(self)

        # keep these properties so we don't re-initialize unless underlying
        # data changes
        self.nodes = None
        self.edges = None

    def __call__(self, graph_data: Tuple[TransitionSystem, PDFA],
                 graph_data_format: str = 'existing_objects') -> Product:
        """
        Returns an initialized Product instance given the graph_data

        graph_data and graph_data_format must match

        :param      graph_data:         The graph configuration file name
        :param      graph_data_format:  The graph data file format.
                                        {'existing_objects'}
                                        (default 'existing_objects')

        :returns:   instance of an initialized Product object

        :raises     ValueError:         checks if graph_data and
                                        graph_data_format have a compatible
                                        data loader
        """

        if graph_data_format == 'existing_objects':
            sys = graph_data[0]
            spec = graph_data[1]
            self._instance = self._from_automata(dynamical_system=sys,
                                                 specification=spec)
        else:
            msg = 'graph_data_format ({}) must be one of: ' + \
                  '"existing_objects"'.format(graph_data_format)
            raise ValueError(msg)

        return self._instance

    def _from_automata(self, dynamical_system: TransitionSystem,
                       specification: PDFA) -> Product:
        """
        Returns an instance of a Product Automaton from existing automata

        :param      dynamical_system:  The dynamical system automaton instance
        :param      specification:     The specification automaton instance

        :returns:   instance of an initialized Product automaton object
        """

        # don't want to destroy the automaton when we pre-process them
        internal_spec = copy.deepcopy(specification)
        internal_dyn_sys = copy.deepcopy(dynamical_system)

        complete_specification = Product._complete_specification(internal_spec)
        complete_specification.draw_IPython()

        augmented_dyn_sys = Product._augment_initial_state(
            internal_dyn_sys,
            complete_specification)
        augmented_dyn_sys.draw_IPython()

        config_data = Product._compute_product(augmented_dyn_sys,
                                               complete_specification)

        # saving these so we can just return initialized instances if the
        # underlying data has not changed
        self.nodes = config_data['nodes']
        self.edges = config_data['edges']

        self._instance = Product(
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
