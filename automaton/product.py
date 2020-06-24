from typing import Tuple
from bidict import bidict

# local packages
from wombats.factory.builder import Builder
from .transition_system import TransitionSystem
from .pdfa import PDFA
from .base import (Automaton, NXNodeList, NXEdgeList, Node, Nodes,
                   Observation, Symbol, Symbols)

# define these type defs for method annotation type hints
TS_Trans_Data = Tuple[Node, Observation]


class Product(Automaton):

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

        # need to start with a fully initialized automaton
        super().__init__(nodes, edges, symbol_display_map,
                         alphabet_size, num_states, start_state,
                         final_transition_sym=final_transition_sym,
                         smooth_transitions=False,
                         is_stochastic=False,
                         state_observation_key='observation',
                         can_have_accepting_nodes=False,
                         edge_weight_key=None)

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
