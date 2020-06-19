import os
from typing import Hashable, List, Tuple
from bidict import bidict

# local packages
from wombats.factory.builder import Builder
from .base import (Automaton, NXNodeList, NXEdgeList, Node, Observation,
                   Symbols)

# define these type defs for method annotation type hints
TS_Trans_Data = Tuple[Node, Observation]


class TransitionSystem(Automaton):

    def __init__(self,
                 nodes: NXNodeList,
                 edges: NXEdgeList,
                 symbol_display_map: bidict,
                 alphabet_size: int,
                 num_states: int,
                 num_obs: int,
                 start_state,
                 final_transition_sym=-1) -> 'TransitionSystem':
        """
        Constructs a new instance of an Automaton object.

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

        self._num_obs = num_states
        """number of state observations in TS obs. space"""

    def transition(self, curr_state, input_symbol: str) -> TS_Trans_Data:

        next_state = self._transition_map[(curr_state, input_symbol)]
        observation = self._get_node_data(curr_state, 'observation')

        return next_state, observation

    def run(self, word: Symbols):
        pass

    def _set_state_acceptance(self, curr_state: Node) -> None:
        """
        Sets the state acceptance property for the given state.

        TS doesn't accept anything, so this just passes
        """
        pass


class TSBuilder(Builder):
    """
    Implements the generic automaton builder class for TransitionSystem objects
    """

    def __init__(self) -> 'TSBuilder':
        """
        Constructs a new instance of the TSBuilder
        """

        # need to call the super class constructor to gain its properties
        Builder.__init__(self)

        # keep these properties so we don't re-initialize unless underlying
        # data changes
        self.nodes = None
        self.edges = None

    def __call__(self, graph_data: str,
                 graph_data_format: str = 'yaml') -> TransitionSystem:
        """
        Returns an initialized TransitionSystem instance given the graph_data

        graph_data and graph_data_format must match

        :param      graph_data:         The graph configuration file name
        :param      graph_data_format:  The graph data file format. (default
                                        'yaml') {'yaml'}

        :returns:   instance of an initialized TransitionSystem object

        :raises     ValueError:         checks if graph_data and
                                        graph_data_format have a compatible
                                        data loader
        """

        _, file_extension = os.path.splitext(graph_data)

        allowed_exts = ['.yaml', '.yml']
        if file_extension in allowed_exts:
            config_data = self.load_YAML_config_data(graph_data)
        else:
            msg = 'graph_data ({}) is not a ({}) file'
            raise ValueError(msg.format(graph_data, allowed_exts))

        nodes_have_changed = (self.nodes != config_data['nodes'])
        edges_have_changed = (self.edges != config_data['edges'])
        no_instance_loaded_yet = (self._instance is None)

        if no_instance_loaded_yet or nodes_have_changed or edges_have_changed:

            # nodes and edge_list must be in the format needed by:
            #   - networkx.add_nodes_from()
            #   - networkx.add_edges_from()
            final_transition_sym = config_data['final_transition_sym']
            (symbol_display_map,
             states,
             edges) = Automaton._convert_states_edges(config_data['nodes'],
                                                      config_data['edges'],
                                                      final_transition_sym,
                                                      is_stochastic=False)

            # saving these so we can just return initialized instances if the
            # underlying data has not changed
            self.nodes = states
            self.edges = edges

            self._instance = TransitionSystem(
                nodes=self.nodes,
                edges=self.edges,
                symbol_display_map=symbol_display_map,
                alphabet_size=config_data['alphabet_size'],
                num_states=config_data['num_states'],
                num_obs=config_data['num_obs'],
                start_state=config_data['start_state'],
                final_transition_sym=final_transition_sym)

            return self._instance
