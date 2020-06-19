# 3rd-party packages
import pygraphviz
import re
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
from networkx.drawing import nx_agraph
from typing import Hashable
from bidict import bidict

# local packages
from wombats.factory.builder import Builder
from .base import Automaton, NXNodeList, NXEdgeList, Node, Symbol


class FDFA(Automaton):
    """
    This class describes a frequency deterministic finite automaton (fdfa).

    built on networkx, so inherits node and edge data structure definitions

    Node Attributes
    -----------------
        - final_frequency: final state frequency for each node.
                           Number of times that a trace ended in that state.
        - in_frequency:    in "flow" of state frequency for each node
                           total times that state was visited with incoming
                           transitions.
        - out_frequency:   out "flow" of state frequency for each node
                           total times that state was visited with outgoing
                           transitions.
        - self_frequency:  the frequency of self transition for each node.
        - trans_distribution: None, just there for consistency with PDFA
        - is_accepting: None, just there for consistency with PDFA

    Edge Properties
    -----------------
        - symbol: the symbol value emitted when the edge is traversed
        - frequency: the number of times the edge was traversed
    """

    def __init__(self, nodes: NXNodeList,
                 edges: NXEdgeList,
                 symbol_display_map: bidict,
                 alphabet_size: int,
                 num_states: int,
                 start_state: Hashable,
                 final_transition_sym=-1) -> 'FDFA':
        """
        Constructs a new instance of a FDFA object.

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
        :param      final_transition_sym:  representation of the empty string /
                                           symbol (a.k.a. lambda) (default -1)
        """

        # need to start with a fully initialized automaton
        super().__init__(nodes, edges, symbol_display_map,
                         alphabet_size, num_states, start_state,
                         smooth_transitions=False,
                         is_stochastic=False,
                         final_transition_sym=final_transition_sym,
                         final_weight_key='final_frequency',
                         can_have_accepting_nodes=False,
                         edge_weight_key='frequency')

    @classmethod
    def load_flexfringe_data(cls: 'FDFA', graph: nx.MultiDiGraph,
                             final_transition_sym: Symbol) -> dict:
        """
        reads in graph configuration data from a flexfringe dot file

        :param      cls:                   The "class instance" this method
                                           belongs to (not object instance)
        :param      graph:                 The nx graph with the flexfringe
                                           fdfa model loaded in
        :param      final_transition_sym:  representation of the empty string /
                                           symbol (a.k.a. lambda) (default -1)

        :returns:   configuration data dictionary for the fdfa
        :rtype:     dictionary
        """

        ff_nodes = graph.nodes(data=True)
        ff_edges = graph.edges(data=True)

        nodes, node_ID_to_node_label = cls.convert_flexfringe_nodes(ff_nodes)
        (symbol_display_map,
         edges, symbols) = cls.convert_flexfringe_edges(ff_edges,
                                                        final_transition_sym,
                                                        node_ID_to_node_label)
        root_node_label = '0'
        config_data = {
            'nodes': nodes,
            'edges': edges,
            'symbol_display_map': symbol_display_map,
            'alphabet_size': len(symbols.keys()),
            'num_states': len(nodes),
            'start_state': node_ID_to_node_label[root_node_label]}

        return config_data

    @staticmethod
    def convert_flexfringe_nodes(flexfringe_nodes: dict) -> (NXNodeList, dict):
        """
        converts node data from a flexfringe (FF) dot file into the internal
        node format needed by networkx.add_nodes_from()

        :param      flexfringe_nodes:  The flexfringe node list mapping node
                                       labels to node attributes

        :returns:   node list as expected by networkx.add_nodes_from(),
                    a dict mapping FF node IDs to FF state labels

        :raises     ValueError:        can't read in "blue" flexfringe nodes,
                                       as they are theoretically undefined for
                                       this class right now
        """

        nodes = {}
        node_ID_to_node_label = {}

        for node_ID, node_data in flexfringe_nodes:

            if 'label' not in node_data:
                continue

            state_label = re.findall(r'\d+', node_data['label'])

            # we can't add blue nodes to our graph
            if 'style' in node_data:
                if 'dotted' in node_data['style']:
                    err = ('node = {} from flexfringe is blue,'
                           ' reading in blue states is not'
                           ' currently supported').format(node_data)
                    raise ValueError(err)

            new_node_label = state_label[0]
            new_node_data = {'final_frequency': 0,
                             'trans_distribution': None,
                             'isAccepting': None}

            nodes[new_node_label] = new_node_data
            node_ID_to_node_label[node_ID] = new_node_label

        return nodes, node_ID_to_node_label

    @staticmethod
    def convert_flexfringe_edges(flexfringeEdges: NXEdgeList,
                                 final_transition_sym: Symbol,
                                 node_ID_to_node_label: dict) -> (bidict,
                                                                  NXEdgeList,
                                                                  dict):
        """
        converts edges read in from flexfringe (FF) dot files into the internal
        edge format needed by networkx.add_edges_from()

        :param      flexfringeEdges:        The flexfringe edge list mapping
                                            edges labels to edge attributes
        :param      final_transition_sym:   representation of the empty string
                                            / symbol (a.k.a. lambda)
        :param      node_ID_to_node_label:  mapping from FF node ID to FF node
                                            label

        :returns:   symbol_display_map - bidirectional mapping of hashable
                                         symbols, to a unique integer index in
                                         the symbol map.
                    edge list as expected by networkx.add_edges_from(),
                    dictionary of symbol counts
        """

        edges = []
        all_symbols = {}

        symbol_count = 0
        symbol_display_map = bidict({})
        for src_FF_node_ID, dest_FF_node_ID, edge_data in flexfringeEdges:

            new_edge_data = {}

            if 'label' not in edge_data:
                continue

            transitionData = re.findall(r'(\d+):(\d+)', edge_data['label'])
            symbols, frequencies = zip(*transitionData)

            for symbol, frequency in transitionData:

                symbol = int(symbol)

                # need to store new symbols in a map for display
                if symbol not in symbol_display_map:
                    symbol_count += 1
                    symbol_display_map[symbol] = symbol_count

                # want to keep track of frequency of all symbols
                if symbol in all_symbols:
                    all_symbols[symbol] += 1
                else:
                    all_symbols[symbol] = 1

                new_edge_data = {'symbol': symbol,
                                 'frequency': int(frequency)}

                src_FF_node_label = node_ID_to_node_label[src_FF_node_ID]
                dest_FF_node_label = node_ID_to_node_label[dest_FF_node_ID]
                new_edge = (src_FF_node_label,
                            dest_FF_node_label,
                            new_edge_data)

                edges.append(new_edge)

        # we need to add the empty / final symbol to the display map
        # for completeness
        symbol_display_map[final_transition_sym] = final_transition_sym

        return symbol_display_map, edges, all_symbols

    def _set_state_acceptance(self, curr_state: Node) -> None:
        """
        Sets the state acceptance property for the given state.

        FDFA doesn't accept anything, so this just passes
        """
        pass

    def _compute_node_data_properties(self, curr_node: Node) -> None:
        """
        Sets all state frequencies for each node in an initialized FDFA

        requires self.nodes and self.edges to be properly loaded into nx data
        structures

        :warning this overrides the base _compute_node_data_properties method
                 in the Automaton

        :param      curr_node:   The node to set properties for

        :returns:   None

        :raises     ValueError:  checks if the final frequency is less than 0,
                                 indicating something wrong with the edge
                                 frequency data
        """

        number_trans_in, _ = self._compute_node_flow(curr_node,
                                                     flow_type='in')
        (number_trans_out,
         number_self_trans) = self._compute_node_flow(curr_node,
                                                      flow_type='out')

        # the final frequency is simply the number of times that you
        # transitioned into a state and then did not leave it.
        #
        # inflow and outflow must not include self transitions, as it self
        # transitions are not true flow
        curr_node_final_freq = number_trans_in - number_trans_out

        # all flow comes from the root node, so it is the only node allowed
        # to "create" transitions
        is_root_node = (curr_node == self.start_state)
        if curr_node_final_freq < 0 and not is_root_node:
            err = 'current node ({}) final frequency ({}) should ' + \
                  'not be less than 0. This means there were more ' +\
                  'outgoing transitions ({}) than incoming ' +\
                  'transitions ({}).'
            raise ValueError(err.format(curr_node, curr_node_final_freq,
                                        number_trans_out, number_trans_in))
        elif is_root_node:

            # only set absolute value of frequency "flow" for the root
            # node, as it is the only node allowed to create frequency
            # flow
            #
            # @warning this shit doesn't really work right now
            curr_node_final_freq = abs(curr_node_final_freq)

        self._set_node_data(curr_node,
                            'final_frequency', curr_node_final_freq)
        self._set_node_data(curr_node,
                            'self_frequency', number_self_trans)
        self._set_node_data(curr_node,
                            'in_frequency', number_trans_in)
        self._set_node_data(curr_node,
                            'out_frequency', number_trans_out)

    def _compute_node_flow(self, curr_node: Node,
                           flow_type: str) -> (int, int):
        """
        Calculates frequency (in/out)flow at the current node

        :param      curr_node:   The node to compute the flow at
        :param      flow_type:   The flow type {'in', 'out'}


        :returns:   The node's (in/out)flow, the node's self-transition flow

        :raises     ValueError:  checks if flow_type is an supported setting
        """

        allowed_flow_types = ['in', 'out']
        if flow_type not in allowed_flow_types:
            msg = ('selected flow_type ({}) not one of '
                   'allowed flow_types: {}').format(flow_type,
                                                    allowed_flow_types)
            raise ValueError(msg)

        if flow_type == 'in':
            nodes = self.predecessors(curr_node)
        elif flow_type == 'out':
            nodes = self.successors(curr_node)

        number_trans = 0
        number_self_trans = 0
        for node in nodes:

            if flow_type == 'in':
                curr_edges = self.get_edge_data(node, curr_node)
            elif flow_type == 'out':
                curr_edges = self.get_edge_data(curr_node, node)

            for _, curr_out_edge_data in curr_edges.items():
                frequency = curr_out_edge_data['frequency']

                # don't want to count flow from self-loops, as they
                # cannot create flow and thus should only be counted as a
                # possible choice and not as (in/out)flow
                if node == curr_node:
                    number_self_trans += frequency
                else:
                    number_trans += frequency

        return number_trans, number_self_trans


class FDFABuilder(Builder):
    """
    Implements the generic automaton builder class for FDFA objects
    """

    def __init__(self) -> 'FDFABuilder':
        """
        Constructs a new instance of the FDFABuilder
        """

        # need to call the super class constructor to gain its properties
        Builder.__init__(self)

        # keep these properties so we don't re-initialize unless underlying
        # data changes
        self.nodes = None
        self.edges = None

    def __call__(self, graph_data: str,
                 graph_data_format: str = 'dot_string') -> FDFA:
        """
        Returns an initialized FDFA instance given the graph_data

        graph_data and graph_data_format must match

        :param      graph_data:         The string containing graph data. Could
                                        be a filename or just the raw data
        :param      graph_data_format:  The graph data file format.
                                        (default 'dot_file')
                                        {'dot_file', 'dot_string'}

        :returns:   instance of an initialized FDFA object

        :raises     ValueError:         checks if graph_data and
                                        graph_data_format have a
                                        compatible data loader
        """

        if graph_data_format == 'dot_string':
            graph = nx_agraph.from_agraph(pygraphviz.AGraph(string=graph_data))
        elif graph_data_format == 'dot_file':
            graph = read_dot(graph_data)
        else:
            msg = 'graph_data_format ({}) must be one of: "dot_file", ' + \
                  '"dot_string"'.format(graph_data_format)
            raise ValueError(msg)

        # these are not things that are a part of flexfringe's automaton
        # data model, so give them default values
        final_transition_sym = -1
        config_data = FDFA.load_flexfringe_data(graph, final_transition_sym)

        nodes_have_changed = (self.nodes != config_data['nodes'])
        edges_have_changed = (self.edges != config_data['edges'])
        no_instance_loaded_yet = (self._instance is None)

        if no_instance_loaded_yet or nodes_have_changed or edges_have_changed:

            self._instance = FDFA(
                nodes=config_data['nodes'],
                edges=config_data['edges'],
                symbol_display_map=config_data['symbol_display_map'],
                alphabet_size=config_data['alphabet_size'],
                num_states=config_data['num_states'],
                final_transition_sym=final_transition_sym,
                start_state=config_data['start_state'])

        return self._instance
