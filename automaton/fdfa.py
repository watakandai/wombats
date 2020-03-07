# 3rd-party packages
import pygraphviz
from networkx.drawing.nx_pydot import read_dot
from networkx.drawing import nx_agraph
import re

# local packages
from wombats.factory.builder import Builder
from .stochastic_automaton import StochasticAutomaton


class FDFA(StochasticAutomaton):
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

    def __init__(self, nodes: list, edges: list, alphabet_size: int,
                 num_states: int, start_state, beta: float=0.95,
                 final_transition_sym=-1) -> 'FDFA':
        """
        Constructs a new instance of a FDFA object.

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
        :type       start_state:           same type as FDFA.nodes node object
        :param      beta:                  the final state probability needed
                                           for a state to accept. Not used for
                                           FDFA (default 0.95)
        :type       beta:                  Float
        :param      final_transition_sym:  representation of the empty string /
                                           symbol (a.k.a. lambda) (default -1)
        :type       final_transition_sym:  same type as FDFA.edges symbol
                                           property
        """

        # need to start with a fully initialized automaton
        super().__init__(nodes, edges, alphabet_size, num_states,
                         start_state, beta=0.95, final_transition_sym=-1)

        self._initialize_node_edge_properties(
            final_weight_key='final_frequency',
            can_have_accepting_nodes=False,
            edge_weight_key='frequency')

    @classmethod
    def load_flexfringe_data(cls: 'FDFA', graph) -> dict:
        """
        reads in graph configuration data from a flexfringe dot file

        :param      cls:    The "class instance" this method belongs to (not
                            object instance)
        :type       cls:    FDFA
        :param      graph:  The graph with the flexfringe fdfa model
        :type       graph:  networkx (MultiDi/Di)graph

        :returns:   configuration data dictionary for the fdfa
        :rtype:     dictionary
        """

        ff_nodes = graph.nodes(data=True)
        ff_edges = graph.edges(data=True)

        nodes, node_ID_to_node_label = cls.convert_flexfringe_nodes(ff_nodes)
        edges, symbols = cls.convert_flexfringe_edges(ff_edges,
                                                      node_ID_to_node_label)
        root_node_label = '0'
        config_data = {
            'nodes': nodes,
            'edges': edges,
            'alphabet_size': len(symbols.keys()),
            'num_states': len(nodes),
            'start_state': node_ID_to_node_label[root_node_label],
            # these are not things that are a part of flexfringe's automaton
            # data model, so give them default values
            'beta': 0.95,
            'final_transition_sym': -1}

        return config_data

    @staticmethod
    def convert_flexfringe_nodes(flexfringe_nodes: dict) -> (list, dict):
        """
        converts a node list from a flexfringe (FF) dot file into the internal
        node format needed by networkx.add_nodes_from()

        :param      flexfringe_nodes:  The flexfringe node list mapping node
                                       labels to node attributes
        :type       flexfringe_nodes:  dict of dicts

        :returns:   node list as expected by networkx.add_nodes_from(),
                    a dict mapping FF node IDs to FF state labels
        :rtype:     list of tuples: (node label, node
                    attribute dict), dict

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
                    err = ('node = {} from flexfringe is blue,' +
                           ' reading in blue states is not' +
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
    def convert_flexfringe_edges(flexfringeEdges: list,
                                 node_ID_to_node_label: dict) -> (list, dict):
        """
        converts edges read in from flexfringe (FF) dot files into the internal
        edge format needed by networkx.add_edges_from()

        :param      flexfringeEdges:        The flexfringe edge list mapping
                                            edges labels to edge attributes
        :type       flexfringeEdges:        list of tuples of: (src_FF_node_ID,
                                            src_FF_node_ID, edge_data)
        :param      node_ID_to_node_label:  mapping from FF node ID to FF node
                                            label
        :type       node_ID_to_node_label:  dict

        :returns:   edge list as expected by networkx.add_edges_from(),
                    dictionary of symbol counts
        :rtype:     edges - list of tuples: (src node label,
                    dest node label, edge attribute dict),
                    all_symbols - dict
        """

        edges = []
        all_symbols = {}

        for src_FF_node_ID, dest_FF_node_ID, edge_data in flexfringeEdges:

            new_edge_data = {}

            if 'label' not in edge_data:
                continue

            transitionData = re.findall(r'(\d+):(\d+)', edge_data['label'])
            symbols, frequencies = zip(*transitionData)

            for symbol, frequency in transitionData:

                # want to keep track of frequency of all symbols
                if symbol in all_symbols:
                    all_symbols[symbol] += 1
                else:
                    all_symbols[symbol] = 1

                new_edge_data = {'symbol': int(symbol),
                                 'frequency': int(frequency)}

                src_FF_node_label = node_ID_to_node_label[src_FF_node_ID]
                dest_FF_node_label = node_ID_to_node_label[dest_FF_node_ID]
                new_edge = (src_FF_node_label,
                            dest_FF_node_label,
                            new_edge_data)

                edges.append(new_edge)

        return edges, all_symbols

    def _compute_node_data_properties(self) -> None:
        """
        Initializes the node and edge data properties correctly for a fdfa.
        """

        self._set_state_frequencies()

    def _set_state_frequencies(self) -> None:
        """
        Sets all state frequencies for each node in an initialized FDFA

        requires self.nodes and self.edges to be properly loaded into nx data
        structures

        :raises     ValueError:  checks if the final frequency is less than 0,
                                 indicating something wrong with the edge
                                 frequency data
        """

        nodes = self.nodes

        for curr_node in nodes:

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
            is_root_node = (curr_node == self._start_state)
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
                curr_node_final_freq = abs(curr_node_final_freq)

            self._set_node_data(curr_node,
                                'final_frequency', curr_node_final_freq)
            self._set_node_data(curr_node,
                                'self_frequency', number_self_trans)
            self._set_node_data(curr_node,
                                'in_frequency', number_trans_in)
            self._set_node_data(curr_node,
                                'out_frequency', number_trans_out)

    def _compute_node_flow(self, curr_node: str, flow_type: str) -> (int, int):
        """
        Calculates frequency (in/out)flow at the current node

        :param      curr_node:   The node to compute the flow at
        :type       curr_node:   str
        :param      flow_type:   The flow type {'in', 'out'}
        :type       flow_type:   str

        :returns:   The node's (in/out)flow, the node's self-transition flow
        :rtype:     tuple of ints

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
                 graph_data_format: str='dot_string') -> FDFA:
        """
        Returns an initialized FDFA instance given the graph_data

        graph_data and graph_data_format must match

        :param      graph_data:         The string containing graph data. Could
                                        be a filename or just the raw data
        :type       graph_data:         string
        :param      graph_data_format:  The graph data file format.
                                        (default 'dot_file')
                                        {'dot_file', 'dot_string'}
        :type       graph_data_format:  string

        :returns:   instance of an initialized FDFA object
        :rtype:     FDFA

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

        config_data = FDFA.load_flexfringe_data(graph)

        nodes_have_changed = (self.nodes != config_data['nodes'])
        edges_have_changed = (self.edges != config_data['edges'])
        no_instance_loaded_yet = (self._instance is None)

        if no_instance_loaded_yet or nodes_have_changed or edges_have_changed:

            self._instance = FDFA(
                nodes=config_data['nodes'],
                edges=config_data['edges'],
                beta=config_data['beta'],
                alphabet_size=config_data['alphabet_size'],
                num_states=config_data['num_states'],
                final_transition_sym=config_data['final_transition_sym'],
                start_state=config_data['start_state'])

        return self._instance
