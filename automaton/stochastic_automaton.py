# 3rd-party packages
from abc import ABCMeta, abstractmethod
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
import graphviz as gv
from IPython.display import display


class StochasticAutomaton(nx.MultiDiGraph, metaclass=ABCMeta):

    """
    This class describes a automaton with stochastic transistion

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

    def __init__(self, nodes, edge_list, alphabet_size, num_states,
                 start_state, beta=0.95, final_transition_sym=-1):
        """
        Constructs a new instance of an Automaton object.

        :param      nodes:                  dict of node objects to be
                                            converted
        :type       nodes:                  dict of node label to node
                                            propeties
        :param      edge_list:              dictionary adj. list representing
                                            the edge_list
        :type       edge_list:              dict of src node label to dict of
                                            dest label to edge properties
        :param      alphabet_size:          number of symbols in pdfa alphabet
        :type       alphabet_size:          Int
        :param      num_states:             number of states in automaton state
                                            space
        :type       num_states:             Int
        :param      start_state:            unique start state string label of
                                            pdfa
        :type       start_state:            same type as PDFA.nodes node object
        :param      beta:                   the final state probability needed
                                            for a state to accept (default
                                            0.95)
        :type       beta:                   Float
        :param      final_transition_sym:  representation of the empty string
                                            / symbol (a.k.a. lambda) (defualt
                                            -1)
        :type       final_transition_sym:  same type as PDFA.edges symbol
                                            property
        """

        # need to start with a fully initialized networkx digraph
        super().__init__()

        self.add_nodes_from(nodes)
        self.add_edges_from(edge_list)

        self._beta = beta
        """the final state probability needed for a state to accept"""

        self._alphabet_size = alphabet_size
        """number of symbols in pdfa alphabet"""

        self._num_states = num_states
        """number of states in pdfa state space"""

        self._final_transition_sym = final_transition_sym
        """representation of the empty string / symbol (a.k.a. lambda)"""

        self._start_state = start_state
        """unique start state string label of pdfa"""

    def disp_edges(self, graph=None):
        """
        Prints each edge in the graph in an edge-list tuple format

        :param      graph:  The graph to access. Default = None => use instance
        :type       graph:  PDFA, nx.MultiDiGraph, or None
        """

        if graph is None:
            graph = self

        for node, neighbors in graph.adj.items():
            for neighbor, edges in neighbors.items():
                for edge_number, edge_data in edges.items():

                    print(node, neighbor, edge_data)

    def disp_nodes(self, graph=None):
        """
        Prints each node's data view

        :param      graph:  The graph to access. Default = None => use instance
        :type       graph:  PDFA, nx.MultiDiGraph, or None
        """

        if graph is None:
            graph = self

        for node in graph.nodes(data=True):
            print(node)

    def draw_IPython(self):
        """
        Draws the pdfa structure in a way compatible with a jupyter / IPython
        notebook
        """

        dot_string = to_pydot(self).to_string()
        display(gv.Source(dot_string))

    def _initialize_node_edge_properties(self, final_weight_key,
                                         can_have_accepting_nodes,
                                         edge_weight_key):
        """
        Initializes the node and edge data properties correctly for a pdfa.

        :param      final_weight_key:          key in the automaton's node data
                                               corresponding to the weight /
                                               probability of ending in that
                                               node
        :type       final_weight_key:          string
        :param      can_have_accepting_nodes:  Indicates if the automata can
                                               have accepting nodes
        :type       can_have_accepting_nodes:  boolean
        :param      edge_weight_key:           The edge data's "weight" key
        :type       edge_weight_key:           string
        """

        # do batch computations at initialization, as these shouldn't
        # frequently change
        self._compute_node_data_properties()
        self._set_node_labels(final_weight_key, can_have_accepting_nodes)
        self._set_edge_labels(edge_weight_key)

    @abstractmethod
    def _compute_node_data_properties(self):
        """
        Initializes the node and edge data properties.
        """

        return

    def _set_node_labels(self, final_weight_key, can_have_accepting_nodes,
                         graph=None):
        """
        Sets each node's label property for use in graphviz output

        :param      graph:                     The graph to access. Default =
                                               None => use instance (default
                                               None)
        :type       graph:                     {None, PDFA, nx.MultiDiGraph}
        :param      final_weight_key:          key in the automaton's node data
                                               corresponding to the weight /
                                               probability of ending in that
                                               node
        :type       final_weight_key:          string
        :param      can_have_accepting_nodes:  Indicates if the automata can
                                               have accepting nodes
        :type       can_have_accepting_nodes:  boolean
        """

        if graph is None:
            graph = self

        label_dict = {}

        for node_name, node_data in graph.nodes.data():

            final_prob_string = str(node_data[final_weight_key])
            node_dot_label_string = node_name + ': ' + final_prob_string
            graphviz_node_label = {'label': node_dot_label_string}

            if can_have_accepting_nodes:
                is_accepting = node_data['is_accepting']
                if is_accepting:
                    graphviz_node_label.update({'shape': 'doublecircle'})
                else:
                    graphviz_node_label.update({'shape': 'circle'})

            label_dict[node_name] = graphviz_node_label

        nx.set_node_attributes(graph, label_dict)

    def _set_edge_labels(self, edge_weight_key, graph=None):
        """
        Sets each edge's label property for use in graphviz output

        :param      edge_weight_key:  The edge data's "weight" key
        :type       edge_weight_key:  string
        :param      graph:            The graph to access. Default = None =>
                                      use instance (default None)
        :type       graph:            {None, PDFA, nx.MultiDiGraph}
        """

        if graph is None:
            graph = self

        # this needs to be a mapping from edges (node label tuples) to a
        # dictionary of attributes
        label_dict = {}

        for u, v, key, data in graph.edges(data=True, keys=True):

            edge_label_string = str(data['symbol']) + ': ' + \
                str(data[edge_weight_key])

            new_label_property = {'label': edge_label_string}
            node_identifier = (u, v, key)

            label_dict[node_identifier] = new_label_property

        nx.set_edge_attributes(graph, label_dict)

    def _get_node_data(self, node_label, data_key, graph=None):
        """
        Gets the node's data_key data from the graph

        :param      node_label:  The node label
        :type       node_label:  string
        :param      data_key:    The desired node data's key name
        :type       data_key:    string
        :param      graph:       The graph to access. Default = None => use
                                 instance (default None)
        :type       graph:       {None, PDFA, nx.MultiDiGraph}

        :returns:   The node data associated with the node_label and data_key
        :rtype:     type of self.nodes.data()[node_label][data_key]
        """

        if graph is None:
            graph = self

        node_data = graph.nodes.data()

        return node_data[node_label][data_key]

    def _set_node_data(self, node_label, data_key, data, graph=None):
        """
        Sets the node's data_key data from the graph

        :param      node_label:  The node label
        :type       node_label:  string
        :param      data_key:    The desired node data's key name
        :type       data_key:    string
        :param      data:        The data to associate with data_key
        :type       data:        whatever u want bro
        :param      graph:       The graph to access. Default = None => use
                                 instance (default None)
        :type       graph:       {None, PDFA, nx.MultiDiGraph}
        """

        if graph is None:
            graph = self

        node_data = graph.nodes.data()
        node_data[node_label][data_key] = data
