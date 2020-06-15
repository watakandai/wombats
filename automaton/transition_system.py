import os
from typing import Hashable, List, Tuple

# local packages
from wombats.factory.builder import Builder
from .stochastic_automaton import Automaton

# define these type defs for method annotation type hints
NXNodeList = List[Tuple[Hashable, dict]]
NXEdgeList = List[Tuple[Hashable, Hashable, dict]]


class TransitionSystem(Automaton):

    def __init__(self, nodes: NXNodeList, edge_list: NXEdgeList,
                 alphabet_size: int, num_states: int, num_obs: int,
                 start_state) -> 'TransitionSystem':
        """
        Constructs a new instance of an Automaton object.

        :param      nodes:          node list as expected by
                                    networkx.add_nodes_from()
                                    (node label, node attribute dict)
        :param      edge_list:      edge list as expected by
                                    networkx.add_edges_from()
                                    (src node label, dest node label,
                                    edge attribute dict)
        :param      alphabet_size:  number of symbols in system alphabet
        :param      num_states:     number of states in automaton state space
        :param      num_obs:        number of observation symbols
        :param      start_state:    unique start state string label of system
        """

        # need to start with a fully initialized networkx digraph
        super().__init__()

        self.add_nodes_from(nodes)
        self.add_edges_from(edge_list)


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

    def __call__(self, graph_data_file: str,
                 graph_data_format: str = 'yaml') -> TransitionSystem:
        """
        Returns an initialized TransitionSystem instance given the graph_data

        graph_data and graph_data_format must match

        :param      graph_data_file:    The graph configuration file name
        :param      graph_data_format:  The graph data file format. (default
                                        'yaml') {'yaml'}

        :returns:   instance of an initialized TransitionSystem object

        :raises     ValueError:         checks if graph_data and
                                        graph_data_format have a compatible
                                        data loader
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

            # saving these so we can just return initialized instances if the
            # underlying data has not changed
            self.nodes = config_data['nodes']
            self.edges = config_data['edges']

            self._instance = TransitionSystem(
                nodes=self.nodes,
                edge_list=self.edges,
                alphabet_size=config_data['alphabet_size'],
                num_states=config_data['num_states'],
                num_obs=config_data['num_obs'],
                start_state=config_data['start_state'])

            return self._instance
