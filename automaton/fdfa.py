# 3rd-party packages
import networkx as nx
import os
from networkx.drawing.nx_pydot import read_dot
import re

# local packages
from wombats.factory.builder import Builder


class FDFA(nx.MultiDiGraph):
    """
    This class describes a probabilistic deterministic finite automaton (pdfa).

    built on networkx, so inherits node and edge data structure definitions

    Node Attributes
    -----------------
        - final_probability: final state probability for the node
        - transDistribution: a sampled-able function to select the next state
                             and emitted symbol
        - isAccepting: a boolean flag determining whether the pdfa considers
                       the node accepting

    Edge Properties
    -----------------
        - symbol: the numeric symbol value emitted when the edge is traversed
        - probability: the probability of selecting this edge for traversal,
                       given the starting node
    """

    def __init__(self, graphDataFile, graphDataFileFormat='flexfringe'):
        """
        Constructs a new instance of a FDFA object.

        :param      graphDataFile:        The graph configuration file name
        :type       graphDataFile:        filename path string
        :param      graphDataFileFormat:  The graph data file format.
                                          Supported formats:
                                          - 'native'
                                          - 'flexfringe'
                                          (Defualt 'native')
        :type       graphDataFileFormat:  string
        """

        # need to start with a fully initialized networkx digraph
        super().__init__()

    @staticmethod
    def loadFlexFringeConfigData(graphDataFile):
        """
        reads in graph configuration data from a flexfringe dot file

        :param      graphDataFile:  The .dot graph data configuration file name
        :type       graphDataFile:  filename path string

        :returns:   configuration data dictionary for the pdfa
        :rtype:     dictionary of pdfa data and settings
        """

        graph = read_dot(graphDataFile)
        nodes = FDFA.convert_FlexFringeNodes(graph.nodes(data=True))
        edges = FDFA.convert_FlexFringeEdges(graph.edges(data=True))
        # need to set final state probs in nodes

    @staticmethod
    def convert_FlexFringeNodes(flexfringeNodes):
        """
        converts a node list from a flexfringe dot file into the internal node
        format needed by getStatesAndEdges

        :param      flexfringeNodes:  The flexfringe node list mapping node
                                      labels to node attributes
        :type       flexfringeNodes:  dict of dicts

        :returns:   a dict mapping state labels to a dict of node attributes,
                    a dict mapping state labels to flexfringe node IDs,
        :rtype:     dict of dicts,
                    dict

        :raises     ValueError:       can't read in "blue" flexfringe nodes, as
                                      they are theoretically undefined for this
                                      class right now
        """

        nodes = {}
        nodeLabelToNodeIDMap = {}

        for nodeID, nodeData in flexfringeNodes:

            if 'label' not in nodeData:
                continue

            stateLabel = re.findall(r'\d+', nodeData['label'])

            # we can't add blue nodes to our graph
            if 'style' in nodeData:
                if 'dotted' in nodeData['style']:
                    err = ('node = {} from flexfringe is blue,' +
                           ' reading in blue states is not' +
                           ' currently supported').format(nodeData)
                    raise ValueError(err)

            newNodeLabel = 'q' + stateLabel[0]
            newNodeData = {'final_probability': 0.0,
                           'transDistribution': None,
                           'isAccepting': None}
            nodes[newNodeLabel] = newNodeData
            nodeLabelToNodeIDMap[newNodeLabel] = nodeID

        return nodes, nodeLabelToNodeIDMap

    @staticmethod
    def convert_FlexFringeEdges(flexfringeEdges):
        """
        converts edges read in from flexfringe (FF) dot files into the internal
        edge format needed by getStatesAndEdges

        :param      flexfringeEdges:  The flexfringe edge list mapping edges
                                      labels to edge attributes
        :type       flexfringeEdges:  list of tuples of: (srcFFNodeID,
                                      srcFFNodeID, edgeData)

        :returns:   { description_of_the_return_value }
        :rtype:     { return_type_description }
        """

        edges = {}

        for srcFFNodeID, destFFNodeID, edgeData in flexfringeEdges:

            if 'label' not in edgeData:
                continue
            print(srcFFNodeID, destFFNodeID, edgeData)
            transitionData = re.findall(r'(\d+):(\d+)', edgeData['label'])

            symbols = []

            # for symbol, frequency in transitionData:


        return None


class FDFABuilder(Builder):
    """
    Implements the generic automaton builder class for FDFA objects
    """

    def __init__(self):
        """
        Constructs a new instance of the FDFABuilder
        """

        # need to call the super class constructor to gain its properties
        Builder.__init__(self)

        # keep these properties so we don't re-initailize unless underlying
        # data changes
        self.nodes = None
        self.edges = None

    def __call__(self, graphDataFile, graphDataFileFormat='flexfringe'):
        """
        Implements the smart constructor for FDFA

        Only reads the config data once, otherwise just returns the built
        object

        :param      graphDataFile:        The graph configuration file name
        :type       graphDataFile:        filename path string
        :param      graphDataFileFormat:  The graph data file format.
                                          Supported formats:
                                          - 'flexfringe'
                                          (Defualt 'flexfringe')
        :type       graphDataFileFormat:  string

        :returns:   instance of an initialized FDFA object
        :rtype:     FDFA
        """

        _, file_extension = os.path.splitext(graphDataFile)
        if file_extension == '.dot' and graphDataFileFormat == 'flexfringe':
            configData = FDFA.loadFlexFringeConfigData(graphDataFile)
        else:
            errStr = 'graphDataFile ({}) is not a .dot ' + \
                     'file matching the supported filetype(s) for the ' +\
                     'selected graphDataFileFormat ({})'
            raise ValueError(errStr.format(graphDataFile, graphDataFileFormat))

        nodesHaveChanged = (self.nodes != configData['nodes'])
        edgesHaveChanged = (self.edges != configData['edges'])
        noInstanceLoadedYet = (self._instance is None)

        if noInstanceLoadedYet or nodesHaveChanged or edgesHaveChanged:

            self._instance = FDFA(
                states=configData['states'],
                edges=configData['edges'],
                beta=configData['beta'],
                alphabetSize=configData['alphabetSize'],
                numStates=configData['numStates'],
                lambdaTransitionSymbol=configData['lambdaTransitionSymbol'],
                startState=configData['startState'])

        return self._instance
