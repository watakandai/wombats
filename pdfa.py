import networkx as nx
import numpy as np
from scipy import stats
from networkx.drawing.nx_pydot import to_pydot, read_dot
import graphviz as gv
import yaml
from IPython.display import display
import multiprocessing
from joblib import Parallel, delayed
import os
import re

NUM_CORES = multiprocessing.cpu_count()


class PDFA(nx.MultiDiGraph):
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

    def __init__(self, graphDataFile, graphDataFileFormat='native'):
        """
        Constructs a new instance of a PDFA object.

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

        if graphDataFile is None:
            raise TypeError('must have a config file name')

        _, file_extension = os.path.splitext(graphDataFile)

        if file_extension == '.yaml' and graphDataFileFormat == 'native':
            configData = self.loadYAMLConfigData(graphDataFile)
        elif file_extension == '.dot' and graphDataFileFormat == 'flexfringe':
            configData = self.loadFlexFringeConfigData(graphDataFile)
        else:
            errStr = 'graphDataFile ({}) is not a .yaml or .dot ' + \
                     'file matching the supported filetype(s) for the ' +\
                     'selected graphDataFileFormat ({})'
            raise ValueError(errStr.format(graphDataFile, graphDataFileFormat))
        print(configData['edges'])
        # states and edges must be in the format needed by:
        #   - networkx.add_nodes_from()
        #   - networkx.add_edges_from()
        states, \
            edges = self.getStatesAndEdges(configData['nodes'],
                                           configData['edges'])
        if states and edges:
            self.add_nodes_from(states)
            self.add_edges_from(edges)
        else:
            raise ValueError('need non-empty states and edges lists')

        self._nodeProperties = set([k for n in self.nodes
                                    for k in self.nodes[n].keys()])
        """ a set of all of the node propety keys in each nodes' dict """

        self._DEFAULT_BETA = 0.95
        """used to set beta when it is not given in graphDataFile"""

        self.beta = configData['beta']
        """the final state probability needed for a state to accept"""

        self.alphabetSize = configData['alphabetSize']
        """number of symbols in pdfa alphabet"""

        self.numStates = configData['numStates']
        """number of states in pdfa state space"""

        self._DEFAULT_LAMBDA_TRANSITION_SYMBOL = -1
        """used to set the empty string symbol when it is not given in
           graphDataFile"""

        self.lambdaTransitionSymbol = configData['lambdaTransitionSymbol']
        """representation of the empty string / symbol (a.k.a. lambda)"""

        self.startState = configData['startState']
        """unique start state string label of pdfa"""

        # do batch computations at initialization, as these shouldn't
        # frequently change
        self.computeNodeProperties()
        self.setNodeLabels()
        self.setEdgeLabels()

    def loadFlexFringeConfigData(self, graphDataFile):
        """
        reads in graph configuration data from a flexfringe dot file

        :param      graphDataFile:  The .dot graph data configuration file name
        :type       graphDataFile:  filename path string

        :returns:   configuration data dictionary for the pdfa
        :rtype:     dictionary of pdfa data and settings
        """

        graph = read_dot(graphDataFile)
        nodes = self.convert_FlexFringeNodesToPDFANodes(graph.nodes(data=True))
        edges = self.convert_FlexFringeNodesToPDFANodes(graph.edges(data=True))

    def convert_FlexFringeNodesToPDFANodes(self, flexfringeNodes):
        """
        converts a node list from a flexfringe dot file into the internal node
        format needed by getStatesAndEdges

        :param      flexfringeNodes:  The flexfringe node list mapping node
                                      labels to node attributes
        :type       flexfringeNodes:  dict of dicts

        :returns:   a dict mapping state labels to a dict of node attributes
        :rtype:     dict of dicts

        :raises     ValueError:       can't read in "blue" flexfringe nodes, as
                                      they are theoretically undefined for this
                                      class right now
        """

        nodes = {}

        for _, nodeData in flexfringeNodes:

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

        return nodes

    def getStatesAndEdges(self, nodes, adjList):
        """
        Converts node and adjList data from a manually specified YAML config
        file to the format needed by:
            - networkx.add_nodes_from()
            - networkx.add_edges_from()
        
        :param      nodes:    dict of node objects to be converted
        :type       nodes:    dict of node label to node propeties
        :param      adjList:  dictionary adj. list to be converted
        :type       adjList:  dict of src node label to dict of dest label to
                              edge properties
        
        :returns:   properly formated node and edge list containers
        :rtype:     tuple: ( nodes - list of tuples: (node label, node
                    attribute dict), edges - list of tuples: (src node label,
                    dest node label, edge attribute dict) )
        """

        # need to convert the configuration adjacency list given in the config
        # to an edge list given as a 3-tuple of (source, dest, edgeAttrDict)
        edgeList = []
        for sourceNode, destEdgesData in adjList.items():

            # don't need to add any edges if there is no edge data
            if destEdgesData is None:
                continue

            for destNode in destEdgesData:

                symbols = destEdgesData[destNode]['symbols']
                probabilities = destEdgesData[destNode]['probabilities']

                for symbol, probability in zip(symbols, probabilities):

                    edgeData = {'symbol': symbol, 'probability': probability}
                    newEdge = (sourceNode, destNode, edgeData)
                    edgeList.append(newEdge)

        # best convention is to convert dict_items to a list, even though both
        # are iterable
        convertedNodes = list(nodes.items())

        return convertedNodes, edgeList

    def computeNodeProperties(self):
        """
        Calculates the properties for each node.

        currently calculated properties:
            - 'isAccepting'
            - 'transDistribution'
        """

        for node in self.nodes:

            # beta-acceptance property shouldn't change after load in
            self.setStateAcceptance(node, self.beta)

            # if we compute this once, we can sample from each distribution
            self.nodes[node]['transDistribution'] = \
                self.setStateTransDistribution(node, self.edges)

    def setStateAcceptance(self, currState, beta):
        """
        Sets the state acceptance property for the given state.

        If currState's final_probability >= beta, then the state accepts

        :param      currState:  The current state's node label
        :type       currState:  string
        :param      beta:       The cut point final state probability
                                acceptance parameter for the PDFA
        :type       beta:       float
        """

        currFinalProb = self.getNodeData(currState, 'final_probability')

        if currFinalProb >= self.beta:
            stateAccepts = True
        else:
            stateAccepts = False

        self.setNodeData(currState, 'isAccepting', stateAccepts)

    def setStateTransDistribution(self, currState, edges):
        """
        Computes a static state transition distribution for given state

        :param      currState:  The current state label
        :type       currState:  string
        :param      edges:      The networkx edge list
        :type       edges:      list

        :returns:   a function to sample the discrete state transition
                    distribution
        :rtype:     stats.rv_discrete object
        """

        edgeData = edges([currState], data=True)

        edgeDests = [edge[1] for edge in edgeData]
        edgeSymbols = [edge[2]['symbol'] for edge in edgeData]

        # need to add final state probability to dicrete rv dist
        edgeProbs = [edge[2]['probability'] for edge in edgeData]

        currFinalStateProb = self.getNodeData(currState, 'final_probability')

        # adding the final-state sequence end transition to the distribution
        edgeProbs.append(currFinalStateProb)
        edgeDests.append(currState)
        edgeSymbols.append(self.lambdaTransitionSymbol)

        nextSymbolDist = stats.rv_discrete(name='custm',
                                           values=(edgeSymbols, edgeProbs))

        return nextSymbolDist

    def setNodeLabels(self, graph=None):
        """
        Sets each node's label property for use in graphviz output

        :param      graph:  The graph to access. Default = None => use instance
                            (default None)
        :type       graph:  {None, PDFA, nx.MultiDiGraph}
        """

        if graph is None:
            graph = self

        labelDict = {}

        for nodeName, nodeData in graph.nodes.data():

            finalProbString = str(nodeData['final_probability'])
            nodeDotLabelString = nodeName + ': ' + finalProbString
            graphvizNodeLabel = {'label': nodeDotLabelString}

            isAccepting = nodeData['isAccepting']
            if isAccepting:
                graphvizNodeLabel.update({'shape': 'doublecircle'})
            else:
                graphvizNodeLabel.update({'shape': 'circle'})

            labelDict[nodeName] = graphvizNodeLabel

        nx.set_node_attributes(graph, labelDict)

    def setEdgeLabels(self, graph=None):
        """
        Sets each edge's label property for use in graphviz output

        :param      graph:  The graph to access. Default = None => use instance
                            (default None)
        :type       graph:  {None, PDFA, nx.MultiDiGraph}
        """

        if graph is None:
            graph = self

        # this needs to be a mapping from edges (node label tuples) to a
        # dictionary of attributes
        labelDict = {}

        for u, v, key, data in graph.edges(data=True, keys=True):

            edgeLabelString = str(data['symbol']) + ': ' + \
                str(data['probability'])

            newLabelProperty = {'label': edgeLabelString}
            nodeIdentifier = (u, v, key)

            labelDict[nodeIdentifier] = newLabelProperty

        nx.set_edge_attributes(graph, labelDict)

    def chooseNextState(self, currState, random_state=None):
        """
        Chooses the next state based on currState's transition distribution

        :param      currState:     The current state label
        :type       currState:     string
        :param      random_state:  The np.random.RandomState() seed parameter
                                   for sampling from the state transition
                                   distribution. Defaulting to None causes the
                                   seed to reset. (default None)
        :type       random_state:  {None, int, array_like}

        :returns:   The next state's label and the symbol emitted by changing
                    states
        :rtype:     tuple(string, numeric)

        :raises     ValueError:    if more than one non-zero probability
                                   transition from currState under a given
                                   symbol exists
        """

        transDist = self.nodes[currState]['transDistribution']

        # critical step for use with parallelized libraries. This must be reset
        # before sampling, as otherwise each of the threads is using the same
        # seed, and we get lots of duplicated strings
        transDist.random_state = np.random.RandomState(random_state)

        # sampling an action (symbol )from the state-action distribution at
        # currState
        nextSymbol = transDist.rvs(size=1)[0]

        if nextSymbol == self.lambdaTransitionSymbol:
            return currState, self.lambdaTransitionSymbol

        else:
            edgeData = self.edges([currState], data=True)
            nextState = [qNext for qCurr, qNext, data in edgeData
                         if data['symbol'] == nextSymbol]

            if len(nextState) > 1:
                raise ValueError('1 < transitions: ' + str(nextState) +
                                 'from' + currState + ' - not deterministic')
            else:
                return (nextState[0], nextSymbol)

    def generateTrace(self, startState, N, random_state=None):
        """
        Generates a trace from the pdfa starting from startState

        :param      startState:    the state label to start sampling traces
                                   from
        :type       startState:    string
        :param      N:             maximum length of trace
        :type       N:             scalar integer
        :param      random_state:  The np.random.RandomState() seed parameter
                                   for sampling from the state transition
                                   distribution. Defaulting to None causes the
                                   seed to reset. (default None)
        :type       random_state:  {None, int, array_like}

        :returns:   the sequence of symbols emitted and the length of the trace
        :rtype:     tuple(list of strings, integer)
        """

        currState = startState
        lengthOfTrace = 1
        nextState, nextSymbol = self.chooseNextState(currState, random_state)
        sampledTrace = str(nextSymbol)

        while nextSymbol != self.lambdaTransitionSymbol and lengthOfTrace <= N:

            nextState, nextSymbol = self.chooseNextState(currState,
                                                         random_state)

            if nextSymbol == self.lambdaTransitionSymbol:
                break

            sampledTrace += ' ' + str(nextSymbol)
            lengthOfTrace += 1
            currState = nextState

        return sampledTrace, lengthOfTrace

    def drawIPython(self):
        """
        Draws the pdfa structure in a way compatible with a jupyter / IPython
        notebook
        """

        dotString = to_pydot(self).to_string()
        display(gv.Source(dotString))

    def getNodeData(self, nodeLabel, dataKey, graph=None):
        """
        Gets the node's dataKey data from the graph

        :param      nodeLabel:  The node label
        :type       nodeLabel:  string
        :param      dataKey:    The desired node data's key name
        :type       dataKey:    string
        :param      graph:      The graph to access. Default = None => use
                                instance (default None)
        :type       graph:      {None, PDFA, nx.MultiDiGraph}

        :returns:   The node data associated with the nodeLabel and dataKey
        :rtype:     type of self.nodes.data()[nodeLabel][dataKey]
        """

        if graph is None:
            graph = self

        nodeData = graph.nodes.data()

        return nodeData[nodeLabel][dataKey]

    def setNodeData(self, nodeLabel, dataKey, data, graph=None):
        """
        Sets the node's dataKey data from the graph

        :param      nodeLabel:  The node label
        :type       nodeLabel:  string
        :param      dataKey:    The desired node data's key name
        :type       dataKey:    string
        :param      data:       The data to associate with dataKey
        :type       data:       whatever u want bro
        :param      graph:      The graph to access. Default = None => use
                                instance (default None)
        :type       graph:      {None, PDFA, nx.MultiDiGraph}
        """

        if graph is None:
            graph = self

        nodeData = graph.nodes.data()
        nodeData[nodeLabel][dataKey] = data

    def loadYAMLConfigData(self, graphDataFile):
        """
        reads in the simulation parameters from a YAML config file

        :param      graphDataFile:  The YAML graph data configuration file name
        :type       graphDataFile:  filename path string

        :returns:   configuration data dictionary for the pdfa
        :rtype:     dictionary of pdfa data and settings
        """

        with open(graphDataFile, 'r') as stream:
            configData = yaml.load(stream, Loader=yaml.Loader)

        return configData

    def generateTraces(self, numSamples, N):
        """
        generates numSamples random traces from the pdfa

        :param      numSamples:  The number of trace samples to generate
        :type       numSamples:  scalar int

        :returns:   the list of sampled trace strings and a list of the
                    associated trace lengths
        :rtype:     tuple(list(strings), list(integers))
        """

        startState = self.startState

        # make sure the numSamples is an int, so you don't have to wrap shit in
        # an 'int()' every time...
        numSamples = int(numSamples)

        iters = range(0, numSamples)
        results = Parallel(n_jobs=NUM_CORES, verbose=1)(
            delayed(self.generateTrace)(startState, N) for i in iters)

        samples, traceLengths = zip(*results)

        return samples, traceLengths

    def dispEdges(self, graph=None):
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

    def dispNodes(self, graph=None):
        """
        Prints each node's data view

        :param      graph:  The graph to access. Default = None => use instance
        :type       graph:  PDFA, nx.MultiDiGraph, or None
        """

        if graph is None:
            graph = self

        for node in graph.nodes(data=True):
            print(node)

    def writeTracesToFile(self, traces, numSamples, traceLengths, fName):
        """
        Writes trace samples to a file in the abbadingo format for use in
        flexfringe

        :param      traces:        The traces to write to a file
        :type       traces:        list of strings
        :param      numSamples:    The number sampled traces
        :type       numSamples:    integer
        :param      traceLengths:  list of sampled trace lengths
        :type       traceLengths:  list of integers
        :param      fName:         The file name to write to
        :type       fName:         filename string
        """

        # make sure the numSamples is an int, so you don't have to wrap shit in
        # an 'int()' every time...
        numSamples = int(numSamples)

        with open(fName, 'w+') as f:

            # need the header to be:
            # number_of_training_samples size_of_alphabet
            f.write(str(numSamples) + ' ' + str(self.alphabetSize) + '\n')

            for trace, traceLength in zip(traces, traceLengths):
                f.write(self.getAbbadingoString(trace, traceLength,
                                                isPositiveExample=True))

    def getAbbadingoString(self, trace, traceLength, isPositiveExample):
        """
        Returns the Abbadingo (sigh) formatted string given a trace string and
        the label for the trace

        :param      trace:              The trace string to represent in
                                        Abbadingo
        :type       trace:              string
        :param      traceLength:        The trace length
        :type       traceLength:        integer
        :param      isPositiveExample:  Indicates if the trace is a positive
                                        example of the pdfa
        :type       isPositiveExample:  boolean

        :returns:   The abbadingo formatted string for the given trace
        :rtype:     string
        """

        traceLabel = {False: '0', True: '1'}[isPositiveExample]
        return traceLabel + ' ' + str(traceLength) + ' ' + str(trace) + '\n'
