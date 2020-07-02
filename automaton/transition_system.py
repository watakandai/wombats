import os
import collections
from typing import Tuple, Iterable
from bidict import bidict

# local packages
from wombats.factory.builder import Builder
from .base import (Automaton, NXNodeList, NXEdgeList, Node, Nodes,
                   Observation, Symbol, Symbols, DEFAULT_FINAL_TRANS_SYMBOL,
                   DEFAULT_EMPTY_TRANS_SYMBOL)

from wombats.systems import StaticMinigridTSWrapper
from wombats.systems.minigrid import ActionsEnum

# define these type defs for method annotation type hints
TS_Trans_Data = Tuple[Node, Observation]
EnvAct = ActionsEnum
EnvActs = Iterable[ActionsEnum]


class TransitionSystem(Automaton):
    """
    A representation of a transition system automaton (a.k.a. moore machine)

    :param      nodes:                 node list as expected by
                                       networkx.add_nodes_from() (node
                                       label, node attribute dict)
    :param      edges:                 edge list as expected by
                                       networkx.add_edges_from() (src node
                                       label, dest node label, edge
                                       attribute dict)
    :param      symbol_display_map:    bidirectional mapping of
                                       hashable symbols, to a unique
                                       integer index in the symbol map.
                                       Needed to translate between the
                                       indices in the transition
                                       distribution and the hashable
                                       representation which is
                                       meaningful to the user
    :param      alphabet_size:         number of symbols in system alphabet
    :param      num_states:            number of states in automaton state
                                       space
    :param      num_obs:               number of observation symbols
    :param      start_state:           unique start state string label of
                                       system
    :param      final_transition_sym:  representation of the termination
                                       symbol. If not given, will default
                                       to base class default.
    :param      empty_transition_sym:  representation of the empty symbol
                                       (a.k.a. lambda). If not given, will
                                       default to base class default.
    """

    def __init__(
        self,
        nodes: NXNodeList,
        edges: NXEdgeList,
        symbol_display_map: bidict,
        alphabet_size: int,
        num_states: int,
        start_state: Node,
        num_obs: int,
        final_transition_sym: {Symbol, None}=DEFAULT_FINAL_TRANS_SYMBOL,
        empty_transition_sym: {Symbol, None}=DEFAULT_EMPTY_TRANS_SYMBOL
    ) -> 'TransitionSystem':

        # need to start with a fully initialized automaton
        super().__init__(nodes, edges, symbol_display_map,
                         alphabet_size, num_states, start_state,
                         smooth_transitions=False,
                         is_stochastic=False,
                         is_sampleable=True,
                         num_obs=num_obs,
                         final_transition_sym=final_transition_sym,
                         empty_transition_sym=empty_transition_sym,
                         state_observation_key='observation',
                         can_have_accepting_nodes=True,
                         edge_weight_key=None)

    def transition(self, curr_state: Node,
                   input_symbol: str,
                   **get_next_state_kwargs: dict) -> TS_Trans_Data:
        """
        transitions the TS given the current TS state and an input symbol, then
        outputs the state observation

        :param      curr_state:             The current TS state
        :param      input_symbol:           The input TS symbol
        :param      get_next_state_kwargs:  Any additional inputs to
                                            _get_next_state

        :returns:   the next TS state, and the obs
        """

        next_state, _ = self._get_next_state(curr_state, input_symbol,
                                             **get_next_state_kwargs)
        observation = self.observe(next_state)

        return next_state, observation

    def observe(self, curr_state: Node) -> Observation:
        """
        Returns the given state's observation symbol

        :param      curr_state:  The current TS state

        :returns:   observation symbol emitted at curr_state
        """

        return self._get_node_data(curr_state, 'observation')

    def run(self, word: {Symbol, Symbols},
            **get_next_state_kwargs: dict) -> Tuple[Symbols, Nodes]:
        """
        processes a input word and produces a output word & state sequence

        :param      word:                   The word to process
        :param      get_next_state_kwargs:  Any additional inputs to
                                            _get_next_state

        :returns:   output word (list of symbols), list of states visited

        :raises     ValueError:             Catches and re-raises exceptions
                                            from invalid symbol use
        """

        # need to do type-checking / polymorphism handling here
        if isinstance(word, str) or not isinstance(word, collections.Iterable):
            word = [word]

        curr_state = self.start_state
        output_word = [self.observe(curr_state)]
        state_sequence = [curr_state]

        for symbol in word:
            try:
                next_state, observation = self.transition(
                    curr_state, symbol,
                    **get_next_state_kwargs)
            except ValueError as e:
                msg = f'Invalid symbol encountered processesing ' + \
                      f'word: {word}.\ncurrent output word: {output_word}' + \
                      f' \ncurrent state sequence: {state_sequence}'
                raise ValueError(msg) from e

            output_word.append(observation)
            state_sequence.append(next_state)

            curr_state = next_state

        return output_word, state_sequence

    def _set_state_acceptance(self, curr_state: Node) -> None:
        """
        Sets the state acceptance property for the given state.

        TS doesn't accept anything, so this just passes
        """
        pass


class MinigridTransitionSystem(TransitionSystem):

    def __init__(self, **kwargs):

        self.env = kwargs['env']

        # normal TS don't have an 'env'
        kwargs.pop('env', None)
        super().__init__(**kwargs)

    def run(self,
            word: {EnvAct, EnvActs, Symbol, Symbols},
            show_steps: bool = False) -> Tuple[Symbols, Nodes]:
        """
        processes a input word and produces a output word & state sequence

        :param      word:        The word to process

        :returns:   output word (list of symbols), list of states visited

        :raises     ValueError:  Catches and re-raises exceptions from
                                 invalid symbol use
        """

        self.env.reset()

        if show_steps:
            self.env.render_notebook()

        # need to do type-checking / polymorphism handling here
        if isinstance(word, str) or not isinstance(word, collections.Iterable):
            word = [word]

        if isinstance(word[0], self.env.actions):
            word = [self.env.ACTION_ENUM_TO_STR[action] for action in word]

        return super().run(word, show_steps=show_steps)

    def _get_next_state(self, curr_state: Node, symbol: Symbol,
                        show_steps: bool = False) -> Tuple[Node, None]:
        """
        Gets the next state given the current state and the "input" symbol.

        computes this using the underlying environment's step() function

        :param      curr_state:  The current state
        :param      symbol:      The input symbol
        :param      show_steps:  turn on / off displaying pictures of the
                                 environment after getting the symbol

        :returns:   (The next state label, the transition probability)

        :raises     ValueError:  symbol not in curr_state's transition function
        :raises     ValueError:  duplicate symbol in curr_state's transition
                                 function
        """

        (possible_symbols, _) = self._get_trans_probabilities(curr_state)

        if symbol not in possible_symbols:
            msg = ('given symbol ({}) is not found in the '
                   'curr_state\'s ({}) '
                   'transition distribution').format(symbol, curr_state)
            raise ValueError(msg)

        symbol_idx = [i for i, val in enumerate(possible_symbols)
                      if val == symbol]
        num_matched_symbols = len(symbol_idx)
        if num_matched_symbols != 1:
            msg = ('given symbol ({}) is found multiple times in '
                   'curr_state\'s ({}) '
                   'transition distribution').format(symbol, curr_state)
            raise ValueError(msg)

        symbol_probability = None

        curr_state = self.env._get_state_from_str(curr_state)
        action = self.env.ACTION_STR_TO_ENUM[symbol]
        next_pos, next_dir, done = self.env._make_transition(action,
                                                             *curr_state)
        next_state_label = self.env._get_state_str(next_pos, next_dir)

        if show_steps:
            self.env.render_notebook()

        return next_state_label, symbol_probability


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

    def __call__(self, graph_data: {str, StaticMinigridTSWrapper},
                 graph_data_format: str = 'yaml') -> TransitionSystem:
        """
        Returns an initialized TransitionSystem instance given the graph_data

        graph_data and graph_data_format must match

        :param      graph_data:         The graph configuration data
                                        {str, Minigrid environment wrapper}
        :param      graph_data_format:  The graph data file format.
                                        {'yaml', 'minigrid'}

        :returns:   instance of an initialized TransitionSystem object

        :raises     ValueError:         checks if graph_data and
                                        graph_data_format have a compatible
                                        data loader
        """

        if graph_data_format == 'yaml':
            config_data = self._from_yaml(graph_data)
            TS_Type = TransitionSystem
        elif graph_data_format == 'minigrid':
            config_data = self._from_minigrid(graph_data)
            TS_Type = MinigridTransitionSystem
        else:
            msg = 'graph_data_format ({}) must be one of: "yaml", ' + \
                  '"minigrid"'.format(graph_data_format)
            raise ValueError(msg)

        nodes_have_changed = (self.nodes != config_data['nodes'])
        edges_have_changed = (self.edges != config_data['edges'])
        no_instance_loaded_yet = (self._instance is None)

        if no_instance_loaded_yet or nodes_have_changed or edges_have_changed:

            # nodes and edge_list must be in the format needed by:
            #   - networkx.add_nodes_from()
            #   - networkx.add_edges_from()
            if 'final_transition_sym' not in config_data:
                final_transition_sym = DEFAULT_FINAL_TRANS_SYMBOL
            else:
                final_transition_sym = config_data['final_transition_sym']

            if 'empty_transition_sym' not in config_data:
                empty_transition_sym = DEFAULT_EMPTY_TRANS_SYMBOL
            else:
                empty_transition_sym = config_data['empty_transition_sym']

            (symbol_display_map,
             states,
             edges) = Automaton._convert_states_edges(config_data['nodes'],
                                                      config_data['edges'],
                                                      final_transition_sym,
                                                      empty_transition_sym,
                                                      is_stochastic=False)
            config_data['symbol_display_map'] = symbol_display_map

            # saving these so we can just return initialized instances if the
            # underlying data has not changed
            self.nodes = states
            self.edges = edges
            config_data['nodes'] = self.nodes
            config_data['edges'] = self.edges

            self._instance = TS_Type(**config_data)

        return self._instance

    def _from_minigrid(
        self,
        graph_data: StaticMinigridTSWrapper
    ) -> TransitionSystem:

        config_data = graph_data.TS_config_data
        config_data['env'] = graph_data

        return config_data

    def _from_yaml(self, graph_data: str) -> dict:

        _, file_extension = os.path.splitext(graph_data)

        allowed_exts = ['.yaml', '.yml']
        if file_extension in allowed_exts:
            config_data = self.load_YAML_config_data(graph_data)
        else:
            msg = 'graph_data ({}) is not a ({}) file'
            raise ValueError(msg.format(graph_data, allowed_exts))

        return config_data
