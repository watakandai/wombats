from gym_minigrid.minigrid import MiniGridEnv, Grid, Lava, Goal
from gym_minigrid.wrappers import ReseedWrapper, FullyObsWrapper
from gym_minigrid.minigrid import (IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX,
                                   OBJECT_TO_IDX)
from gym_minigrid.register import register

import matplotlib.pyplot as plt
import gym
import itertools
import numpy as np
import re
from collections import defaultdict
from bidict import bidict
from typing import Type, List, Tuple

# define these type defs for method annotation type hints
EnvObs = np.ndarray
Reward = float
Done = bool
StepData = Tuple[EnvObs, Reward, Done, dict]
AgentPos = Tuple[int, int]
AgentDir = int
EnvType = Type[MiniGridEnv]
Minigrid_TSNode = Tuple[AgentPos, AgentDir]
Minigrid_TSEdge = Tuple[Minigrid_TSNode, Minigrid_TSNode]


class StaticMinigridTSWrapper(gym.core.Wrapper):
    """
    Wrapper to define an environment that can be represented as a transition
    system.

    This means that the environment must be STATIC -> no keys or doors opening
    as this would require a reactive synthesis formulation.

    :param      env:    The gym environment to wrap and compute transitions on
    :param      seeds:  The random seeds given to the Minigrid environment, so
                        when the environment is reset(), it remains the same
    """

    env: EnvType
    IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))
    DIR_TO_STRING = bidict({0: 'right', 1: 'down', 2: 'left', 3: 'up'})

    def __init__(self, env: EnvType,
                 seeds: List[int] = [0]) -> 'StaticMinigridTSWrapper':

        # actually creating the minigrid environment with appropriate wrappers
        super().__init__(FullyObsWrapper(ReseedWrapper(env, seeds=seeds)))

        # building some more constant DICTS dynamically from the env data
        self.ACTION_STR_TO_ENUM = {self._get_action_str(action): action
                                   for action in self.env.Actions}
        self.ACTION_ENUM_TO_STR = dict(zip(self.ACTION_STR_TO_ENUM.values(),
                                           self.ACTION_STR_TO_ENUM.keys()))

        # We only compute state observations label maps once here, as the
        # environment MUST BE STATIC in this instance
        obs = self.state_only_obs_reset()
        self.agent_start_pos, self.agent_start_dir = self._get_agent_props()
        (self.obs_str_idxs_map,
         self.cell_obs_map,
         self.cell_to_obs) = self._get_observation_maps(self.agent_start_pos,
                                                        obs)

        # we want to statically compute the data that can be used to build
        # a transition system representation of the environment
        self.TS_config_data = self._get_transition_system_data(
            self.cell_obs_map,
            self.cell_to_obs,
            self.ACTION_ENUM_TO_STR)

    def render_notebook(self) -> None:
        """
        Wrapper for the env.render() that works in notebooks
        """

        plt.imshow(self.env.render(mode='rgb_image'), interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def state_only_obs(self, obs: dict) -> EnvObs:
        """
        Extracts only the grid observation from a step() observation

        This command only works for a MiniGridEnv obj, as their obs:
            obs, reward, done, _ = MiniGridEnbv.step()
        is a dict containing the (full/partially) observable grid observation

        :param      obs:  Full observation received from MiniGridEnbv.step()

        :returns:   The grid-only observation
        """

        cell_obs = obs['image']

        return cell_obs

    def state_only_obs_reset(self) -> EnvObs:
        """
        Resets the environment, but returns the grid-only observation

        :returns:   The grid-only observation after reseting
        """

        obs = self.env.reset()

        return self.state_only_obs(obs)

    def state_only_obs_step(self, action: MiniGridEnv.Actions) -> StepData:
        """
        step()s the environment, but returns only the grid observation

        This command only works for a MiniGridEnv obj, as their obs:
            obs, reward, done, _ = MiniGridEnbv.step()
        is a dict containing the (full/partially) observable grid observation

        :param      action:  The action to take

        :returns:   Normal step() return data, but with obs being only the grid
        """

        obs, reward, done, _ = self.env.step(action)

        return self.state_only_obs(obs), reward, done, {}

    def _get_agent_props(self) -> Tuple[AgentPos, AgentDir]:
        """
        Gets the agent's position and direction in the base environment
        """

        base_env = self.env.unwrapped

        return tuple(base_env.agent_pos), base_env.agent_dir

    def _set_agent_props(self,
                         position: {AgentPos, None}=None,
                         direction: {AgentDir, None}=None) -> None:
        """
        Sets the agent's position and direction in the base environment

        :param      position:   The new agent grid position
        :param      direction:  The new agent direction
        """

        base_env = self.env.unwrapped

        if position is not None:
            base_env.agent_pos = position

        if direction is not None:
            base_env.agent_dir = direction

    def _get_env_prop(self, env_property_name: str):
        """
        Gets the base environment's property.

        :param      env_property_name:  The base environment's property name

        :returns:   The base environment's property.
        """

        base_env = self.env.unwrapped

        return getattr(base_env, env_property_name)

    def _set_env_prop(self, env_property_name: str, env_property) -> None:
        """
        Sets the base environment's property.

        :param      env_property_name:  The base environment's property name
        :param      env_property:       The new base environment property data
        """

        base_env = self.env.unwrapped
        setattr(base_env, env_property_name, env_property)

    def _obs_to_prop_str(self, obs: EnvObs,
                         row_idx: int, col_idx: int) -> str:
        """
        Converts a grid observation array into a string based on Minigrid ENUMs

        :param      obs:      The grid observation
        :param      row_idx:  The row index of the cell to get the obs. string
        :param      col_idx:  The col index of the cell to get the obs. string

        :returns:   verbose, string representation of the state observation
        """

        obj_type, obj_color, obj_state = obs[col_idx, row_idx]
        agent_pos, _ = self._get_agent_props()
        is_agent = (col_idx, row_idx) == tuple(agent_pos)

        prop_string_base = '_'.join([IDX_TO_OBJECT[obj_type],
                                     IDX_TO_COLOR[obj_color]])

        if is_agent:
            return '_'.join([prop_string_base, self.DIR_TO_STRING[obj_state]])
        else:
            return '_'.join([prop_string_base, self.IDX_TO_STATE[obj_state]])

    def _get_action_str(self, action_enum: MiniGridEnv.Actions) -> str:
        """
        Gets a string representation of the action enum constant

        :param      action_enum:  The action enum constant to convert

        :returns:   The action enum's string representation
        """

        action_str = str(action_enum)

        # string version of Action.done is 'Action.done' -> trim
        substring_to_trim = 'Actions.'
        action_str = action_str[len(substring_to_trim):]

        return action_str

    def _make_transition(self, action: MiniGridEnv.Actions,
                         pos: AgentPos,
                         direction: AgentDir) -> Tuple[AgentPos,
                                                       AgentDir,
                                                       Done]:
        """
        Makes a state transition in the environment, assuming the env has state

        :param      action:     The action to take
        :param      pos:        The agent's position
        :param      direction:  The agent's direction

        :returns:   the agent's new state, whether or not step() emitted done
        """

        self._set_agent_props(pos, direction)
        _, _, done, _ = self.state_only_obs_step(action)

        return *self._get_agent_props(), done

    def _get_observation_maps(self, start_pos: AgentPos,
                              obs: EnvObs) -> Tuple[dict, defaultdict, dict]:
        """
        Computes mappings for grid state (cell) observations.

        A cell obs. array consists of [obj_type, obj_color, obj_state], where
        each element is an integer index in a ENUM from the Minigrid env.

            obs_str_idxs_map[cell_obs_str] = np.array(cell obs. array)
            cell_obs_map[cell_obs_str] = list_of((cell_col_idx, cell_row_idx))
            obs_str_idxs_map[(cell_col_idx, cell_row_idx)] = cell_obs_str

        :param      start_pos:  The agent's start position
        :param      obs:        The grid observation

        :returns:   (mapping from cell obs. string -> cell obs. array
                     mapping from cell obs. string -> cell indices
                        NOTE: each key in this dict has a list of values assoc.
                     mapping from cell indices -> cell obs. string)
        """

        obs_str_idxs_map = dict()
        cell_obs_map = defaultdict(list)
        cell_to_obs = dict()

        (num_cols, num_rows, num_cell_props) = obs.shape

        for row_idx in range(num_rows):
            for col_idx in range(num_cols):

                obs_str = self._obs_to_prop_str(obs, row_idx, col_idx)
                obj = obs[col_idx, row_idx][0]

                if IDX_TO_OBJECT[obj] != 'wall':
                    obs_str_idxs_map[obs_str] = tuple(obs[col_idx, row_idx])
                    cell_obs_map[obs_str].append((col_idx, row_idx))
                    cell_to_obs[(col_idx, row_idx)] = obs_str

        # need to remove the agent from the environment
        empty_cell_str = self._get_cell_str('empty', obs_str_idxs_map)
        agent_cell_str = self._get_cell_str('agent', obs_str_idxs_map)

        cell_to_obs[start_pos] = empty_cell_str
        obs_str_idxs_map.pop(agent_cell_str, None)
        cell_obs_map[empty_cell_str].append(start_pos)
        cell_obs_map.pop(agent_cell_str, None)

        return obs_str_idxs_map, cell_obs_map, cell_to_obs

    def _get_cell_str(self, obj_type_str: str, obs_str_idxs_map: dict,
                      only_one_type_of_obj: bool = True) -> {str, List[str]}:
        """
        Gets the observation string(s) associated with each type of object

        :param      obj_type_str:          The object type string
        :param      obs_str_idxs_map:      mapping from cell obs.
                                           string -> cell obs. array
        :param      only_one_type_of_obj:  Whether or not there should only be
                                           one distinct version of this object
                                           in the environment

        :returns:   The cell observation string(s) associated with the object

        :raises     AssertionError:        obj_type_str must be in the ENUM.
        :raises     ValueError:            if there is more than one of an
                                           object when there should only be one
                                           in the env.
        """

        assert obj_type_str in OBJECT_TO_IDX.keys()

        cell_str = [obs_str for obs_str in list(obs_str_idxs_map.keys())
                    if obj_type_str in obs_str]

        if only_one_type_of_obj and len(cell_str) != 1:
            msg = f'there should be exactly one observation string ' + \
                  f'for a {obj_type_str} object. Found {cell_str} in ' + \
                  f'cell oberservations.'
            raise ValueError(msg)
        else:
            cell_str = cell_str[0]

        return cell_str

    def _get_state_str(self, pos: AgentPos, direction: AgentDir) -> str:
        """
        Gets the string label for the automaton state given the agent's state.

        :param      pos:        The agent's position
        :param      direction:  The agent's direction

        :returns:   The state label string.
        """

        return ', '.join([str(pos), self.DIR_TO_STRING[direction]])

    def _get_state_from_str(self, state: str) -> Tuple[AgentPos, AgentDir]:

        m = re.match(r'\(([\d]), ([\d])\), ([a-z]*)', state)

        pos = (int(m.group(1)), int(m.group(2)))
        direction = self.DIR_TO_STRING.inv[m.group(3)]

        return pos, direction

    def _add_node(self, nodes: dict, pos: AgentPos,
                  direction: AgentPos, obs_str: str) -> Tuple[dict, str]:
        """
        Adds a node to the dict of nodes used to initialize an automaton obj.

        :param      nodes:      dict of nodes to build the automaton
                                out of. Must be in the format needed
                                by networkx.add_nodes_from()
        :param      pos:        The agent's position
        :param      direction:  The agent's direction
        :param      obs_str:    The state observation string

        :returns:   (updated dict of nodes, new label for the added node)
        """

        state = self._get_state_str(pos, direction)

        if state not in nodes:
            state_data = {'trans_distribution': None,
                          'observation': obs_str}
            nodes[state] = state_data

        return nodes, state

    def _add_edge(self, nodes: dict, edges: dict,
                  action: MiniGridEnv.Actions,
                  edge: Minigrid_TSEdge,
                  ACTION_ENUM_TO_STR: dict,
                  cell_to_obs: dict) -> Tuple[dict, dict]:
        """
        Adds both nodes to the dict of nodes and to the dict of edges used to
        initialize an automaton obj.

        :param      nodes:               dict of nodes to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_nodes_from()
        :param      edges:               dict of edges to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_edges_from()
        :param      action:              The action taken
        :param      edge:                The edge to add
        :param      ACTION_ENUM_TO_STR:  Mapping from action to it's str
                                         representation
        :param      cell_to_obs:         mapping from cell indices -> cell obs.
                                         string

        :returns:   (updated dict of nodes, updated dict of edges)
        """

        action_str = ACTION_ENUM_TO_STR[action]

        (src, dest,
         src_pos, src_dir,
         dest_pos, dest_dir,
         obs_str_src, obs_str_dest) = self._get_edge_components(edge,
                                                                cell_to_obs)

        nodes, state_src = self._add_node(nodes, src_pos, src_dir, obs_str_src)
        nodes, state_dest = self._add_node(nodes, dest_pos, dest_dir,
                                           obs_str_dest)

        edge_data = {'symbols': [action_str]}
        edge = {state_dest: edge_data}

        if state_src in edges:
            edges[state_src].update(edge)
        else:
            edges[state_src] = edge

        return nodes, edges

    def _get_edge_components(self, edge: Minigrid_TSEdge,
                             cell_to_obs: dict) -> Tuple[Minigrid_TSNode,
                                                         Minigrid_TSNode,
                                                         AgentPos, AgentDir,
                                                         AgentPos, AgentDir,
                                                         str, str]:
        """
        Parses the edge data structure and returns a tuple of unpacked data

        :param      edge:         The edge to unpack
        :param      cell_to_obs:  mapping from cell indices -> cell obs. string

        :returns:   All edge components. Not going to name them all bro-bro
        """

        edge = edge

        src, dest = edge
        src_pos, src_dir = src
        dest_pos, dest_dir = dest

        obs_str_src = cell_to_obs[src_pos]
        obs_str_dest = cell_to_obs[dest_pos]

        return (src, dest, src_pos, src_dir,
                dest_pos, dest_dir, obs_str_src, obs_str_dest)

    def _get_transition_system_data(self, cell_obs_map: defaultdict,
                                    cell_to_obs: dict,
                                    ACTION_ENUM_TO_STR: dict) -> Tuple[dict,
                                                                       dict]:
        """
        Extracts all data needed to build a transition system representation of
        the environment.

        :param      cell_obs_map:        mapping from cell obs. string -> cell
                                         indices NOTE: each key in this dict
                                         has a list of values assoc.
        :param      cell_to_obs:         mapping from cell indices -> cell obs.
                                         string
        :param      ACTION_ENUM_TO_STR:  Mapping from action to it's str
                                         representation

        :returns:   The transition system data.
        """

        possible_nodes = [(cell, direction) for cells in cell_obs_map.values()
                          for cell in cells
                          for direction in self.DIR_TO_STRING.keys()]

        all_possible_edges = itertools.product(possible_nodes, possible_nodes)

        self.reset()
        done_nodes = set()

        nodes = {}
        edges = {}

        for possible_edge in all_possible_edges:
            (src, dest,
             src_pos, src_dir,
             possible_dest_pos, possible_dest_dir,
             obs_str_src,
             obs_str_dest) = self._get_edge_components(possible_edge,
                                                       cell_to_obs)

            if src not in done_nodes:
                for action in self.env.actions:

                    (dest_pos,
                     dest_dir,
                     done) = self._make_transition(action, src_pos, src_dir)

                    dest_pos_is_valid = possible_dest_pos == dest_pos
                    dest_dir_is_valid = possible_dest_dir == dest_dir
                    egde_is_valid = (dest_pos_is_valid and dest_dir_is_valid)

                    if egde_is_valid:
                        nodes, edges = self._add_edge(nodes, edges,
                                                      action, possible_edge,
                                                      ACTION_ENUM_TO_STR,
                                                      cell_to_obs)
                        if done:
                            done_nodes.add(dest)

        # we have moved the agent a bunch, so we should reset it when done
        # extracting all of the data
        self.reset()

        return self._package_data(nodes, edges)

    def _package_data(self, nodes: dict, edges: dict) -> dict:
        """
        Packages up extracted data from the environment in the format needed by
        automaton constructors

        :param      nodes:               dict of nodes to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_nodes_from()
        :param      edges:               dict of edges to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_edges_from()

        :returns:   configuration data dictionary
        """

        config_data = {}

        # can directly compute these from the graph data
        symbols = set()
        state_labels = set()
        observations = set()
        for state, edge in edges.items():
            for _, edge_data in edge.items():
                symbols.update(edge_data['symbols'])
                state_labels.add(state)
                observation = nodes[state]['observation']
                observations.add(observation)

        alphabet_size = len(symbols)
        num_states = len(state_labels)
        num_obs = len(observations)

        config_data['alphabet_size'] = alphabet_size
        config_data['num_states'] = num_states
        config_data['num_obs'] = num_obs
        config_data['nodes'] = nodes
        config_data['edges'] = edges
        config_data['start_state'] = self._get_state_str(self.agent_start_pos,
                                                         self.agent_start_dir)

        return config_data


class LavaComparison(MiniGridEnv):
    """
    Environment to try comparing with MIT Shah paper
    """

    def __init__(
        self,
        width=10,
        height=10,
        agent_start_pos=(3, 5),
        agent_start_dir=0,
        drying_off_task=False
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = [(1, 1), (1, 8), (8, 1), (8, 8)]
        self.drying_off_task = drying_off_task

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        for goal_pos in self.goal_pos:
            self.put_obj(Goal(), *goal_pos)

        # left Lava block
        self.put_obj(Lava(), 1, 3)
        self.put_obj(Lava(), 1, 4)
        self.put_obj(Lava(), 2, 3)
        self.put_obj(Lava(), 2, 4)

        # right Lava block
        self.put_obj(Lava(), 7, 3)
        self.put_obj(Lava(), 7, 4)
        self.put_obj(Lava(), 8, 3)
        self.put_obj(Lava(), 8, 4)

        # bottom left Lava blocking goal
        self.put_obj(Lava(), 1, 7)
        self.put_obj(Lava(), 2, 7)
        self.put_obj(Lava(), 2, 8)

        # bottom right Lava blocking goal
        self.put_obj(Lava(), 8, 7)
        self.put_obj(Lava(), 7, 7)
        self.put_obj(Lava(), 7, 8)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to a green goal squares without touching lava"


class LavaComparison_noDryingOff(LavaComparison):
    def __init__(self):
        super().__init__(drying_off_task=False)


register(
    id='MiniGrid-LavaComparison_noDryingOff-v0',
    entry_point='wombats.systems.minigrid:LavaComparison_noDryingOff'
)
