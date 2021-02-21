from gym_minigrid.minigrid import (MiniGridEnv, Grid, Lava, Floor,
                                   Ball, Key, Door, Goal, Wall, Box)
from gym_minigrid.wrappers import (ReseedWrapper, FullyObsWrapper,
                                   ViewSizeWrapper)
from gym_minigrid.minigrid import (IDX_TO_COLOR, STATE_TO_IDX,
                                   TILE_PIXELS, COLORS, COLOR_TO_IDX)
from gym_minigrid.register import register
from gym_minigrid.rendering import (point_in_rect, point_in_circle,
                                    fill_coords, highlight_img, downsample)

import matplotlib.pyplot as plt
import gym
import numpy as np
import re
import queue
import warnings
from collections import defaultdict
from bidict import bidict
from typing import Type, List, Tuple
from gym import wrappers
from gym.wrappers.monitor import disable_videos
from enum import IntEnum

# define these type defs for method annotation type hints
EnvObs = np.ndarray
CellObs = Tuple[int, int, int]
ActionsEnum = MiniGridEnv.Actions
Reward = float
Done = bool
StepData = Tuple[EnvObs, Reward, Done, dict]
AgentPos = Tuple[int, int]
AgentDir = int
EnvType = Type[MiniGridEnv]
Minigrid_TSNode = Tuple[AgentPos, AgentDir]
Minigrid_TSEdge = Tuple[Minigrid_TSNode, Minigrid_TSNode]
Minigrid_Edge_Unpacked = Tuple[Minigrid_TSNode, Minigrid_TSNode, AgentPos,
                               AgentDir, AgentPos, AgentDir, str, str]

MINIGRID_TO_GRAPHVIZ_COLOR = {'red': 'firebrick',
                              'green': 'darkseagreen1',
                              'blue': 'steelblue1',
                              'purple': 'mediumpurple1',
                              'yellow': 'yellow',
                              'grey': 'gray60'}

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen': 0,
    'empty': 1,
    'wall': 2,
    'floor': 3,
    'door': 4,
    'key': 5,
    'ball': 6,
    'box': 7,
    'goal': 8,
    'lava': 9,
    'agent': 10,
    'carpet': 11,
    'water': 12,
}
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

GYM_MONITOR_LOG_DIR_NAME = 'minigrid_env_logs'


class ModifyActionsWrapper(gym.core.Wrapper):
    """
    This class allows you to modify the action space and behavior of the agent

    :param      env:           The gym environment to wrap
    :param      actions_type:  The actions type string
                               {'static', 'simple_static', 'diag_static',
                                'default'}
                               'static':
                               use a directional agent only capable of going
                               forward and turning
                               'simple_static':
                               use a non-directional agent which can only move
                               in cardinal directions in the grid
                               'default':
                               use an agent which has the default MinigridEnv
                               actions, suitable for dynamic environments.
    """

    # Enumeration of possible actions
    # as this is a static environment, we will only allow for movement actions
    # For a simple environment, we only allow the agent to move:
    # North, South, East, or West
    class SimpleStaticActions(IntEnum):
        # move in this direction on the grid
        north = 0
        south = 1
        east = 2
        west = 3

    SIMPLE_ACTION_TO_DIR_IDX = {SimpleStaticActions.north: 3,
                                SimpleStaticActions.south: 1,
                                SimpleStaticActions.east: 0,
                                SimpleStaticActions.west: 2}

    # Enumeration of possible actions
    # as this is a static environment, we will only allow for movement actions
    # For a simple environment, we only allow the agent to move:
    # Northeast, Northwest, Southeast, or Southwest
    class DiagStaticActions(IntEnum):
        # move in this direction on the grid
        northeast = 0
        northwest = 1
        southeast = 2
        southwest = 3

    DIAG_ACTION_TO_POS_DELTA = {
        DiagStaticActions.northeast: (1, -1),
        DiagStaticActions.northwest: (-1, -1),
        DiagStaticActions.southeast: (1, 1),
        DiagStaticActions.southwest: (-1, 1)}

    # Enumeration of possible actions
    # as this is a static environment, we will only allow for movement actions
    class StaticActions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(self, env: EnvType, actions_type: str = 'static'):

        # actually creating the minigrid environment with appropriate wrappers
        super().__init__(env)

        self._allowed_actions_types = set(['static', 'simple_static',
                                           'diag_static', 'default'])
        if actions_type not in self._allowed_actions_types:
            msg = f'actions_type ({actions_type}) must be one of: ' + \
                  f'{actions_type}'
            raise ValueError(msg)

        # Need to change the Action enumeration in the base environment.
        # This also changes the "step" behavior, so we also change that out
        # to match the new set of actions
        self._actions_type = actions_type

        if actions_type == 'static':
            actions = ModifyActionsWrapper.StaticActions
            step_function = self._step_default
        elif actions_type == 'simple_static':
            actions = ModifyActionsWrapper.SimpleStaticActions
            step_function = self._step_simple_static
        elif actions_type == 'diag_static':
            actions = ModifyActionsWrapper.DiagStaticActions
            step_function = self._step_diag_static
        elif actions_type == 'default':
            actions = MiniGridEnv.Actions
            step_function = self._step_default

        self.unwrapped.actions = actions
        self._step_function = step_function

        # Actions are discrete integer values
        num_actions = len(self.unwrapped.actions)
        self.unwrapped.action_space = gym.spaces.Discrete(num_actions)

        # building some more constant DICTS dynamically from the env data
        self.ACTION_STR_TO_ENUM = {self._get_action_str(action): action
                                   for action in self.unwrapped.actions}
        self.ACTION_ENUM_TO_STR = dict(zip(self.ACTION_STR_TO_ENUM.values(),
                                           self.ACTION_STR_TO_ENUM.keys()))

    def step(self, action: IntEnum) -> StepData:

        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        base_env.step_count += 1

        done, reward = self._step_function(action)

        if base_env.step_count >= base_env.max_steps:
            done = True

        obs = base_env.gen_obs()

        return obs, reward, done, {}

    def _step_diag_static(self, action: IntEnum) -> Tuple[Done, Reward]:

        reward = 0
        done = False

        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        start_pos = base_env.agent_pos

        # a diagonal action is really just two simple actions :)
        pos_delta = ModifyActionsWrapper.DIAG_ACTION_TO_POS_DELTA[action]

        # Get the contents of the new cell of the agent
        new_pos = tuple(np.add(start_pos, pos_delta))
        new_cell = base_env.grid.get(*new_pos)

        if new_cell is None or new_cell.can_overlap():
            base_env.agent_pos = new_pos
        if new_cell is not None and new_cell.type == 'goal':
            done = True
            reward = base_env._reward()
        if new_cell is not None and new_cell.type == 'lava':
            done = True

        return done, reward

    def _step_simple_static(self, action: IntEnum) -> Tuple[Done, Reward]:

        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        reward = 0
        done = False

        # save the original direction so we can reset it after moving
        old_dir = base_env.agent_dir
        new_dir = ModifyActionsWrapper.SIMPLE_ACTION_TO_DIR_IDX[action]
        base_env.agent_dir = new_dir

        # Get the contents of the cell in front of the agent
        fwd_pos = base_env.front_pos
        fwd_cell = base_env.grid.get(*fwd_pos)

        if fwd_cell is None or fwd_cell.can_overlap():
            base_env.agent_pos = fwd_pos
        if fwd_cell is not None and fwd_cell.type == 'goal':
            done = True
            reward = base_env._reward()
        if fwd_cell is not None and fwd_cell.type == 'lava':
            done = True

        # reset the direction of the agent, as it really cannot change
        # direction
        base_env.agent_dir = old_dir

        return done, reward

    def _step_default(self, action: IntEnum) -> Tuple[Done, Reward]:

        reward = 0
        done = False

        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        # Get the position in front of the agent
        fwd_pos = base_env.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = base_env.grid.get(*fwd_pos)
        # Rotate left
        if action == base_env.actions.left:
            base_env.agent_dir -= 1
            if base_env.agent_dir < 0:
                base_env.agent_dir += 4

        # Rotate right
        elif action == base_env.actions.right:
            base_env.agent_dir = (base_env.agent_dir + 1) % 4

        # Move forward
        elif action == base_env.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                base_env.agent_pos = fwd_pos
            if fwd_cell is not None and fwd_cell.type == 'goal':
                done = True
                reward = base_env._reward()
            if fwd_cell is not None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == base_env.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if base_env.carrying is None:
                    base_env.carrying = fwd_cell
                    base_env.carrying.cur_pos = np.array([-1, -1])
                    base_env.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == base_env.actions.drop:
            if not fwd_cell and base_env.carrying:
                base_env.grid.set(*fwd_pos, base_env.carrying)
                base_env.carrying.cur_pos = fwd_pos
                base_env.carrying = None

        # Toggle/activate an object
        elif action == base_env.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(base_env, fwd_pos)

        # Done action (not used by default)
        elif action == base_env.actions.done:
            pass

        else:
            assert False, "unknown action"

        return done, reward

    def _get_action_str(self, action_enum: ActionsEnum) -> str:
        """
        Gets a string representation of the action enum constant

        :param      action_enum:  The action enum constant to convert

        :returns:   The action enum's string representation
        """

        return self.unwrapped.actions._member_names_[action_enum]


class StaticMinigridTSWrapper(gym.core.Wrapper):
    """
    Wrapper to define an environment that can be represented as a transition
    system.

    This means that the environment must be STATIC -> no keys or doors opening
    as this would require a reactive synthesis formulation.

        :param      env:                   The gym environment to wrap and
                                           compute transitions on
        :param      seeds:                 The random seeds given to the
                                           Minigrid environment, so when the
                                           environment is reset(), it remains
                                           the same.
        :param      actions_type:          The actions type string
                                           {'static', 'simple_static',
                                           'diag_static', 'default'}
                                           'static': use a directional agent
                                           only capable of going forward and
                                           turning
                                           'simple_static': use a
                                           non-directional agent which can only
                                           move in cardinal directions in the
                                           grid
                                           'diag_static': use a
                                           non-directional agent which can only
                                           move in intercardinal directions
                                           (diagonally) in the grid
                                           'default': use an agent which
                                           has the default MinigridEnv actions,
                                           suitable for dynamic environments.
        :param      monitor_log_location:  The location to save gym env
                                           monitor logs & videos
    """

    env: EnvType
    IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))
    DIR_TO_STRING = bidict({0: 'right', 1: 'down', 2: 'left', 3: 'up'})

    def __init__(
        self,
        env: EnvType,
        seeds: List[int] = [0],
        actions_type: str = 'static',
        monitor_log_location: str = GYM_MONITOR_LOG_DIR_NAME
    ) -> 'StaticMinigridTSWrapper':

        self.monitor_log_location = monitor_log_location
        self._force_monitor = False
        self._resume_monitor = True
        self._uid_monitor = None
        self._mode = None

        self._allowed_actions_types = set(['static', 'simple_static',
                                           'diag_static', 'default'])
        if actions_type not in self._allowed_actions_types:
            msg = f'actions_type ({actions_type}) must be one of: ' + \
                  f'{self._allowed_actions_types}'
            raise ValueError(msg)

        if actions_type == 'simple_static' or actions_type == 'diag_static':
            env.directionless_agent = True
        elif actions_type == 'static' or actions_type == 'default':
            env.directionless_agent = False

        env = ViewSizeWrapper(env, agent_view_size=3)
        env = ModifyActionsWrapper(env, actions_type)
        env = FullyObsWrapper(ReseedWrapper(env, seeds=seeds))
        env = wrappers.Monitor(env, self.monitor_log_location,
                               video_callable=False,
                               force=self._force_monitor,
                               resume=self._resume_monitor,
                               mode=self._mode)

        # actually creating the minigrid environment with appropriate wrappers
        super().__init__(env)
        self.actions = self.unwrapped.actions

        # We only compute state observations label maps once here, as the
        # environment MUST BE STATIC in this instance
        obs = self.state_only_obs_reset()
        self.agent_start_pos, self.agent_start_dir = self._get_agent_props()

        (self.obs_str_idxs_map,
         self.cell_obs_map,
         self.cell_to_obs) = self._get_observation_maps(self.agent_start_pos,
                                                        obs)

        self.reset()

    def render_notebook(self) -> None:
        """
        Wrapper for the env.render() that works in notebooks
        """

        plt.imshow(self.env.render(mode='rgb_image', tile_size=64),
                   interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def reset(self, new_monitor_file: bool = False, **kwargs) -> np.ndarray:
        """
        Wrapper for the reset function that manages the monitor wrapper

        :param      new_monitor_file:  whether to create a new monitor file
        :param      kwargs:            The keywords arguments to pass on to the
                                       next wrapper's reset()

        :returns:   env observation
        """

        self.close()
        self._start_monitor(new_monitor_file)
        observation = self.env.reset(**kwargs)

        return observation

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

    def state_only_obs_step(self, action: ActionsEnum) -> StepData:
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
                         col_idx: int, row_idx: int) -> str:
        """
        Converts a grid observation array into a string based on Minigrid ENUMs

        :param      obs:      The grid observation
        :param      col_idx:  The col index of the cell to get the obs. string
        :param      row_idx:  The row index of the cell to get the obs. string

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

    def _make_transition(self, action: ActionsEnum,
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

        return (*self._get_agent_props(), done)

    def _get_obs_str_of_start_cell(self, start_obs: EnvObs) -> str:
        """
        Gets the cell observation string for the start (reset) state

        if returns None, then the agent cannot leave the start state and the
        environment is broken lol.

        :param      start_obs:   The gridcell observation matrix at reset()

        :returns:   a full-cell observation at the agent's reset state

        :raises     ValueError:  If the agent can't do anything at reset()
        """

        self.reset()
        init_state = (self.agent_start_pos, self.agent_start_dir)

        # sigh.... Well, we know that if you can't move within 3 steps, then
        # the environment is completely unsolvable, or you start on the goal
        # state.
        for a1 in self.actions:
            (agent_pos1,
             agent_dir1,
             done) = self._make_transition(a1, *init_state)
            s1 = (agent_pos1, agent_dir1)

            if done:
                self.reset()

            for a2 in self.actions:
                (agent_pos2,
                 agent_dir2,
                 done) = self._make_transition(a2, *s1)
                s2 = (agent_pos2, agent_dir2)

                if done:
                    self.reset()

                for a3 in self.actions:
                    self._set_agent_props(*s2)
                    obs, _, done, _ = self.state_only_obs_step(a3)
                    agent_pos3, _ = self._get_agent_props()
                    at_new_cell = agent_pos3 != init_state[0]

                    if at_new_cell:
                        obs_str = self._obs_to_prop_str(obs, *init_state[0])
                        self.reset()
                        return obs_str

                    if done:
                        self.reset()

        msg = f'No actions allow the agent to make any progress in the env.'
        raise ValueError(msg)

    def _get_observation_maps(self, start_pos: AgentPos,
                              obs: EnvObs) -> Tuple[bidict, defaultdict, dict]:
        """
        Computes mappings for grid state (cell) observations.

        A cell obs. array consists of [obj_type, obj_color, obj_state], where
        each element is an integer index in a ENUM from the Minigrid env.

            obs_str_idxs_map[cell_obs_str] = np.array(cell obs. array)
            cell_obs_map[cell_obs_str] = list_of((cell_col_idx, cell_row_idx))
            cell_to_obs[(cell_col_idx, cell_row_idx)] = cell_obs_str

        :param      start_pos:  The agent's start position
        :param      obs:        The grid observation

        :returns:   (mapping from cell obs. string -> cell obs. array
                     mapping from cell obs. string -> cell indices
                        NOTE: each key in this dict has a list of values assoc.
                     mapping from cell indices -> cell obs. string)
        """

        obs_str_idxs_map = bidict()
        cell_obs_map = defaultdict(list)
        cell_to_obs = dict()

        (num_cols, num_rows, num_cell_props) = obs.shape

        for row_idx in range(num_rows):
            for col_idx in range(num_cols):

                obs_str = self._obs_to_prop_str(obs, col_idx, row_idx)
                obj = obs[col_idx, row_idx][0]

                is_agent = IDX_TO_OBJECT[obj] == 'agent'
                is_wall = IDX_TO_OBJECT[obj] == 'wall'

                if not is_agent and not is_wall:
                    obs_str_idxs_map[obs_str] = tuple(obs[col_idx, row_idx])
                    cell_obs_map[obs_str].append((col_idx, row_idx))
                    cell_to_obs[(col_idx, row_idx)] = obs_str

        # need to add the agent's start cell observation to the environment
        start_cell_obs_str = self._get_obs_str_of_start_cell(obs)
        start_col, start_row = start_pos

        obs_str_idxs_map[start_cell_obs_str] = tuple(obs[start_col, start_row])
        cell_obs_map[start_cell_obs_str].append((start_col, start_row))
        cell_to_obs[(start_col, start_row)] = start_cell_obs_str

        return obs_str_idxs_map, cell_obs_map, cell_to_obs

    def _get_cell_str(self, obj_type_str: str, obs_str_idxs_map: bidict,
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

        if only_one_type_of_obj and len(cell_str) > 1:
            msg = f'there should be exactly one observation string ' + \
                  f'for a {obj_type_str} object. Found {cell_str} in ' + \
                  f'cell observations.'
            raise ValueError(msg)
        elif only_one_type_of_obj and len(cell_str) == 0:
            msg = f'could not find any {obj_type_str} objects.'
            warnings.warn(msg, RuntimeWarning)
            return None
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
        """
        Gets the agent's state components from the state string representation

        :param      state:  The state label string

        :returns:   the agent's grid cell position, the agent's direction index
        """

        m = re.match(r'\(([\d]*), ([\d]*)\), ([a-z]*)', state)

        pos = (int(m.group(1)), int(m.group(2)))
        direction = self.DIR_TO_STRING.inv[m.group(3)]

        return pos, direction

    def _get_state_obs_from_state_str(self, state: str) -> CellObs:
        """
        Return the cell observation at the given cell from the cell's
        state string

        :param      state:  The state string to get the obs. array from

        :returns:   The state obs from state string.
        """

        agent_pos, _ = self._get_state_from_str(state)
        cell_obs_str = self.cell_to_obs[agent_pos]
        cell_obs_arr = self.obs_str_idxs_map[cell_obs_str]

        return cell_obs_arr

    def _get_state_obs_color(self, state: str) -> str:

        cell_obs_arr = self._get_state_obs_from_state_str(state)

        return IDX_TO_COLOR[cell_obs_arr[1]]

    def _add_node(self, nodes: dict, pos: AgentPos,
                  direction: AgentPos, obs_str: str) -> Tuple[dict, str]:
        """
        Adds a node to the dict of nodes used to initialize an automaton obj.

        :param      nodes:             dict of nodes to build the automaton out
                                       of. Must be in the format needed by
                                       networkx.add_nodes_from()
        :param      pos:               The agent's position
        :param      direction:         The agent's direction
        :param      obs_str:           The state observation string

        :returns:   (updated dict of nodes, new label for the added node)
        """

        state = self._get_state_str(pos, direction)
        color = self._get_state_obs_color(state)
        empty_cell_str = self._get_cell_str('empty', self.obs_str_idxs_map)

        if obs_str == empty_cell_str:
            color = 'gray'
        else:
            color = MINIGRID_TO_GRAPHVIZ_COLOR[color]

        goal_cell_str = self._get_cell_str('goal', self.obs_str_idxs_map)
        if goal_cell_str is not None:
            is_goal = obs_str == goal_cell_str
        else:
            is_goal = False

        if state not in nodes:
            state_data = {'trans_distribution': None,
                          'observation': obs_str,
                          'is_accepting': is_goal,
                          'color': color}
            nodes[state] = state_data

        return nodes, state

    def _add_edge(self, nodes: dict, edges: dict,
                  action: ActionsEnum,
                  edge: Minigrid_TSEdge) -> Tuple[dict, dict, str, str]:
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

        :returns:   (updated dict of nodes, updated dict of edges)
        """

        action_str = self.ACTION_ENUM_TO_STR[action]

        (src, dest,
         src_pos, src_dir,
         dest_pos, dest_dir,
         obs_str_src, obs_str_dest) = self._get_edge_components(edge)

        nodes, state_src = self._add_node(nodes, src_pos, src_dir, obs_str_src)
        nodes, state_dest = self._add_node(nodes, dest_pos, dest_dir,
                                           obs_str_dest)

        edge_data = {'symbols': [action_str]}
        edge = {state_dest: edge_data}

        if state_src in edges:
            if state_dest in edges[state_src]:
                existing_edge_data = edges[state_src][state_dest]
                existing_edge_data['symbols'].extend(edge_data['symbols'])
                edges[state_src][state_dest] = existing_edge_data
            else:
                edges[state_src].update(edge)
        else:
            edges[state_src] = edge

        return nodes, edges, state_src, state_dest

    def _get_edge_components(self,
                             edge: Minigrid_TSEdge) -> Minigrid_Edge_Unpacked:
        """
        Parses the edge data structure and returns a tuple of unpacked data

        :param      edge:         The edge to unpack

        :returns:   All edge components. Not going to name them all bro-bro
        """

        edge = edge

        src, dest = edge
        src_pos, src_dir = src
        dest_pos, dest_dir = dest

        obs_str_src = self.cell_to_obs[src_pos]
        obs_str_dest = self.cell_to_obs[dest_pos]

        return (src, dest, src_pos, src_dir,
                dest_pos, dest_dir, obs_str_src, obs_str_dest)

    def extract_transition_system(self) -> Tuple[dict, dict]:
        """
        Extracts all data needed to build a transition system representation of
        the environment.

        :returns:   The transition system data.
        """

        self.reset()

        nodes = {}
        edges = {}

        init_state_label = self._get_state_str(self.agent_start_pos,
                                               self.agent_start_dir)

        search_queue = queue.Queue()
        search_queue.put(init_state_label)
        visited = set()
        done_states = set()

        while not search_queue.empty():

            curr_state_label = search_queue.get()
            visited.add(curr_state_label)
            src_pos, src_dir = self._get_state_from_str(curr_state_label)

            for action in self.actions:
                if curr_state_label not in done_states:

                    (dest_pos,
                     dest_dir,
                     done) = self._make_transition(action, src_pos, src_dir)

                    possible_edge = ((src_pos, src_dir), (dest_pos, dest_dir))

                    (nodes, edges,
                     _,
                     dest_state_label) = self._add_edge(nodes, edges,
                                                        action, possible_edge)

                    # don't want to add outgoing transitions from states that
                    # we know are done to the TS, as these are wasted space
                    if done:
                        done_states.add(dest_state_label)

                        # need to reset after done, to clear the 'done' state
                        self.reset()

                    if dest_state_label not in visited:
                        search_queue.put(dest_state_label)
                        visited.add(dest_state_label)

        # we have moved the agent a bunch, so we should reset it when done
        # extracting all of the data.
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

        for node in nodes.keys():
            observation = nodes[node]['observation']
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

    def _toggle_video_recording(self, record_video: {bool, None}=None) -> None:
        """
        Turns on / off the video monitoring for the underlying Minigrid env

        :param      record_video:  setting for environment monitoring.
                                   If not given, will toggle the current video
                                   recording state
        """

        if record_video is None:
            turn_off_video = self.env._video_enabled()
        else:
            turn_off_video = not record_video

        if turn_off_video:
            self.env.video_callable = disable_videos
        else:
            self.env.video_callable = lambda episode_id: True

    def _start_monitor(self, new_monitor_file: bool) -> None:
        """
        (Re)-Starts a the env's monitor wrapper

        :param      new_monitor_file:  whether to create a new Monitor file

        :returns:   basically re-runs the Monitor's __init__ function
        """

        env = self.env

        if new_monitor_file:
            env.videos = []
            env.stats_recorder = None
            env.video_recorder = None
            env.enabled = False
            env.episode_id = 0
            env._monitor_id = None

            self.env._start(self.env.directory,
                            self.env.video_callable,
                            self._force_monitor,
                            self._resume_monitor,
                            self.env.write_upon_reset,
                            self._uid_monitor,
                            self._mode)
        else:

            self.env._start(self.env.directory,
                            self.env.video_callable,
                            self._force_monitor,
                            self._resume_monitor,
                            self.env.write_upon_reset,
                            self._uid_monitor,
                            self._mode)

    def _get_video_path(self) -> str:
        """
        Gets the current video recording's full path.

        :returns:   The video path.
        """

        return self.env.video_recorder.path


class NoDirectionAgentGrid(Grid):
    """
    This class overrides the drawing of direction-less agents
    """

    tile_cache = {}

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

    def render(
        self,
        tile_size,
        agent_pos=None,
        agent_dir=None,
        highlight_mask=None
    ):
        """
        Render this grid at a given scale

        NOTE: overridden here to change the tile rendering to be the class' own

        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height),
                                      dtype=np.bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))

                # CHANGED: Grid.render_tile(...) to self.render_tile(...)
                tile_img = self.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    @classmethod
    def render_tile(
        cls,
        obj,
        agent_dir=None,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3,
        white_background=False
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        if white_background:
            img = np.full(shape=(tile_size * subdivs, tile_size * subdivs, 3),
                          fill_value=255,
                          dtype=np.uint8)
        else:
            img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3),
                           dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            cir_fn = point_in_circle(cx=0.5, cy=0.5, r=0.3)
            fill_coords(img, cir_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img


class LavaComparison(MiniGridEnv):
    """
    Environment to try comparing with Sheshia paper
    """

    def __init__(
        self,
        width=10,
        height=10,
        agent_start_pos=(3, 5),
        agent_start_dir=0,
        drying_off_task=False,
        path_only_through_water=False,
        second_goal_task=False,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = [(1, 1), (1, 8), (8, 8)]
        self.drying_off_task = drying_off_task
        self.directionless_agent = False
        self.path_only_through_water = path_only_through_water
        self.second_goal_task = second_goal_task

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        if self.directionless_agent:
            self.grid = NoDirectionAgentGrid(width, height)
        else:
            self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        for goal_pos in self.goal_pos:
            self.put_obj(Floor(color='green'), *goal_pos)

        if self.second_goal_task:
            self.put_obj(Floor(color='purple'), 8, 1)
        else:
            if self.drying_off_task:
                self.put_obj(Floor(color='green'), 8, 1)
            else:
                self.put_obj(Water(), 8, 1)

        # top left Lava block
        self.put_obj(Lava(), 1, 3)
        self.put_obj(Lava(), 1, 4)
        self.put_obj(Lava(), 2, 3)
        self.put_obj(Lava(), 2, 4)

        # top right Lava block
        self.put_obj(Lava(), 7, 3)
        self.put_obj(Lava(), 7, 4)
        self.put_obj(Lava(), 8, 3)
        self.put_obj(Lava(), 8, 4)

        # bottom left Lava blocking goal
        self.put_obj(Lava(), 1, 7)
        self.put_obj(Lava(), 2, 7)
        self.put_obj(Lava(), 2, 8)

        # place the water
        if self.drying_off_task:
            if self.path_only_through_water:
                # new top left
                self.put_obj(Lava(), 3, 3)
                self.put_obj(Lava(), 1, 2)
                self.put_obj(Lava(), 2, 2)
                self.put_obj(Lava(), 2, 1)

                # new top right
                self.put_obj(Lava(), 6, 3)
                self.put_obj(Lava(), 7, 2)
                self.put_obj(Lava(), 8, 2)
                self.put_obj(Lava(), 7, 1)

            self.put_obj(Water(), 4, 6)
            self.put_obj(Water(), 4, 5)
            self.put_obj(Water(), 4, 4)
            self.put_obj(Water(), 4, 3)
            self.put_obj(Water(), 5, 6)
            self.put_obj(Water(), 5, 5)
            self.put_obj(Water(), 5, 4)
            self.put_obj(Water(), 5, 3)

            # bottom carpet
            self.put_obj(Carpet(), 3, 1)
            self.put_obj(Carpet(), 4, 1)
            self.put_obj(Carpet(), 5, 1)
            self.put_obj(Carpet(), 6, 1)

            # top carpet
            self.put_obj(Carpet(), 3, 8)
            self.put_obj(Carpet(), 4, 8)
            self.put_obj(Carpet(), 5, 8)
            self.put_obj(Carpet(), 6, 8)

        if self.path_only_through_water:
            # opened up bottom right Lava blocking goal
            self.put_obj(Lava(), 6, 7)
        else:
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

        self.mission = 'get to a green goal squares, don"t touch lava, ' + \
                       'must dry off if you get wet'


class AlternateLavaComparison(MiniGridEnv):
    """
    Very different Environment to the Seshia Paper to show environmental indep.
    """

    def __init__(
        self,
        narrow=False,
        path_only_through_water=False
    ):

        self.width = 20

        if narrow:
            self.corridor_size = 1
            self.height = 9
        else:
            self.corridor_size = 2
            self.height = 13

        self.agent_start_pos = (2, self.height - 2)
        self.agent_start_dir = 0
        self.num_empty_left_side_cells = 2 * self.corridor_size
        self.path_only_through_water = path_only_through_water

        self.directionless_agent = False

        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=4 * self.width * self.height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        if self.directionless_agent:
            self.grid = NoDirectionAgentGrid(width, height)
        else:
            self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        corridor_size = self.corridor_size
        wall = 1
        first_empty_col = wall
        first_empty_row = wall
        last_empty_col = self.width - (wall * 2)
        last_empty_row = self.height - (wall * 2)
        num_empty_left_side_cells = self.num_empty_left_side_cells

        # place the water blocks by flooding the whole area
        water_start_row = 2 * wall + corridor_size
        water_end_row = last_empty_row
        water_col_start = first_empty_col + num_empty_left_side_cells + wall
        corridor_base_len = self.width - water_col_start - wall - corridor_size

        for row in range(water_start_row, water_end_row + 1):
            if row < water_end_row - 1:
                self.grid.horz_wall(water_col_start, row,
                                    length=corridor_base_len - 1,
                                    obj_type=Water)
            else:
                self.grid.horz_wall(water_col_start, row,
                                    length=corridor_base_len - 1 - 1,
                                    obj_type=Water)

        # generate the horiz. corridor walls
        water_corridor_bottom_row = (2 * corridor_size) + wall

        middle_wall_length = corridor_base_len - corridor_size - wall
        bottom_wall_length = corridor_base_len - corridor_size - wall

        top_wall_row = water_corridor_bottom_row - corridor_size
        middle_wall_row = top_wall_row + corridor_size + wall
        bottom_wall_row = middle_wall_row + corridor_size + wall

        self.grid.horz_wall(water_col_start, top_wall_row,
                            length=corridor_base_len - 1)
        self.grid.horz_wall(water_col_start + wall, middle_wall_row,
                            length=middle_wall_length - 1)
        self.grid.horz_wall(water_col_start + corridor_size + wall,
                            bottom_wall_row,
                            length=bottom_wall_length - 1)

        # generate the vert. corridor walls
        left_vert_wall_col = water_col_start
        right_vert_wall_col = last_empty_col - corridor_size - wall
        right_vert_wall_length = bottom_wall_row - top_wall_row + 2 * wall

        self.grid.vert_wall(left_vert_wall_col, middle_wall_row)
        self.grid.vert_wall(right_vert_wall_col + wall, top_wall_row,
                            length=right_vert_wall_length - 1)

        # place the carpet square
        # carpet_col = water_col_start + corridor_base_len - 1
        # carpet_row = water_end_row
        carpet_col = water_col_start + corridor_base_len + wall
        carpet_row = middle_wall_row
        self.put_obj(Carpet(), carpet_col, carpet_row)

        # place a recharge square in the bottom-right corner
        # goal_col, goal_row = (carpet_col + 2), carpet_row
        goal_col, goal_row = (carpet_col), water_end_row
        self.put_obj(Floor(color='green'), goal_col, goal_row)

        # lava blocks
        lava_start_col = first_empty_col + corridor_size
        lava_end_col = left_vert_wall_col - corridor_size
        lava_length = right_vert_wall_length - wall
        lava_start_row = top_wall_row

        for col in range(lava_start_col, lava_end_col):
            self.grid.vert_wall(col, lava_start_row, length=lava_length - 1,
                                obj_type=Lava)

        # blocking the route around the maze if we want to force the agent
        # through water
        if self.path_only_through_water:
            new_lava_start_col = lava_start_col
            new_lava_end_col = left_vert_wall_col
            new_lava_start_row = first_empty_row
            new_lava_end_row = corridor_size
            new_lava_length = new_lava_end_col - new_lava_start_row + 1

            for row in range(new_lava_start_row, new_lava_end_row + 1):
                self.grid.horz_wall(new_lava_start_col, row,
                                    length=new_lava_length - 1, obj_type=Lava)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = 'get to a green goal squares, don"t touch lava, ' + \
                       'must dry off if you get wet'


class MyDistShift(MiniGridEnv):
    """
    Customized distributional shift environment.
    """

    def __init__(
        self,
        width=6,
        height=5,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        strip2_row=3
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_1_pos = (width - 2, 1)
        self.goal_2_pos = (width - 2, height - 2)
        self.strip2_row = strip2_row

        self.directionless_agent = False

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        # create an empty grid with different types of agents
        if self.directionless_agent:
            self.grid = NoDirectionAgentGrid(width, height)
        else:
            self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the two goal squares in the bottom-right corner
        self.put_obj(Floor(color='green'), *self.goal_1_pos)
        self.put_obj(Floor(color='purple'), *self.goal_2_pos)

        # Place the lava rows
        for i in range(self.width - 4):
            self.grid.set(2 + i, 1, Lava())
            self.grid.set(2 + i, self.strip2_row, Lava())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to both the green and purple squares"


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'floor':
            v = Floor(color)
        elif obj_type == 'ball':
            v = Ball(color)
        elif obj_type == 'key':
            v = Key(color)
        elif obj_type == 'box':
            v = Box(color)
        elif obj_type == 'door':
            v = Door(color, is_open, is_locked)
        elif obj_type == 'goal':
            v = Goal()
        elif obj_type == 'lava':
            v = Lava()
        elif obj_type == 'carpet':
            v = Carpet()
        elif obj_type == 'water':
            v = Water()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Carpet(WorldObj):
    """
    Yellow carpet (floor) tile the agent can walk over
    """

    def __init__(self):
        super().__init__('carpet', color='yellow')

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Water(WorldObj):
    """
    A floor tile with water on it that the agent can walk over
    """

    def __init__(self):
        super().__init__('water', color='blue')

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 3
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class LavaComparison_noDryingOff(LavaComparison):
    def __init__(self):
        super().__init__(drying_off_task=False)


class LavaComparison_seshia(LavaComparison):
    def __init__(self):
        super().__init__(drying_off_task=True)


class LavaComparison_SeshiaSecondGoal(LavaComparison):
    def __init__(self):
        super().__init__(drying_off_task=True,
                         second_goal_task=True)


class LavaComparison_SeshiaOnlyWaterPath(LavaComparison):
    def __init__(self):
        super().__init__(drying_off_task=True, path_only_through_water=True)


class AlternateLavaComparison_AllCorridorsOpen_Wide(AlternateLavaComparison):
    def __init__(self):
        super().__init__(narrow=False, path_only_through_water=False)


class AlternateLavaComparison_OnlyWaterPath_Wide(AlternateLavaComparison):
    def __init__(self):
        super().__init__(narrow=False, path_only_through_water=True)


class AlternateLavaComparison_AllCorridorsOpen_Narrow(AlternateLavaComparison):
    def __init__(self):
        super().__init__(narrow=True, path_only_through_water=False)


class AlternateLavaComparison_OnlyWaterPath_Narrow(AlternateLavaComparison):
    def __init__(self):
        super().__init__(narrow=True, path_only_through_water=True)


class TwoDifferentPaths(MiniGridEnv):
    """
    Customized environment with two different paths.
    (One has nothing on its way and the other has water on its way)
    """

    def __init__(
        self,
        width=6,
        height=6,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        strip2_row=3
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = (1, height-2)
        self.strip2_row = strip2_row

        self.directionless_agent = False

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # create an empty grid with different types of agents
        if self.directionless_agent:
            self.grid = NoDirectionAgentGrid(width, height)
        else:
            self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the two goal squares in the bottom-right corner
        self.put_obj(Floor(color='green'), *self.goal_pos)

        # Place the lava
        self.grid.set(2, 2, Lava())
        self.grid.set(3, 2, Lava())
        self.grid.set(3, 3, Lava())

        # Place the water
        self.grid.set(1, 3, Water())
        self.grid.set(2, 3, Water())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Reach the green floor"


register(
    id='MiniGrid-LavaComparison_noDryingOff-v0',
    entry_point='wombats.systems.minigrid:LavaComparison_noDryingOff'
)

register(
    id='MiniGrid-LavaComparison_seshia-v0',
    entry_point='wombats.systems.minigrid:LavaComparison_seshia'
)

register(
    id='MiniGrid-LavaComparison_SeshiaSecondGoal-v0',
    entry_point='wombats.systems.minigrid:LavaComparison_SeshiaSecondGoal'
)

register(
    id='MiniGrid-LavaComparison_SeshiaOnlyWaterPath-v0',
    entry_point='wombats.systems.minigrid:LavaComparison_SeshiaOnlyWaterPath'
)

register(
    id='MiniGrid-AlternateLavaComparison_AllCorridorsOpen_Wide-v0',
    entry_point='wombats.systems.minigrid:AlternateLavaComparison_AllCorridorsOpen_Wide'
)

register(
    id='MiniGrid-AlternateLavaComparison_OnlyWaterPath_Wide-v0',
    entry_point='wombats.systems.minigrid:AlternateLavaComparison_OnlyWaterPath_Wide'
)

register(
    id='MiniGrid-AlternateLavaComparison_AllCorridorsOpen_Narrow-v0',
    entry_point='wombats.systems.minigrid:AlternateLavaComparison_AllCorridorsOpen_Narrow'
)

register(
    id='MiniGrid-AlternateLavaComparison_OnlyWaterPath_Narrow-v0',
    entry_point='wombats.systems.minigrid:AlternateLavaComparison_OnlyWaterPath_Narrow'
)

register(
    id='MiniGrid-MyDistShift-v0',
    entry_point='wombats.systems.minigrid:MyDistShift'
)

register(
    id='MiniGrid-TwoDifferentPaths-v0',
    entry_point='wombats.systems.minigrid:TwoDifferentPaths'
)
