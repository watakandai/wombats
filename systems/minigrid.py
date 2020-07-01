from gym_minigrid.minigrid import MiniGridEnv, Grid, Lava, Goal
from gym_minigrid.wrappers import ReseedWrapper, FullyObsWrapper
from gym_minigrid.minigrid import (IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX,
                                   OBJECT_TO_IDX)
from gym_minigrid.register import register

import matplotlib.pyplot as plt
import gym
import itertools
import numpy as np
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


class StaticMinigridWombatsWrapper(gym.core.Wrapper):

    env: EnvType
    IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))
    DIR_TO_STRING = bidict({0: 'right', 1: 'down', 2: 'left', 3: 'up'})

    def __init__(self, env: EnvType,
                 seeds: List[int] = [0]) -> 'StaticMinigridWombatsWrapper':

        # actually creating the minigrid environment with appropriate wrappers
        super().__init__(FullyObsWrapper(ReseedWrapper(env, seeds=seeds)))

        # building some more constant DICTS dynamically from the env data
        self.ACTION_STR_TO_ENUM = {self.get_action_str(action): action
                                   for action in self.env.Actions}
        self.ACTION_ENUM_TO_STR = dict(zip(self.ACTION_STR_TO_ENUM.values(),
                                           self.ACTION_STR_TO_ENUM.keys()))

        # We only compute state observations label maps once here, as the
        # environment MUST BE STATIC in this instance
        obs = self.state_only_obs_reset()
        self.start_pos, self.start_dir = self.get_agent_props()
        (self.obs_str_idxs_map,
         self.cell_obs_map,
         self.cell_to_obs) = self.get_observation_maps(self.start_pos, obs)

        # we want to statically compute the data that can be used to build
        # a transition system representation of the environment
        (self.nodes,
         self.edges) = self.get_transition_system_data(self.cell_obs_map,
                                                       self.cell_to_obs,
                                                       self.ACTION_ENUM_TO_STR)

    def render_notebook(self) -> None:

        plt.imshow(self.env.render(mode='rgb_image'), interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def state_only_obs(self, obs: dict) -> EnvObs:

        cell_obs = obs['image']

        return cell_obs

    def state_only_obs_reset(self) -> EnvObs:

        obs = self.env.reset()

        return self.state_only_obs(obs)

    def state_only_obs_step(self, action: MiniGridEnv.Actions) -> StepData:

        obs, reward, done, _ = self.env.step(action)

        return self.state_only_obs(obs), reward, done, {}

    def get_agent_props(self) -> Tuple[AgentPos, AgentDir]:

        base_env = self.env.unwrapped

        return tuple(base_env.agent_pos), base_env.agent_dir

    def set_agent_props(self,
                        position: {AgentPos, None}=None,
                        direction: {AgentDir, None}=None) -> None:

        base_env = self.env.unwrapped

        if position is not None:
            base_env.agent_pos = position

        if direction is not None:
            base_env.agent_dir = direction

    def get_env_prop(self, env_property_name: str):

        base_env = self.env.unwrapped

        return getattr(base_env, env_property_name)

    def set_env_prop(self, env_property_name: str, env_property) -> None:

        base_env = self.env.unwrapped
        setattr(base_env, env_property_name, env_property)

    def obs_to_prop_str(self, obs: EnvObs,
                        row_idx: int, col_idx: int) -> str:

        obj_type, obj_color, obj_state = obs[col_idx, row_idx]
        agent_pos, _ = self.get_agent_props()
        is_agent = (col_idx, row_idx) == tuple(agent_pos)

        prop_string_base = '_'.join([IDX_TO_OBJECT[obj_type],
                                     IDX_TO_COLOR[obj_color]])

        if is_agent:
            return '_'.join([prop_string_base, self.DIR_TO_STRING[obj_state]])
        else:
            return '_'.join([prop_string_base, self.IDX_TO_STATE[obj_state]])

    def get_action_str(self, action_enum: MiniGridEnv.Actions) -> str:

        action_str = str(action_enum)

        # string version of Action.done is 'Action.done' -> trim
        substring_to_trim = 'Actions.'
        action_str = action_str[len(substring_to_trim):]

        return action_str

    def get_observation_maps(self, start_pos: AgentPos,
                             obs: EnvObs) -> Tuple[dict, defaultdict, dict]:

        obs_str_idxs_map = dict()
        cell_obs_map = defaultdict(list)
        cell_to_obs = dict()

        (num_cols, num_rows, num_cell_props) = obs.shape

        for row_idx in range(num_rows):
            for col_idx in range(num_cols):

                obs_str = self.obs_to_prop_str(obs, row_idx, col_idx)
                obj = obs[col_idx, row_idx][0]

                if IDX_TO_OBJECT[obj] != 'wall':
                    obs_str_idxs_map[obs_str] = tuple(obs[col_idx, row_idx])
                    cell_obs_map[obs_str].append((col_idx, row_idx))
                    cell_to_obs[(col_idx, row_idx)] = obs_str

        # need to remove the agent from the environment
        empty_cell_str = self.get_cell_str('empty', obs_str_idxs_map)
        agent_cell_str = self.get_cell_str('agent', obs_str_idxs_map)

        cell_to_obs[start_pos] = empty_cell_str
        obs_str_idxs_map.pop(agent_cell_str, None)
        cell_obs_map[empty_cell_str].append(start_pos)
        cell_obs_map.pop(agent_cell_str, None)

        return obs_str_idxs_map, cell_obs_map, cell_to_obs

    def get_cell_str(self, obj_type_str: str, obs_str_idxs_map: dict,
                     only_one_type_of_obj: bool = True) -> str:

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

    def add_node(self, nodes: dict, pos: AgentPos,
                 direction: AgentPos, obs_str: str) -> Tuple[dict, dict]:

        state = ', '.join([str(pos), self.DIR_TO_STRING[direction]])

        if state not in nodes:
            state_data = {'trans_distribution': None,
                          'observation': obs_str}
            nodes[state] = state_data

        return nodes, state

    def add_edge(self, nodes: dict, edges: dict,
                 action: MiniGridEnv.Actions,
                 possible_edge: Minigrid_TSEdge,
                 ACTION_ENUM_TO_STR: dict,
                 cell_to_obs: dict) -> Tuple[dict, dict]:

        action_str = ACTION_ENUM_TO_STR[action]

        (src, dest,
         src_pos, src_dir,
         dest_pos, dest_dir,
         obs_str_src, obs_str_dest) = self.get_egde_components(possible_edge,
                                                               cell_to_obs)

        nodes, state_src = self.add_node(nodes, src_pos, src_dir, obs_str_src)
        nodes, state_dest = self.add_node(nodes, dest_pos, dest_dir,
                                          obs_str_dest)

        edge_data = {'symbols': [action_str]}
        edge = {state_dest: edge_data}

        if state_src in edges:
            edges[state_src].update(edge)
        else:
            edges[state_src] = edge

        return nodes, edges

    def get_egde_components(self, edge: Minigrid_TSEdge,
                            cell_to_obs: dict) -> Tuple[Minigrid_TSNode,
                                                        Minigrid_TSNode,
                                                        AgentPos, AgentDir,
                                                        AgentPos, AgentDir,
                                                        str, str]:

        edge = edge
        src = edge[0]
        dest = edge[1]
        src_pos = src[0]
        src_dir = src[1]
        dest_pos = dest[0]
        dest_dir = dest[1]
        obs_str_src = cell_to_obs[src_pos]
        obs_str_dest = cell_to_obs[dest_pos]

        return (src, dest, src_pos, src_dir,
                dest_pos, dest_dir, obs_str_src, obs_str_dest)

    def get_transition_system_data(self, cell_obs_map: defaultdict,
                                   cell_to_obs: dict,
                                   ACTION_ENUM_TO_STR: dict) -> Tuple[dict,
                                                                      dict]:

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
             obs_str_dest) = self.get_egde_components(possible_edge,
                                                      cell_to_obs)

            if src not in done_nodes:
                for action in self.env.actions:

                    self.set_agent_props(src_pos, src_dir)
                    _, _, done, _ = self.state_only_obs_step(action)
                    dest_pos, dest_dir = self.get_agent_props()

                    dest_pos_is_valid = possible_dest_pos == dest_pos
                    dest_dir_is_valid = possible_dest_dir == dest_dir
                    egde_is_valid = (dest_pos_is_valid and dest_dir_is_valid)
                    if egde_is_valid:
                        nodes, edges = self.add_edge(nodes, edges,
                                                     action, possible_edge,
                                                     ACTION_ENUM_TO_STR,
                                                     cell_to_obs)
                        if done:
                            done_nodes.add(dest)

        # we have moved the agent a bunch, so we should reset it when done
        # extracting all of the data
        self.reset()

        return nodes, edges


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
