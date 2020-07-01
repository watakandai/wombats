from gym_minigrid.minigrid import *
from gym_minigrid.register import register


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
