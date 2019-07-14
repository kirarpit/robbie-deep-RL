from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import time
import warnings

from src.utils import merge_dicts, init_dict
from src.maze import Maze

# Type of channels
WALL_CHANNEL = 0
CAN_CHANNEL = 1
ROBOT_CHANNEL = 2

# Actions to idx mapping
ACT_STAY = 0
ACT_PICKUP = 1
ACT_N = 2
ACT_S = 3
ACT_E = 4
ACT_W = 5
ACT_RAND = 6
NUM_ACTS = 7

ENV_DEFAULT_CONFIG = {
    "width": 10,
    "height": 10,
    # Can specify name of the files inside the folder "mazes" without the extension. For example, "four_rooms".
    "maze": None,
    "num_robots": 2,
    "num_cans": 10,
    "max_steps": 200,
    # Possible options are "von_neumann" and "box".
    "vision_type": "von_neumann",
    "vision_radius": 1,
    "alpha": 0,
    "rewards": {
        "robot_wall_collision": -1,
        "can_picked": 1,
        "can_pick_failed": -1,
    }
}


class CanPickingEnv(MultiAgentEnv):
    def __init__(self, config):
        # check if all env_config keys are in default config
        custom_keys = config.keys()
        if not all(key in ENV_DEFAULT_CONFIG for key in custom_keys):
            raise KeyError("Custom environment configuration not found in default configuration.")
        self.config = merge_dicts(ENV_DEFAULT_CONFIG, config)
        self._process_and_validate_config()

        self.grid = None
        self.robot_positions = None
        self.can_cnt = 0
        self._timestep = 0

        # required attributes for Gym environments
        self.action_space = spaces.Discrete(NUM_ACTS)
        self.observation_space = self.get_observation_space()
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {'render.modes': ['human', 'rgb_array']}
        self.spec = None
        self.viewer = None

    def _process_and_validate_config(self):
        if self.config["maze"] is not None:
            maze = Maze(self.config["maze"])
            self.config["height"] = maze.height
            self.config["width"] = maze.width
            self.config["maze"] = maze
            warnings.warn("Maze associated parameters like height and width were overridden because of custom maze!")

        if self.config["vision_type"] not in ["von_neumann", "box"]:
            raise ValueError("Vision type {} not supported".format(self.config["vision_type"]))

        if self.config["vision_type"] == "von_neumann" and self.config["vision_radius"] != 1:
            raise ValueError("Vision radius != 1 not supported in case of Von Neumann Neighborhood")

    def get_observation_space(self):
        num_channels = int(self.config["num_robots"] > 1) + 2
        num_cells = None

        if self.config["vision_type"] == "von_neumann":
            num_cells = 5
        elif self.config["vision_type"] == "box":
            num_cells = (2*self.config["vision_radius"] + 1) ** 2

        return spaces.Box(0, self.config["num_robots"], (num_channels*num_cells, ), dtype=np.int)

    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        self.can_cnt = 0
        self._timestep = 0
        width = self.config["width"]
        height = self.config["height"]

        # channels for wall, cans and robots in that order
        grid = np.zeros((3, height+2, width+2), dtype=int)

        # Initialize the walls of the grid
        grid[WALL_CHANNEL, :, [0, -1]] = 1
        grid[WALL_CHANNEL, [0, -1], :] = 1

        # Get wall cells from the custom maze file if given
        if self.config["maze"] is not None:
            wall_cells = self.config["maze"].get_walls()
            wall_cells = (wall_cells[0]+1, wall_cells[1]+1)
            grid[WALL_CHANNEL][wall_cells] = 1

        # Generate cans positions
        available_positions = np.where(grid[WALL_CHANNEL] != 1)
        available_positions = list(zip(*available_positions))
        np.random.shuffle(available_positions)
        can_locations = available_positions[:self.config["num_cans"]]
        grid[CAN_CHANNEL][tuple(zip(*can_locations))] = 1

        # Generate robots positions
        grid[ROBOT_CHANNEL, 1, 1] = self.config["num_robots"]
        self.robot_positions = [(1, 1)]*self.config["num_robots"]
        self.grid = grid
        return self.get_observations()

    def step(self, action_dict):
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.
        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        self._timestep += 1
        rewards = init_dict(self.config["num_robots"], 0)
        dones = init_dict(self.config["num_robots"], False)
        dones["__all__"] = False
        can_picks = {}

        for idx, action in action_dict.items():
            if action >= NUM_ACTS:
                raise ValueError("Invalid action {}!".format(action))

            r, c = self.robot_positions[idx]

            if action == ACT_RAND:
                action = np.random.choice([ACT_N, ACT_E, ACT_W, ACT_S])

            if action == ACT_N:
                rn, cn = (r - 1, c)
            elif action == ACT_S:
                rn, cn = (r + 1, c)
            elif action == ACT_W:
                rn, cn = (r, c - 1)
            elif action == ACT_E:
                rn, cn = (r, c + 1)
            else:
                rn, cn = (r, c)

            # if wall collision
            if self.grid[WALL_CHANNEL][rn][cn] == 1:
                rewards[idx] += self.config["rewards"]["robot_wall_collision"]
                rn, cn = (r, c)

            # Update the robot's position.
            if (r, c) != (rn, cn):
                self.update_robot_pos(idx, (rn, cn))

            if action == ACT_PICKUP:
                if self.grid[CAN_CHANNEL][r][c] == 1:
                    if (r, c) in can_picks:
                        can_picks[(r, c)].append(idx)
                    else:
                        can_picks[(r, c)] = [idx]
                else:
                    rewards[idx] += self.config["rewards"]["can_pick_failed"]

        # Reward the agents
        for (r, c), agent_ids in can_picks.items():
            for agent_id in agent_ids:
                partial_reward = self.config["rewards"]["can_picked"]/len(agent_ids)
                rewards[agent_id] += partial_reward

                """
                Reward other players according to coop or comp modes
                ------
                -> alpha = 0 means zero-sum
                -> alpha = -1 means competitive
                -> alpha = 1 means cooperative
                """
                for i in [x for x in range(self.config["num_robots"]) if x != agent_id]:
                    rewards[i] += self.config["alpha"] * partial_reward

            # Remove successfully picked can
            self.grid[CAN_CHANNEL][r][c] = 0
            self.can_cnt += 1

        if self.can_cnt >= self.config["num_cans"] or self._timestep >= self.config["max_steps"]:
            dones = init_dict(self.config["num_robots"], True)
            dones["__all__"] = True

        return self.get_observations(), rewards, dones, {}

    def get_observations(self):
        obs_dict = {}

        for idx, robot_pos in enumerate(self.robot_positions):
            # removing the current robot to take observation
            self.grid[ROBOT_CHANNEL, robot_pos[0], robot_pos[1]] -= 1

            obs = []
            num_channels = int(self.config["num_robots"] > 1) + 2

            if self.config["vision_type"] == "von_neumann":
                neighbors = self.get_von_neumann_neighs(robot_pos)
                for neighbor in neighbors:
                    obs.append(self.grid[:num_channels, neighbor[0], neighbor[1]])

            elif self.config["vision_type"] == "box":
                obs = np.zeros((num_channels, 2*self.config["vision_radius"]+1, 2*self.config["vision_radius"]+1))
                start_row = 0
                start_col = 0

                row_min = robot_pos[0] - self.config["vision_radius"]
                col_min = robot_pos[1] - self.config["vision_radius"]

                if row_min < 0:
                    start_row += abs(row_min)
                    row_min = max(0, row_min)

                if col_min < 0:
                    start_col += abs(col_min)
                    col_min = max(0, col_min)

                row_max = min(robot_pos[0] + self.config["vision_radius"], self.config["height"]) + 1
                col_max = min(robot_pos[1] + self.config["vision_radius"], self.config["width"]) + 1

                end_row = start_row + row_max - row_min
                end_col = start_col + col_max - col_min
                obs[:num_channels, start_row:end_row, start_col:end_col] =\
                    self.grid[:num_channels, row_min:row_max, col_min:col_max]

            obs_dict[idx] = np.ravel(obs)

            # adding the current robot back
            self.grid[ROBOT_CHANNEL, robot_pos[0], robot_pos[1]] += 1

        return obs_dict

    def update_robot_pos(self, idx, new_pos):
        old_pos = self.robot_positions[idx]
        self.grid[ROBOT_CHANNEL, old_pos[0], old_pos[1]] -= 1
        self.grid[ROBOT_CHANNEL, new_pos[0], new_pos[1]] += 1
        self.robot_positions[idx] = new_pos

    def get_von_neumann_neighs(self, pos):
        result = [pos]

        neighs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        for neigh in neighs:
            new_pos = (pos[0]+neigh[0], pos[1]+neigh[1])
            if 0 <= new_pos[0] < self.config["height"]+2 and 0 <= new_pos[1] <= self.config["width"]+2:
                result.append(new_pos)

        return result

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        height = self.config["height"]
        width = self.config["width"]
        sf = 25

        if self.viewer is None:
            self.viewer = rendering.Viewer((width + 2) * sf, (height + 2) * sf)

            # put walls
            r, c = np.where(self.grid[WALL_CHANNEL] == 1)
            for pos in list(zip(r, c)):
                pos = (pos[1] * sf + sf // 2, (height + 1 - pos[0]) * sf + sf // 2)
                square = rendering.make_circle(sf // 2, res=4)
                square.set_color(0, 0, 255)
                transform = rendering.Transform()
                square.add_attr(transform)
                transform.set_translation(*pos)
                self.viewer.add_geom(square)

        # put can items
        r, c = np.where(self.grid[CAN_CHANNEL] == 1)
        for pos in list(zip(r, c)):
            pos = (pos[1] * sf + sf // 2 + 1, (height + 1 - pos[0]) * sf + sf // 2 + 1)
            circle = self.viewer.draw_circle(sf // 2, color=(0, 255, 0))
            transform = rendering.Transform()
            circle.add_attr(transform)
            transform.set_translation(*pos)

        # put robots
        for pos in self.robot_positions:
            pos = (pos[1] * sf + sf // 2 + 1, (height + 1 - pos[0]) * sf + sf // 2 + 1)
            circle = self.viewer.draw_circle(sf // 2, color=(255, 0, 0))
            transform = rendering.Transform()
            circle.add_attr(transform)
            transform.set_translation(*pos)

        time.sleep(1/20)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
