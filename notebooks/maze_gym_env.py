import os

import random


class Discrete:
    def __init__(self, num_actions: int):
        """ Discrete action space for num_actions.
        Discrete(4) can be used as encoding moving in one of the cardinal directions.
        """
        self.n = num_actions

    def sample(self):
        return random.randint(0, self.n - 1)


class Environment:

    seeker, goal = (0, 0), (4, 4)
    info = {'seeker': seeker, 'goal': goal}

    def __init__(self,  *args, **kwargs):
        self.action_space = Discrete(4)
        self.observation_space = Discrete(5*5)

    def reset(self):
        """Reset seeker and goal positions, return observations."""
        self.seeker = (0, 0)
        self.goal = (4, 4)

        return self.get_observation()

    def get_observation(self):
        """Encode the seeker position as integer"""
        return 5 * self.seeker[0] + self.seeker[1]

    def get_reward(self):
        """Reward finding the goal"""
        return 1 if self.seeker == self.goal else 0

    def is_done(self):
        """We're done if we found the goal"""
        return self.seeker == self.goal

    def step(self, action):
        """Take a step in a direction and return all available information."""
        if action == 0:  # move down
            self.seeker = (min(self.seeker[0] + 1, 4), self.seeker[1])
        elif action == 1:  # move left
            self.seeker = (self.seeker[0], max(self.seeker[1] - 1, 0))
        elif action == 2:  # move up
            self.seeker = (max(self.seeker[0] - 1, 0), self.seeker[1])
        elif action == 3:  # move right
            self.seeker = (self.seeker[0], min(self.seeker[1] + 1, 4))
        else:
            raise ValueError("Invalid action")

        return self.get_observation(), self.get_reward(), self.is_done(), self.info

    def render(self, *args, **kwargs):
        """Render the environment, e.g. by printing its representation."""
        os.system('cls' if os.name == 'nt' else 'clear')
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except Exception:
            pass
        grid = [['| ' for _ in range(5)] + ["|\n"] for _ in range(5)]
        grid[self.goal[0]][self.goal[1]] = '|G'
        grid[self.seeker[0]][self.seeker[1]] = '|S'
        print(''.join([''.join(grid_row) for grid_row in grid]))


import gym
from gym.spaces import Discrete


class GymEnvironment(Environment, gym.Env):
    def __init__(self, *args, **kwargs):
        """Make our original `Environment` a gym `Env`."""
        super().__init__(*args, **kwargs)


gym_env = GymEnvironment()
