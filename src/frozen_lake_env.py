# Required Libraries
import gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

# housekeeping
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plt.style.use('ggplot')

class FrozenLakeEnv():
    
    def __init__(self, render_mode: Union[str, None] = None, seed:int=0, plot_rewards_bool: bool = True) -> None:
        # MDP environment
        self.observation = None
        self.info = None
        self.reward = None
        self.terminated: bool = None
        self.truncated: bool = None

        # make the environment
        self._seed: int = seed
        self.render_mode: bool = render_mode

        # graphing vars
        self.plot_rewards_bool = plot_rewards_bool
        if plot_rewards_bool:
            self.reward_running: float = 0.0
            self.rewards: np.array = np.array([])
        

    def init_environment(self) -> None:
        self.env = gym.make(
                            'FrozenLake-v1', 
                            map_name="8x8",
                            render_mode=self.render_mode
                            )
        self.observation, self.info = self.env.reset(seed=self.seed)

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, seed:int) -> None:
        self.env.seed(seed)

    def reset(self, seed:int=0) -> tuple:
        if self.plot_rewards_bool:
            self.rewards = np.append(self.rewards, self.reward_running)
            self.reward_running = 0.0
        self.terminated = False
        self.truncated = False
        self.observation, self.info = self.env.reset(seed=seed)
        return self.observation, self.info

    def get_action_space(self) -> gym.spaces.Discrete:
        return self.env.action_space

    def get_observation_space(self) -> gym.spaces.Box:
        return self.env.observation_space
    
    def get_reward_range(self) -> tuple:
        return self.env.reward_range

    
    def step(self, action:int) -> tuple:
        self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(action)
        self.reward_running += self.reward
        return self.observation, self.reward, self.terminated, self.truncated, self.info
    
    def quit(self) -> None:
        self.env.close()


    def plot_rewards(self, bin_size:int = 10_000, show: bool = True, save: bool = True) -> None:
        rewards_rolling_average = np.convolve(self.rewards, np.ones(bin_size), 'valid') / bin_size
        
        plt.plot(rewards_rolling_average)
        plt.title('Frozen Lake Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')

        if save:
            plt.savefig('figs/frozen_lake_rewards.png')

        if show:
            plt.show()        
