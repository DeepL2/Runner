import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np

class YamaXRealForwardWalk(gym.Env):
    def __init__(self, action_dim=14, obs_dim=14+3):
        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset(self):
        return None # initial obs

    def _step(self, action):
        return None # state, reward, done, info

