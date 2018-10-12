import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np

from servoarray import ServoArray
from mpu6050 import mpu6050

import math

class YamaXRealForwardWalk(gym.Env):
    def __init__(self, action_dim=14, obs_dim=14+3, imu_address=0x68, sa_address=0x40, min_pulse=95, max_pulse=425):
        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        self._seed()

        self.sa = ServoArray(1, sa_address, min_pulse, max_pulse)
        self.sa.auto_clip(True)
        self.imu = mpu6050(imu_address)

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state = self.calc_state()
        return self.state # initial obs

    def _step(self, action):
        self.apply_action(action)

        self.state = self.calc_state()

        return np.array(self.state), 0, False, {} # state, reward, done, info

    def _render(self, mode, close):
        return

    def calc_state(self):
        acc = self.imu.get_accel_data()
        roll  = math.atan2(acc['y'], math.sqrt(acc['x']**2 + acc['z']**2))
        pitch = math.atan2(acc['x'], math.sqrt(acc['y']**2 + acc['z']**2))
        yaw   = math.atan2(acc['z'], math.sqrt(acc['x']**2 + acc['z']**2))
        return self.joint_states() + [roll, pitch, yaw]

    def joint_states(self):
        return self.sa[:14]

    def apply_action(self, action):
        self.sa[:14] = self.sa[:14] + action


