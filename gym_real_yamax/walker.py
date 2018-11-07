import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np

from icsservo import IOProvider
from Adafruit_ADXL345 import ADXL345

import math

class YamaXRealForwardWalk(gym.Env):
    def __init__(self, action_dim=14, obs_dim=14+3, imu_address=0x54, ics_port="/dev/serial0", ics_en=26):
        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        self._seed()

        self.ics_io = IOProvider(ics_port, ics_en)
        self.servos = [self.ics_io.servo(i) for i in range(10)]
        self.imu = ADXL345(imu_address)

        self.stands = np.array([
            -1.2211,
             0.8722,
             0,
             0.5233,
            -0.1744,
            -0.2616,
             0.1395,
             0.4361,
             0.2616,
             0.1744,
             0.5233,
            -0.2616,
            -0.9594,
             1.0117
            ])

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.sa[:14] = self.stands
        self.state = self.calc_state()
        return self.state # initial obs

    def _step(self, action):
        self.apply_action(action)

        self.state = self.calc_state()

        return self.state, 0, False, {} # state, reward, done, info

    def _render(self, mode, close):
        return

    def calc_state(self):
        acc = self.imu.get_accel_data()
        roll  = math.atan2(acc['y'], math.sqrt(acc['x']**2 + acc['z']**2))
        pitch = math.atan2(acc['x'], math.sqrt(acc['y']**2 + acc['z']**2))
        yaw   = math.atan2(acc['z'], math.sqrt(acc['x']**2 + acc['z']**2))
        return np.append(self.joint_states(), [roll, pitch, yaw])

    def joint_states(self):
        return self.sa[:14]

    def apply_action(self, action):
        self.sa[:14] = self.sa[:14] + action


