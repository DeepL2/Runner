from gym.envs.registration import register

register(
    id='YamaXRealForwardWalk-v0',
    entry_point='gym_real_yamax:YamaXRealForwardWalk',
    max_episode_steps=1000
)

from gym_real_yamax.walker import YamaXRealForwardWalk  # noqa: E402, F401
