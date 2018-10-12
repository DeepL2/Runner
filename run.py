import gym_real_yamax

from baselines.run import build_env, train, parse_cmdline_kwargs
from baselines.common.cmd_util import common_arg_parser

import numpy as np


def main():
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    extra_args = parse_cmdline_kwargs(unknown_args)

    model, env = train(args, extra_args)
    env.close()

    env = build_env(args)
    obs = env.reset()

    def initialize_placeholders(nlstm=128, **kwargs):
        return np.zeros((args.num_env or 1, 2*nlstm)), np.zeros((1))
    state, dones = initialize_placeholders(**extra_args)
    while True:
        actions, _, state, _ = model.step(obs, S=state, M=dones)
        obs, _, done, _ = env.step(actions)
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done

        if done:
            obs = env.reset()

    env.close()
