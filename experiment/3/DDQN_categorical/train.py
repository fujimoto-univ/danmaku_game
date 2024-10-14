import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.absolute()) + "/bullet")

import numpy as np
import torch as th
import pfrl
from bullet.env import BulletEnv
from DDQNCategorical import DDQNCategorical
from bullet.wrapper import GetPlayerAroundView
from pfrl.replay_buffers import ReplayBuffer


th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True


def main():
    env = BulletEnv()
    env = GetPlayerAroundView(env, shape=(84, 84))
    name = "DDQN_categorical_2stack_07_08"
    log_dir = pathlib.Path(__file__).parent.absolute()

    np.random.seed(0)
    th.manual_seed(0)
    th.cuda.manual_seed(0)

    max_steps = 50_000_000
    eval_interval = 500_000

    q_func = th.nn.Sequential(
        pfrl.nn.LargeAtariCNN(n_input_channels=2, n_output_channels=512),
        pfrl.q_functions.DistributionalFCStateQFunctionWithDiscreteAction(
            512,
            len(env.ACTION),
            DDQNCategorical.n_atoms,
            DDQNCategorical.v_min,
            DDQNCategorical.v_max,
            n_hidden_channels=0,
            n_hidden_layers=0,
        ),
    )

    replay_buffer = ReplayBuffer(
        10**6,
        num_steps=1,
    )

    agent = DDQNCategorical(
        env=env,
        name=name,
        q_func=q_func,
        replay_buffer=replay_buffer,
        log_dir=log_dir,
        test=False,
        max_steps=max_steps,
        eval_interval=eval_interval
    )
    agent.learn()


if __name__ == "__main__":
    main()
