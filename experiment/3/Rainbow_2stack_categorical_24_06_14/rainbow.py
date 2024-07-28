import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.absolute()))

import numpy as np
import torch as th
from bullet.env import BulletEnv
from bullet.wrapper import GetPlayerAroundView
from pfrl.q_functions import DuelingDQN
from pfrl.replay_buffers import PrioritizedReplayBuffer
from rainbow_categorical import RainbowAgentNoCategorical


th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True


def main():
    env = BulletEnv()
    env = GetPlayerAroundView(env, shape=(84, 84))
    name = "Rainbow_2stack_categorical_06_14"
    log_dir = pathlib.Path(__file__).parent.absolute()

    np.random.seed(0)
    th.manual_seed(0)
    th.cuda.manual_seed(0)

    max_steps = 50_000_000
    eval_interval = 500_000

    q_func = DuelingDQN(
        len(env.ACTION),
        n_input_channels=2
    )

    betasteps = max_steps / RainbowAgentNoCategorical.netwrok_update_interval
    replay_buffer = PrioritizedReplayBuffer(
        10**6,
        alpha=0.5,
        beta0=0.4,
        betasteps=betasteps,
        num_steps=3,
        normalize_by_max="memory",
    )

    agent = RainbowAgentNoCategorical(
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
