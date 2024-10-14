import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.absolute()) + "/bullet")

import numpy as np
import torch as th
from bullet.env import BulletEnv
from bullet.rainbow_components import RainbowAgent
from bullet.wrapper import GetPlayerAroundView
from pfrl.q_functions import DistributionalDuelingDQN
from pfrl.replay_buffers import PrioritizedReplayBuffer


th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True


def main():
    env = BulletEnv()
    env = GetPlayerAroundView(env, shape=(84, 84))
    name = "Rainbow_2stack_multi_10_01"
    log_dir = pathlib.Path(__file__).parent.absolute()

    np.random.seed(1)
    th.manual_seed(1)
    th.cuda.manual_seed(1)

    max_steps = 50_000_000
    eval_interval = 500_000

    q_func = DistributionalDuelingDQN(
        len(env.ACTION),
        RainbowAgent.n_atoms,
        RainbowAgent.v_min,
        RainbowAgent.v_max,
        n_input_channels=2
    )

    betasteps = max_steps / RainbowAgent.netwrok_update_interval
    replay_buffer = PrioritizedReplayBuffer(
        10**6,
        alpha=0.5,
        beta0=0.4,
        betasteps=betasteps,
        num_steps=1,
        normalize_by_max="memory",
    )

    agent = RainbowAgent(
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
