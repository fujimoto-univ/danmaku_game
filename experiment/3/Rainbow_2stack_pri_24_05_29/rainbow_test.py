import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.absolute()))

import numpy as np
import torch as th
from bullet.env import BulletEnv
from bullet.rainbow_components import RainbowAgent
from bullet.tester import Tester
from bullet.wrapper import GetPlayerAroundView
from pfrl.q_functions import DistributionalDuelingDQN
from pfrl.replay_buffers import ReplayBuffer


th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True


def main():

    env = BulletEnv()
    env = GetPlayerAroundView(env, shape=(84, 84))

    name = "Rainbow_2stack_pri_05_29"

    log_dir = pathlib.Path(__file__).parent.absolute() / "test"
    model_path = pathlib.Path(__file__).parent.absolute() / "best_model"

    log_dir.mkdir(exist_ok=False)
    assert model_path.exists()

    np.random.seed(0)
    th.manual_seed(0)
    th.cuda.manual_seed(0)

    max_steps = 1_000_000
    eval_interval = 500_000

    q_func = DistributionalDuelingDQN(
        len(env.ACTION),
        RainbowAgent.n_atoms,
        RainbowAgent.v_min,
        RainbowAgent.v_max,
        n_input_channels=2
    )

    replay_buffer = ReplayBuffer(
        10**6,
        num_steps=3,
    )

    agent = RainbowAgent(
        env=env,
        name=name,
        q_func=q_func,
        replay_buffer=replay_buffer,
        log_dir=log_dir,
        test=True,
        max_steps=max_steps,
        eval_interval=eval_interval
    )

    tester = Tester(env, agent)
    tester.test(model_path=model_path, log_path=log_dir)


if __name__ == "__main__":
    main()
