import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.absolute()))

import numpy as np
import torch as th
import pfrl
from bullet.env import BulletEnv
from bullet.min_dist_test import MinDistWrapper, Tester
from DDQNCategorical import DDQNCategorical
from bullet.wrapper import GetPlayerAroundView
from pfrl.replay_buffers import ReplayBuffer

th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True


def main():

    env = BulletEnv()
    env = MinDistWrapper(env)
    env = GetPlayerAroundView(env, shape=(84, 84))

    name = "DDQN_categorical_2stack_07_08"

    log_dir = pathlib.Path(__file__).parent.absolute() / "min_dist_test"
    model_path = pathlib.Path(__file__).parent.absolute() / "best_model"

    log_dir.mkdir(exist_ok=False)
    assert model_path.exists()

    np.random.seed(0)
    th.manual_seed(0)
    th.cuda.manual_seed(0)

    max_steps = 1
    eval_interval = 0

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
        test=True,
        max_steps=max_steps,
        eval_interval=eval_interval
    )

    tester = Tester(env, agent)
    tester.test(model_path=model_path, log_path=log_dir)


if __name__ == "__main__":
    main()
