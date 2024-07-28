import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.absolute()))

import numpy as np
import torch as th
from bullet.agent import DDQNAgent
from bullet.env import BulletEnv
from bullet.net import DQNnet2ch
from bullet.replay_buffer import ReplayBufferNp
from bullet.wrapper import GetPlayerAroundView


th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True


def main():
    env = BulletEnv()
    env = GetPlayerAroundView(env, shape=(84, 84))
    name = "DDQN_2stack_05_02"
    log_dir = pathlib.Path(__file__).parent.absolute()
    model_dir = pathlib.Path(__file__).parent.absolute() / "old_model"

    np.random.seed(0)
    th.manual_seed(0)
    th.cuda.manual_seed(0)

    max_steps = 50_000_000
    eval_interval = 500_000

    qnet = DQNnet2ch(output_dim=len(env.ACTION))
    target_qnet = DQNnet2ch(output_dim=len(env.ACTION))
    replay_buffer = ReplayBufferNp(shape=(2, 84, 84))
    model = DDQNAgent(
        env=env,
        name=name,
        qnet=qnet,
        target_qnet=target_qnet,
        replay_buffer=replay_buffer,
        log_dir=log_dir,
        test=False,
        max_steps=max_steps,
        eval_interval=eval_interval
    )
    model.load(model_dir)
    model.learn()


if __name__ == "__main__":
    main()