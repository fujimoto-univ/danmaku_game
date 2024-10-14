import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.absolute()) + "/bullet")

import numpy as np
import torch as th
from bullet.agent import DDQNAgent
from bullet.env import BulletEnv
from bullet.net import DQNnet1ch
from bullet.replay_buffer import ReplayBufferNp
from bullet.wrapper import GrayFrameStack, NpNewAxis, ObsResize


th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True


def main():
    env = BulletEnv()
    env = ObsResize(env, shape=(84, 84))
    env = GrayFrameStack(env, shape=(84, 84))
    env = NpNewAxis(env)
    name = "DDQN_1stack_05_02"
    log_dir = pathlib.Path(__file__).parent.absolute()

    np.random.seed(1)
    th.manual_seed(1)
    th.cuda.manual_seed(1)

    max_steps = 50_000_000
    eval_interval = 500_000

    qnet = DQNnet1ch(output_dim=len(env.ACTION))
    target_qnet = DQNnet1ch(output_dim=len(env.ACTION))
    replay_buffer = ReplayBufferNp(shape=(1, 84, 84))
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
    model.learn()


if __name__ == "__main__":
    main()
