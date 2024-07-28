import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.absolute()))

import numpy as np
import torch as th

from bullet.agent import DDQNAgent
from bullet.env import BulletEnv
from bullet.net import DQNnet4ch
from bullet.replay_buffer import ReplayBufferNp
from bullet.tester import Tester
from bullet.wrapper import ObsResize, FrameStack


th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True


def main():

    env = BulletEnv()
    env = ObsResize(env, shape=(84, 84))
    env = FrameStack(env, shape=(84, 84))

    name = "DDQN_4stack_05_02"

    log_dir = pathlib.Path(__file__).parent.absolute() / "test"
    model_path = pathlib.Path(__file__).parent.absolute() / "best_model"

    log_dir.mkdir(exist_ok=False)
    assert model_path.exists()

    np.random.seed(0)
    th.manual_seed(0)
    th.cuda.manual_seed(0)

    max_steps = 1_000_000
    eval_interval = 500_000

    qnet = DQNnet4ch(output_dim=len(env.ACTION))
    target_qnet = DQNnet4ch(output_dim=len(env.ACTION))
    replay_buffer = ReplayBufferNp(shape=(1, 84, 84))
    agent = DDQNAgent(
        env=env,
        name=name,
        qnet=qnet,
        target_qnet=target_qnet,
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
