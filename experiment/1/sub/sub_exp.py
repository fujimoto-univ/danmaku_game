import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.absolute()) + "/bullet")

import random
import numpy as np
import torch as th
from bullet.env import BulletEnv
from bullet.min_dist_test import MinDistWrapper, Tester

th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True


class OnlyRight:
    def __init__(self):
        return

    def load(self, path):
        return

    def eval_mode(self):
        return

    def get_action(self, obs, epsilon):
        return 1


class RadomAction:
    def __init__(self):
        return

    def load(self, path):
        return

    def eval_mode(self):
        return

    def get_action(self, obs, epsilon):
        return random.randint(0, 4)


def right():

    env = BulletEnv()
    env = MinDistWrapper(env)

    log_dir = pathlib.Path(__file__).parent.absolute() / "right"

    log_dir.mkdir(exist_ok=False)

    np.random.seed(0)
    th.manual_seed(0)
    th.cuda.manual_seed(0)

    agent = OnlyRight()

    tester = Tester(env, agent)
    tester.test(model_path="", log_path=log_dir)


def random_action():
    env = BulletEnv()
    env = MinDistWrapper(env)

    log_dir = pathlib.Path(__file__).parent.absolute() / "random"

    log_dir.mkdir(exist_ok=False)

    random.seed(0)
    np.random.seed(0)
    th.manual_seed(0)
    th.cuda.manual_seed(0)

    agent = RadomAction()

    tester = Tester(env, agent)
    tester.test(model_path="", log_path=log_dir)


if __name__ == "__main__":
    right()
    random_action()
