import pathlib
import pickle
import time
import logging
from abc import ABC, abstractmethod

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import tqdm
from wrapper import BulletEnv
from replay_buffer import AbsReplayBuffer


class AbsEvaluator(ABC):
    EVAL_EPISODE = 500
    SEED = np.arange(EVAL_EPISODE)

    eval_times = 0
    max_score = -np.inf

    eval_step: int
    eval_interval: int

    env: BulletEnv
    logger: logging.Logger
    writer: SummaryWriter

    def eval_if_step(self, check_step: int) -> bool:
        """指定したステップ数に達しているかを確認し、達している場合は評価を行う
        seed 0~500を用いる

        :param check_step: 現在のステップ数
        :return: 最高スコアを更新した場合はTrue
        """
        if check_step < self.eval_step:
            return False
        self.eval_step += self.eval_interval
        return self.eval(check_step)

    def eval(self, step: int) -> bool:

        self.eval_mode()

        self.eval_times += 1
        total_rewards = []
        tmp_reward = 0
        episode = 0
        idx = 0

        obs = self.env.reset(seed=self.SEED[idx])
        while True:

            action = self.get_action(obs)
            obs, reward, done, _ = self.env.step(action)
            tmp_reward += reward
            if done:
                self.logger.info(f"test: {self.eval_times}, episode: {episode}, R: {tmp_reward}")
                total_rewards.append(tmp_reward)
                tmp_reward = 0
                episode += 1
                idx += 1
                if episode >= self.EVAL_EPISODE:
                    break
                obs = self.env.reset(seed=self.SEED[idx])

        eval_stats = dict(
            mean=np.mean(total_rewards),
            median=np.median(total_rewards),
            max=np.max(total_rewards),
            min=np.min(total_rewards),
            stdev=np.std(total_rewards) if len(total_rewards) > 1 else 0.0,
        )
        self.record_tb_stats(
            summary_writer=self.writer,
            eval_stats=eval_stats, t=step
        )
        check_best = False
        if self.max_score < eval_stats["mean"]:
            self.max_score = eval_stats["mean"]
            check_best = True

        self.train_mode()
        return check_best

    def get_writer(self, log_dir: pathlib.Path) -> SummaryWriter:
        """tensorboardのwriterを取得する

        :param log_dir: ログを保存するディレクトリ
        :return: SummaryWriter
        """
        writer = SummaryWriter(log_dir=log_dir)
        layout = {
            "Aggregate Charts": {
                "mean w/ min-max": [
                    "Margin",
                    ["eval/mean", "eval/min", "eval/max"],
                ],
                "mean +/- std": [
                    "Margin",
                    ["eval/mean", "extras/meanplusstdev", "extras/meanminusstdev"],
                ],
            }
        }
        writer.add_custom_scalars(layout)
        return writer

    def record_tb_stats(self, summary_writer: SummaryWriter, eval_stats: dict, t: int):
        """tensorboardに評価結果を記録する

        :param summary_writer: SummaryWriter
        :param eval_stats: 評価結果
        :param t: 現在のステップ数
        """
        cur_time = time.time()

        for stat in ("mean", "median", "max", "min", "stdev"):
            value = eval_stats[stat]
            summary_writer.add_scalar("eval/" + stat, value, t, cur_time)

        summary_writer.flush()
        return

    def close(self):
        self.writer.close()
        return

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> int:
        raise NotImplementedError

    @abstractmethod
    def eval_mode(self):
        raise NotImplementedError

    @abstractmethod
    def train_mode(self):
        raise NotImplementedError


class DDQNEvaluator(AbsEvaluator):
    """stepごとに評価を行い、最高スコアを更新した場合にモデルを保存する
    """

    def __init__(self, env: BulletEnv, qnet: nn.Module, log_dir: pathlib.Path, eval_interval: int, logger: logging.Logger):
        self.env = env
        self.qnet = qnet
        self.eval_interval = eval_interval
        self.logger = logger

        self.eval_step = eval_interval
        self.writer = self.get_writer(log_dir)

    def get_action(self, obs: np.ndarray) -> int:
        obs = obs / 255
        obs = obs[np.newaxis, :, :, :].astype(np.float32)
        obs = th.tensor(obs, device="cuda")

        with th.no_grad():
            q = self.qnet(obs).detach().cpu().numpy()
            action = np.argmax(q.squeeze())
        return action

    def eval_mode(self):
        self.qnet.train(False)
        return

    def train_mode(self):
        self.qnet.train(True)
        return


class AbsAgent(ABC):
    env: BulletEnv
    evaluator: AbsEvaluator

    max_steps: int
    name: str

    def set_dirs(self, log_dir: pathlib.Path):
        """ディレクトリを設定する

        :param log_dir: ログを保存するディレクトリ
        :param name: エージェントの名前
        """

        self.model_dir_path = log_dir / "model"
        self.best_model_dir_path = log_dir / "best_model"

        self.model_dir_path.mkdir(exist_ok=False)
        self.best_model_dir_path.mkdir(exist_ok=False)

        logging.basicConfig(level=20, filename=log_dir / "log.log", filemode="a+")
        self.logger = logging.getLogger(__name__)
        return

    def learn(self):
        """学習を行う
        """
        step = 0
        episode = 1
        seed = self.evaluator.EVAL_EPISODE + 1

        tmp_reward = 0  # episode

        obs = self.env.reset(seed=seed)
        tqdm_bar = tqdm.tqdm(total=self.max_steps, initial=step, desc="training")

        while step < self.max_steps:
            step += 1

            action = self.get_action(obs, step)
            # 正規化かつコピーしたndarrayを返す
            next_obs, reward, done, _ = self.env.step(action)

            self.observe(obs, action, reward, next_obs, done)

            tmp_reward += reward

            obs = np.copy(next_obs)
            self.update_network(step)
            self.sync_target_network(step)

            if done:
                tqdm_bar.update(step - tqdm_bar.n)
                check_save = self.evaluator.eval_if_step(step)
                if check_save:
                    self.save_model(self.best_model_dir_path, self.name, step)
                self.logger.info(f"step: {step}, episode: {episode}, R: {tmp_reward}")
                episode += 1
                seed += 1
                tmp_reward = 0
                obs = self.env.reset(seed=seed)

        check_save = self.evaluator.eval(step)
        if check_save:
            self.save_model(self.best_model_dir_path, self.name, step)

        self.evaluator.close()
        tqdm_bar.close()
        self.save_model(self.model_dir_path, self.name, step)
        return

    @abstractmethod
    def observe(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        raise NotImplementedError

    @abstractmethod
    def update_network(self, step: int):
        raise NotImplementedError

    @abstractmethod
    def get_action(self, obs: np.ndarray, step: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def sync_target_network(self, step: int):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, model_path: pathlib.Path, model_name: str, steps: int):
        raise NotImplementedError

    @abstractmethod
    def load(self, model_path: pathlib.Path):
        raise NotImplementedError

    @abstractmethod
    def eval_mode(self):
        raise NotImplementedError


class DDQNAgent(AbsAgent):
    gamma = 0.99
    batch_size = 32

    target_update_interval = 10_000
    start_update = 20_000
    netwrok_update_interval = 4
    decay_steps = 1_000_000

    determistic = False

    def __init__(
            self,
            env: BulletEnv,
            name: str,
            qnet: nn.Module,
            target_qnet: nn.Module,
            replay_buffer: AbsReplayBuffer,
            log_dir: pathlib.Path,
            test: bool = False,
            max_steps=100_000_000,
            eval_interval=1_000_000):
        self.set_params(env, name, max_steps)
        self.set_models(qnet, target_qnet, replay_buffer)

        if not test:
            self.set_dirs(log_dir)
            self.evaluator = DDQNEvaluator(env=self.env, qnet=self.qnet, log_dir=log_dir, eval_interval=eval_interval, logger=self.logger)
        else:
            self.determistic = True
        return

    def set_params(self, env: BulletEnv, name: str, max_steps: int):
        """パラメータを設定する

        :param env: 環境
        :param name: エージェントの名前
        :param max_steps: 最大ステップ数
        """
        self.env = env
        self.name = name
        self.max_steps = max_steps

        self.epsilon_scheduler = (
            lambda steps: max(1.0 - (0.9 * steps / self.decay_steps), 0.1))
        self.max_steps = max_steps + 1
        self.action_space = len(self.env.ACTION)
        return

    def set_models(self, qnet: nn.Module, target_qnet: nn.Module, replay_buffer: AbsReplayBuffer):
        """モデルを設定する

        :param qnet: Qネットワーク
        :param target_qnet: ターゲットQネットワーク
        :param replay_buffer: リプレイバッファ
        """
        self.qnet = qnet.float().cuda()
        self.target_qnet = target_qnet.float().cuda()
        self.sync_target_network(0)

        self.qnet.train(True)
        self.target_qnet.train(False)

        for p in self.target_qnet.parameters():
            p.requires_grad = False

        self.replay_buffer = replay_buffer
        self.optimizer = th.optim.Adam(self.qnet.parameters(), lr=0.00025, eps=0.01 / self.batch_size)
        return

    def observe(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self.replay_buffer.push(obs, action, reward, next_obs, done)
        return

    def update_network(self, step: int):
        """ネットワークを更新する
        """
        if step % self.netwrok_update_interval != 0:
            return

        if len(self.replay_buffer) < self.start_update:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.get_minibatch(batch_size=32)

        states = (states / 255).astype(np.float32)
        next_states = (next_states / 255).astype(np.float32)

        states_th = th.tensor(states, device="cuda")
        rewards_th = th.tensor(rewards, device="cuda")
        next_states_th = th.tensor(next_states, device="cuda")
        dones_th = th.tensor(dones, device="cuda")

        qs = self.qnet(states_th)
        q = qs[np.arange(self.batch_size), actions]
        with th.no_grad():
            next_qs = self.qnet(next_states_th)
            best_action = th.argmax(next_qs, axis=1)
            next_q_max = self.target_qnet(next_states_th)[np.arange(self.batch_size), best_action]
            target_q = rewards_th + self.gamma * (1 - dones_th) * next_q_max

        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(q, target_q)
        loss.backward()
        # 勾配爆発を防ぐ
        th.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10)
        self.optimizer.step()

        return

    def get_action(self, state: np.ndarray, step: int) -> int:
        """epsilon-greedyで行動を選択する

        :param state: 現在の状態
        :param epsilon: epsilon
        :return: 行動
        """
        epsilon = self.epsilon_scheduler(step)

        if np.random.rand() < epsilon and not self.determistic:
            action = np.random.randint(self.action_space)
        else:
            state = state / 255
            state = state[np.newaxis, :, :, :].astype(np.float32)
            state = th.tensor(state, device="cuda")

            self.qnet.train(False)
            with th.no_grad():
                q = self.qnet(state).detach().cpu().numpy()
                action = np.argmax(q.squeeze())
            self.qnet.train(True)
        return action

    def sync_target_network(self, step: int):
        if step % self.target_update_interval == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())
        return

    def save_model(self, model_path: pathlib.Path, model_name: str, steps: int):
        """モデルを保存する

        :param model_path: 保存先のディレクトリ
        :param model_name: モデルの名前
        :param steps: 現在のステップ数
        :param best_score: 最高スコア
        """
        if not model_path.exists():
            model_path.mkdir(exist_ok=False)
        try:
            path = model_path / (model_name + ".pth")
            th.save({
                'steps': steps,
                'model_state_dict': self.qnet.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)
        except Exception:
            with open("error_net.pickle", "wb") as f:
                pickle.dump(self.qnet.state_dict(), f)
            with open("error_opt.pickle", "wb") as f:
                pickle.dump(self.optimizer.state_dict(), f)
            with open("error.txt", "a") as f:
                f.write(f"steps: {steps}\n{self.qnet.state_dict()} \n\n")
        return

    def load(self, model_dir: pathlib.Path):
        """モデルを読み込む

        :param model_path: モデルのパス
        """
        model_path = model_dir / (self.name + ".pth")
        checkpoint: dict = th.load(model_path)
        self.qnet.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.sync_target_network(0)
        return

    def eval_mode(self):
        self.qnet.train(False)
        return
