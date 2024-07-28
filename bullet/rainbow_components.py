from typing import Generic, Optional, Sequence, Tuple, TypeVar, List, Deque
import collections
import pickle
import zlib
import pathlib
import logging
import torch as th
import torch.nn as nn
import numpy as np
import pfrl
from pfrl import agents, explorers, replay_buffers
from pfrl.replay_buffers.replay_buffer import ReplayBuffer
from pfrl.collections.prioritized import SumTreeQueue, MinTreeQueue
from pfrl.utils.random import sample_n_k

from env import BulletEnv
from replay_buffer import AbsReplayBuffer
from agent import AbsEvaluator, AbsAgent


T = TypeVar("T")


class PrioritizedBufferCompress(Generic[T]):
    def __init__(
        self,
        capacity: Optional[int] = None,
        wait_priority_after_sampling: bool = True,
        initial_max_priority: float = 1.0,
    ):
        self.capacity = capacity
        self.data: Deque = collections.deque()
        self.priority_sums = SumTreeQueue()
        self.priority_mins = MinTreeQueue()
        self.max_priority = initial_max_priority
        self.wait_priority_after_sampling = wait_priority_after_sampling
        self.flag_wait_priority = False

    def __len__(self) -> int:
        return len(self.data)

    def append(self, value: T, priority: Optional[float] = None) -> None:
        if self.capacity is not None and len(self) == self.capacity:
            self.popleft()
        if priority is None:
            # Append with the highest priority
            priority = self.max_priority

        compressed = zlib.compress(pickle.dumps(value))
        self.data.append(compressed)

        self.priority_sums.append(priority)
        self.priority_mins.append(priority)

    def popleft(self) -> T:
        assert len(self) > 0
        self.priority_sums.popleft()
        self.priority_mins.popleft()
        compressed = self.data.popleft()
        return pickle.loads(zlib.decompress(compressed))

    def _sample_indices_and_probabilities(
        self, n: int, uniform_ratio: float
    ) -> Tuple[List[int], List[float], float]:
        total_priority: float = self.priority_sums.sum()
        min_prob = self.priority_mins.min() / total_priority
        indices = []
        priorities = []
        if uniform_ratio > 0:
            # Mix uniform samples and prioritized samples
            n_uniform = np.random.binomial(n, uniform_ratio)
            un_indices, un_priorities = self.priority_sums.uniform_sample(
                n_uniform, remove=self.wait_priority_after_sampling
            )
            indices.extend(un_indices)
            priorities.extend(un_priorities)
            n -= n_uniform
            min_prob = uniform_ratio / len(self) + (1 - uniform_ratio) * min_prob

        pr_indices, pr_priorities = self.priority_sums.prioritized_sample(
            n, remove=self.wait_priority_after_sampling
        )
        indices.extend(pr_indices)
        priorities.extend(pr_priorities)

        probs = [
            uniform_ratio / len(self) + (1 - uniform_ratio) * pri / total_priority
            for pri in priorities
        ]
        return indices, probs, min_prob

    def sample(
        self, n: int, uniform_ratio: float = 0
    ) -> Tuple[List[T], List[float], float]:
        """Sample data along with their corresponding probabilities.

        Args:
            n (int): Number of data to sample.
            uniform_ratio (float): Ratio of uniformly sampled data.
        Returns:
            sampled data (list)
            probabitilies (list)
        """
        assert not self.wait_priority_after_sampling or not self.flag_wait_priority
        indices, probabilities, min_prob = self._sample_indices_and_probabilities(
            n, uniform_ratio=uniform_ratio
        )
        sampled = [pickle.loads(zlib.decompress(self.data[i])) for i in indices]
        self.sampled_indices = indices
        self.flag_wait_priority = True
        return sampled, probabilities, min_prob

    def set_last_priority(self, priority: Sequence[float]) -> None:
        assert not self.wait_priority_after_sampling or self.flag_wait_priority
        assert all([p > 0.0 for p in priority])
        assert len(self.sampled_indices) == len(priority)
        for i, p in zip(self.sampled_indices, priority):
            self.priority_sums[i] = p
            self.priority_mins[i] = p
            self.max_priority = max(self.max_priority, p)
        self.flag_wait_priority = False
        self.sampled_indices = []

    def _uniform_sample_indices_and_probabilities(
        self, n: int
    ) -> Tuple[List[int], List[float]]:
        indices = list(sample_n_k(len(self.data), n))
        probabilities = [1 / len(self)] * len(indices)
        return indices, probabilities


class PrioritizedReplayBufferCompress(replay_buffers.PrioritizedReplayBuffer, replay_buffers.PriorityWeightError):
    """Stochastic Prioritization

    https://arxiv.org/pdf/1511.05952.pdf Section 3.3
    proportional prioritization

    Args:
        capacity (int): capacity in terms of number of transitions
        alpha (float): Exponent of errors to compute probabilities to sample
        beta0 (float): Initial value of beta
        betasteps (int): Steps to anneal beta to 1
        eps (float): To revisit a step after its error becomes near zero
        normalize_by_max (bool): Method to normalize weights. ``'batch'`` or
            ``True`` (default): divide by the maximum weight in the sampled
            batch. ``'memory'``: divide by the maximum weight in the memory.
            ``False``: do not normalize
    """

    def __init__(
        self,
        capacity=None,
        alpha=0.6,
        beta0=0.4,
        betasteps=2e5,
        eps=0.01,
        normalize_by_max=True,
        error_min=0,
        error_max=1,
        num_steps=1,
    ):
        self.capacity = capacity
        assert num_steps > 0
        self.num_steps = num_steps
        self.memory = PrioritizedBufferCompress(capacity=capacity)
        self.last_n_transitions = collections.defaultdict(
            lambda: collections.deque([], maxlen=num_steps)
        )
        replay_buffers.PriorityWeightError.__init__(
            self,
            alpha,
            beta0,
            betasteps,
            eps,
            normalize_by_max,
            error_min=error_min,
            error_max=error_max,
        )

    def sample(self, n):
        assert len(self.memory) >= n
        sampled, probabilities, min_prob = self.memory.sample(n)
        weights = self.weights_from_probabilities(probabilities, min_prob)
        for e, w in zip(sampled, weights):
            e[0]["weight"] = w
        return sampled

    def update_errors(self, errors):
        self.memory.set_last_priority(self.priority_from_errors(errors))


class RainbowEvaluator(AbsEvaluator):
    def __init__(self, env: BulletEnv, agent: pfrl.agent.Agent, log_dir: pathlib.Path, eval_interval: int, logger: logging.Logger):
        self.env = env
        self.agent = agent
        self.eval_interval = eval_interval
        self.logger = logger

        self.eval_step = eval_interval
        self.writer = self.get_writer(log_dir)

    def get_action(self, obs: np.ndarray) -> int:
        return self.agent.act(obs)

    def eval_mode(self):
        self.agent.training = False
        return

    def train_mode(self):
        self.agent.training = True
        return


class RainbowAgent(AbsAgent):
    gamma = 0.99
    batch_size = 32

    decay_steps = 1_000_000
    replay_start_size = 8 * 10**4
    target_update_interval = 32_000
    netwrok_update_interval = 4

    n_atoms = 51
    v_max = 10
    v_min = -10

    def __init__(
            self,
            env: BulletEnv,
            name: str,
            q_func: nn.Module,
            replay_buffer: AbsReplayBuffer,
            log_dir: pathlib.Path,
            test: bool = False,
            max_steps=100_000_000,
            eval_interval=1_000_000):
        self.set_params(env, name, max_steps)
        self.set_models(q_func, replay_buffer)
        if not test:
            self.set_dirs(log_dir)
            self.evaluator = RainbowEvaluator(env, self.agent, log_dir, eval_interval, self.logger)
        else:
            self.deterministic = True
        return

    def set_params(self, env: BulletEnv, name: str, max_steps: int):
        """パラメータを設定する

        :param env: 環境
        :param name: エージェントの名前
        :param max_steps: 最大ステップ数
        """
        self.env = env
        self.name = name
        self.max_steps = max_steps + 1

        self.action_space = len(self.env.ACTION)

    def set_models(self, q_func: nn.Module, replay_buffer: ReplayBuffer):

        explorer = explorers.LinearDecayEpsilonGreedy(
            start_epsilon=1.0,
            end_epsilon=0.1,
            decay_steps=self.decay_steps,
            random_action_func=(lambda: np.random.randint(self.action_space))
        )
        opt = th.optim.Adam(q_func.parameters(), 6.25e-5, eps=1.5 * 10**-4)

        def phi(x):
            # Feature extractor
            return np.asarray(x, dtype=np.float32) / 255
        Agent = agents.CategoricalDoubleDQN
        self.agent = Agent(
            q_func,
            opt,
            replay_buffer,
            gpu=0,
            gamma=self.gamma,
            explorer=explorer,
            minibatch_size=self.batch_size,
            replay_start_size=self.replay_start_size,
            target_update_interval=self.target_update_interval,
            update_interval=self.netwrok_update_interval,
            batch_accumulator="mean",
            phi=phi,
        )
        return

    def observe(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        return self.agent.observe(next_obs, reward, done, False)

    def update_network(self, step: InterruptedError):
        return

    def get_action(self, obs: np.ndarray, step: float) -> int:
        return self.agent.act(obs)

    def sync_target_network(self, step: int):
        return

    def save_model(self, model_path: pathlib.Path, model_name: str, steps: int):
        self.agent.save(model_path)
        return

    def load(self, model_dir: pathlib.Path):
        self.agent.load(model_dir)
        return

    def eval_mode(self):
        self.agent.training = False
        return
