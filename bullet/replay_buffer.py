from __future__ import annotations
import pickle
import zlib
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class AbsReplayBuffer(ABC):
    MAX_LEN = 1_000_000

    @abstractmethod
    def push(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        """トランジションをバッファに追加する

        :param obs: observation
        :param action: action
        :param reward: reward
        :param next_obs: next observation
        :param done: done
        """
        raise NotImplementedError

    @abstractmethod
    def get_minibatch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """バッファからミニバッチを取得する

        :param batch_size: ミニバッチのサイズ
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class ReplayBufferCompress(AbsReplayBuffer):

    def __init__(self):
        self.buffer = deque(maxlen=self.MAX_LEN)

    def __len__(self):
        return len(self.buffer)

    def push(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        transition = (obs, action, reward, next_obs, done)
        exp = Experience(*transition)
        self.buffer.append(zlib.compress(pickle.dumps(exp)))
        return

    def get_minibatch(self, batch_size: int):
        indices = np.random.randint(0, len(self.buffer), batch_size)
        selected_experiences: list[Experience] = [pickle.loads(zlib.decompress(self.buffer[idx])) for idx in indices]

        states = np.array([exp.state for exp in selected_experiences], dtype=np.uint8)
        actions = np.array([exp.action for exp in selected_experiences], dtype=np.int8)
        rewards = np.array([exp.reward for exp in selected_experiences], dtype=np.int8)
        next_states = np.array([exp.next_state for exp in selected_experiences], dtype=np.uint8)
        dones = np.array([exp.done for exp in selected_experiences], dtype=np.uint8)

        return states, actions, rewards, next_states, dones


class ReplayBufferNp(AbsReplayBuffer):

    def __init__(self, shape):
        self.states = np.zeros((self.MAX_LEN, *(shape)), dtype=np.uint8)
        self.actions = np.zeros(self.MAX_LEN, dtype=np.int8)
        self.rewards = np.zeros(self.MAX_LEN, dtype=np.int8)
        self.next_states = np.zeros((self.MAX_LEN, *(shape)), dtype=np.uint8)
        self.dones = np.zeros(self.MAX_LEN, dtype=np.uint8)

        self.pos = 0
        self.full = False

    def push(self, obs, action, reward, next_obs, done):
        self.states[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_obs
        self.dones[self.pos] = done

        self.pos += 1

        if self.pos == self.MAX_LEN:
            self.pos = 0
            self.full = True

    def get_minibatch(self, batch_size):
        indices = np.random.randint(0, len(self), batch_size)

        states = np.copy(self.states[indices])
        next_states = np.copy(self.next_states[indices])
        rewards = np.copy(self.rewards[indices])
        dones = np.copy(self.dones[indices])

        return (states, self.actions[indices], rewards, next_states, dones)

    def __len__(self):
        return self.MAX_LEN if self.full else self.pos
