import numpy as np
import cv2
import warnings

from env import AbsEnv, BulletEnv
warnings.simplefilter("ignore")


def grayscale_frames_priority(obs0: np.ndarray, obs1: np.ndarray, obs2: np.ndarray, obs3: np.ndarray) -> np.ndarray:
    """4つの画像を合成する

    :param obs0: 画像0
    :param obs1: 画像1
    :param obs2: 画像2
    :param obs3: 画像3
    :return: 合成した画像
    """
    img = np.copy(obs0)
    diff = np.where((obs1 != 0) & (img == 0))
    img[diff] = obs1[diff] / 2
    diff = np.where((obs2 != 0) & (img == 0))
    img[diff] = obs2[diff] / 4
    diff = np.where((obs3 != 0) & (img == 0))
    img[diff] = obs3[diff] / 8
    return img


def around_player_pos(screen: np.ndarray, player_pos: np.ndarray) -> np.ndarray:
    """プレイヤーの周り84*84の画像を取得する

    :param obs: 画像
    :param player_pos: プレイヤーの位置
    :return: プレイヤーの周りの画像
    """
    x, y = player_pos + 5
    h, w = screen.shape

    left = max(0, x - 42)
    top = max(0, y - 42)
    right = min(w, x + 42)
    bottom = min(h, y + 42)

    patch = np.zeros((84, 84), dtype=np.uint8)
    patch_top = max(0, 42 - y + top)
    patch_bottom = min(84, 42 + bottom - y)
    patch_left = max(0, 42 - x + left)
    patch_right = min(84, 42 + right - x)

    img_top = max(0, y - 42)
    img_bottom = min(h, y + 42)
    img_left = max(0, x - 42)
    img_right = min(w, x + 42)

    patch[patch_top:patch_bottom, patch_left:patch_right] = screen[img_top:img_bottom, img_left:img_right]
    return patch


class FrameSkip(AbsEnv):
    def __init__(self, env: AbsEnv, skip: int):
        self.env = env
        self.skip = skip

    def reset(self, seed=None):
        obs = self.env.reset(seed)
        total_reward = 0
        for _ in range(self.skip):
            obs, reward, done, _ = self.env.step(0)
            total_reward += reward
            if done:
                break

        return self.observation(obs)

    def observation(self, observation):
        return observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class ObsResize(AbsEnv):
    def __init__(self, env: AbsEnv, shape: tuple):
        self.env = env
        self.shape = shape

    def reset(self, seed=None):
        obs = self.env.reset(seed)
        return self.observation(obs)

    def observation(self, observation):
        observation_resize = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation_resize

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class GrayFrameStack(AbsEnv):
    def __init__(self, env: AbsEnv, shape: tuple):
        self.env = env
        self.shape = shape
        self.obs0 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs1 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs2 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs3 = np.zeros(shape=self.shape, dtype=np.uint8)

    def reset(self, seed=None):
        obs = self.env.reset(seed)

        self.obs0 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs1 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs2 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs3 = np.zeros(shape=self.shape, dtype=np.uint8)

        return self.observation(obs)

    def observation(self, observation):
        self.obs3 = self.obs2
        self.obs2 = self.obs1
        self.obs1 = self.obs0
        self.obs0 = observation
        observation_merged = grayscale_frames_priority(self.obs0, self.obs1, self.obs2, self.obs3)
        return observation_merged

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class FrameStack(AbsEnv):
    def __init__(self, env: AbsEnv, shape: tuple):
        self.env = env
        self.shape = shape
        self.obs0 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs1 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs2 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs3 = np.zeros(shape=self.shape, dtype=np.uint8)

    def reset(self, seed=None):
        obs = self.env.reset(seed)

        self.obs0 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs1 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs2 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs3 = np.zeros(shape=self.shape, dtype=np.uint8)

        return self.observation(obs)

    def observation(self, observation):
        self.obs3 = self.obs2
        self.obs2 = self.obs1
        self.obs1 = self.obs0
        self.obs0 = observation
        observation_stack = np.stack([self.obs0, self.obs1, self.obs2, self.obs3], axis=0)
        return observation_stack

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class GetPlayerAroundView(AbsEnv):
    """プレイヤーの周りの画像を取得する
    """

    def __init__(self, env: BulletEnv, shape=(84, 84)):
        self.env = env
        self.player_pos = env.player_pos

        self.shape = shape

        self.obs0 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs1 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs2 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs3 = np.zeros(shape=self.shape, dtype=np.uint8)

    def reset(self, seed=None):
        obs = self.env.reset(seed)

        self.obs0 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs1 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs2 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.obs3 = np.zeros(shape=self.shape, dtype=np.uint8)

        self.player_pos = self.env.player_pos

        return self.observation(obs)

    def observation(self, observation):
        observation_0 = around_player_pos(observation, self.player_pos)

        observation_resize = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        self.obs3 = self.obs2
        self.obs2 = self.obs1
        self.obs1 = self.obs0
        self.obs0 = observation_resize

        observation_1 = grayscale_frames_priority(self.obs0, self.obs1, self.obs2, self.obs3)

        observation_stack = np.array([observation_0, observation_1])
        return observation_stack

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.player_pos = self.env.player_pos
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class NpNewAxis(AbsEnv):
    def __init__(self, env: AbsEnv):
        self.env = env

    def reset(self, seed=None):
        obs = self.env.reset(seed)
        return self.observation(obs)

    def observation(self, observation):
        return observation[np.newaxis, :, :]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)
