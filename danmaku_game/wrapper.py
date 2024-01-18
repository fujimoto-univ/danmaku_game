import numpy as np
import cv2
import torch as th
import warnings
import torchvision.transforms as T
from copy import deepcopy
warnings.simplefilter("ignore")


# class NoopWrapper:
#     def __init__(self, env: MyEnvManual):
#         self.env = env
#         self.skip = self.env.enemy_manager.enemy_list[0].ENEMY_SHOOT_TIMING_DELAY

#     def reset(self):
#         _ = self.env.reset()
#         for _ in range(np.random.randint(1, self.skip + 1)):
#             obs = self.env.step(0)
#         return obs

#     def step(self, action):
#         return self.env.step(action)

#     def observation(self, observation):
#         return self.env.observation(observation)

#     def __getattr__(self, name):
#         return getattr(self.env, name)


WEIGHT = 255 * 15 / 8


def grayscale_frames(obs0: np.ndarray, obs1: np.ndarray, obs2: np.ndarray, obs3: np.ndarray):
    img = obs0 + obs1 / 2 + obs2 / 4 + obs3 / 8
    img = 255 * img / WEIGHT
    img = img.astype(np.uint8)
    return img


def grayscale_frames_priority(obs0: np.ndarray, obs1: np.ndarray, obs2: np.ndarray, obs3: np.ndarray):
    img = obs0
    diff = np.where((obs1 != 0) & (img == 0))
    img[diff] = obs1[diff] / 2
    diff = np.where((obs2 != 0) & (img == 0))
    img[diff] = obs2[diff] / 4
    diff = np.where((obs3 != 0) & (img == 0))
    img[diff] = obs3[diff] / 8
    return img


def around_player_pos(obs, player_pos):
    """84*84の画像の中心をプレイヤーの位置にする

    Args:
        obs (ndarray): _description_
        player_pos (ndarray): _description_

    Returns:
        ndarray: _description_
    """
    x, y = player_pos + 5
    h, w = obs.shape

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

    patch[patch_top:patch_bottom, patch_left:patch_right] = obs[img_top:img_bottom, img_left:img_right]
    return patch.copy()


# class SkipFrame:
#     def __init__(self, env, skip):
#         self.env = env
#         self._skip = skip

#     def step(self, action):
#         total_reward = 0.0
#         done = False
#         for i in range(self._skip):
#             obs, reward, done, info = self.env.step(action)
#             total_reward += reward
#             if done:
#                 break
#         return obs, total_reward, done, info

#     def __getattr__(self, name):
#         return getattr(self.env, name)

class NpCopy:
    def __init__(self, env):
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return deepcopy(obs)

    def observation(self, observation):
        return observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return deepcopy(obs), reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class SkipFirstFrame:
    """最初のフレームをスキップする
    """
    def __init__(self, env, frame_skip):
        self.env = env
        self.frame_skip = frame_skip

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.frame_skip):
            obs, _, _, _ = self.env.step(0)
        return obs

    def observation(self, observation):
        return observation

    def step(self, action):
        return self.env.step(action)

    def __getattr__(self, name):
        return getattr(self.env, name)


class ToTensor:
    """tensorに変換する
    """
    def __init__(self, env):
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def observation(self, observation):
        observation = np.transpose(observation[:, :, np.newaxis], (2, 0, 1))
        observation = th.tensor(observation.copy(), dtype=th.float32)
        return observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class ObsGrayScaleAndResizeNormTensor:
    """
    ObsGrayScaleAndResize (0, 1)に正規化、リサイズする
    output: (1, 84, 84)
    """
    def __init__(self, env, shape):
        self.env = env
        self.shape = shape

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def observation(self, observation):
        trans = T.Compose([T.Resize(self.shape), T.Normalize(0, 255)])
        observation = trans(observation)
        return observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class ObsGrayScaleAndResizeNormNp:
    """
    ObsGrayScaleAndResize (0, 1)に正規化、リサイズする
    output: (84, 84)
    """
    def __init__(self, env, shape):
        self.env = env
        self.shape = shape

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def observation(self, observation):
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        observation = observation / 255
        return observation.astype(np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class ObsGrayScaleAndResizeNp:
    """
    ObsGrayScaleAndResize リサイズする
    output: (84, 84)
    """
    def __init__(self, env, shape):
        self.env = env
        self.shape = shape

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def observation(self, observation):
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class GrayFrameStack:
    def __init__(self, env, shape):
        self.env = env
        self.shape = shape
        self.old_obs0 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs1 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs2 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs3 = np.zeros(shape=self.shape, dtype=np.uint8)

    def reset(self):
        obs = self.env.reset()
        self.old_obs0 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs1 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs2 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs3 = np.zeros(shape=self.shape, dtype=np.uint8)
        return self.observation(obs)

    def observation(self, observation):
        observation = grayscale_frames_priority(observation, self.old_obs0, self.old_obs1, self.old_obs2)
        self.old_obs0 = observation
        self.old_obs1 = self.old_obs0
        self.old_obs2 = self.old_obs1
        self.old_obs3 = self.old_obs2
        return observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class FrameStack:
    def __init__(self, env, shape):
        self.env = env
        self.shape = shape
        self.old_obs0 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs1 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs2 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs3 = np.zeros(shape=self.shape, dtype=np.uint8)

    def reset(self):
        obs = self.env.reset()
        self.old_obs0 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs1 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs2 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs3 = np.zeros(shape=self.shape, dtype=np.uint8)
        return self.observation(obs)

    def observation(self, observation):
        self.old_obs0 = observation
        self.old_obs1 = self.old_obs0
        self.old_obs2 = self.old_obs1
        self.old_obs3 = self.old_obs2
        observation = np.stack([self.old_obs0, self.old_obs1, self.old_obs2, self.old_obs3])
        return observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class FrameStackTensor:
    """4フレームをスタックする
    """
    def __init__(self, env, shape):
        self.env = env
        self.shape = shape
        self.old_obs0 = th.zeros(size=self.shape)
        self.old_obs1 = th.zeros(size=self.shape)
        self.old_obs2 = th.zeros(size=self.shape)
        self.old_obs3 = th.zeros(size=self.shape)

    def reset(self):
        obs = self.env.reset()
        self.old_obs0 = th.zeros(size=self.shape)
        self.old_obs1 = th.zeros(size=self.shape)
        self.old_obs2 = th.zeros(size=self.shape)
        self.old_obs3 = th.zeros(size=self.shape)
        return self.observation(obs)

    def observation(self, observation):
        self.old_obs0 = observation
        self.old_obs1 = self.old_obs0
        self.old_obs2 = self.old_obs1
        self.old_obs3 = self.old_obs2
        observation = th.stack([self.old_obs0, self.old_obs1, self.old_obs2, self.old_obs3])
        return observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class GetPlayerAroundView:
    """プレイヤーの周りの画像を取得する
    """
    def __init__(self, env, shape=(84, 84)):
        self.env = env
        self.player_pos = env.player_pos

        self.shape = shape
        self.old_obs0 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs1 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs2 = np.zeros(shape=self.shape, dtype=np.uint8)

    def reset(self):
        obs = self.env.reset()
        self.player_pos = self.env.player_pos

        self.old_obs0 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs1 = np.zeros(shape=self.shape, dtype=np.uint8)
        self.old_obs2 = np.zeros(shape=self.shape, dtype=np.uint8)

        return self.observation(obs)

    def observation(self, observation):
        observation_0 = around_player_pos(observation, self.player_pos)

        observation_1 = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)

        observation = grayscale_frames_priority(observation_1, self.old_obs0, self.old_obs1, self.old_obs2)
        self.old_obs0 = observation
        self.old_obs1 = self.old_obs0
        self.old_obs2 = self.old_obs1

        observation = np.array([observation_0, observation])
        return observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.player_pos = self.env.player_pos
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class NpNewAxis:
    def __init__(self, env):
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def observation(self, observation):
        return observation[np.newaxis, :, :]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)
