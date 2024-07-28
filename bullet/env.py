from __future__ import annotations
from abc import ABC, abstractmethod

import cv2
import numpy as np

# 画面サイズ
WINDOW_SIZE = (383, 423)


class AbsEnv(ABC):
    @abstractmethod
    def reset(self, seed=None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def observation(self, observation) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def step(self, action) -> tuple:
        raise NotImplementedError


class Enemy:
    ENEMY_SPEED = 5
    # 値は1~20
    ENEMY_SHOOT_TIMING_DELAY = 21

    # 値は3~10
    BULLET_SPEED_MAX = 11
    BULLET_SPEED_MIN = 3
    BULLET_SPEED_DIFF = BULLET_SPEED_MAX - BULLET_SPEED_MIN
    # 値は5~10
    BULLET_RADIUS_MAX = 7
    BULLET_RADIUS_MIN = 3
    # 値は1~10
    BULLET_NUM = 11
    BULLET_DELAY = 8

    def __init__(self, n_way: int, random_state: np.random.RandomState = None) -> None:
        # 射出方向の数
        self.n_way = n_way
        self.random_state = random_state

        # カウンター
        self.cnt: int = 0
        # 奇数弾は1, 偶数弾は0
        self.e_type = 1 if n_way % 2 == 1 else 0
        # 0以上で停止、1以上で弾を生成
        self.stay_flag = 0
        # pos, vec, timing, posは重心の座標
        self.e_info: list[np.ndarray, np.ndarray, int] = self._gen_enemy()
        # bulletの生成関数
        self.b_gen_func = None

    def _get_random_int(self, low: int, high: int) -> int:
        if self.random_state is not None:
            return self.random_state.randint(low, high)
        return np.random.randint(low, high)

    def _gen_enemy(self) -> list[np.ndarray, np.ndarray, int]:
        """エネミーの生成

        :return: 初期位置、速度、弾の生成タイミング
        """
        sgn = self._get_random_int(0, 2)
        pos = np.array([sgn * WINDOW_SIZE[0], self._get_random_int(0, WINDOW_SIZE[1] // 10)], dtype=np.float32)
        vec = np.array([np.sign(-sgn + 0.5) * self.ENEMY_SPEED, 0], dtype=np.float32)
        timing = self._get_random_int(self.ENEMY_SHOOT_TIMING_DELAY, 3 * self.ENEMY_SHOOT_TIMING_DELAY)
        return [pos, vec, timing]

    def update(self, player_pos: np.ndarray) -> tuple[bool, list[list[np.ndarray, int, np.ndarray]]]:
        """エネミーの更新
        弾を出すかなどの処理を行う

        :param player_pos: プレイヤーの位置
        :raises ValueError: etypeが0, 1で定義されていない場合
        :return: エネミーが存在しているかどうか、弾の情報
        """
        # timingで弾を生成する
        bullets = None
        if self.cnt == self.e_info[2]:
            if self.e_type == 0:
                self.b_gen_func, num_iter = self._gen_bullet_scatter(self.n_way)
            elif self.e_type == 1:
                self.b_gen_func, num_iter = self._gen_bullet_aim(self.n_way)
            else:
                raise ValueError("etype is not defined")
            self.stay_flag = num_iter

        self.cnt += 1

        if self.stay_flag:
            if self.cnt % self.BULLET_DELAY == 0:
                self.stay_flag -= 1
                bullets = self.b_gen_func(player_pos)
        else:
            if np.any(self.e_info[0] < 0) or np.any(self.e_info[0] > WINDOW_SIZE[0: 2]):
                return False, None
            self.e_info[0] += self.e_info[1]

        return True, bullets

    def _gen_bullet_aim(self, n_way: int) -> tuple[callable, int]:
        """エネミーの追尾弾の生成

        :param n_way: 弾の方向
        :return: 弾の生成関数、生成回数
        """
        num_iter = self._get_random_int(1, self.BULLET_NUM)
        radius = self._get_random_int(self.BULLET_RADIUS_MIN, self.BULLET_RADIUS_MAX)
        angles = (2 * np.pi * np.arange(n_way) / n_way) - np.pi / 2
        pos = np.column_stack((np.cos(angles), np.sin(angles))) * radius + self.e_info[0]
        length = self._get_random_int(self.BULLET_SPEED_MIN, self.BULLET_SPEED_MAX)

        def gen_bullet(player_pos) -> list[list[np.ndarray, int, np.ndarray]]:
            # 初期位置をradius分だけずらしてプレイヤーの方向に弾を発射する player_posは重心の座標に設定する
            # player_pos: shape(2,)
            vec = player_pos - pos
            vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
            vec = vec * length
            # pos, radius, vec
            bullets = [[pos[i].copy(), radius, vec[i]] for i in range(len(pos))]
            return bullets

        return gen_bullet, num_iter

    def _gen_bullet_scatter(self, n_way: int) -> tuple[callable, int]:
        """エネミーの散弾の生成

        :param n_way: 弾の方向
        :return: 弾の生成関数、生成回数
        """
        num_iter = self._get_random_int(1, self.BULLET_NUM)
        radius = self._get_random_int(self.BULLET_RADIUS_MIN, self.BULLET_RADIUS_MAX)
        angles = (2 * np.pi * np.arange(n_way) / n_way) - np.pi / 2
        vec_n = np.column_stack((np.cos(angles), np.sin(angles)))
        pos = vec_n * radius + self.e_info[0]
        length = self._get_random_int(self.BULLET_SPEED_MIN, self.BULLET_SPEED_MAX)

        def gen_bullet(player_pos) -> list[list[np.ndarray, int, np.ndarray]]:
            # プレイヤーの方向に弾を発射する
            vec = vec_n * length
            bullets = [[pos[i].copy(), radius, vec[i]] for i in range(len(pos))]
            return bullets

        return gen_bullet, num_iter


class EnemyManager:
    N_WAY_MAX = 12
    # 1更新当たりの敵の生成確率
    PROBABILITY = 0.1

    def __init__(self, random_state: np.random.RandomState = None) -> None:
        self.random_state = random_state
        # pos, radius, vec
        self.bullet_list: list[list[np.ndarray, int, np.ndarray]] = []
        self.enemy_list: list[Enemy] = []

        self._start_gen_enemy()

    def _get_random_int(self, low: int, high: int) -> int:
        if self.random_state is not None:
            return self.random_state.randint(low, high)
        return np.random.randint(low, high)

    def _get_random_float(self) -> float:
        if self.random_state is not None:
            return self.random_state.rand()
        return np.random.rand()

    def _start_gen_enemy(self):
        """敵の初期生成
        """
        m_way = self._get_random_int(1, self.N_WAY_MAX)
        self.enemy_list.append(Enemy(n_way=m_way, random_state=self.random_state))
        return

    def update(self, player_pos: np.ndarray) -> tuple[list[Enemy], list[list[np.ndarray, int, np.ndarray]]]:
        """敵の更新

        :param player_pos: プレイヤーの位置
        :return: 敵のリスト、弾のリスト
        """
        tmp_enemy_list: list[Enemy] = []
        tmp_bullet_list: list[list[np.ndarray, int, np.ndarray]] = []

        for bullet in self.bullet_list:
            bullet[0] += bullet[2]
            if 0 < bullet[0][0] < WINDOW_SIZE[0] or 0 < bullet[0][1] < WINDOW_SIZE[1]:
                tmp_bullet_list.append(bullet)

        for enemy in self.enemy_list:
            check, bullets = enemy.update(player_pos=player_pos)
            if check:
                tmp_enemy_list.append(enemy)
                if bullets:
                    tmp_bullet_list.extend(bullets)

        if self._get_random_float() < self.PROBABILITY:
            n_way = self._get_random_int(1, self.N_WAY_MAX)
            tmp_enemy_list.append(Enemy(n_way=n_way, random_state=self.random_state))

        self.enemy_list = tmp_enemy_list
        self.bullet_list = tmp_bullet_list

        return self.enemy_list, self.bullet_list


class BulletEnv(AbsEnv):
    # 幅、高さの順
    MOVE_SIZE = tuple([i - 1 for i in WINDOW_SIZE[0: 2]])
    MOVE_SPEED = 5
    PLAYER_SIZE = 10
    ENEMY_SIZE = 10

    ACTION = (MOVE_SPEED * np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int8)).tolist()

    def __init__(self, render_mode="rgb_array"):
        self.render_mode = render_mode
        self.reset()
        return

    def reset(self, seed=None) -> np.ndarray:
        """環境のリセット

        :return: 初期画面
        """
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = None

        self.player_pos = np.array([WINDOW_SIZE[0] // 2, (9 * WINDOW_SIZE[1]) // 10], dtype=np.uint16)
        self.enemy_manager = EnemyManager(self.random_state)

        screen = self._draw(self.player_pos, self.enemy_manager.enemy_list, self.enemy_manager.bullet_list)
        return screen

    def step(self, action: int) -> tuple[np.ndarray, int, bool, dict]:
        """環境の更新

        :param action: エージェントの行動
        :return: 画面、報酬、終了フラグ、情報
        """
        action = self.ACTION[action]

        self.player_pos = self.player_pos + action
        check_player_pos = self._check_player_pos(self.player_pos)

        if not check_player_pos:
            self.player_pos = self.player_pos - action

        new_enemies_list, new_bullets = self.enemy_manager.update(self.player_pos)
        screen = self._draw(self.player_pos, new_enemies_list, new_bullets)
        check_collide = self._check_collision(screen=screen, player_position=self.player_pos)

        if check_collide:
            reward = -1
            done = True
        else:
            reward = 1
            done = False

        return self.observation(screen), reward, done, {}

    def observation(self, screen: np.ndarray) -> np.ndarray:
        """観測の取得

        :param screen: 画面
        :return: 画面
        """
        return screen

    def _draw(self, player_position: np.ndarray, enemies_list: list[Enemy], bullet_list: list[list[np.ndarray, int, np.ndarray]]) -> np.ndarray:
        """画面の描画

        :param screen: 画面
        :param player_position: プレイヤーの位置
        :param enemies_list: 敵のリスト
        :param bullet_list: 弾のリスト
        :return: 画面
        """
        screen = np.zeros(shape=(WINDOW_SIZE[1], WINDOW_SIZE[0]), dtype=np.uint8)
        cv2.rectangle(screen, (player_position[0], player_position[1], self.PLAYER_SIZE, self.PLAYER_SIZE), 255, thickness=-1)
        for enemy in enemies_list:
            cv2.fillConvexPoly(screen, self._calc_triangle(enemy.e_info[0]), 255)
        for bullet in bullet_list:
            cv2.circle(screen, tuple(bullet[0].astype(np.int16)), bullet[1], 255, thickness=-1)
        return screen

    def render(self) -> np.ndarray:
        """画面の表示

        :return: 画面, None
        """
        screen = self._draw(self.player_pos, self.enemy_manager.enemy_list, self.enemy_manager.bullet_list)
        if self.render_mode == "human":
            cv2.imshow("screen", screen)
            cv2.waitKey(1)
            return
        elif self.render_mode == "rgb_array":
            return screen
        return

    def _check_player_pos(self, player_position: np.ndarray) -> bool:
        """プレイヤーの位置のチェック

        :param player_position: プレイヤーの位置
        :return: プレイヤーが画面内にいるかどうか
        """
        return not (np.any(player_position <= 0) or np.any(player_position + self.PLAYER_SIZE >= self.MOVE_SIZE))

    def _check_collision(self, screen: np.ndarray, player_position: np.ndarray) -> bool:
        """プレイヤーと敵の衝突判定

        :param screen: 画面
        :param player_position: プレイヤーの位置
        :return: 衝突しているかどうか
        """
        player_coods = [player_position[0], player_position[1], player_position[0] + self.PLAYER_SIZE, player_position[1] + self.PLAYER_SIZE]

        check_screen = screen[player_coods[1] - 1: player_coods[3] + 1, player_coods[0] - 1: player_coods[2] + 1]
        if int(np.sum(check_screen) / 255) > self.PLAYER_SIZE ** 2:
            return True
        else:
            return False

    def _calc_triangle(self, center: np.ndarray) -> np.ndarray:
        """正三角形の座標の計算

        :param center: 重心の座標
        :return: 正三角形の座標
        """
        height = int(np.sqrt(3) / 2 * self.ENEMY_SIZE)
        pt1 = (center[0], center[1] - height // 2)
        pt2 = (center[0] - self.ENEMY_SIZE // 2, center[1] + height // 2)
        pt3 = (center[0] + self.ENEMY_SIZE // 2, center[1] + height // 2)

        return np.array([pt1, pt2, pt3], np.int32)

    def close(self):
        cv2.destroyAllWindows()
        return
