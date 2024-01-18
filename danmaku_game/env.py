import cv2
import numpy as np
import datetime
# 画面サイズ
WINDOW_SIZE = (383, 423)


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

    def __init__(self, n_way) -> None:
        # 射出方向の数
        self.n_way = n_way
        self.cnt: int = 0
        # 奇数弾は1, 偶数弾は0
        self.e_type = 1 if n_way % 2 == 1 else 0
        self.stay_flag = 0
        # pos, vec, timing, posは重心の座標
        self.e_info: list[np.ndarray, np.ndarray, int] = self.gen_enemy()
        # bulletの生成関数
        self.b_gen_func = None

    def gen_enemy(self):
        sgn = np.random.randint(0, 2)
        pos = np.array([sgn * WINDOW_SIZE[0], np.random.randint(0, WINDOW_SIZE[1] // 10)], dtype=np.float32)
        vec = np.array([np.sign(-sgn + 0.5) * self.ENEMY_SPEED, 0], dtype=np.float32)
        timing = np.random.randint(self.ENEMY_SHOOT_TIMING_DELAY, 3 * self.ENEMY_SHOOT_TIMING_DELAY)
        return [pos, vec, timing]

    def update(self, player_pos) -> tuple[bool, list[list[np.ndarray, int, np.ndarray]]]:
        # timingで弾を生成する
        bullets = None
        if self.cnt == self.e_info[2]:
            if self.e_type == 0:
                self.b_gen_func, num_iter = self.gen_bullet_scatter(self.n_way)
            elif self.e_type == 1:
                self.b_gen_func, num_iter = self.gen_bullet_aim(self.n_way)
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

    def gen_bullet_aim(self, n_way):
        num_iter = np.random.randint(1, self.BULLET_NUM)
        radius = np.random.randint(self.BULLET_RADIUS_MIN, self.BULLET_RADIUS_MAX)
        angles = (2 * np.pi * np.arange(n_way) / n_way) - np.pi / 2
        pos = np.column_stack((np.cos(angles), np.sin(angles))) * radius + self.e_info[0]
        length = np.random.randint(self.BULLET_SPEED_MIN, self.BULLET_SPEED_MAX)

        def gen_bullet(player_pos):
            # 初期位置をradius分だけずらしてプレイヤーの方向に弾を発射する player_posは重心の座標に設定する
            # player_pos: shape(2,)
            vec = player_pos - pos
            vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
            vec = vec * length
            # pos, radius, vec
            bullets = [[pos[i].copy(), radius, vec[i]] for i in range(len(pos))]
            return bullets

        return gen_bullet, num_iter

    def gen_bullet_scatter(self, n_way):
        num_iter = np.random.randint(1, self.BULLET_NUM)
        radius = np.random.randint(self.BULLET_RADIUS_MIN, self.BULLET_RADIUS_MAX)
        angles = (2 * np.pi * np.arange(n_way) / n_way) - np.pi / 2
        vec_n = np.column_stack((np.cos(angles), np.sin(angles)))
        pos = vec_n * radius + self.e_info[0]
        length = np.random.randint(self.BULLET_SPEED_MIN, self.BULLET_SPEED_MAX)

        def gen_bullet(player_pos):
            # プレイヤーの方向に弾を発射する
            vec = vec_n * length
            bullets = [[pos[i].copy(), radius, vec[i]] for i in range(len(pos))]
            return bullets

        return gen_bullet, num_iter


class EnemyManager:
    # 最大値は11
    N_WAY_MAX = 12
    PROBABILITY = 0.1

    def __init__(self) -> None:
        # pos, radius, vec
        self.bullet_list: list[list[np.ndarray, int, np.ndarray]] = []
        self.enemy_list: list[Enemy] = []

        # 確認用
        # self.stats = {"vect": [], "angle": []}
        self.start_gen_enemy()

    def start_gen_enemy(self):
        m_way = np.random.randint(1, self.N_WAY_MAX)
        self.enemy_list.append(Enemy(n_way=m_way))
        return

    def update(self, player_pos):
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
        # 要検証
        if np.random.rand() < self.PROBABILITY:
            n_way = np.random.randint(1, self.N_WAY_MAX)
            tmp_enemy_list.append(Enemy(n_way=n_way))

        self.enemy_list = tmp_enemy_list
        self.bullet_list = tmp_bullet_list

        return self.enemy_list, self.bullet_list


class MyEnvManual:
    # 幅、高さの順
    MOVE_SIZE = tuple([i - 1 for i in WINDOW_SIZE[0: 2]])
    # MOVE_SPEED = 7
    SLOW_MOVE_SPEED = 5
    PLAYER_SIZE = 10
    ENEMY_SIZE = 10

    # BASE_ACTION = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [0, 0]], dtype=np.int8)
    # ACTION = (MOVE_SPEED * BASE_ACTION).tolist() + (SLOW_MOVE_SPEED * BASE_ACTION).tolist()
    # ACTION = (SLOW_MOVE_SPEED * BASE_ACTION).tolist()
    ACTION = (SLOW_MOVE_SPEED * np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int8)).tolist()

    def __init__(self, render_mode="rgb_array"):
        """
        observation_space: エージェントが観測する環境の状態の値の範囲
        action_space: エージェントが取る行動の値の範囲

        player_position: プレイヤーの位置、rectの左上の座標
        bullet: 弾の情報

        """
        self.render_mode = render_mode
        self.tmp_step = 0
        self.reset()

    def reset(self):
        # playerは10*10の正方形
        self.tmp_step = 0
        screen = np.zeros(shape=(WINDOW_SIZE[1], WINDOW_SIZE[0]), dtype=np.uint8)
        self.player_pos = np.array([WINDOW_SIZE[0] // 2, (9 * WINDOW_SIZE[1]) // 10], dtype=np.uint16)
        self.enemy_manager = EnemyManager()
        self.draw(screen, self.player_pos, self.enemy_manager.enemy_list, self.enemy_manager.bullet_list)

        return screen

    def step(self, action_index):
        self.tmp_step += 1
        screen = np.zeros(shape=(WINDOW_SIZE[1], WINDOW_SIZE[0]), dtype=np.uint8)

        action = self.ACTION[action_index]

        self.player_pos = self.player_pos + action

        check_player_pos = self.check_player_pos(self.player_pos)

        if not check_player_pos:
            self.player_pos = self.player_pos - action

        new_enemies_list, new_bullets = self.enemy_manager.update(self.player_pos)
        self.draw(screen, self.player_pos, new_enemies_list, new_bullets)
        check_collide = self.check_collision(screen=screen, player_position=self.player_pos)

        if check_collide:
            reward = -1
            done = True
        else:
            reward = 1
            done = False

        if self.tmp_step > 3000:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("truncated.txt", "a") as f:
                f.write(f"3000step, {time}\n")

        return self.observation(screen), reward, done, {}

    def observation(self, screen):
        return screen

    def draw(self, screen, player_position: np.ndarray, enemies_list: list[Enemy], bullet_list: list[list[np.ndarray, int, np.ndarray]]):
        cv2.rectangle(screen, (player_position[0], player_position[1], self.PLAYER_SIZE, self.PLAYER_SIZE), 255, thickness=-1)
        for enemy in enemies_list:
            cv2.fillConvexPoly(screen, self.calc_triangle(enemy.e_info[0]), 255)
        for bullet in bullet_list:
            cv2.circle(screen, tuple(bullet[0].astype(np.int16)), bullet[1], 255, thickness=-1)
        return screen

    def render(self):
        screen = np.zeros(shape=(WINDOW_SIZE[1], WINDOW_SIZE[0]), dtype=np.uint8)

        self.draw(screen, self.player_pos, self.enemy_manager.enemy_list, self.enemy_manager.bullet_list)
        if self.render_mode == "human":
            cv2.imshow("screen", screen)
            cv2.waitKey(1)
            return
        elif self.render_mode == "rgb_array":
            return screen
        return

    def check_player_pos(self, player_position: np.ndarray):
        return not (np.any(player_position <= 0) or np.any(player_position + self.PLAYER_SIZE >= self.MOVE_SIZE))

    def check_collision(self, screen, player_position):
        player_coods = [player_position[0], player_position[1], player_position[0] + self.PLAYER_SIZE, player_position[1] + self.PLAYER_SIZE]

        check_screen = screen[player_coods[1] - 1: player_coods[3] + 1, player_coods[0] - 1: player_coods[2] + 1]
        if int(np.sum(check_screen) / 255) > self.PLAYER_SIZE ** 2:
            return True
        else:
            return False

    def calc_triangle(self, center):
        height = int(np.sqrt(3) / 2 * self.ENEMY_SIZE)
        pt1 = (center[0], center[1] - height // 2)
        pt2 = (center[0] - self.ENEMY_SIZE // 2, center[1] + height // 2)
        pt3 = (center[0] + self.ENEMY_SIZE // 2, center[1] + height // 2)

        # 正三角形を描画
        return np.array([pt1, pt2, pt3], np.int32)

    def close(self):
        # print(np.mean(np.array(self.bullet.stats["vect"]), axis=0), np.mean(np.array(self.bullet.stats["angle"])))
        cv2.destroyAllWindows()
        return
