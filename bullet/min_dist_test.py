import pathlib

import numpy as np
import imageio
import tqdm
from env import AbsEnv, BulletEnv
from agent import AbsAgent


class MinDistWrapper(AbsEnv):
    def __init__(self, env: BulletEnv):
        self.env = env
        self.min_dist = []
        self.move_sum = 0

    def reset(self, seed=None):
        obs = self.env.reset(seed)
        self.min_dist = []
        self.move_sum = 0
        return self.observation(obs)

    def observation(self, observation):
        return observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        p_pos = self.env.player_pos.copy()
        b_pos = [bullet[0] for bullet in self.env.enemy_manager.bullet_list]
        if len(b_pos) != 0:
            dist = np.linalg.norm(np.array(b_pos) - p_pos, axis=1)
            self.min_dist.append(min(dist))
        player_pos = p_pos + self.env.ACTION[action]
        check = self.env._check_player_pos(player_pos)
        if check and (action != 0):
            self.move_sum += 1
        if done:
            info["min_dist"] = self.min_dist
            info["epi_move"] = self.move_sum
        obs = self.observation(obs)
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class Tester:
    def __init__(self, env: BulletEnv, agent: AbsAgent):
        self.env = env
        self.agent = agent

    def test(self, model_path: pathlib.Path, log_path: pathlib.Path, num_episodes=500, init_seed=0):
        assert self.env.render_mode == "rgb_array", "render_mode must be 'rgb_array'"
        assert log_path.exists(), f"{log_path} does not exist"

        self.agent.load(model_path)
        self.agent.eval_mode()

        frames = []
        log_txt = ""

        total_rewards = []
        min_dists = 0
        cnt = 0
        move_list = []
        tmp_reward = 0
        episode = 0
        seed = init_seed
        bar = tqdm.tqdm(total=num_episodes)

        obs = self.env.reset(seed=seed)
        while episode < num_episodes:
            if episode < 20:
                frames.append(self.env.render())
            action = self.agent.get_action(obs, np.inf)
            obs, reward, done, info = self.env.step(action)
            tmp_reward += reward
            if done:
                log_txt += f"episode: {episode}, R: {tmp_reward}\n"
                for k, v in info.items():
                    if k == "min_dist":
                        for d in v:
                            min_dists = (cnt / (cnt + 1)) * min_dists + (d / (cnt + 1))
                            cnt += 1
                        log_txt += f"{k}: {np.mean(v)}, "
                    elif k == "epi_move":
                        log_txt += f"{k}: {v}, "
                        move_list.append(v)
                log_txt += "\n"
                total_rewards.append(tmp_reward)
                bar.update(1)
                tmp_reward = 0
                episode += 1
                seed += 1
                obs = self.env.reset(seed=seed)

        bar.close()
        imageio.mimsave(log_path / "test.gif", frames, fps=30)

        eval_stats = dict(
            mean=np.mean(total_rewards),
            median=np.median(total_rewards),
            max=np.max(total_rewards),
            min=np.min(total_rewards),
            stdev=np.std(total_rewards) if len(total_rewards) > 1 else 0.0,
        )
        log_txt += f"mean: {eval_stats['mean']}, median: {eval_stats['median']}, max: {eval_stats['max']}, min: {eval_stats['min']}, stdev: {eval_stats['stdev']}, min_dist: {min_dists}, move: {np.mean(move_list)}\n"
        with open(log_path / "test.txt", "w") as f:
            f.write(log_txt)
        return
