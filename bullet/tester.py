import pathlib

import numpy as np
import imageio
import tqdm

from env import BulletEnv
from agent import AbsAgent


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
        tmp_reward = 0
        episode = 0
        seed = init_seed
        bar = tqdm.tqdm(total=num_episodes)

        obs = self.env.reset(seed=seed)
        while episode < num_episodes:
            if episode < 20:
                frames.append(self.env.render())
            action = self.agent.get_action(obs, np.inf)
            obs, reward, done, _ = self.env.step(action)
            tmp_reward += reward
            if done:
                log_txt += f"episode: {episode}, R: {tmp_reward}\n"
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
        log_txt += f"mean: {eval_stats['mean']}, median: {eval_stats['median']}, max: {eval_stats['max']}, min: {eval_stats['min']}, stdev: {eval_stats['stdev']}\n"
        with open(log_path / "test.txt", "w") as f:
            f.write(log_txt)
        return
