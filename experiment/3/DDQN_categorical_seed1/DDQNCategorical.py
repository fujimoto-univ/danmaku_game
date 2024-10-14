import numpy as np
import torch as th
import torch.nn as nn
from pfrl import agents, explorers
from pfrl.replay_buffers.replay_buffer import ReplayBuffer
from bullet.rainbow_components import RainbowAgent


class DDQNCategorical(RainbowAgent):
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
