import torch as th
from torch import nn


class DQNnet4ch(nn.Module):
    def __init__(self, output_dim: int = 5):
        super(DQNnet4ch, self).__init__()
        # 入力は84 * 84 * 4

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        self.net.apply(self._init_weights)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            nn.init.constant_(module.bias, 0)


class DQNnet1ch(nn.Module):
    def __init__(self, output_dim: int = 5):
        super(DQNnet1ch, self).__init__()
        # 入力は84 * 84 * 4

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        self.net.apply(self._init_weights)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            nn.init.constant_(module.bias, 0)


class DQNnet2ch(nn.Module):
    def __init__(self, output_dim: int = 5):
        super(DQNnet2ch, self).__init__()
        # 入力は84 * 84 * 4

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        self.net.apply(self._init_weights)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
