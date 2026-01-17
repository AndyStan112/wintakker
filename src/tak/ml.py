import random
from tak.core import WHITE, BLACK, Result

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self, in_planes=10):
        super().__init__()
        self.c1 = nn.Conv2d(in_planes, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.h1 = nn.Linear(64, 128)
        self.h2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = x.mean(dim=(2, 3))
        x = F.relu(self.h1(x))
        x = torch.tanh(self.h2(x))
        return x


class MockValueNet:
    def predict(self, state, player):
        return random.uniform(-0.2, 0.2)


class TakMLEvaluator:
    def __init__(self, net=None):
        self.net = net or MockValueNet()

    def evaluate(self, state, player):
        result = state.compute_result(1 - state.to_move)
        if result != Result.ONGOING:
            if result == Result.DRAW:
                return 0
            if (result == Result.WHITE_WIN and player == WHITE) or (
                result == Result.BLACK_WIN and player == BLACK
            ):
                return 1e6
            return -1e6
        return self.net.predict(state, player)
