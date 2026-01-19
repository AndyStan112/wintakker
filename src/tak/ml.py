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
        result = state.compute_result(state.to_move)
        if result != Result.ONGOING:
            if result == Result.DRAW:
                return 0
            if (result == Result.WHITE_WIN and player == WHITE) or (
                result == Result.BLACK_WIN and player == BLACK
            ):
                return 1e6
            return -1e6
        return self.net.predict(state, player)


import math
import random
from tak.core import IllegalMove, Result


class TakMCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

        self.children = {}
        self.priors = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0

        self.untried_moves = None

    def expand(self, move, next_state):
        child = TakMCTSNode(next_state, parent=self)
        self.children[move] = child
        return child

    def update(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N


class TakMCTS:
    def __init__(self, evaluator, movegen, simulations=800, c_puct=1.5):
        self.evaluator = evaluator
        self.movegen = movegen
        self.simulations = simulations
        self.c_puct = c_puct

    def choose_move(self, root_state):
        root = TakMCTSNode(root_state.clone())

        root.untried_moves = list(self.movegen.generate_moves(root_state))

        for _ in range(self.simulations):
            node = root
            state = root_state.clone()

            while node.untried_moves == [] and len(node.children) > 0:
                node = self._select_child(node)
                self._apply_move(state, node.parent_move)

            if node.untried_moves:
                move = node.untried_moves.pop()
                next_state = state.clone()
                self._apply_move(next_state, move)
                child = node.expand(move, next_state)
                child.parent_move = move
                child.untried_moves = list(self.movegen.generate_moves(next_state))
                node = child

            value = self._evaluate(state, node.state)

            while node is not None:
                node.update(value if state.to_move != node.state.to_move else -value)
                node = node.parent

        best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        return best_move

    def _select_child(self, node):
        best_score = -float("inf")
        best_child = None

        for move, child in node.children.items():
            U = self.c_puct * math.sqrt(node.N) / (1 + child.N)
            score = child.Q + U
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _apply_move(self, state, move):
        if move[0] == "place":
            _, x, y, k = move
            return state.place(x, y, k)
        else:
            _, x, y, dx, dy, cnt, drops = move
            return state.move_stack(x, y, dx, dy, cnt, drops)

    def _evaluate(self, original_state, simulated_state):
        result = simulated_state.compute_result(original_state.to_move)

        if result != Result.ONGOING:
            if result == Result.DRAW:
                return 0.0
            if result == Result.WHITE_WIN:
                return 1.0 if original_state.to_move == 1 else -1.0
            if result == Result.BLACK_WIN:
                return 1.0 if original_state.to_move == 0 else -1.0

        return float(self.evaluator.evaluate(simulated_state, original_state.to_move))
