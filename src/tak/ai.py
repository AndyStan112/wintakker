import math
from tak.core import (
    WHITE,
    BLACK,
    FLAT,
    STANDING,
    CAPSTONE,
    Result,
    IllegalMove,
    piece_player,
    piece_kind,
)


class TakWeights:
    def __init__(self, **kwargs):
        self.tempo = kwargs.get("tempo", 5)
        self.top_flat = kwargs.get("top_flat", 40)
        self.standing = kwargs.get("standing", 20)
        self.capstone = kwargs.get("capstone", 30)
        self.center = kwargs.get("center", 4)
        self.stack_captive = kwargs.get("stack_captive", 5)
        self.cap_mobility = kwargs.get("cap_mobility", 1)
        self.road_potential = kwargs.get("road_potential", 10)
        self.flat_control = kwargs.get("flat_control", 3)

    def as_dict(self):
        return {
            "tempo": self.tempo,
            "top_flat": self.top_flat,
            "standing": self.standing,
            "capstone": self.capstone,
            "center": self.center,
            "stack_captive": self.stack_captive,
            "cap_mobility": self.cap_mobility,
            "road_potential": self.road_potential,
            "flat_control": self.flat_control,
        }


class TakFeatures:
    KEYS = (
        "tempo",
        "top_flat",
        "standing",
        "capstone",
        "center",
        "stack_captive",
        "cap_mobility",
        "road_potential",
        "flat_control",
    )

    def extract(self, state, player):
        opp = 1 - player
        n = state.size
        center_min = 1
        center_max = n - 2

        f = dict.fromkeys(self.KEYS, 0)
        f["tempo"] = 1

        for y in range(n):
            for x in range(n):
                st = state.board[y][x]
                if not st:
                    continue

                top = st[-1]
                owner = piece_player(top)
                sign = 1 if owner == player else -1
                kind = piece_kind(top)

                if kind == FLAT:
                    f["top_flat"] += sign
                elif kind == STANDING:
                    f["standing"] += sign
                elif kind == CAPSTONE:
                    f["capstone"] += sign

                if center_min <= x <= center_max and center_min <= y <= center_max:
                    f["center"] += sign

                if len(st) > 1:
                    f["stack_captive"] += sign * (len(st) - 1)

                if kind == CAPSTONE:
                    f["cap_mobility"] += sign * self._cap_mobility(state, x, y, len(st))

        f["road_potential"] = self._road_count(state, player) - self._road_count(
            state, opp
        )

        f["flat_control"] = self._flat_control(state, player) - self._flat_control(
            state, opp
        )

        return f

    def as_vector(self, features):
        return [features[k] for k in self.KEYS]

    def _cap_mobility(self, state, x, y, height):
        count = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cx, cy = x, y
            for _ in range(height):
                cx += dx
                cy += dy
                if not state.inside(cx, cy):
                    break
                dst = state.board[cy][cx]
                if dst and piece_kind(dst[-1]) in (STANDING, CAPSTONE):
                    break
                count += 1
        return count

    def _road_count(self, state, player):
        n = state.size
        cnt = 0
        for y in range(n):
            for x in range(n):
                st = state.board[y][x]
                if not st:
                    continue
                top = st[-1]
                if piece_player(top) == player and piece_kind(top) in (FLAT, CAPSTONE):
                    cnt += 1
        return cnt

    def _flat_control(self, state, player):
        n = state.size
        ctrl = 0
        for y in range(n):
            for x in range(n):
                if state.board[y][x]:
                    continue
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if not state.inside(nx, ny):
                        continue
                    st = state.board[ny][nx]
                    if not st:
                        continue
                    top = st[-1]
                    if piece_player(top) == player and piece_kind(top) == FLAT:
                        ctrl += 1
                        break
        return ctrl


class TakEvaluator:
    def __init__(self, weights=None, features=None):
        self.w = weights or TakWeights()
        self.f = features or TakFeatures()

    def evaluate(self, state, player):
        feats = self.f.extract(state, player)
        return sum(getattr(self.w, k) * feats[k] for k in feats)


class TakMoveGenerator:
    def generate_moves(self, state):
        n = state.size
        player = state.to_move
        moves = []

        for y in range(n):
            for x in range(n):
                if state.board[y][x]:
                    continue
                opening = state.move_number < 2

                if opening:
                    moves.append(("place", x, y, FLAT))
                else:
                    if state.stones_remaining[player] > 0:
                        moves.append(("place", x, y, FLAT))
                        moves.append(("place", x, y, STANDING))
                    if state.caps_remaining[player] > 0:
                        moves.append(("place", x, y, CAPSTONE))

        for y in range(n):
            for x in range(n):
                stack = state.board[y][x]
                if not stack:
                    continue
                if piece_player(stack[-1]) != player:
                    continue

                height = len(stack)
                max_carry = min(height, n)

                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    for carry in range(1, max_carry + 1):
                        for drops in self._drop_partitions(carry, n):
                            moves.append(("move", x, y, dx, dy, carry, drops))

        return moves

    def _drop_partitions(self, stones, max_len):
        """
        Generate all legal drop sequences:
        - sum(drops) == stones
        - each drop >= 1
        - len(drops) <= board size
        """
        result = []

        def dfs(remaining, path):
            if remaining == 0:
                result.append(path)
                return
            if len(path) == max_len:
                return
            for i in range(1, remaining + 1):
                dfs(remaining - i, path + [i])

        dfs(stones, [])
        return result


class TakSearch:
    def __init__(self, evaluator=None, movegen=None, max_depth=2):
        self.evaluator = evaluator or TakEvaluator()
        self.movegen = movegen or TakMoveGenerator()
        self.max_depth = max_depth

    def choose_move(self, state):
        best_move = None
        best_score = -math.inf
        player = state.to_move

        for move in self.movegen.generate_moves(state):
            try:
                new_state = state.clone()
                result = self._apply(new_state, move)

                score = self._minimax(
                    new_state, self.max_depth - 1, -math.inf, math.inf, False, player
                )
            except IllegalMove:
                continue

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _minimax(self, state, depth, alpha, beta, maximizing, player):
        result = state.compute_result(1 - state.to_move)
        if result != Result.ONGOING:
            if result == Result.DRAW:
                return 0
            return (
                1e6
                if (result == Result.WHITE_WIN and player == WHITE)
                or (result == Result.BLACK_WIN and player == BLACK)
                else -1e6
            )

        if depth == 0:
            return self.evaluator.evaluate(state, player)

        moves = self.movegen.generate_moves(state)

        if maximizing:
            best = -math.inf
            for m in moves:
                try:
                    ns = state.clone()
                    self._apply(ns, m)
                except IllegalMove:
                    continue
                score = self._minimax(ns, depth - 1, alpha, beta, False, player)
                best = max(best, score)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best

        else:
            best = math.inf
            for m in moves:
                try:
                    ns = state.clone()
                    self._apply(ns, m)
                except IllegalMove:
                    continue
                score = self._minimax(ns, depth - 1, alpha, beta, True, player)
                best = min(best, score)
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return best

    def _apply(self, state, move):
        if move[0] == "place":
            _, x, y, k = move
            return state.place(x, y, k)
        else:
            _, x, y, dx, dy, cnt, drops = move
            return state.move_stack(x, y, dx, dy, cnt, drops)
