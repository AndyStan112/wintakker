import sqlite3
import random
import torch
from torch.utils.data import IterableDataset
from tak.core import TakState, Result, WHITE, BLACK, IllegalMove
from tak.encoder import (
    split_playtak_notation,
    parse_playtak_action,
    encode_state_value_input,
)


def _result_to_value(result, player):
    if result == Result.DRAW:
        return 0.0
    if (result == Result.WHITE_WIN and player == WHITE) or (
        result == Result.BLACK_WIN and player == BLACK
    ):
        return 1.0
    return -1.0


def _db_result_to_result_str(s):
    if s is None:
        return None
    return str(s).strip().upper()


def _parse_game_result(result_str):
    if not result_str:
        return None
    r = result_str.upper()
    if r.startswith("R-0") or r.startswith("1-0"):
        return Result.WHITE_WIN
    if r.startswith("0-R") or r.startswith("0-1"):
        return Result.BLACK_WIN
    if r.startswith("1/2") or "DRAW" in r:
        return Result.DRAW
    if r.startswith("F-0"):
        return Result.WHITE_WIN
    if r.startswith("0-F"):
        return Result.BLACK_WIN
    return None


class TakValueStream(IterableDataset):
    def __init__(
        self,
        db_path,
        sizes=None,
        sample_per_game=24,
        seed=0,
        resume_game_id=None,
        resume_within=0,
    ):
        super().__init__()
        self.db_path = db_path
        self.sizes = set(sizes) if sizes else None
        self.sample_per_game = int(sample_per_game)
        self.seed = int(seed)

        self.resume_game_id = resume_game_id
        self.resume_within = int(resume_within)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0
        num_workers = 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        q = "SELECT id, size, notation, result FROM games"
        args = []
        where_clauses = []
        if self.sizes:
            qs = ",".join(["?"] * len(self.sizes))
            where_clauses.append(f"size IN ({qs})")
            args.extend(list(sorted(self.sizes)))

        if self.resume_game_id is not None:
            where_clauses.append("id >= ?")
            args.append(int(self.resume_game_id))

        if where_clauses:
            q += " WHERE " + " AND ".join(where_clauses)

        q += " ORDER BY id"

        for row in cur.execute(q, args):
            row_id = int(row["id"])

            if num_workers > 1 and (row_id % num_workers) != worker_id:
                continue

            size = int(row["size"])
            notation = row["notation"]
            result_str = _db_result_to_result_str(row["result"])
            final = _parse_game_result(result_str)
            if final is None:
                continue

            try:
                state = TakState.from_size(size)
            except Exception:
                continue

            moves = split_playtak_notation(notation)
            if not moves:
                continue

            positions = []
            players = []

            ok = True
            for mv in moves:
                try:
                    action = parse_playtak_action(mv)
                    if action is None:
                        continue
                    positions.append(state.clone())
                    players.append(state.to_move)
                    if action[0] == "place":
                        _, x, y, k = action
                        state.place(x, y, k)
                    else:
                        _, x, y, dx, dy, cnt, drops = action
                        state.move_stack(x, y, dx, dy, cnt, drops)
                except (IllegalMove, ValueError):
                    ok = False
                    break

            if not ok or not positions:
                continue

            k = min(self.sample_per_game, len(positions))

            per_game_rng = random.Random(self.seed + row_id)
            idxs = list(range(len(positions)))
            per_game_rng.shuffle(idxs)
            idxs = idxs[:k]

            if self.resume_game_id is not None and row_id == int(self.resume_game_id):
                if self.resume_within and self.resume_within > 0:
                    idxs = idxs[self.resume_within :]

            for sample_idx, i in enumerate(idxs):
                s = positions[i]
                p = players[i]
                x = encode_state_value_input(s, p)
                y = torch.tensor([_result_to_value(final, p)], dtype=torch.float32)

                yield x, y, row_id, int(sample_idx)

        con.close()
