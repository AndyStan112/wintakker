#!/usr/bin/env python3
"""Preprocess games.db into sharded memmap dataset for fast training.

Writes per-shard binary memmap files and a meta.json describing shards.

Usage: uv run scripts/preprocess_dataset.py --db games.db --out-dir preproc --shards 8
"""
import argparse
import os
import json
import sqlite3
import random
import math
import multiprocessing as mp
import numpy as np

from tak.encoder import (
    split_playtak_notation,
    parse_playtak_action,
    encode_state_value_input,
)
from tak.core import TakState, IllegalMove, WHITE, BLACK, Result


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


def _result_to_value(result, player):
    if result == Result.DRAW:
        return 0.0
    if (result == Result.WHITE_WIN and player == WHITE) or (
        result == Result.BLACK_WIN and player == BLACK
    ):
        return 1.0
    return -1.0


def count_samples(db_path, sizes, samples_per_game):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    q = "SELECT id, size, notation, result FROM games"
    args = []
    if sizes:
        qs = ",".join(["?"] * len(sizes))
        q += f" WHERE size IN ({qs})"
        args = list(sorted(sizes))
    total = 0
    for row in cur.execute(q, args):
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
        total += min(samples_per_game, len(positions))
    con.close()
    return total


def count_samples_by_worker(db_path, workers, samples_per_game):
    counts = [0] * workers
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    q = "SELECT id, size, notation, result FROM games ORDER BY id"
    for row in cur.execute(q, []):
        row_id = int(row["id"])
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
        k = min(samples_per_game, len(positions))
        counts[row_id % workers] += k
    con.close()
    return counts


def preprocess(db_path, out_dir, shards, samples_per_game, seed, dtype):
    os.makedirs(out_dir, exist_ok=True)
    sizes = None
    print("counting total samples...")
    total = count_samples(db_path, sizes, samples_per_game)
    print(f"total samples to write: {total}")

    if shards <= 1:
        shard_sizes = [total]
    else:
        base = total // shards
        rem = total % shards
        shard_sizes = [base + (1 if i < rem else 0) for i in range(shards)]

    dtype_np = np.float16 if dtype == "float16" else np.float32

    shard_files = []
    X_maps = []
    Y_maps = []
    G_maps = []
    GI_maps = []
    for i, sz in enumerate(shard_sizes):
        xf = os.path.join(out_dir, f"X_{i}.dat")
        yf = os.path.join(out_dir, f"Y_{i}.dat")
        gf = os.path.join(out_dir, f"GID_{i}.dat")
        gif = os.path.join(out_dir, f"GIDX_{i}.dat")
        X = np.memmap(xf, dtype=dtype_np, mode="w+", shape=(sz, 10, 8, 8))
        Y = np.memmap(yf, dtype=np.float32, mode="w+", shape=(sz,))
        G = np.memmap(gf, dtype=np.int32, mode="w+", shape=(sz,))
        GI = np.memmap(gif, dtype=np.int16, mode="w+", shape=(sz,))
        shard_files.append({"X": xf, "Y": yf, "G": gf, "GI": gif, "size": sz})
        X_maps.append(X)
        Y_maps.append(Y)
        G_maps.append(G)
        GI_maps.append(GI)

    workers = getattr(preprocess, "workers", 1)
    if workers is None:
        workers = 1

    def worker_fill(worker_id, worker_count, xfile, yfile, gfile, gifile, size):

        X_w = np.memmap(xfile, dtype=dtype_np, mode="r+", shape=(size, 10, 8, 8))
        Y_w = np.memmap(yfile, dtype=np.float32, mode="r+", shape=(size,))
        G_w = np.memmap(gfile, dtype=np.int32, mode="r+", shape=(size,))
        GI_w = np.memmap(gifile, dtype=np.int16, mode="r+", shape=(size,))
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        q = f"SELECT id, size, notation, result FROM games WHERE (id % ?) = ? ORDER BY id"
        write_idx = 0
        for row in cur.execute(q, (worker_count, worker_id)):
            row_id = int(row["id"])
            size_r = int(row["size"])
            notation = row["notation"]
            result_str = _db_result_to_result_str(row["result"])
            final = _parse_game_result(result_str)
            if final is None:
                continue
            try:
                state = TakState.from_size(size_r)
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
            k = min(samples_per_game, len(positions))
            per_game_rng = random.Random(seed + row_id)
            idxs = list(range(len(positions)))
            per_game_rng.shuffle(idxs)
            idxs = idxs[:k]
            for sample_idx, i in enumerate(idxs):
                if write_idx >= size:
                    break
                x = encode_state_value_input(
                    positions[i], players[i], pad=8, device="cpu"
                ).numpy()
                y = float(_result_to_value(final, players[i]))
                X_w[write_idx] = x.astype(dtype_np)
                Y_w[write_idx] = y
                G_w[write_idx] = row_id
                GI_w[write_idx] = sample_idx
                write_idx += 1
        con.close()
        X_w.flush()
        Y_w.flush()
        G_w.flush()
        GI_w.flush()

    if getattr(preprocess, "workers", 1) and preprocess.workers > 1:
        procs = []
        for i, s in enumerate(shard_files):
            sz = int(s["size"])
            if sz == 0:
                continue
            p = mp.Process(
                target=worker_fill,
                args=(i, preprocess.workers, s["X"], s["Y"], s["G"], s["GI"], sz),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        q = "SELECT id, size, notation, result FROM games ORDER BY id"
        args = []
        write_idx = 0
        shard_idx = 0
        offset_in_shard = 0
        rng = random.Random(seed)
        for row in cur.execute(q, args):
            row_id = int(row["id"])
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

            k = min(samples_per_game, len(positions))
            per_game_rng = random.Random(seed + row_id)
            idxs = list(range(len(positions)))
            per_game_rng.shuffle(idxs)
            idxs = idxs[:k]

            for sample_idx, i in enumerate(idxs):
                if write_idx >= total:
                    break
                while (
                    shard_idx < len(shard_files)
                    and offset_in_shard >= shard_files[shard_idx]["size"]
                ):
                    shard_idx += 1
                    offset_in_shard = 0
                if shard_idx >= len(shard_files):
                    raise RuntimeError("ran out of shard space")

                x = encode_state_value_input(
                    positions[i], players[i], pad=8, device="cpu"
                ).numpy()
                y = float(_result_to_value(final, players[i]))
                X_maps[shard_idx][offset_in_shard] = x.astype(dtype_np)
                Y_maps[shard_idx][offset_in_shard] = y
                G_maps[shard_idx][offset_in_shard] = row_id
                GI_maps[shard_idx][offset_in_shard] = sample_idx
                write_idx += 1
                offset_in_shard += 1
                if write_idx % 10000 == 0:
                    print(f"written {write_idx}/{total} samples")

        con.close()
        for m in X_maps + Y_maps + G_maps + GI_maps:
            m.flush()

    meta = {
        "total_samples": total,
        "dtype": "float16" if dtype == "float16" else "float32",
        "shape": [10, 8, 8],
        "samples_per_game": samples_per_game,
        "seed": seed,
        "shards": shard_files,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("preprocessing done, wrote", total, "samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of parallel worker processes to use (each worker writes one shard)",
    )
    ap.add_argument("--samples-per-game", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    args = ap.parse_args()

    preprocess.workers = max(1, int(args.workers))
    shards = args.shards
    if preprocess.workers > 1 and shards == 1:
        shards = preprocess.workers

    preprocess(
        args.db, args.out_dir, shards, args.samples_per_game, args.seed, args.dtype
    )


if __name__ == "__main__":
    main()
