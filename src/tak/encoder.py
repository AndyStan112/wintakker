import re
import torch
from tak.core import FLAT, STANDING, CAPSTONE

_files = "ABCDEFGH"


def _coord_to_xy(c):
    c = c.strip().upper()
    x = _files.index(c[0])
    y = int(c[1:]) - 1
    return x, y


def parse_playtak_action(text):
    t = text.strip()
    if not t:
        return None
    parts = [p for p in t.split() if p]
    head = parts[0].upper()
    if head == "P":
        coord = parts[1].upper()
        kind = FLAT
        if len(parts) >= 3:
            k = parts[2].upper()
            if k == "W":
                kind = STANDING
            elif k == "C":
                kind = CAPSTONE
        x, y = _coord_to_xy(coord)
        return ("place", x, y, kind)

    if head == "M":
        fr = parts[1].upper()
        to = parts[2].upper()
        drops = [int(p) for p in parts[3:]]
        fx, fy = _coord_to_xy(fr)
        tx, ty = _coord_to_xy(to)
        dx = tx - fx
        dy = ty - fy
        if dx != 0 and dy != 0:
            raise ValueError("Diagonal move")
        if dx == 0 and dy == 0:
            raise ValueError("Zero move")
        step_dx = 0 if dx == 0 else (1 if dx > 0 else -1)
        step_dy = 0 if dy == 0 else (1 if dy > 0 else -1)
        count = sum(drops)
        return ("move", fx, fy, step_dx, step_dy, count, drops)

    raise ValueError("Unknown action")


def split_playtak_notation(notation):
    if not notation:
        return []
    parts = [p.strip() for p in notation.split(",")]
    return [p for p in parts if p]


def encode_state_value_input(state, player, pad=8, device="cpu"):
    n = state.size
    x = torch.zeros((10, pad, pad), dtype=torch.float32, device=device)

    x[8, :, :] = 1.0 if state.to_move == player else 0.0
    x[9, :, :] = float(n) / 8.0

    for yy in range(n):
        for xx in range(n):
            st = state.board[yy][xx]
            if not st:
                continue
            top = st[-1]
            owner = top >> 2
            kind = top & 0b11
            base = 0 if owner == player else 3
            if kind == FLAT:
                x[base + 0, yy, xx] = 1.0
            elif kind == STANDING:
                x[base + 1, yy, xx] = 1.0
            elif kind == CAPSTONE:
                x[base + 2, yy, xx] = 1.0
            h = len(st)
            x[6, yy, xx] = min(h, 8) / 8.0

    x[7, :, :] = (state.stones_remaining[player] + state.caps_remaining[player]) / 60.0
    return x
