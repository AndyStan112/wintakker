import argparse
import json
import re
import torch

from tak.core import (
    TakState,
    Result,
    IllegalMove,
    WHITE,
    BLACK,
    FLAT,
    STANDING,
    CAPSTONE,
    piece_player,
    piece_kind,
)
from tak.ai import TakEvaluator, TakMoveGenerator, TakSearch, TakWeights
from tak.ml import TakMLEvaluator, ValueNet
from tak.encoder import encode_state_value_input

PLAYER_NAMES = {
    WHITE: "White",
    BLACK: "Black",
}

DEFAULT_ML_MODEL = "final.pt"
DEFAULT_MINIMAX_WEIGHTS = "tak_weights.json"
DEFAULT_DEPTH = 3


def coord_to_indices(coord, size):
    if len(coord) < 2:
        raise ValueError("Invalid coordinate")
    x = ord(coord[0].lower()) - ord("a")
    y = int(coord[1:]) - 1
    if not (0 <= x < size and 0 <= y < size):
        raise ValueError("Coordinate out of bounds")
    return x, y


def parse_move(text, size):
    s = text.strip()
    if not s:
        raise ValueError("Empty input")
    if s.lower() in ("q", "quit", "exit", "resign"):
        return ("quit",)

    s = s.replace(" ", "")
    m = re.fullmatch(r"([FSC])?([a-h][1-8])", s, re.IGNORECASE)
    if m:
        kind = (
            FLAT
            if not m.group(1) or m.group(1).upper() == "F"
            else (STANDING if m.group(1).upper() == "S" else CAPSTONE)
        )
        x, y = coord_to_indices(m.group(2), size)
        return ("place", x, y, kind)

    m = re.fullmatch(r"(\d+)?([a-h][1-8])([<>+\-])(\d+)", s)
    if m:
        count = int(m.group(1)) if m.group(1) else sum(map(int, m.group(4)))
        drops = list(map(int, m.group(4)))
        x, y = coord_to_indices(m.group(2), size)
        dx, dy = {">": (1, 0), "<": (-1, 0), "+": (0, 1), "-": (0, -1)}[m.group(3)]
        return ("move", x, y, dx, dy, count, drops)

    raise ValueError("Could not parse move")


def piece_char(piece):
    p = piece_player(piece)
    k = piece_kind(piece)
    return {
        (WHITE, FLAT): "w",
        (WHITE, STANDING): "W",
        (WHITE, CAPSTONE): "C",
        (BLACK, FLAT): "b",
        (BLACK, STANDING): "B",
        (BLACK, CAPSTONE): "K",
    }[(p, k)]


def print_board(state):
    n = state.size

    def stack_str(stack):
        return "." if not stack else "(" + "".join(piece_char(p) for p in stack) + ")"

    board = [[stack_str(state.board[y][x]) for x in range(n)] for y in range(n)]
    w = max(len(c) for r in board for c in r)

    print("   " + " ".join(chr(ord("a") + i).center(w) for i in range(n)))
    for y in range(n - 1, -1, -1):
        print(f"{y+1:2d} " + " ".join(board[y][x].center(w) for x in range(n)))

    print()
    for p in (WHITE, BLACK):
        print(
            f"{PLAYER_NAMES[p]}: stones={state.stones_remaining[p]}, caps={state.caps_remaining[p]}"
        )
    print(f"To move: {PLAYER_NAMES[state.to_move]}\n")


def load_minimax(path):
    try:
        with open(path) as f:
            return TakEvaluator(TakWeights(**json.load(f)))
    except Exception as e:
        print(f"Failed to load weights {path}: {e}")
        return TakEvaluator()


def load_ml(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(path, map_location=device)
        sd = ckpt.get("state_dict", ckpt)
        net = ValueNet()
        net.load_state_dict(sd)
        net.eval()

        class Adapter:
            def predict(self, state, player):
                x = encode_state_value_input(
                    state, player, pad=8, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    return float(net(x).view(-1).item())

        print(f"Loaded ML model {path} on {device}")
        return TakMLEvaluator(net=Adapter())
    except Exception as e:
        print(f"Failed to load ML model {path}: {e}")
        return TakMLEvaluator()


def configure_player(color, args):
    type_attr = f"{color}_type"
    path_attr = f"{color}_path"
    depth_attr = f"{color}_depth"

    cfg = {
        "type": getattr(args, type_attr),
        "path": getattr(args, path_attr),
        "depth": getattr(args, depth_attr),
    }

    if cfg["type"] is None:
        print(f"\n{PLAYER_NAMES[WHITE if color=='white' else BLACK]} player:")
        print("1. Manual")
        print("2. Minimax")
        print("3. ML")
        choice = input("Select (1-3): ").strip()
        cfg["type"] = {"1": "manual", "2": "minimax", "3": "ml"}.get(choice, "manual")

        if cfg["type"] != "manual":
            d = input(f"Depth (default {DEFAULT_DEPTH}): ").strip()
            cfg["depth"] = int(d) if d else DEFAULT_DEPTH

            default_path = (
                DEFAULT_MINIMAX_WEIGHTS
                if cfg["type"] == "minimax"
                else DEFAULT_ML_MODEL
            )
            p = input(f"Path (default {default_path}): ").strip()
            cfg["path"] = p if p else default_path

    if cfg["depth"] is None:
        cfg["depth"] = DEFAULT_DEPTH

    if cfg["path"] is None:
        cfg["path"] = (
            DEFAULT_MINIMAX_WEIGHTS if cfg["type"] == "minimax" else DEFAULT_ML_MODEL
        )

    return cfg


def main():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--size", type=int, help="Board size 3–8")

    parser.add_argument("--w", dest="white_type", choices=["manual", "minimax", "ml"])
    parser.add_argument("--b", dest="black_type", choices=["manual", "minimax", "ml"])

    parser.add_argument("--w-path", dest="white_path")
    parser.add_argument("--b-path", dest="black_path")

    parser.add_argument("--w-depth", dest="white_depth", type=int)
    parser.add_argument("--b-depth", dest="black_depth", type=int)

    args, _ = parser.parse_known_args()

    try:
        if args.size:
            size = args.size
        else:
            size = int(input("Board size (3-8, default 5): ") or 5)

        state = TakState.from_size(size)
    except Exception as e:
        print(e)
        return

    white = configure_player("white", args)
    black = configure_player("black", args)

    engines = {}

    for color, cfg in [(WHITE, white), (BLACK, black)]:
        for color, cfg in [(WHITE, white), (BLACK, black)]:
            if cfg["type"] == "minimax":
                evaluator = load_minimax(cfg["path"])
            elif cfg["type"] == "ml":
                evaluator = load_ml(cfg["path"])
            else:
                continue
            engines[color] = TakSearch(
                evaluator=evaluator,
                movegen=TakMoveGenerator(),
                max_depth=cfg["depth"],
            )

    result = Result.ONGOING

    while result == Result.ONGOING:
        print_board(state)
        p = state.to_move

        if p in engines:
            print(f"{PLAYER_NAMES[p]} (AI) thinking...")
            move = engines[p].choose_move(state)
            print(f"AI plays: {move}")
            parsed = move
        else:
            try:
                parsed = parse_move(input(f"{PLAYER_NAMES[p]} move: "), state.size)
            except Exception as e:
                print(e)
                continue

        if parsed[0] == "quit":
            return

        try:
            result = (
                state.place(*parsed[1:])
                if parsed[0] == "place"
                else state.move_stack(*parsed[1:])
            )
        except IllegalMove as e:
            print("Illegal move:", e)

    print_board(state)
    print("Result:", result.name)


if __name__ == "__main__":
    main()
