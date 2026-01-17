import argparse
import json
import re
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
from tak.ai import TakEvaluator, TakSearch, TakWeights
from tak.ml import TakMLEvaluator, ValueNet
import torch
from tak.encoder import encode_state_value_input

PLAYER_NAMES = {
    WHITE: "White",
    BLACK: "Black",
}
LOAD_WEIGHTS = False


def coord_to_indices(coord, size):
    if len(coord) < 2:
        raise ValueError("Invalid coordinate")
    file_char = coord[0].lower()
    if not ("a" <= file_char <= "h"):
        raise ValueError("Invalid file")
    x = ord(file_char) - ord("a")
    try:
        rank = int(coord[1:])
    except ValueError:
        raise ValueError("Invalid rank")
    y = rank - 1
    if not (0 <= x < size and 0 <= y < size):
        raise ValueError("Coordinate out of bounds")
    return x, y


def parse_move(text, size):
    s = text.strip()
    if not s:
        raise ValueError("Empty input")
    low = s.lower()
    if low in ("q", "quit", "exit", "resign"):
        return ("quit",)
    s = s.replace(" ", "")
    m = re.fullmatch(r"([FSC])?([a-h][1-8])", s, re.IGNORECASE)
    if m:
        piece_char = m.group(1)
        coord = m.group(2)
        if piece_char is None or piece_char.upper() == "F":
            kind = FLAT
        elif piece_char.upper() == "S":
            kind = STANDING
        else:
            kind = CAPSTONE
        x, y = coord_to_indices(coord, size)
        return ("place", x, y, kind)
    m = re.fullmatch(r"(\d+)?([a-h][1-8])([<>+\-])(\d+)", s, re.IGNORECASE)
    if m:
        count_str, coord, dir_char, drops_str = m.groups()
        count = int(count_str) if count_str is not None else None
        drops = [int(ch) for ch in drops_str]
        if any(d <= 0 for d in drops):
            raise ValueError("Drops must be positive digits")
        if count is None:
            count = sum(drops)
        x, y = coord_to_indices(coord, size)
        if dir_char == ">":
            dx, dy = 1, 0
        elif dir_char == "<":
            dx, dy = -1, 0
        elif dir_char == "+":
            dx, dy = 0, 1
        else:
            dx, dy = 0, -1
        return ("move", x, y, dx, dy, count, drops)
    raise ValueError("Could not parse move")


def piece_char(piece):
    player = piece_player(piece)
    kind = piece_kind(piece)
    if player == WHITE:
        if kind == FLAT:
            return "w"
        if kind == STANDING:
            return "W"
        return "C"
    else:
        if kind == FLAT:
            return "b"
        if kind == STANDING:
            return "B"
        return "K"


def print_board(state):
    n = state.size

    def stack_str(stack):
        if not stack:
            return "."
        return "(" + "".join(piece_char(p) for p in stack) + ")"

    board_repr = [[stack_str(state.board[y][x]) for x in range(n)] for y in range(n)]

    cell_width = max(len(cell) for row in board_repr for cell in row)

    def center(cell):
        return cell.center(cell_width)

    header = "   " + " ".join(center(chr(ord("a") + i)) for i in range(n))
    print(header)

    for row in range(n - 1, -1, -1):
        line = " ".join(center(board_repr[row][col]) for col in range(n))
        print(f"{row+1:2d} {line}")

    print()
    for player in (WHITE, BLACK):
        print(
            f"{PLAYER_NAMES[player]}: stones={state.stones_remaining[player]}, caps={state.caps_remaining[player]}"
        )
    print(f"To move: {PLAYER_NAMES[state.to_move]}")
    print()


def main():

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model",
        default="final.pt",
        help="path to trained model file (default final.pt)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="search max depth for the AI (default 3)",
    )
    parsed_args, _ = parser.parse_known_args()
    model_path = parsed_args.model
    search_depth = parsed_args.depth

    try:
        size_str = input("Board size (3-8, default 5): ").strip()
    except EOFError:
        return

    if not size_str:
        size = 5
    else:
        try:
            size = int(size_str)
        except ValueError:
            print("Invalid size, using 5")
            size = 5

    try:
        state = TakState.from_size(size)
    except ValueError as e:
        print(e)
        return

    print("\nChoose mode:")
    print("1. Local 2-player game")
    print("2. Minimax (default weights)")
    print("3. Minimax (loaded weights)")
    print("4. ML AI (mock NN)")
    print("5. AI vs AI (minimax default)")
    mode = input("Select option (1-5): ").strip()

    ai_white = False
    ai_black = False
    ai_type = None

    if mode in ("2", "3", "4"):
        side = input("Play as (W/B)? ").strip().lower()
        if side == "w":
            ai_black = True
        elif side == "b":
            ai_white = True
        else:
            ai_black = True

        if mode == "2":
            ai_type = "minimax_default"
        elif mode == "3":
            ai_type = "minimax_loaded"
        else:
            ai_type = "ml"

    elif mode == "5":
        ai_white = True
        ai_black = True
        ai_type = "minimax_default"

    weights_dict = {}
    with open("tak_weights.json") as f:
        weights_dict = json.load(f)

    if ai_type == "minimax_loaded":
        weights = TakWeights(**weights_dict)
        evaluator = TakEvaluator(weights=weights)
    elif ai_type == "ml":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            ckpt = torch.load(model_path, map_location=device)
            state_dict = ckpt.get("state_dict", ckpt)
            net = ValueNet()
            net.load_state_dict(state_dict)
            net.eval()

            class NetAdapter:
                def __init__(self, module, device="cpu"):
                    self.module = module.to(device)
                    self.device = device

                def predict(self, state, player):
                    x = encode_state_value_input(
                        state, player, pad=8, device=self.device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        out = self.module(x)
                    return float(out.view(-1).item())

            evaluator = TakMLEvaluator(net=NetAdapter(net, device=device))
            print(f"loaded ML model from {model_path} on device {device}")
        except Exception as e:
            print(
                f"failed to load model {model_path}: {e}; falling back to mock ML evaluator"
            )
            evaluator = TakMLEvaluator()
    else:
        evaluator = TakEvaluator()

    ai = TakSearch(evaluator=evaluator, max_depth=search_depth)

    print("Enter moves in a PTN-like format.")
    print("Placement: a1 or Fa1, Sa1, Ca1 etc. Default is flat if no letter.")
    print("Movement: 3a1>12 means pick up 3 from a1, move right, drop 1 then 2.")
    print("You can omit the leading count: a1>12 means pick up 3.")
    print("Type 'quit' to exit.")

    result = Result.ONGOING

    while result == Result.ONGOING:
        print_board(state)
        player = state.to_move
        player_name = PLAYER_NAMES[player]

        if (player == WHITE and ai_white) or (player == BLACK and ai_black):
            print(f"{player_name} (AI) thinking...")
            move = ai.choose_move(state)
            print(f"AI plays: {move}")
            parsed = move
        else:

            try:
                move_text = input(f"{player_name} move: ")
            except EOFError:
                break
            try:
                parsed = parse_move(move_text, state.size)
            except ValueError as e:
                print("Input error:", e)
                continue

        if parsed[0] == "quit":
            print("Game ended by player.")
            return

        try:
            if parsed[0] == "place":
                _, x, y, kind = parsed
                result = state.place(x, y, kind)
            else:
                _, x, y, dx, dy, count, drops = parsed
                result = state.move_stack(x, y, dx, dy, count, drops)
        except IllegalMove as e:
            print("Illegal move:", e)
            continue

    print_board(state)
    if result == Result.DRAW:
        print("Game is a draw.")
    elif result == Result.WHITE_WIN:
        print("White wins.")
    elif result == Result.BLACK_WIN:
        print("Black wins.")


if __name__ == "__main__":
    main()
