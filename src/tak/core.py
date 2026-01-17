from enum import IntEnum


WHITE = 0
BLACK = 1

FLAT = 0
STANDING = 1
CAPSTONE = 2


def make_piece(player, kind):
    return (player << 2) | kind


def piece_player(piece):
    return piece >> 2


def piece_kind(piece):
    return piece & 0b11


PIECE_TABLE = {
    3: (10, 0),
    4: (15, 0),
    5: (21, 1),
    6: (30, 1),
    7: (40, 2),
    8: (50, 2),
}


class Result(IntEnum):
    ONGOING = 0
    WHITE_WIN = 1
    BLACK_WIN = 2
    DRAW = 3


class IllegalMove(Exception):
    pass


class TakState:
    def __init__(self, size, stones_per_player, caps_per_player):
        if size < 3 or size > 8:
            raise ValueError("Board size must be between 3 and 8")
        self.size = size
        self.board = [[[] for _ in range(size)] for _ in range(size)]
        self.to_move = WHITE
        self.stones_remaining = [stones_per_player, stones_per_player]
        self.caps_remaining = [caps_per_player, caps_per_player]
        self.move_number = 0

    @classmethod
    def from_size(cls, size):
        if size not in PIECE_TABLE:
            raise ValueError("Unsupported board size")
        stones, caps = PIECE_TABLE[size]
        return cls(size, stones, caps)

    def clone(self):
        other = TakState.__new__(TakState)
        other.size = self.size
        other.board = [[stack.copy() for stack in row] for row in self.board]
        other.to_move = self.to_move
        other.stones_remaining = self.stones_remaining.copy()
        other.caps_remaining = self.caps_remaining.copy()
        other.move_number = self.move_number
        return other

    def inside(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def get_stack(self, x, y):
        if not self.inside(x, y):
            raise IndexError("Out of bounds")
        return self.board[y][x]

    def place(self, x, y, kind):
        if not self.inside(x, y):
            raise IllegalMove("Out of bounds")
        stack = self.board[y][x]
        if stack:
            raise IllegalMove("Square not empty")

        opening = self.move_number < 2
        current = self.to_move
        owner = (1 - current) if opening else current

        if opening:
            if kind != FLAT:
                raise IllegalMove("Only flat placements allowed on the first two moves")
            if self.stones_remaining[owner] <= 0:
                raise IllegalMove("No stones left")
            self.stones_remaining[owner] -= 1
            piece = make_piece(owner, FLAT)
        else:
            if kind == CAPSTONE:
                if self.caps_remaining[owner] <= 0:
                    raise IllegalMove("No capstones left")
                self.caps_remaining[owner] -= 1
                piece = make_piece(owner, CAPSTONE)
            else:
                if self.stones_remaining[owner] <= 0:
                    raise IllegalMove("No stones left")
                self.stones_remaining[owner] -= 1
                if kind not in (FLAT, STANDING):
                    raise IllegalMove("Invalid piece type")
                piece = make_piece(owner, kind)

        stack.append(piece)
        last_player = owner
        self.to_move = 1 - self.to_move
        self.move_number += 1
        return self.compute_result(last_player)

    def move_stack(self, x, y, dx, dy, count, drops):
        if self.move_number < 2:
            raise IllegalMove("No stack moves allowed on the first two moves")

        if dx == 0 and dy == 0:
            raise IllegalMove("Zero move")
        if not self.inside(x, y):
            raise IllegalMove("Out of bounds")
        if count <= 0:
            raise IllegalMove("Count must be positive")
        if count > self.size:
            raise IllegalMove("Cannot carry that many stones")
        if not drops:
            raise IllegalMove("Drops required")
        if sum(drops) != count:
            raise IllegalMove("Drops do not sum to count")
        stack = self.board[y][x]
        if len(stack) < count:
            raise IllegalMove("Not enough stones in stack")
        top = stack[-1]
        player = self.to_move
        if piece_player(top) != player:
            raise IllegalMove("Stack not controlled by player")
        new_board = [[s.copy() for s in row] for row in self.board]
        stack = new_board[y][x]
        carried = stack[-count:]
        new_board[y][x] = stack[:-count]
        cx, cy = x, y
        for i, drop_n in enumerate(drops):
            cx += dx
            cy += dy
            if not self.inside(cx, cy):
                raise IllegalMove("Move goes off board")
            if drop_n <= 0:
                raise IllegalMove("Invalid drop count")
            if drop_n > len(carried):
                raise IllegalMove("Not enough stones to drop")
            dest_stack = new_board[cy][cx]
            dest_top = dest_stack[-1] if dest_stack else None
            drop_pieces = carried[:drop_n]
            carried = carried[drop_n:]
            if dest_top is not None:
                dest_owner = piece_player(dest_top)
                dest_kind = piece_kind(dest_top)
                if dest_kind in (STANDING, CAPSTONE):
                    dropping_capstone = (
                        len(drop_pieces) == 1 and piece_kind(drop_pieces[0]) == CAPSTONE
                    )
                    last_step = i == len(drops) - 1
                    alone = len(carried) == 0
                    if (
                        dest_kind == STANDING
                        and dropping_capstone
                        and last_step
                        and alone
                    ):
                        dest_stack[-1] = make_piece(dest_owner, FLAT)
                        dest_stack.append(drop_pieces[0])
                        carried = []
                        break
                    else:
                        raise IllegalMove("Cannot move onto standing stone or capstone")
            dest_stack.extend(drop_pieces)
        if carried:
            raise IllegalMove("Did not drop all stones")
        self.board = new_board
        last_player = player
        self.to_move = 1 - self.to_move
        self.move_number += 1
        return self.compute_result(last_player)

    def is_road_cell(self, player, x, y):
        stack = self.board[y][x]
        if not stack:
            return False
        top = stack[-1]
        if piece_player(top) != player:
            return False
        kind = piece_kind(top)
        return kind in (FLAT, CAPSTONE)

    def has_road(self, player):
        n = self.size
        visited = [[False] * n for _ in range(n)]
        stack = []
        for y in range(n):
            if self.is_road_cell(player, 0, y):
                visited[y][0] = True
                stack.append((0, y))
        while stack:
            x, y = stack.pop()
            if x == n - 1:
                return True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < n
                    and 0 <= ny < n
                    and not visited[ny][nx]
                    and self.is_road_cell(player, nx, ny)
                ):
                    visited[ny][nx] = True
                    stack.append((nx, ny))
        visited = [[False] * n for _ in range(n)]
        stack = []
        for x in range(n):
            if self.is_road_cell(player, x, 0):
                visited[0][x] = True
                stack.append((x, 0))
        while stack:
            x, y = stack.pop()
            if y == n - 1:
                return True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < n
                    and 0 <= ny < n
                    and not visited[ny][nx]
                    and self.is_road_cell(player, nx, ny)
                ):
                    visited[ny][nx] = True
                    stack.append((nx, ny))
        return False

    def count_flats(self, player):
        n = self.size
        total = 0
        for y in range(n):
            for x in range(n):
                stack = self.board[y][x]
                if not stack:
                    continue
                top = stack[-1]
                if piece_player(top) == player and piece_kind(top) == FLAT:
                    total += 1
        return total

    def board_full(self):
        for row in self.board:
            for stack in row:
                if not stack:
                    return False
        return True

    def compute_result(self, last_player):
        white_road = self.has_road(WHITE)
        black_road = self.has_road(BLACK)
        if white_road or black_road:
            if white_road and black_road:
                return Result.WHITE_WIN if last_player == WHITE else Result.BLACK_WIN
            if white_road:
                return Result.WHITE_WIN
            return Result.BLACK_WIN
        full = self.board_full()
        white_out = self.stones_remaining[WHITE] + self.caps_remaining[WHITE] == 0
        black_out = self.stones_remaining[BLACK] + self.caps_remaining[BLACK] == 0
        if full or white_out or black_out:
            w_flats = self.count_flats(WHITE)
            b_flats = self.count_flats(BLACK)
            if w_flats > b_flats:
                return Result.WHITE_WIN
            if b_flats > w_flats:
                return Result.BLACK_WIN
            return Result.DRAW
        return Result.ONGOING
