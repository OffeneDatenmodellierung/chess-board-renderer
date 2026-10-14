#!/usr/bin/env python3
"""Board positions to render, one FEN placement per line.

Two sources, mixable:

* ``--random N``: N positions from random games (random legal moves from
  the start, stopped after a random number of plies). Everything about
  them is ours.
* ``--pgn FILE``: positions sampled from real games, e.g. the Lichess open
  database (https://database.lichess.org, released as CC0). ``.pgn`` or
  ``.pgn.zst`` (needs ``zstandard``).

    uv run positions.py --random 3000 --pgn lichess_db_standard_rated_2013-01.pgn.zst \
        --from-pgn 7000 --out positions.txt

Run with uv (``uv run positions.py …``), which installs python-chess.
"""

from __future__ import annotations

import argparse
import io
import random
import sys

import chess
import chess.pgn


def random_positions(n: int, rng: random.Random, max_plies: int) -> list[str]:
    out = []
    while len(out) < n:
        board = chess.Board()
        # Weighted towards the middlegame, where most boards are photographed.
        plies = min(int(rng.triangular(0, max_plies, max_plies * 0.35)), max_plies)
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            # Captures a little more often than chance, so piece counts thin
            # out as games really do.
            captures = [m for m in moves if board.is_capture(m)]
            pick = captures if captures and rng.random() < 0.3 else moves
            board.push(rng.choice(pick))
        out.append(board.board_fen())
    return out


def open_pgn(path: str) -> io.TextIOBase:
    if path.endswith(".zst"):
        try:
            import zstandard
        except ImportError:
            sys.exit("reading .zst needs zstandard: run with uv run positions.py …")
        raw = open(path, "rb")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def pgn_positions(path: str, n: int, rng: random.Random, per_game: int) -> list[str]:
    """Up to ``per_game`` random positions from each game until there are ``n``."""
    out: list[str] = []
    with open_pgn(path) as f:
        while len(out) < n:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            fens = []
            for move in game.mainline_moves():
                board.push(move)
                fens.append(board.board_fen())
            if fens:
                out.extend(rng.sample(fens, min(per_game, len(fens))))
    if len(out) < n:
        print(f"note: {path} gave only {len(out)} positions", file=sys.stderr)
    return out[:n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--random", type=int, default=0, help="positions from random games")
    p.add_argument("--max-plies", type=int, default=140)
    p.add_argument("--pgn", help="a PGN file (or .pgn.zst) to sample real games from")
    p.add_argument("--from-pgn", type=int, default=0, help="positions to take from --pgn")
    p.add_argument("--per-game", type=int, default=3, help="at most this many positions per PGN game")
    p.add_argument("--start", type=int, default=0, help="also this many copies of the starting position")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rng = random.Random(a.seed)
    fens = random_positions(a.random, rng, a.max_plies)
    if a.pgn and a.from_pgn:
        fens += pgn_positions(a.pgn, a.from_pgn, rng, a.per_game)
    fens += [chess.STARTING_BOARD_FEN] * a.start
    rng.shuffle(fens)
    with open(a.out, "w") as f:
        f.writelines(fen + "\n" for fen in fens)
    print(f"wrote {len(fens)} positions to {a.out}")


if __name__ == "__main__":
    main()
