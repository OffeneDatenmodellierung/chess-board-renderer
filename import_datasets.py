#!/usr/bin/env python3
"""Turn openly licensed chess board datasets into the board-reader trainer's
``jsonl`` format: ``photos/…`` plus ``labels.jsonl`` with one line per photo,
``{"image", "placement" (FEN, White's side), "group", "split"?}``.

Each source keeps its own licence: see DATASETS.md for where to download
them, their licences, and the attribution lines the model card must carry.

    uv run import_datasets.py samryan18 path/to/chess-dataset   ~/chess-data/samryan18
    uv run import_datasets.py chesscog  path/to/osf-render-dir   ~/chess-data/chesscog
    uv run --extra hf import_datasets.py felix -                       ~/chess-data/felix-boards

Photos are scaled down to --max-side (default 1600) and turned upright from
their EXIF orientation: the trainer scales to 392–644 anyway, and 6000-pixel
originals only slow it down.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

FEN_ROW = re.compile(r"^[pnbrqkPNBRQK1-8]+$")


def valid_placement(fen: str) -> bool:
    rows = fen.split("/")
    if len(rows) != 8:
        return False
    for row in rows:
        if not FEN_ROW.match(row) or sum(int(c) if c.isdigit() else 1 for c in row) != 8:
            return False
    return True


class Writer:
    """Writes photos and labels.jsonl into ``out``, skipping photos already
    written (so an import can be run again)."""

    def __init__(self, out: Path, max_side: int):
        self.out = out
        self.max_side = max_side
        (out / "photos").mkdir(parents=True, exist_ok=True)
        self.labels_path = out / "labels.jsonl"
        self.seen = set()
        if self.labels_path.exists():
            with self.labels_path.open() as f:
                self.seen = {json.loads(line)["image"] for line in f if line.strip()}
        self.labels = self.labels_path.open("a")
        self.written = self.skipped = 0

    def add(self, name: str, image, placement: str, group: str, split: str | None = None):
        if not valid_placement(placement):
            print(f"skipping {name}: not a board: {placement!r}", file=sys.stderr)
            self.skipped += 1
            return
        rel = f"photos/{name}.jpg"
        if rel in self.seen:
            return
        img = image if isinstance(image, Image.Image) else Image.open(image)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((self.max_side, self.max_side), Image.LANCZOS)
        img.save(self.out / rel, quality=92)
        line = {"image": rel, "placement": placement, "group": group}
        if split:
            line["split"] = split
        self.labels.write(json.dumps(line) + "\n")
        self.seen.add(rel)
        self.written += 1
        if self.written % 250 == 0:
            print(f"  {self.written} photos")

    def close(self):
        self.labels.close()
        print(f"wrote {self.written} photos to {self.out} ({len(self.seen)} in all, {self.skipped} skipped)")


def samryan18(src: Path, w: Writer):
    """github.com/samryan18/chess-dataset: 500 photos (Nikon, top-down),
    each named by its FEN placement with '-' for '/'. The originals, not the
    edge-detected 'preprocessed' copies."""
    folder = src / "labeled_originals"
    if not folder.is_dir():
        sys.exit(f"no labeled_originals/ in {src}: clone github.com/samryan18/chess-dataset")
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        # A suffix like "_4" marks another photo of the same position.
        placement = re.sub(r"_\d+$", "", p.stem).replace("-", "/")
        # No game ids: photos of the same game differ by a move, so group by
        # the pieces' count and kings' squares to keep near-duplicates
        # together.
        w.add(f"sr-{stable_id(p.stem)}", p, placement, group=f"samryan18:{kings_and_count(placement)}")


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def kings_and_count(placement: str) -> str:
    squares, n = [], 0
    for r, row in enumerate(placement.split("/")):
        f = 0
        for c in row:
            if c.isdigit():
                f += int(c)
                continue
            n += 1
            if c in "Kk":
                squares.append(f"{c}{f}{7 - r}")
            f += 1
    return f"{''.join(squares)}-{n}"


SPLITS = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


def split_from(parts) -> str | None:
    """The split named by a folder or zip in a path, if any."""
    for part in reversed(list(parts)):
        stem = Path(part).stem.lower()
        if stem in SPLITS:
            return SPLITS[stem]
    return None


def chesscog(src: Path, w: Writer):
    """The chesscog renders (osf.io/xf3ka): N.png with N.json holding
    "fen" (the board, absolute), "white_turn" (the side the camera is on),
    corners and boxes. Read as downloaded, from its zip files (OSF's
    train.zip, val.zip, test.zip), or from folders they were unzipped into.
    Splits from the zip or folder names."""
    found = 0

    def take(where: str, stem: str, label: dict, image, split):
        nonlocal found
        fen = label.get("fen")
        if not fen:
            return
        found += 1
        key = f"{split or where}-{stem}"
        w.add(f"cc-{key}", image, fen.split()[0], group=f"chesscog:{key}", split=split)

    zips = [src] if src.suffix == ".zip" else sorted(src.rglob("*.zip"))
    for z in zips:
        before = found
        with zipfile.ZipFile(z) as zf:
            names = set(zf.namelist())
            for name in sorted(n for n in names if n.endswith(".json")):
                base = name[: -len(".json")]
                img = next((base + e for e in (".png", ".jpg") if base + e in names), None)
                if img is None:
                    continue
                try:
                    label = json.loads(zf.read(name))
                except json.JSONDecodeError:
                    continue
                parts = [z.name, *Path(name).parts[:-1]]
                take(z.stem, Path(base).name, label, io.BytesIO(zf.read(img)), split_from(parts))
        print(f"  {z.name}: {found - before} labelled renders")
    if src.is_dir():
        for j in sorted(src.rglob("*.json")):
            img = next((j.with_suffix(e) for e in (".png", ".jpg") if j.with_suffix(e).exists()), None)
            if img is None:
                continue
            try:
                label = json.loads(j.read_text())
            except json.JSONDecodeError:
                continue
            take(j.parent.name, j.stem, label, img, split_from(j.relative_to(src).parts[:-1]))
    if not found:
        sys.exit(f"no labelled renders (a .json with \"fen\" beside each image) in {src}, nor in zip files there")


def felix(src: str, w: Writer):
    """huggingface.co/datasets/felixfabricius/robot-chess-commentator-squares,
    subset "boards": 371 top-down (perspective-corrected) boards with
    board_fen. Only rows whose FEN matches the pixels are kept; the setup id
    is the group."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        sys.exit(f"felix needs Hugging Face's datasets (uv run --extra hf import_datasets.py felix …): {e}")
    name = src if src not in ("-", "") else "felixfabricius/robot-chess-commentator-squares"
    ds = load_dataset(name, "boards")
    for split, rows in ds.items():
        ours = {"validation": "val"}.get(split, split)
        for row in rows:
            if row.get("fen_matches_pixels") is False:
                continue
            img = row["warped_image"]
            if isinstance(img, dict) and img.get("bytes"):
                img = Image.open(io.BytesIO(img["bytes"]))
            w.add(
                f"fx-{ours}-{row['image_id']}",
                img,
                row["board_fen"].split()[0],
                group=f"felix:{ours}:{row['setup_id']}",
                split=ours,
            )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", choices=["samryan18", "chesscog", "felix"])
    p.add_argument("src", help="the downloaded dataset (felix: '-' to fetch it from Hugging Face)")
    p.add_argument("out", type=Path)
    p.add_argument("--max-side", type=int, default=1600)
    a = p.parse_args()
    w = Writer(a.out.expanduser(), a.max_side)
    try:
        if a.source == "samryan18":
            samryan18(Path(a.src).expanduser(), w)
        elif a.source == "chesscog":
            chesscog(Path(a.src).expanduser(), w)
        else:
            felix(a.src, w)
    finally:
        w.close()


if __name__ == "__main__":
    main()
