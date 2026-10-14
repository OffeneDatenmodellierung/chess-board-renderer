# Openly licensed chess board datasets

These datasets can train a board reader that may be shipped. Each needs its
licence noted in open-chess's `crates/chess-vision-train/datasets.toml`.
CC BY and MIT also need their attribution line, which the trainer copies
into the model card.

| Dataset | What | Labels | Licence | Use as |
|---|---|---|---|---|
| [samryan18/chess-dataset](https://github.com/samryan18/chess-dataset) | 500 real photos (Nikon, from above, one green vinyl board) | FEN in each file name | MIT | `jsonl` via `import_datasets.py samryan18` |
| [chesscog renders](https://doi.org/10.17605/OSF.IO/XF3KA) (Wölflein & Arandjelović) | About 5,000 3D renders, various angles and lighting | FEN in a JSON beside each image | CC BY 4.0 | `jsonl` via `import_datasets.py chesscog` |
| [robot-chess-commentator-squares](https://huggingface.co/datasets/felixfabricius/robot-chess-commentator-squares), subset `boards` (Felix Fabricius) | 371 real boards, already straightened to top-down | `board_fen` | CC BY 4.0 | `jsonl` via `import_datasets.py felix` |
| [Annotated Image Dataset for Fenit](https://data.mendeley.com/datasets/wpp9xnbcp6) (Binus University) | Real photos | Piece boxes and the board's 4 corners (CVAT) | CC BY 4.0 | `photos` (unlabelled) for now. An importer that turns boxes and corners into FEN is next, once we've seen the files |
| [Chess Pieces](https://universe.roboflow.com/joseph-nelson/chess-pieces-new) (Roboflow, joseph-nelson) | About 290 real photos, one fixed angle | Piece boxes only | CC0 | `photos` (unlabelled): for a teacher to read |

Left out: Kaggle's "Chess Positions" (2D diagrams drawn with Chess.com's
artwork; the licence is doubtful), and ChessReD (CC BY-NC-SA, so
internal-only).

## Getting them

```sh
# samryan18 (4.4 GB of 6000-pixel photos; the import scales them to 1600)
git clone --depth 1 https://github.com/samryan18/chess-dataset
uv run import_datasets.py samryan18 chess-dataset ~/chess-data/samryan18

# chesscog: download from https://osf.io/xf3ka (or
# `uvx --from osfclient osf -p xf3ka clone chesscog-osf`), unzip, then
uv run import_datasets.py chesscog chesscog-osf ~/chess-data/chesscog

# felix: fetched from Hugging Face (the `hf` extra)
uv run --extra hf import_datasets.py felix - ~/chess-data/felix-boards

# Fenit and Roboflow: download (Roboflow: any export format), then put the
# images in a folder; the trainer reads the folder as `photos`.
```

The imports can be run again: photos already written are skipped. Views of
one game or setup share a `group`, so they stay in one split.

## datasets.toml entries

These are in open-chess's `datasets.toml`.

- `samryan18`, `chesscog` and `fenit-photos` are marked verified.
- `felix-boards` and `roboflow-chess-pieces` wait for you to read their
  licence on the source page and set `licence_verified = true`.

| Name | Attribution |
|---|---|
| `samryan18` | Chess Board Dataset, © 2019 Samuel Ryan, Mukund Venkateswaran, Kurt Convey, Michael Deng, github.com/samryan18/chess-dataset, MIT |
| `chesscog` | Chess recognition rendered dataset, G. Wölflein and O. Arandjelović, OSF, doi:10.17605/OSF.IO/XF3KA, CC BY 4.0 |
| `felix-boards` | robot-chess-commentator-squares (boards), Felix Fabricius, Hugging Face, CC BY 4.0 |
| `fenit-photos` | Annotated Image Dataset for Fenit, Binus University, Mendeley Data, doi:10.17632/wpp9xnbcp6.1, CC BY 4.0 |
| `roboflow-chess-pieces` | Chess Pieces, Roboflow Universe (joseph-nelson), CC0 1.0 |
