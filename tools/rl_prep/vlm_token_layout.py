"""Where the visual tokens of a scene sit in the policy's input sequence (NumPy only).

The released image prompt is  boi, img, <rows of visual tokens joined by eol>, eof, eoi  (one block per frame). Visual tokens are the ids in
[bov, eov]. Rows are counted by the eol tokens, frames by the eof tokens. Nothing here assumes an 18 x 32 grid: it reports what is there so a
smoke test can confirm it, and `require_grid` turns a mismatch into an error for the production run.
"""
from collections import Counter

import numpy as np


def grid_positions(ids, bov, eov, eol_id, eof_id=None):
    """Return an int array [n, 4] of (sequence index, frame, row, col) for every visual token, plus a summary dict."""
    ids = np.asarray(ids)
    out = []
    frame = row = col = 0
    widths = Counter()
    rows_per_frame = Counter()
    other_between = 0
    in_block = False
    for i, t in enumerate(ids.tolist()):
        if t == eol_id:
            if col > 0:
                widths[col] += 1
                row += 1
                col = 0
            continue
        if eof_id is not None and t == eof_id:
            if col > 0:
                widths[col] += 1
                row += 1
                col = 0
            if row > 0:
                rows_per_frame[row] += 1
                frame += 1
                row = 0
            continue
        if bov <= t <= eov:
            out.append((i, frame, row, col))
            col += 1
            in_block = True
        elif in_block and col > 0:
            other_between += 1      # a non-visual token inside a row: the layout assumption is broken
    if col > 0:
        widths[col] += 1
        row += 1
    if row > 0:
        rows_per_frame[row] += 1
        frame += 1
    pos = np.array(out, dtype=np.int32).reshape(-1, 4)
    summary = {"n_visual": int(len(pos)), "frames": int(frame), "rows_per_frame": dict(rows_per_frame),
               "row_widths": dict(widths), "foreign_tokens_inside_rows": int(other_between)}
    return pos, summary


def require_grid(summary, rows, cols, frames=1):
    ok = (summary["frames"] == frames and summary["rows_per_frame"] == {rows: frames}
          and summary["row_widths"] == {cols: rows * frames} and summary["foreign_tokens_inside_rows"] == 0
          and summary["n_visual"] == rows * cols * frames)
    if not ok:
        raise ValueError(f"visual layout {summary} differs from the expected {frames} frame(s) of {rows} x {cols}")


def random_projection(d_in, d_out=128, seed=0):
    """Fixed Gaussian projection with entries of variance 1/d_out (Johnson-Lindenstrauss); regenerated from the seed wherever it is needed."""
    return np.random.default_rng(seed).normal(0.0, 1.0 / np.sqrt(d_out), size=(d_in, d_out)).astype(np.float32)


def projection_fingerprint(proj):
    return float(np.round(proj[:4, :4].sum(), 6))


def require_frames(summary, frames, tokens_per_frame):
    """The video prompt stores each frame as ONE run of tokens (no eol between rows): `frames` runs of `tokens_per_frame` tokens."""
    ok = (summary["frames"] == frames and summary["n_visual"] == frames * tokens_per_frame
          and summary["rows_per_frame"] == {1: frames} and summary["row_widths"] == {tokens_per_frame: frames}
          and summary["foreign_tokens_inside_rows"] == 0)
    if not ok:
        raise ValueError(f"visual layout {summary} differs from the expected {frames} frames of {tokens_per_frame} contiguous tokens")


def frame_grid(pos, height, width):
    """(frame, row, col) of every visual token from its position inside the frame's run; frames are flattened row-major ([h, w].flatten())."""
    pos = np.asarray(pos)
    k = pos[:, 3]
    if k.max() >= height * width or (pos[:, 2] != 0).any():
        raise ValueError("tokens do not form single runs of at most height*width tokens")
    return np.stack([pos[:, 1], k // width, k % width], axis=1).astype(np.int16)
