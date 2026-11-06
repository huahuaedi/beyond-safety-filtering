import unittest

import numpy as np

import vlm_token_layout as L

BOV, EOV, EOL, EOF, TXT = 1000, 1999, 7, 8, 5


def image(rows, cols, frames=1):
    ids = [TXT, TXT]
    for f in range(frames):
        for r in range(rows):
            ids += list(range(BOV + r * cols, BOV + (r + 1) * cols))
            if r < rows - 1:
                ids.append(EOL)
        ids.append(EOF)
    return ids + [TXT, TXT]


class LayoutTest(unittest.TestCase):
    def test_grid_rows_cols_and_indices(self):
        ids = image(3, 4)
        pos, s = L.grid_positions(ids, BOV, EOV, EOL, EOF)
        self.assertEqual(s["n_visual"], 12)
        self.assertEqual(s["rows_per_frame"], {3: 1}); self.assertEqual(s["row_widths"], {4: 3})
        self.assertEqual(pos[0].tolist(), [2, 0, 0, 0])
        self.assertEqual(pos[-1][2:].tolist(), [2, 3])
        L.require_grid(s, 3, 4)
        with self.assertRaises(ValueError):
            L.require_grid(s, 18, 32)

    def test_two_frames_and_foreign_tokens(self):
        pos, s = L.grid_positions(image(2, 3, frames=2), BOV, EOV, EOL, EOF)
        self.assertEqual(s["frames"], 2); self.assertEqual(sorted(set(pos[:, 1])), [0, 1])
        broken = image(2, 3); broken.insert(4, TXT)
        _, s2 = L.grid_positions(broken, BOV, EOV, EOL, EOF)
        self.assertGreater(s2["foreign_tokens_inside_rows"], 0)
        with self.assertRaises(ValueError):
            L.require_grid(s2, 2, 3)

    def test_projection_is_reproducible_and_roughly_norm_preserving(self):
        a, b = L.random_projection(4096, 128, 0), L.random_projection(4096, 128, 0)
        np.testing.assert_array_equal(a, b)
        x = np.random.default_rng(1).normal(size=(50, 4096)).astype(np.float32)
        ratio = np.linalg.norm(x @ a, axis=1) / np.linalg.norm(x, axis=1)
        self.assertTrue(0.8 < ratio.mean() < 1.2)
        self.assertEqual(L.projection_fingerprint(a), L.projection_fingerprint(b))

    def test_video_prompt_layout_and_row_major_grid(self):
        # two frames of 3 x 4 tokens, each frame one contiguous run, frames separated by eof (the released video prompt)
        ids = [TXT, TXT] + list(range(BOV, BOV + 12)) + [EOF] + list(range(BOV + 12, BOV + 24)) + [EOF, TXT]
        pos, s = L.grid_positions(ids, BOV, EOV, EOL, EOF)
        L.require_frames(s, 2, 12)
        grid = L.frame_grid(pos, 3, 4)
        self.assertEqual(grid.shape, (24, 3))
        self.assertEqual(grid[0].tolist(), [0, 0, 0]); self.assertEqual(grid[5].tolist(), [0, 1, 1])
        self.assertEqual(grid[11].tolist(), [0, 2, 3]); self.assertEqual(grid[12].tolist(), [1, 0, 0])
        with self.assertRaises(ValueError):
            L.require_frames(s, 1, 12)
        with self.assertRaises(ValueError):
            L.frame_grid(pos, 2, 4)


if __name__ == "__main__":
    unittest.main()
