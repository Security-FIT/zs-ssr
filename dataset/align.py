"""Estimate the constant time offset between a VC sample and its LibriSpeech original.

Most synthesizers keep the source timing frame for frame, but some (Train-1, Train-4) shift
the whole utterance by a roughly constant offset. Both recordings are reduced to a 1-D energy
contour (each model uses its own front-end) and the offset is the lag at which the two
contours have the highest Pearson correlation. Used by every trainer's `--align` and by
`duration_check.py`.
"""

from __future__ import annotations

import numpy as np


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (frames) of contour `a` against contour `b` maximizing their Pearson correlation
    over +/-`max_lag`, and that correlation.

    Positive lag: `a` is delayed, drop its first `lag` frames to align.
    Negative lag: drop the first `-lag` frames of `b`.
    Each lag is scored on the overlapping region only. Returns (0, -1.0), i.e. no shift, if no
    lag leaves 16 overlapping frames. Ties keep the most negative lag, so a flat (all-silent)
    contour gives -`max_lag`.
    """
    best = (0, -1.0)
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b) if lag >= 0 else (a, b[-lag:])
        m = min(len(x), len(y))
        if m < 16:  # too little overlap to trust the correlation
            continue
        # Pearson correlation: dot product of the mean-centered contours over the product of
        # their norms (the epsilon only guards a flat, all-silent contour)
        x, y = x[:m] - x[:m].mean(), y[:m] - y[:m].mean()
        c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8))
        if c > best[1]:
            best = (lag, c)
    return best


def main():
    """Self-test: a contour delayed by 7 frames is found at lag +7, and the reverse at -7."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal(300)
    a = np.concatenate([rng.standard_normal(7), b])
    assert best_lag(a, b, 20)[0] == 7, best_lag(a, b, 20)
    assert best_lag(b, a, 20)[0] == -7, best_lag(b, a, 20)
    print("[align.py] best_lag OK")


if __name__ == "__main__":
    main()
