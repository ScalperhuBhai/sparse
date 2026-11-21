"""Event-domain noise rejection.

Three stages, cheapest first — this ordering is what buys the latency headroom:

1. Hot-pixel mask      O(N)  — a handful of pixels emit a large share of events.
2. Refractory period   O(N)  — collapses per-pixel bursts to one event.
3. Background activity O(kN) — an event survives only if a spatial neighbour
                               fired within `dt_us`. Isolated shot noise dies,
                               spatially-coherent target signal lives.

Everything is chunk-vectorised NumPy: no per-event Python loop, no GPU.

## Why the background-activity filter is configurable

The classic Delbruck formulation asks "did one of my eight neighbours fire in
the last few milliseconds?" That is the right question for a bright, extended,
fast-moving edge, which is what the synthetic scene contains and what the
filter was originally tuned on.

It is the wrong question for a faint resident space object on a tracked
telescope. Such a target is an unresolved point source: it lands on one or two
pixels, it emits on the order of ten events per second, and consecutive events
at the same location are tens of milliseconds apart. A 5 ms neighbour-only
window rejects essentially all of it — measured on EBSSA, roughly one event in
273 survived inside windows that contained a labelled object.

So the neighbourhood radius, the self-term and the correlation time are all
parameters rather than constants, and each observing regime carries its own
values in `profiles.py`. Setting `dt_us <= 0` bypasses the stage entirely.
"""
from __future__ import annotations

import numpy as np


def hot_pixel_mask(ev: np.ndarray, shape, n_sigma: float = 8.0, min_count: int = 50,
                   neighbour_ratio: float = 0.0):
    """Boolean (H, W) mask of pixels whose event count is a gross outlier.

    ## Why a count outlier is not enough on a tracked telescope

    A stuck pixel and a near-stationary point source produce the same thing in a
    per-pixel count image: one pixel, far above every other. On a sidereally
    tracked telescope a geostationary RSO moves at a fraction of a pixel per
    second, so it sits on the same two pixels for the whole recording and this
    test deletes it — measured on EBSSA `refslew_2106_atis`, where **every one
    of 2,243 ground-truth samples fell within one pixel of a masked pixel** and
    the recording scored F1 0.000 with the target already gone before the
    detector ran.

    The two cases separate on **spatial support**. A stuck pixel is an isolated
    spike: its neighbours sit at the background level, because the defect is in
    one sensor cell. A point source has a PSF and sub-pixel jitter, so its
    neighbours are elevated too. Measured across every labelled EBSSA recording,
    best-neighbour count as a fraction of the candidate's own count:

        median on a labelled object   1.307
        median elsewhere              0.117

    An order of magnitude, so `neighbour_ratio` spares a candidate whose best
    neighbour reaches that fraction of its own count. 0.0 disables the test and
    reproduces the classic behaviour exactly, which is what the frozen
    `synthetic` profile uses.
    """
    H, W = shape
    counts = np.bincount((ev["y"].astype(np.int64) * W + ev["x"]),
                         minlength=H * W).reshape(H, W)
    nz = counts[counts > 0]
    if nz.size == 0:
        return np.zeros((H, W), dtype=bool)
    med = np.median(nz)
    mad = np.median(np.abs(nz - med)) + 1e-6
    thresh = max(med + n_sigma * 1.4826 * mad, min_count)
    mask = counts > thresh
    if neighbour_ratio > 0 and mask.any():
        # best neighbour count per pixel, excluding the pixel itself: a max
        # over the 8 shifts of the count image. O(HW), once, at calibration.
        pad = np.pad(counts, 1, mode="constant")
        best = np.zeros_like(counts)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                if (dy, dx) == (1, 1):
                    continue
                np.maximum(best, pad[dy:dy + H, dx:dx + W], out=best)
        supported = best >= neighbour_ratio * counts
        mask &= ~supported
    return mask


def apply_hot_pixel_mask(ev: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return ev[~mask[ev["y"], ev["x"]]]


def refractory(ev: np.ndarray, shape, period_us: int = 1000) -> np.ndarray:
    """Drop events from a pixel that fired less than `period_us` ago."""
    H, W = shape
    last = np.full(H * W, -(1 << 62), dtype=np.int64)
    idx = ev["y"].astype(np.int64) * W + ev["x"]
    keep = np.ones(len(ev), dtype=bool)
    # chunked to stay vectorised while keeping causality across chunks
    for s in range(0, len(ev), 4096):
        sl = slice(s, s + 4096)
        i, t = idx[sl], ev["t"][sl]
        k = (t - last[i]) >= period_us
        keep[sl] = k
        np.maximum.at(last, i[k], t[k])
    return ev[keep]


def _offsets(radius: int, include_self: bool):
    r = int(radius)
    offs = [(dy, dx)
            for dy in range(-r, r + 1)
            for dx in range(-r, r + 1)
            if include_self or (dy, dx) != (0, 0)]
    return offs


class BackgroundActivityFilter:
    """Nearest-neighbour temporal-correlation filter (Delbruck-style).

    Keeps event e at (x, y, t) only if some pixel in its support fired within
    `dt_us`. Implemented as a k-way gather against a padded time surface —
    O(kN) with no Python-level event loop.

    Parameters
    ----------
    radius : int
        Chebyshev radius of the support. 1 gives the classic 8-neighbourhood;
        2 gives 24 neighbours, which is what an unresolved point source with a
        one-pixel jitter actually needs.
    include_self : bool
        Whether a previous event at the *same* pixel counts as support. False
        is the classic formulation and is correct when the signal is an
        extended moving edge. True is correct when the signal is a stationary
        or slow point source, where the target's own pixel is the only pixel
        that ever fires. Turning it on weakens rejection of single-pixel shot
        noise, which is why the hot-pixel mask runs first.
    dt_us : int
        Correlation time. Must exceed the target's own inter-event interval,
        not the sensor's noise interval. `<= 0` bypasses the filter.
    """

    def __init__(self, shape, dt_us: int = 5000, chunk: int = 2048,
                 radius: int = 1, include_self: bool = False):
        self.H, self.W = shape
        self.dt_us = int(dt_us)
        self.chunk = int(chunk)
        self.radius = int(radius)
        self.include_self = bool(include_self)
        self.pad = max(1, self.radius)
        self.offs = _offsets(self.radius, self.include_self)
        self.T = np.full((self.H + 2 * self.pad, self.W + 2 * self.pad),
                         -(1 << 62), dtype=np.int64)

    def reset(self):
        self.T.fill(-(1 << 62))

    def __call__(self, ev: np.ndarray) -> np.ndarray:
        if len(ev) == 0 or self.dt_us <= 0:
            return ev
        T, p = self.T, self.pad
        keep = np.empty(len(ev), dtype=bool)
        offs = self.offs
        for s in range(0, len(ev), self.chunk):
            sl = slice(s, s + self.chunk)
            yy = ev["y"][sl].astype(np.int64) + p
            xx = ev["x"][sl].astype(np.int64) + p
            t = ev["t"][sl]
            best = T[yy + offs[0][0], xx + offs[0][1]].copy()
            for dy, dx in offs[1:]:
                np.maximum(best, T[yy + dy, xx + dx], out=best)
            keep[sl] = (t - best) <= self.dt_us
            T[yy, xx] = t          # update after the query: strictly causal
        return ev[keep]


class Pipeline:
    """Convenience wrapper: hot-pixel mask (fitted once) + refractory + BAF."""

    def __init__(self, shape, dt_us=5000, refractory_us=1000, hot_mask=None,
                 baf_radius=1, baf_include_self=False,
                 hot_sigma=8.0, hot_min_count=50, hot_neighbour_ratio=0.0):
        self.shape = shape
        self.baf = BackgroundActivityFilter(shape, dt_us=dt_us, radius=baf_radius,
                                            include_self=baf_include_self)
        self.refractory_us = refractory_us
        self.hot_mask = hot_mask
        self.hot_sigma = float(hot_sigma)
        self.hot_min_count = int(hot_min_count)
        self.hot_neighbour_ratio = float(hot_neighbour_ratio)

    def fit_hot_pixels(self, ev: np.ndarray):
        self.hot_mask = hot_pixel_mask(ev, self.shape, n_sigma=self.hot_sigma,
                                       min_count=self.hot_min_count,
                                       neighbour_ratio=self.hot_neighbour_ratio)
        return self

    def __call__(self, ev: np.ndarray) -> np.ndarray:
        if self.hot_mask is not None and len(ev):
            ev = apply_hot_pixel_mask(ev, self.hot_mask)
        if self.refractory_us:
            ev = refractory(ev, self.shape, self.refractory_us)
        return self.baf(ev)
